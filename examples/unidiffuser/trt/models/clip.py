from collections import OrderedDict, UserDict
from dataclasses import fields, dataclass
from typing import Tuple, Any
import numpy as np
import tensorrt as trt
from tensorrt_llm.parameter import Tensor, constant
from tensorrt_llm.functional import slice, matmul, softmax, ACT2FN
from tensorrt_llm.layers import LayerNorm, Embedding, Linear
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm._utils import np_dtype_to_trt


class ModelOutput(OrderedDict):
    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not isinstance(first_field, Tensor):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
    

@dataclass
class BaseModelOutput(ModelOutput):
    last_hidden_state: Tensor = None
    hidden_states: Tuple[Tensor] = None
    attentions: Tuple[Tensor] = None
    
    
@dataclass
class BaseModelOutputWithPooling(ModelOutput):
    last_hidden_state: Tensor = None
    pooler_output: Tensor = None
    hidden_states: Tuple[Tensor] = None
    attentions: Tuple[Tensor] = None
    
    
class CLIPTextEmbeddings(Module):
    def __init__(self, hidden_size, vocab_size, max_position_embeddings, dtype):
        super().__init__()
        embed_dim = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.token_embedding = Embedding(vocab_size, embed_dim, dtype=dtype)
        self.position_embedding = Embedding(max_position_embeddings, embed_dim, dtype=dtype)

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None) -> Tensor:
        seq_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[0]
        if position_ids is None:
            position_ids = slice(constant(np.expand_dims(np.arange(self.max_position_embeddings), axis=0).astype(np.int32)), starts=[0, 0], sizes=[1, seq_length])
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return embeddings


class CLIPAttention(Module):
    def __init__(self, hidden_size, num_attention_heads, dtype):
        super().__init__()
        self.dtype = dtype
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).")
        self.scale = self.head_dim**-0.5

        self.k_proj = Linear(self.embed_dim, self.embed_dim, dtype=self.dtype)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, dtype=self.dtype)
        self.q_proj = Linear(self.embed_dim, self.embed_dim, dtype=self.dtype)
        self.out_proj = Linear(self.embed_dim, self.embed_dim, dtype=self.dtype)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return tensor.view(shape=[bsz, seq_len, self.num_heads, self.head_dim]).transpose(1, 2)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor = None, 
                causal_attention_mask: Tensor = None, output_attentions = False):
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = [bsz * self.num_heads, -1, self.head_dim]
        query_states = self._shape(query_states, tgt_len, bsz).view(shape=proj_shape)
        key_states = key_states.view(shape=proj_shape)
        value_states = value_states.view(shape=proj_shape)

        src_len = key_states.shape[1]
        attn_weights = matmul(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}")

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}")
            attn_weights = attn_weights.view(shape=[bsz, self.num_heads, tgt_len, src_len]) + causal_attention_mask
            attn_weights = attn_weights.view(shape=[bsz * self.num_heads, tgt_len, src_len])

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights.view(shape=[bsz, self.num_heads, tgt_len, src_len]) + attention_mask
            attn_weights = attn_weights.view(shape=[bsz * self.num_heads, tgt_len, src_len])

        attn_weights = softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(shape=[bsz, self.num_heads, tgt_len, src_len])
            attn_weights = attn_weights_reshaped.view(shape=[bsz * self.num_heads, tgt_len, src_len])
        else:
            attn_weights_reshaped = None
        attn_output = matmul(attn_weights, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}")

        attn_output = attn_output.view(shape=[bsz, self.num_heads, tgt_len, self.head_dim])
        attn_output = attn_output.transpose(dim0=1, dim1=2)
        attn_output = attn_output.view(shape=[bsz, tgt_len, embed_dim])
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped


class CLIPMLP(Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, dtype):
        super().__init__()
        self.dtype = dtype
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = Linear(hidden_size, intermediate_size, dtype=self.dtype)
        self.fc2 = Linear(intermediate_size, hidden_size, dtype=self.dtype)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
    
class CLIPEncoderLayer(Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, hidden_act, dtype):
        super().__init__()
        self.dtype = dtype
        self.embed_dim = hidden_size
        self.self_attn = CLIPAttention(self.embed_dim, num_attention_heads, self.dtype)
        self.layer_norm1 = LayerNorm(self.embed_dim, dtype=self.dtype)
        self.mlp = CLIPMLP(self.embed_dim, intermediate_size, hidden_act, self.dtype)
        self.layer_norm2 = LayerNorm(self.embed_dim, dtype=self.dtype)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, causal_attention_mask: Tensor, output_attentions = False):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, 
                                                     causal_attention_mask=causal_attention_mask, output_attentions=output_attentions)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    

class CLIPEncoder(Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, 
                 hidden_act, num_hidden_layers, output_attentions, output_hidden_states, use_return_dict, dtype):
        super().__init__()
        self.use_return_dict = use_return_dict
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.layers = ModuleList(
            [CLIPEncoderLayer(hidden_size, num_attention_heads, intermediate_size, hidden_act, dtype=dtype) for _ in range(num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds, attention_mask: Tensor = None, causal_attention_mask: Tensor = None, 
                output_attentions: bool = None, output_hidden_states: bool = None, return_dict: bool = None):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states, attention_mask, causal_attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class CLIPTextTransformer(Module):
    def __init__(self, hidden_size, vocab_size, max_position_embeddings, num_attention_heads, intermediate_size, 
                 hidden_act, num_hidden_layers, output_attentions, output_hidden_states, use_return_dict, np_dtype):
        super().__init__()
        embed_dim = hidden_size
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        self.np_dtype = np_dtype
        self.dtype = np_dtype_to_trt(np_dtype)
        self.embeddings = CLIPTextEmbeddings(hidden_size, vocab_size, max_position_embeddings, self.dtype)
        self.encoder = CLIPEncoder(hidden_size, num_attention_heads, intermediate_size, 
                 hidden_act, num_hidden_layers, output_attentions, output_hidden_states, use_return_dict, self.dtype)
        self.final_layer_norm = LayerNorm(embed_dim, dtype=self.dtype)

    def forward(self, input_ids: Tensor = None, attention_mask: Tensor = None, position_ids: Tensor = None, 
                output_attentions: bool = None, output_hidden_states: bool = None, return_dict: bool = None):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        bsz, seq_len = input_ids.size()
        input_ids = input_ids.view(shape=[bsz, seq_len])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        return last_hidden_state

    def _build_causal_attention_mask(self, bsz, seq_len):        
        mask = np.empty(shape=[bsz, seq_len, seq_len], dtype=np.float32)
        # mask.fill(np.finfo(np.float32).min)
        mask.fill(float("-inf"))
        mask = np.triu(mask, k=1)
        mask = np.expand_dims(mask, axis=1)
        return constant(mask)
        