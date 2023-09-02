import torch
from cuda import cudart
from trt_infer import TRTInfer
from transformers import CLIPTokenizer

prompts = ["a dog under the sea"]
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
batch_encoding = tokenizer(prompts, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
input_ids = batch_encoding["input_ids"].type(torch.int32).contiguous().cuda()

clip = TRTInfer("outputs/clip_float16.trt")
clip_input_data_ptr = clip.inputs[0]["tensor"].data_ptr()
clip_input_data_size = clip.inputs[0]["size"]
cudart.cudaMemcpy(clip_input_data_ptr, input_ids.data_ptr(), clip_input_data_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
clip.infer()
print(clip.outputs[0]["tensor"].shape)
print(clip.outputs[0]["tensor"].cpu().numpy())
