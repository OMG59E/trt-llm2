## Unidiffuser 模型优化加速:zap:
### [Unidiffuser: One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale](https://arxiv.org/abs/2303.06555)
[![](https://img.shields.io/badge/Github-TensorRT%20LLM-blue)](https://github.com/NVIDIA/TensorRT)
[![](https://img.shields.io/badge/Github-TensorRT-blue)](https://github.com/NVIDIA/TensorRT)
[![](https://img.shields.io/badge/%E9%98%BF%E9%87%8C%E5%A4%A9%E6%B1%A0-TensorRT%20Hackathon%202023-blue)](https://tianchi.aliyun.com/competition/entrance/532108/introduction)
[![](https://img.shields.io/badge/NVIDIA-TensorRT%20CookBook%20CN-blue)](https://github.com/NVIDIA/trt-samples-for-hackathon-cn)
[![](https://img.shields.io/badge/B%E7%AB%99-GodV%20TensorRT%E6%95%99%E7%A8%8B-blue)](https://www.bilibili.com/video/BV1jj411Z7wG/?spm_id_from=333.337.search-card.all.click&vd_source=7cd071f968d19705aeb3d6a72130d7cf)
[![](https://img.shields.io/badge/Github-Unidiffuser-blue)](https://github.com/thu-ml/unidiffuser)
### 总述

- 本次复赛选择的模型为：清华朱军团队开源Unidiffuser，首个基于Transformer的多模态扩散大模型。该论文提出了一个为多模态设计的概率建模框架 UniDiffuser，并采用该团队提出的基于 transformer 的网络架构 U-ViT，在开源的大规模图文数据集 LAION-5B 上训练了一个十亿参数量的模型，使得一个底层模型能够高质量地完成多种生成任务。如：文生图，图生文、图文联合生成、无条件图文生成、图文改写等，大幅提升文图内容的生产效率，也进一步提升了生成式模型的应用想象力。本次选择其中"文生图"任务进行优化加速，至于选择该模型的主要原因没有大语言模型的相关概念，而该模型与初赛模型相类似更易入手。其原始模型的相关链接如下：

<div align=center>

|名称|参考连接|
|-|-|
|![](https://img.shields.io/badge/ICML2023-Unidiffuser-179bd3)|<https://arxiv.org/abs/2303.06555>|
|![](https://img.shields.io/badge/Github-Unidiffuser-blue)|<https://github.com/thu-ml/unidiffuser>|
|![zhihu](https://img.shields.io/badge/zhihu-知乎中文解读-179bd3)| <https://zhuanlan.zhihu.com/p/614696522>|

</div>

<div align=center>

|Model|PyTorch|FP32|FP16|
|-|-|-|-|
|clip + captution_encoder|6.891| |1.650|
|uvit|213.541| |88.519|
|decoder|334.181| |47.523|
|pipeline|11018.846||4475.347|

</div>

请简练地概括项目的主要贡献，使读者可以快速理解并复现你的工作，包括：

- 介绍本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的参赛题目（请给出上述链接），并介绍具体选题是什么（参见“选题得分”小节，应为如下之一：1，2，3，4，2+4，3+4）
    - 如果是优化新模型，原始模型的名称及链接，并对该模型做个简要介绍
- 优化效果（例如给出精度和加速比），简单给出关键的数字即可，在这里不必详细展开
- 在Docker里面代码编译、运行步骤的完整说明

```shell
git clone https://gitee.com/xingwg/trt-llm2.git
cd trt-llm2/examples/unidiffuser/trt
python hf_unidiffuser_convert.py   # 预先将模型下载到trt-llm2/examples/unidiffuser/pytorch/models
python build_clip.py   # 编译CLIP fp16
python build_uvit.py  # 编译UViT fp16
cd plugin && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release ..
cd ../../ && python build_decoder.py
python run.py
```

### 主要开发工作

Unidiffuser模型的"文生图"任务共包含4个模型，分别是:

- clip - [https://huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)(代码自行下载，无需手动)
- caption_decoder - [https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/caption_decoder.pth](https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/caption_decoder.pth)
- uvit - [https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/uvit_v1.pth](https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/uvit_v1.pth)
- autoencoder - [https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/autoencoder_kl.pth](https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/autoencoder_kl.pth)

"文生图"任务的整个流程包括三步如下：

<div align=center>

<img src=docs/whiteboard_exported_image.png width=70% />

</div>

#### 开发工作的难点

本次通过TRT-LLM手动搭建了三个模型分别是CLIP(内联了caption_decoder部分)、UViT、以及autoencoder中的decoder部分，三个模型的主要主要算子是embedding、attention、layernorm、mlp、linear、groupnorm构成。

- 对于没有经验的玩家和LLM模型处理经验的，摸索TensorRT-LLM并应用本身就有点难度。
- 开发过程中出现精度误差时，需要逐层手动mark，进行比对定位，比较麻烦和耗时，大多时间消耗在这里。
- 通过plugin支持解决fp16精度下groupnorm精度损失大的问题。

### 开发与优化过程

这一部分是报告的主体。请把自己假定为老师，为 TensorRT 或 TensorRT-LLM 的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的 TensorRT 或 TensorRT-LLM 模型。或者你是如何一步步通过修改哪些模块添加了新feature的。

建议：

- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

### 优化效果

这一部分介绍你的工作在云主机上的运行效果。如果是优化模型，需要分两部分说明：

- 精度：报告与原始模型进行精度对比测试的结果，验证精度达标。
  - 如果选用TensorRT-LLM，请跑summarize任务并使用 [Rouge](https://huggingface.co/spaces/evaluate-metric/rouge) 来对比模型优化前后的精度差距。如果精度良好，原始模型与优化模型的Rouge score的差异一般在1以内。例子见 TensorRT-LLM docker 中 /root/workspace/tensorrt_llm_july-release-v1/examples/gpt/summarize.py
  - 如果选用TensorRT，这里的精度测试指的是针对“原始模型”和“TensorRT优化模型”分别输出的数据（tensor）进行数值比较。请给出绝对误差和相对误差的统计结果（至少包括最大值、平均值与中位数）。
    - 使用训练好的权重和有意义的输入数据更有说服力。如果选手使用了随机权重和输入数据，请在这里注明。
    - 在精度损失较大的情况下，鼓励选手用训练好的权重和测试数据集对模型优化前与优化后的准确度指标做全面比较，以增强说服力。
- 性能：例如可以用图表展示不同batch size或sequence length下性能加速效果（考虑到可能模型可能比较大，可以只给batch size为1的数据）
  - 一般用原始模型作为baseline
  - 一般提供模型推理时间的加速比即可；若能提供压力测试下的吞吐提升则更好。

请注意：

- 相关测试代码也需要包含在代码仓库中，可被复现。
- 请写明云主机的软件硬件环境，方便他人参考。

### Bug报告（可选）

暂无

### 送分题答案（可选）

- 送分题1

```bash
root@trt2023:~/workspace/tensorrt_llm_july-release-v1/examples/gpt# python3 run.py --max_output_len=8
Input: Born in north-east France, Soyer trained as a
Output:  chef and eventually became a chef at a
root@trt2023:~/workspace/tensorrt_llm_july-release-v1/examples/gpt#
```

- 送分题2

```bash
[08/22/2023-09:26:59] [TRT-LLM] [I] TensorRT-LLM (total latency: 3.029069662094116 sec)
[08/22/2023-09:26:59] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[08/22/2023-09:26:59] [TRT-LLM] [I]   rouge1 : 21.869322054781037
[08/22/2023-09:26:59] [TRT-LLM] [I]   rouge2 : 6.258925475911645
[08/22/2023-09:26:59] [TRT-LLM] [I]   rougeL : 16.755771650012953
[08/22/2023-09:26:59] [TRT-LLM] [I]   rougeLsum : 18.68034777724496
[08/22/2023-09:26:59] [TRT-LLM] [I] Hugging Face (total latency: 14.837929248809814 sec)
[08/22/2023-09:26:59] [TRT-LLM] [I] HF beam 0 result
[08/22/2023-09:27:00] [TRT-LLM] [I]   rouge1 : 18.182978950152904
[08/22/2023-09:27:00] [TRT-LLM] [I]   rouge2 : 5.166241888544473
[08/22/2023-09:27:00] [TRT-LLM] [I]   rougeL : 14.851620358520162
[08/22/2023-09:27:00] [TRT-LLM] [I]   rougeLsum : 16.95757748412272
```

### 经验与体会（可选）

欢迎在这里总结经验，抒发感慨。