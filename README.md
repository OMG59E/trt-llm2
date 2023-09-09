## Unidiffuser 模型优化加速:zap:
### [Unidiffuser: One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale](https://arxiv.org/abs/2303.06555)
[![](https://img.shields.io/badge/Github-TensorRT%20LLM-blue)](https://github.com/NVIDIA/TensorRT)
[![](https://img.shields.io/badge/Github-TensorRT-blue)](https://github.com/NVIDIA/TensorRT)
[![](https://img.shields.io/badge/%E9%98%BF%E9%87%8C%E5%A4%A9%E6%B1%A0-TensorRT%20Hackathon%202023-blue)](https://tianchi.aliyun.com/competition/entrance/532108/introduction)
[![](https://img.shields.io/badge/NVIDIA-TensorRT%20CookBook%20CN-blue)](https://github.com/NVIDIA/trt-samples-for-hackathon-cn)
[![](https://img.shields.io/badge/B%E7%AB%99-GodV%20TensorRT%E6%95%99%E7%A8%8B-blue)](https://www.bilibili.com/video/BV1jj411Z7wG/?spm_id_from=333.337.search-card.all.click&vd_source=7cd071f968d19705aeb3d6a72130d7cf)
[![](https://img.shields.io/badge/Github-Unidiffuser-blue)](https://github.com/thu-ml/unidiffuser)
### 总述

本次复赛选择的模型为：清华朱军团队开源Unidiffuser，首个基于Transformer的多模态扩散大模型。该论文提出了一个为多模态设计的概率建模框架 UniDiffuser，并采用该团队提出的基于 transformer 的网络架构 U-ViT，在开源的大规模图文数据集 LAION-5B 上训练了一个十亿参数量的模型，使得一个底层模型能够高质量地完成多种生成任务。如：文生图，图生文、图文联合生成、无条件图文生成、图文改写等，大幅提升文图内容的生产效率，也进一步提升了生成式模型的应用想象力。本次选择其中"文生图"任务进行优化加速，至于选择该模型的主要原因没有大语言模型的相关概念，而该模型与初赛模型相类似更易入手。其原始模型的相关链接如下：

<div align=center>

|名称|参考连接|
|-|-|
|![](https://img.shields.io/badge/ICML2023-Unidiffuser-179bd3)|<https://arxiv.org/abs/2303.06555>|
|![](https://img.shields.io/badge/Github-Unidiffuser-blue)|<https://github.com/thu-ml/unidiffuser>|
|![zhihu](https://img.shields.io/badge/zhihu-知乎中文解读-179bd3)| <https://zhuanlan.zhihu.com/p/614696522>|

</div>

经过**fp16 + batch + cudaGraph**优化后，最终获得各模型加速比分别为**CLIP**部分约**4.2**倍、**UViT**部分**2.4**倍、**Decoder**部分**4.7**倍，整个**pipeline**加速比约为**2.4**倍；同时使用初赛的**PD**评估方法，测试40个prompt + seed得到平均PD得分**6.883**。

**编译、运行、测试步骤如下**：

```shell
git clone https://gitee.com/xingwg/trt-llm2.git
cd trt-llm2/examples/unidiffuser/pytorch
# 预先将模型下载到trt-llm2/examples/unidiffuser/pytorch/models，下载地址见“主要开发工作部分”
python simplest_text2img_pipeline.py  # 执行torch pipeline, 同时生成40张参考图片
cd ../trt
python hf_unidiffuser_convert.py   
python build_clip.py   # 编译CLIP fp16
python build_uvit.py  # 编译UViT fp16
cd plugin && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release ..  # 编译groupnorm插件
cd ../../ && python build_decoder.py  # 编译 decoder fp16
python run.py   # 执行trt pipeline, 同时生成40张测试图片
python compute_score.py  # 进行PD评估
```

### 主要开发工作

Unidiffuser模型的"文生图"任务共包含4个模型，分别是:

- **clip** - [https://huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)(代码自行下载，无需手动)
- **caption_decoder** - [https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/caption_decoder.pth](https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/caption_decoder.pth)
- **uvit** - [https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/uvit_v1.pth](https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/uvit_v1.pth)
- **autoencoder** - [https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/autoencoder_kl.pth](https://huggingface.co/thu-ml/unidiffuser-v1/blob/main/autoencoder_kl.pth)

"文生图"任务的整个流程包括三步如下：

<div align=center>

<img src=docs/whiteboard_exported_image.png width=70% />

</div>

- 通过TRT-LLM手动搭建了三个模型分别是CLIP(内联了caption_decoder部分)、UViT、以及autoencoder中的decoder部分，三个模型的主要算子是**embedding、attention、layernorm、mlp、linear、groupnorm**构成。

- 对于没有经验的玩家和LLM模型处理经验的，摸索TensorRT-LLM并应用本身就有点难度。
- 开发过程中出现精度误差时，需要逐层手动mark，进行比对定位，比较麻烦和耗时，大多时间消耗在这里。
- 通过plugin支持解决fp16精度下groupnorm精度损失大的问题。

### 开发与优化过程

- 首先需要了解原模型整个“文生图”任务的pipeline，本次比赛通过简化原代码最终获得了最简化的pipeline，见代码**unidiffuser/pytorch/simplest_text2img_pipeline.py**，同时实现onnx模型导出的代码，方便查看模型graph，见代码**unidiffuser/pytorch/export_onnx.py**

- 通过对简化原模型的pipeline，对模型有了初步了解，接下来通过TensorRT-LLM的API手动搭建模型，在搭建模型之前可以通过TensorRT-LLM自带的示例对其进行初步的摸索。TensorRT-LLM实现一个模型需要三步，**第一步手动搭建模型**，**第二步转换保存原模型权重**，**第三步构建模型**。

- 整个pipeline共三个模型CLIP、UViT、Decoder，手搓后的模型见**trt/models/clip.py**、**trt/models/uvit.py**、**trt/models/decoder.py**。
TensorRT-LLM的主要主要算子实现在**tensorrt_llm.layers**、以及**tensorrt_llm.functional**搭建模型较为常用，这里有坑**模型构造函数中不能存在constant tensor**。模型构建完成后，需要转换保存权重，这一步较为简单，见代码**trt/hf_unidiffuser_convert.py**，可能下载模型会有科学上网问题。

- 接下来逐模型尝试fp32构建模型和原模型进行校验，校验方式可以dump出文件，比较两者差值的最大最小值，当然最简单的是直接print，但容易观察不到，带偏方向。其中CLIP较为顺利，fp32/fp16均未遇到精度问题；UViT遇到精度问题，通过debug发现属于手搓过程的错误，debug的方法是**逐层mark_output**进行比对定位，最终fp32/fp16均无精度问题；Decoder在fp32下构建精度无问题，fp16精度误差很大，通过debug定位后，发现是groupnorm产生的误差，通过增加groupNormPlugin解决精度问题，同时减少了reformatting操作减少了decoder推理时延，plugin实现见目录**trt/plugin**，以及文件**trt/plugin.py**，插件实现来自TensorRT-8.5，这里就不展开介绍了。

- 观察uvit的模型结构发现，其中可以优化合并两个推理分支进行batch，提高GPU利用效率和显存利用效率。

<div align=center>

<img src=docs/uvit.png width=70% />

</div>



这一部分是报告的主体。请把自己假定为老师，为 TensorRT 或 TensorRT-LLM 的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的 TensorRT 或 TensorRT-LLM 模型。或者你是如何一步步通过修改哪些模块添加了新feature的。

建议：

- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

### 优化效果

<div align=center>

|Model|PyTorch-FP32|PyTorch-FP16(baseline, uvit-fp16)|TRT-FP32 + CudaGraph|TRT-FP16|TRT-FP16 + CudaGraph|
|-|-|-|-|-|-|
|clip + captution_encoder|6.996|6.936|2.468|2.135|1.631|
|uvit|270.409|211.132|254.555|103.029|88.406|
|decoder|258.488|223.529|137.656|53.686|47.137|
|pipeline|13786.638|10787.720|12868.122|5207.594|4469.315|

</div>

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