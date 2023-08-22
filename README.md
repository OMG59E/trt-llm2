## Unidiffuser 模型优化加速:zap:
### [Unidiffuser: One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale](https://arxiv.org/abs/2303.06555)
[![](https://img.shields.io/badge/Github-TensorRT%20LLM-blue)](https://github.com/NVIDIA/TensorRT)
[![](https://img.shields.io/badge/Github-TensorRT-blue)](https://github.com/NVIDIA/TensorRT)
[![](https://img.shields.io/badge/%E9%98%BF%E9%87%8C%E5%A4%A9%E6%B1%A0-TensorRT%20Hackathon%202023-blue)](https://tianchi.aliyun.com/competition/entrance/532108/introduction)
[![](https://img.shields.io/badge/NVIDIA-TensorRT%20CookBook%20CN-blue)](https://github.com/NVIDIA/trt-samples-for-hackathon-cn)
[![](https://img.shields.io/badge/B%E7%AB%99-GodV%20TensorRT%E6%95%99%E7%A8%8B-blue)](https://www.bilibili.com/video/BV1jj411Z7wG/?spm_id_from=333.337.search-card.all.click&vd_source=7cd071f968d19705aeb3d6a72130d7cf)
[![](https://img.shields.io/badge/Github-Unidiffuser-blue)](https://github.com/thu-ml/unidiffuser)
### 总述

- 本次复赛选择的模型为：清华朱军团队开源Unidiffuser，首个基于Transformer的多模态扩散大模型。该论文提出了一个为多模态设计的概率建模框架 UniDiffuser，并采用该团队提出的基于 transformer 的网络架构 U-ViT，在开源的大规模图文数据集 LAION-5B 上训练了一个十亿参数量的模型，使得一个底层模型能够高质量地完成多种生成任务。如：文生图，图生文、图文联合生成、无条件图文生成、图文改写等，大幅提升文图内容的生产效率，也进一步提升了生成式模型的应用想象力。本次选择其中文生图任务进行优化加速，其原始模型的相关链接如下：

<div align=center>

|名称|参考连接|
|-|-|
|![](https://img.shields.io/badge/ICML2023-Unidiffuser-179bd3)|<https://arxiv.org/abs/2303.06555>|
|![](https://img.shields.io/badge/Github-Unidiffuser-blue)|<https://github.com/thu-ml/unidiffuser>|
|![zhihu](https://img.shields.io/badge/zhihu-知乎中文解读-179bd3)| <https://zhuanlan.zhihu.com/p/614696522>|

</div>


请简练地概括项目的主要贡献，使读者可以快速理解并复现你的工作，包括：

- 介绍本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的参赛题目（请给出上述链接），并介绍具体选题是什么（参见“选题得分”小节，应为如下之一：1，2，3，4，2+4，3+4）
    - 如果是优化新模型，原始模型的名称及链接，并对该模型做个简要介绍
- 优化效果（例如给出精度和加速比），简单给出关键的数字即可，在这里不必详细展开
- 在Docker里面代码编译、运行步骤的完整说明
  - 请做到只要逐行运行你给的命令，就能把代码跑起来

### 主要开发工作

#### 开发工作的难点

请在这一节里总结你的工作难点与亮点。
- 如果使用 TensorRT 进行优化，请介绍一下在模型在导出时、或用polygraphy/trtexec解析时，或在使用TensorRT中，遇到了什么问题并解决了。换句话说，针对这个模型，我们为什么需要额外的工程手段。
- 如果使用 TensorRT-LLM 进行优化，描述以下方面可供选手参考：如果搭建了新模型， 请介绍模型结构有无特别之处，在模型的搭建过程中使用了什么算子，有没有通过plugin支持的新算子。如果支持新feature，请介绍这个feature具体需要修改哪些模块才能实现。如果优化已有模型，请介绍模型性能瓶颈以及解决方法。另外还可以包含工程实现以及debug过程中的难点。

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

提交bug是对TensorRT/TensorRT-LLM的另一种贡献。发现的TensorRT/TensorRT-LLM或cookbook、或文档和教程相关bug，请提交到[github issues](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues)，并请在这里给出链接。  

对于每个bug，请标记上hackathon2023标签，并写好正文：

- 对于cookbook或文档和教程相关bug，说清楚问题即可，不必很详细。
- 对于TensorRT bug，首先确认在云主机上使用NGC docker + TensorRT 9.0.0.1可复现。
- 然后填写如下模板，并请导师复核确认（前面“评分标准”已经提到，确认有效可得附加分）：
  - Environment
    - TensorRT 9.0.0.1
    - Versions of CUDA, CUBLAS, CuDNN used
    - Container used
    - NVIDIA driver version
  - Reproduction Steps
    - Provide detailed reproduction steps for the issue here, including any commands run on the command line.
  - Expected Behavior
    - Provide a brief summary of the expected behavior of the software. Provide output files or examples if possible.
  - Actual Behavior
    - Describe the actual behavior of the software and how it deviates from the expected behavior. Provide output files or examples if possible.
  - Additional Notes
    - Provide any additional context here you think might be useful for the TensorRT team to help debug this issue (such as experiments done, potential things to investigate).

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