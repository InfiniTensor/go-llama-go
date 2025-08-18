# Go, Llama, go!

本仓库为 2025 年夏季 InfiniTensor 大模型与人工智能系统训练营 Triton & 九齿方向专业阶段的作业。

## 任务说明

本仓库含有 `llama.py` 和 `infer.py` 两个主要文件，其中 `llama.py` 写有一个基础的 Llama 3 模型架构，`infer.py` 是其推理启动脚本。可以使用以下命令查询 `infer.py` 的使用方法：

```shell
python infer.py --help
```

例如，我们可以通过以下命令，使用位于 `/data/shared/models/` 的 `Llama-3.2-1B` 模型推理两段 prompt。其中我们设定了推理最多只能生成 `64` 个新 token，并且运行在 `cuda` 设备上。该推理会总共进行 `4` 次，其中 `1` 次作为预热，其余 `3` 次用于性能测量。

```shell
python infer.py --model /data/shared/models/Llama-3.2-1B/ --prompts "The emergence of deep learning domain-specific languages (DSLs) has substantially reduced the obstacles in developing high-performance, cross-platform compute kernels, but current DSLs" "Driven by recent advancements in the AI industry, the AI accelerator sector has increasingly diversified, with vendors developing their own hardware architectures and programming models, such as NVIDIA" --max-new-tokens 64 --device cuda --num-warmup-iterations 1 --num-profiling-iterations 3
```

以上命令运行结束后将打印出 JSON 格式的输出，其中 `"texts"` 里会存放所有输出的文本，`"num_tokens_per_second"` 会存放测量出的性能。

我们的任务就是在保持推理有序进行的同时，加快其进度。具体而言，即是保持 `"texts"` 一定程度上符合逻辑的同时，增大 `"num_tokens_per_second"`。

注：服务器可能使用 [Slurm](https://slurm.schedmd.com/documentation.html) 等工作负载管理器，所以可能需要在指令前方加入 `srun --cpus-per-task=16 --mem=64G --gres=gpu:1` 等以完成运行。

## 限制条件

由于本方向为 Triton & 九齿，所以优化应当全部来源于对 Triton 或者九齿的使用。像是使用 `torch.nn.functional.rms_norm` 直接替换 `RMSNorm.forward` 中的内容，或者一些系统层面的修改，如加入 KV cache 等，将视为违规。但是使用 Triton 或者九齿进行算子融合等优化，从而替换模型中单个或多个结构是被允许的。

## 评分方式

作业将通过两阶段筛选机制进行评审：

1. 基础要求：性能提升数值达到 80%，且生成出的文本均保持相当的逻辑水平。
2. 进阶筛选：在满足基础要求的提交中，性能提升数值不低于平均值减去两个标准差。

## 答疑环节

如果有任何问题，像是某项改动是否允许，或者对评分标准有疑惑，欢迎在群里进行咨询。
