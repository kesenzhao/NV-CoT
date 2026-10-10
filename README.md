# Thinking with Images as Continuous Actions: Numerical Visual Chain-of-Thought



## 更新

* 我们支持了Qwen3-VL-2B和Qwen3-Vl-8B
* 我们优化了架构，支持调用api作为判分模型，提高了训练效率。
* 我们提供了更多数值头方案，支持上下文输入。


## 1. 环境配置

```bash
conda create -n deepeyes-joint python=3.10 -y
conda activate deepeyes-joint
cd /path/to/DeepEyes-Joint-GRPO-OpenSource
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install packaging ninja wheel psutil
MAX_JOBS=4 python -m pip install flash-attn==2.7.0.post2 --no-build-isolation
python -c 'import torch, transformers, flash_attn; print(torch.__version__, transformers.__version__, torch.cuda.device_count())'
python -m unittest discover -s . -p 'test_*.py' -v
```

## 2. 下载 checkpoint 和数据集

公开来源：

- 基础模型：
  - [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
  - [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- 训练数据：[ChenShawn/DeepEyes-Datasets-47k](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k/tree/main)。
- 评估数据：[craigwu/vstar_bench](https://huggingface.co/datasets/craigwu/vstar_bench/tree/main)。
- ckpt：
  - [NVCoT-Qwen3-VL-2B](https://huggingface.co/kesenZhaoNTU/NVCoT-Qwen3-VL-2B-Joint-GRPO-Step32)
  - [NVCoT-Qwen3-VL-8B](https://huggingface.co/kesenZhaoNTU/NVCoT-Qwen3-VL-8B-Joint-GRPO-Step32)

下载到共享目录：

```bash
python scripts/download.py
# 或分开下载；中断后可重跑相同命令
python scripts/download.py --only model
python scripts/download.py --only train
python scripts/download.py --only eval
```

## 3. 配置训练奖励判分服务

调用api 判分

- `JUDGE_BACKEND=openai`：OpenAI-compatible `/chat/completions`，`JUDGE_BASE_URL` 包含 `/v1`。
- `JUDGE_BACKEND=gemini`：Bearer 认证的 Gemini-compatible `/models/{model}:generateContent`，base URL 包含 `/v1beta`

在两台机器设置同一服务和模型，并提供密钥：

```bash
# 其余非密钥配置写在 config.env 中
read -rsp 'Judge API key: ' JUDGE_API_KEY; echo
export JUDGE_API_KEY
# node 0 运行一次真实预检（会调用服务）
python probe_judge.py
```

原实验服务的模型标识为 `gemini-3.8-flash-high`；判分失败会重试并最终报错，不会用假奖励继续训练。

3.1 可选：自部署 Qwen2.5-72B-Instruct 判分服务

还提供 72B judge 路线，模型为
[Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)，
具体参考 [github.com/Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes)。

在 judge 机器单独创建环境，避免 vLLM 修改训练环境依赖：

```bash
conda create -n deepeyes-judge72b python=3.10 -y
conda activate deepeyes-judge72b
python -m pip install vllm==0.8.2
export JUDGE_MODEL_DIR=/shared/deepeyes/assets/Qwen2.5-72B-Instruct
huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir "$JUDGE_MODEL_DIR"
# 已有完整权重可直接将 JUDGE_MODEL_DIR 指向它，无需下载
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/start_judge_72b.sh
```

## 4. 正式训练入口

```bash
# node 0
python scripts/pipeline.py --rank 0 --id run001 --until 128
# node 1（相同的新 id，同时启动）
python scripts/pipeline.py --rank 1 --id run001 --until 128
```

## 5. 来源与许可证

保留原工程 Apache-2.0 `LICENSE`，代码来源和导出修改说明见 `THIRD_PARTY_NOTICES.md`。感谢 [DeepEyes](https://github.com/Visual-Agent/DeepEyes)、[verl](https://github.com/volcengine/verl) 和 Qwen。模型、数据集需分别遵循各自来源的许可。


## 6. 引用

```
@article{zhao2026thinking,
  title={Thinking with Images as Continuous Actions: Numerical Visual Chain-of-Thought},
  author={Zhao, Kesen and Zhu, Beier and Zhou, Junbao and Zhu, Xingyu and Yue, Zhongqi and Zhang, Hanwang},
  journal={arXiv preprint arXiv:2602.23959},
  year={2026}
}
```
