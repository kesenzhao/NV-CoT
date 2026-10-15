# [NeurIPS 2026 Oral] Thinking with Images as Continuous Actions: Numerical Visual Chain-of-Thought

<p align="center">
  <strong>NeurIPS 2026 Oral &middot; Top 0.4%</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.23959"><img src="https://img.shields.io/badge/arXiv-2602.23959-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv paper" height="20"></a>
  <a href="https://kesenzhao.github.io/NV-CoT/"><img src="https://img.shields.io/badge/Project-Page-0078D4.svg?logo=googlechrome&logoColor=white" alt="Project Page" height="20"></a>
  <a href="https://huggingface.co/kesenZhaoNTU/NVCoT-Qwen3-VL-2B-Joint-GRPO-Step32"><img src="https://img.shields.io/badge/Hugging%20Face-2B-FFD21E.svg?logo=huggingface&logoColor=black" alt="Hugging Face 2B model" height="20"></a>
  <a href="https://huggingface.co/kesenZhaoNTU/NVCoT-Qwen3-VL-8B-Joint-GRPO-Step32"><img src="https://img.shields.io/badge/Hugging%20Face-8B-FFD21E.svg?logo=huggingface&logoColor=black" alt="Hugging Face 8B model" height="20"></a>
</p>

## Updates

- 🎉 NV-CoT has been accepted to **NeurIPS 2026 as an Oral presentation (Top 0.4%)**!
- Added support for Qwen3-VL-2B and Qwen3-VL-8B.
- Optimized the architecture to support API-based judge models, improving training efficiency.
- Added more numerical-head designs with support for contextual inputs.

## 1. Environment Setup

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

## 2. Download Checkpoints and Datasets

Public sources:

- Base models:
  - [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
  - [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- Training data: [ChenShawn/DeepEyes-Datasets-47k](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k/tree/main).
- Evaluation data: [craigwu/vstar_bench](https://huggingface.co/datasets/craigwu/vstar_bench/tree/main).
- Checkpoints:
  - [NVCoT-Qwen3-VL-2B](https://huggingface.co/kesenZhaoNTU/NVCoT-Qwen3-VL-2B-Joint-GRPO-Step32)
  - [NVCoT-Qwen3-VL-8B](https://huggingface.co/kesenZhaoNTU/NVCoT-Qwen3-VL-8B-Joint-GRPO-Step32)

Download them to a shared directory:

```bash
python scripts/download.py
# Alternatively, download each component separately. You can rerun the same command after an interruption.
python scripts/download.py --only model
python scripts/download.py --only train
python scripts/download.py --only eval
```

## 3. Configure the Reward Judge Service for Training

Use an API-based judge:

- `JUDGE_BACKEND=openai`: an OpenAI-compatible `/chat/completions` endpoint; `JUDGE_BASE_URL` must include `/v1`.
- `JUDGE_BACKEND=gemini`: a Gemini-compatible `/models/{model}:generateContent` endpoint using Bearer authentication; the base URL must include `/v1beta`.

Configure the same service and model on both machines, and provide the API key:

```bash
# Store all non-secret settings in config.env.
read -rsp 'Judge API key: ' JUDGE_API_KEY; echo
export JUDGE_API_KEY
# Run one real preflight check on node 0. This will call the service.
python probe_judge.py
```

The model identifier used by the original experiments was `gemini-3.8-flash-high`. If judging fails, the request will be retried and eventually raise an error; training will not continue with fabricated rewards.

### 3.1 Optional: Self-Hosted Qwen2.5-72B-Instruct Judge Service

A 72B judge option is also provided, using
[Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct).
For details, see [github.com/Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes).

Create a separate environment on the judge machine to prevent vLLM from modifying dependencies in the training environment:

```bash
conda create -n deepeyes-judge72b python=3.10 -y
conda activate deepeyes-judge72b
python -m pip install vllm==0.8.2
export JUDGE_MODEL_DIR=/shared/deepeyes/assets/Qwen2.5-72B-Instruct
huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir "$JUDGE_MODEL_DIR"
# If the complete model weights are already available, point JUDGE_MODEL_DIR to them instead; no download is required.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/start_judge_72b.sh
```

## 4. Training Entry Point

```bash
# Node 0
python scripts/pipeline.py --rank 0 --id run001 --until 128
# Node 1 (start it at the same time with the same new ID)
python scripts/pipeline.py --rank 1 --id run001 --until 128
```

## 5. Sources and License

The original project's Apache-2.0 `LICENSE` is retained. See `THIRD_PARTY_NOTICES.md` for code provenance and a description of the exported modifications. We thank [DeepEyes](https://github.com/Visual-Agent/DeepEyes), [verl](https://github.com/volcengine/verl), and Qwen. Models and datasets remain subject to the licenses of their respective sources.

## 6. Citation

```
@article{zhao2026thinking,
  title={Thinking with Images as Continuous Actions: Numerical Visual Chain-of-Thought},
  author={Zhao, Kesen and Zhu, Beier and Zhou, Junbao and Zhu, Xingyu and Yue, Zhongqi and Zhang, Hanwang},
  journal={arXiv preprint arXiv:2602.23959},
  year={2026}
}
```
