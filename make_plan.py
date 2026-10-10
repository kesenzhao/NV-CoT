"""Freeze settings before any RL optimization or benchmark measurement."""
import hashlib
import json
import os
from pathlib import Path
from reward import REWARD_SPEC

from config import ROOT, CODE, MODEL

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()

def atomic_json(path,obj):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

assert not (ROOT/'plan.json').exists()
data=json.loads((ROOT/'data/manifest.json').read_text())
assert data['passed'] and data['train_sha256']==sha(ROOT/'data/train.jsonl')
assert json.loads((ROOT/'smoke.json').read_text())['passed']
assert json.loads((ROOT/'judge_preflight.json').read_text())['passed']
plan=dict(schema_version=1,vlm_frozen=False,independent_roi_encoding=False,reference_model=None,
    seed=20261003,vlm_lr=1e-6,head_lr=1e-5,rollouts_per_prompt=4,prompts_per_step=16,
    temperature=.8,top_p=1.,top_k=0,train_max_pixels=1048576,train_max_turn_tokens=768,
    train_max_context=12288,max_turns=5,max_crops=3,action_raw_std=.15,max_residual=.15,
    min_crop_size=.02,gradient_clip=1.,weight_decay=0.,ppo_epochs=1,clip=.2,kl_coefficient=0.,
    reward='0.8*API_binary_correctness -0.2*protocol_failure +1.2*correct_executed_tool',
    reward_spec=REWARD_SPEC,
    reward_api_model=json.loads((ROOT/'judge_preflight.json').read_text())['model'],
    effective_group_policy='Retry fresh16-question batches up to4times only when ALL groups have zero variance; never fake a step',
    train_sha256=data['train_sha256'],train_data_manifest_sha256=sha(ROOT/'data/manifest.json'),
    base_model=str(MODEL),
    initial_head='fresh seeded random; final projection normal(std=0.001), bias=0; seed=20261003',
    initial_head_sha256=None,
    target_steps=list(range(32,129,32)),eval_steps=list(range(0,129,32)),world_size=16,
    precision='BF16 model weights/gradients with rank0 CPU-offloaded FP32 master weights and unbiased stochastic BF16 rounding, BF16 autocast, gradient checkpointing',
    optimizer='CPUOffloadAdafactor: FP32 master, stochastic BF16 rounding, beta2_decay=-0.8, eps=(None,0.001), d=1.0',
    code_sha256={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*') if p.suffix in ('.py','.sh') and p.is_file() and 'runs' not in p.relative_to(CODE).parts},
    note='Fresh online rollouts from current in-process VLM and head every update. No frozen feature cache. '
         'Manual synchronous gradient averaging is equivalent to data parallelism and supports unequal tool-call counts. '
         'Single on-policy pass: text old logprobs recomputed on unchanged behavior weights; ratio starts1 but gradients flow. '
         'Full-model+head+optimizer+per-rank RNG checkpoints; fixed32-step evaluation intervals before each resume. '
         'V* labels are evaluation-only; all191cases,greedy,strict protocol,errors count wrong.')
atomic_json(ROOT/'plan.json',plan);print('PLAN_READY',sha(ROOT/'plan.json'),flush=True)
