"""Download public assets and record resolved revisions (resumable HF cache)."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import MODEL, RAW_DATA, VSTAR_DIR, ROOT
from huggingface_hub import HfApi, snapshot_download
p=argparse.ArgumentParser()
p.add_argument('--only',choices=('all','model','train','eval'),default='all')
p.add_argument('--model-revision',default='main')
p.add_argument('--train-revision',default='main')
p.add_argument('--eval-revision',default='main')
a=p.parse_args()
specs=[('model','Qwen/Qwen3-VL-8B-Instruct','model',MODEL,a.model_revision,None),
       ('train','ChenShawn/DeepEyes-Datasets-47k','dataset',RAW_DATA,a.train_revision,['data_0.1.2_visual_toolbox_v2.parquet','data_v0.8_visual_toolbox_v2.parquet']),
       ('eval','craigwu/vstar_bench','dataset',VSTAR_DIR,a.eval_revision,['direct_attributes/*','relative_position/*'])]
for kind,repo,repo_type,dest,revision,patterns in specs:
    if a.only not in ('all',kind):continue
    resolved=HfApi().repo_info(repo,repo_type=repo_type,revision=revision).sha
    snapshot_download(repo,repo_type=repo_type,revision=resolved,local_dir=dest,allow_patterns=patterns)
    (ROOT/f'download_{kind}.json').write_text(json.dumps(dict(repo=repo,revision=resolved),indent=2))
    print(kind,resolved,dest)
