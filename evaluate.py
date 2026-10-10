"""Greedy V* with updated VLM AND head; weights verified before generation."""
import argparse
import json
import re
import time
from PIL import Image
import torch
from policy import *
from rollout import sample,public_trace
from iou_metrics import localization

def main():
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=tuple(range(0,129,32)),required=True)
    p.add_argument('--shard',type=int,required=True);a=p.parse_args();assert 0<=a.shard<16
    checkpoint=ROOT/'checkpoints'/f'step{a.step}' if a.step else None
    checkpoint_hash=None
    if checkpoint:
        manifest=json.loads((checkpoint/'manifest.json').read_text())
        assert manifest['step']==a.step and not manifest['vlm_frozen'] and manifest['replicas_equal']
        for name,value in manifest['files_sha256'].items():
            if name not in ('training_state.pt','optimizer_state.pt'):assert sha(checkpoint/name)==value
        checkpoint_hash=sha(checkpoint/'manifest.json')
    vlm,head,processor=load_policy(checkpoint,training=False)
    rows=baseline().dataset()[a.shard::16];constants=baseline().constants()
    out=ROOT/f'eval_step{a.step}';out.mkdir(exist_ok=True)
    path=out/f'shard{a.shard}.jsonl';started=time.monotonic()
    with path.open('x') as f:
        for index,item in enumerate(rows):
            with Image.open(item['image']) as im:image=im.convert('RGB')
            options='\n'+''.join(f'{chr(65+i)}. {option}\n' for i,option in enumerate(item['options']))
            user=f"Question: {item['question']}\nOptions: {options}\n"+constants['USER_PROMPT_V2']
            t=sample(vlm,head,processor,image,user,version=a.step,training=False)
            # Read GT only AFTER generation; never expose boxes to the policy.
            annotation_path=Path(item['image']).with_suffix('.json')
            assert sha(annotation_path)==item['annotation_sha256']
            loc=localization(json.loads(annotation_path.read_text()),image.size,t['crop_trace'])
            match=baseline().OPTION.match(t['answer']);letter=match.group(1).upper() if match else None
            record=dict(**item,**public_trace(t),step=a.step,shard=a.shard,predicted_option=letter,
                        correct=letter=='A',checkpoint_manifest_sha256=checkpoint_hash,localization=loc)
            f.write(json.dumps(record)+'\n');f.flush()
            print('VSTAR',a.step,a.shard,index+1,'/',len(rows),flush=True)
    atomic_json(out/f'shard{a.shard}.summary.json',dict(step=a.step,shard=a.shard,count=len(rows),
        checkpoint_manifest_sha256=checkpoint_hash,results_sha256=sha(path),seconds=time.monotonic()-started))

if __name__=='__main__':main()
