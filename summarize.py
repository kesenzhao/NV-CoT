import argparse
import collections
import json
from policy import *
from iou_metrics import localization,summarize_localization
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=tuple(range(0,129,32)),required=True);a=p.parse_args()
out=ROOT/f'eval_step{a.step}';rows=[]
expected={r['id']:r for r in baseline().dataset()}
checkpoint_hash=sha(ROOT/'checkpoints'/f'step{a.step}'/'manifest.json') if a.step else None
for shard in range(16):
    path=out/f'shard{shard}.jsonl';summary=json.loads((out/f'shard{shard}.summary.json').read_text())
    assert summary['results_sha256']==sha(path) and summary['checkpoint_manifest_sha256']==checkpoint_hash
    part=[json.loads(x) for x in path.read_text().splitlines()];assert len(part)==summary['count']
    for row in part:
        assert row['step']==a.step and row['shard']==shard and row['checkpoint_manifest_sha256']==checkpoint_hash
        assert all(row[k]==v for k,v in expected[row['id']].items())
        assert row['correct']==(row['predicted_option']=='A')
        annotation_path=Path(row['image']).with_suffix('.json')
        assert sha(annotation_path)==row['annotation_sha256']
        with Image.open(row['image']) as im:
            assert row['localization']==localization(json.loads(annotation_path.read_text()),im.size,row['crop_trace'])
    rows.extend(part)
assert len(rows)==len({r['id'] for r in rows})==191 and {r['id'] for r in rows}==set(expected)
def stats(rows):return dict(count=len(rows),correct=sum(r['correct'] for r in rows),
    accuracy=sum(r['correct'] for r in rows)/len(rows),errors=sum(r['error'] is not None for r in rows),
    unresolved=sum(r['predicted_option'] is None for r in rows),tool_calls=sum(r['tools'] for r in rows))
report=dict(step=a.step,identity_audit_passed=True,vlm_frozen=False,independent_roi_encoding=False,
    checkpoint_manifest_sha256=checkpoint_hash,overall=stats(rows),
    localization=summarize_localization(rows),
    localization_by_category={c:summarize_localization([r for r in rows if r['category']==c]) for c in ('direct_attributes','relative_position')},
    by_category={c:stats([r for r in rows if r['category']==c]) for c in ('direct_attributes','relative_position')},
    error_counts=dict(collections.Counter(r['error'] for r in rows if r['error'])),
    protocol='191cases,greedy,5turns,3crops,1536tokens/turn,original-image0..1000coordinates,stricttags;allerrorscountwrong')
atomic_json(out/'summary.json',report);print(json.dumps(report,indent=2),flush=True)
