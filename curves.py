"""Live global committed-step CSV and SVG curves; standard library only."""
import csv
import io
import json
from pathlib import Path

from config import ROOT


def records(path):
    if not path.exists():return []
    # Ignore the last incomplete line while a worker is appending.
    text=path.read_text();lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines if line]


def atomic(path,text):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)


def write_csv(path,rows):
    if not rows:return
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic(path,out.getvalue())


def svg(path,rows,panels,max_step=128,tick_step=32,width=960,horizontal=False):
    if not rows:return
    height=285 if horizontal else 260*len(panels);compact=width<400
    canvas_width=width*len(panels) if horizontal else width
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{height}" viewBox="0 0 {canvas_width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>']
    colors=['#2563eb','#dc2626','#059669','#7c3aed']
    for p,(title,keys) in enumerate(panels):
        if horizontal:out.append(f'<g transform="translate({p*width},0)">')
        top=40 if horizontal else p*260+40;bottom=top+170;left=45 if compact else 80;right=width-(10 if compact else 30)
        values=[float(r[k]) for r in rows for k in keys if r.get(k) is not None]
        lo=min([0.]+values);hi=max([.01]+values);span=hi-lo
        def xy(r,k):return (left+(right-left)*r['step']/max_step,bottom-(float(r[k])-lo)/span*170)
        if compact and title.startswith('Reward'):title='Reward (steps 1-32)'
        out.append(f'<text x="{10 if compact else 80}" y="{top-20}" font-family="sans-serif" font-size="{12 if compact else 16}">{title}</text>')
        for j in range(5):
            y=bottom-j*170/4;v=lo+j*span/4
            out.append(f'<path d="M{left} {y} H{right}" stroke="#ddd"/><text x="10" y="{y+4}" font-size="{9 if compact else 12}">{v:.3f}</text>')
        for step in range(0,max_step+1,tick_step):
            x=left+(right-left)*step/max_step
            out.append(f'<text x="{x-8}" y="{bottom+18}" font-size="11">{step}</text>')
        for i,k in enumerate(keys):
            points=' '.join(f'{x:.2f},{y:.2f}' for r in rows if r.get(k) is not None for x,y in [xy(r,k)])
            out.append(f'<polyline points="{points}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')
            lx=45 if compact else 80+i*220
            ly=bottom+32+i*14 if compact else bottom+38
            out.append(f'<text x="{lx}" y="{ly}" fill="{colors[i]}" font-size="{9 if compact else 12}">{k}</text>')
        if horizontal:out.append('</g>')
    out.append('</svg>');atomic(path,'\n'.join(out))


def main():
    out=ROOT/'curves';out.mkdir(exist_ok=True)
    metrics=[]
    for directory in sorted(ROOT.glob('train_*_*')):
        if not directory.is_dir():continue
        updates=[{r['step']:r for r in records(directory/f'rank{rank}.updates.jsonl')} for rank in range(16)]
        committed=set.intersection(*(set(u) for u in updates))
        traces=[{} for _ in range(16)]
        for rank in range(16):
            for row in records(directory/f'rank{rank}.jsonl'):
                if row['step'] in committed:
                    # Later recollection waves supersede zero-variance batches.
                    traces[rank][row['step']]=row
        for step in sorted(committed):
            if not all(step in t for t in traces):continue
            trajectories=[t for rank in range(16) for t in traces[rank][step]['trajectories']]
            assert len(trajectories)==64
            def mean(fn):return sum(fn(t) for t in trajectories)/64
            metrics.append(dict(step=step,trajectories=64,
                reward_mean=mean(lambda t:t['reward']),
                correctness_rate=mean(lambda t:t['answer_correct']),
                tool_bonus_mean=mean(lambda t:t['reward_components']['tool_bonus']),
                protocol_failure_rate=mean(lambda t:float(t['error'] is not None)),
                tool_use_rate=mean(lambda t:float(t['tools']>0)),
                calls_per_trajectory=mean(lambda t:t['tools']),
                correct_tool_rate=mean(lambda t:float(t['tools']>0 and t['answer_correct']>0.5)),
                effective_groups=updates[0][step]['effective_groups'],
                step_seconds=max(u[step]['seconds'] for u in updates)))
    metrics.sort(key=lambda r:r['step']);assert len({r['step'] for r in metrics})==len(metrics)
    write_csv(out/'training_steps.csv',metrics)
    # Preserve the original full-history plot and save the32-step view separately.
    svg(out/'training.svg',metrics,[('Reward (64 online trajectories per committed optimizer step)', ['reward_mean','tool_bonus_mean']),
        ('Fractions', ['correctness_rate','tool_use_rate','protocol_failure_rate']),
        ('Tool calls', ['calls_per_trajectory'])])
    svg(out/'training_first32.svg',[r for r in metrics if 1<=r['step']<=32],[('Reward (steps 1-32; 64 online trajectories per step)', ['reward_mean','tool_bonus_mean']),
        ('Fractions', ['correctness_rate','tool_use_rate','protocol_failure_rate']),
        ('Tool calls', ['calls_per_trajectory'])],max_step=32,tick_step=8,width=192,horizontal=True)
    evaluations=[]
    for p in ROOT.glob('eval_step*/summary.json'):
        r=json.loads(p.read_text());loc=r['localization']
        evaluations.append(dict(step=r['step'],accuracy=r['overall']['accuracy'],
            tool_use_rate=loc['tool_samples']/191,
            coarse_iou=loc['coarse']['mean_best_union_iou_all'],
            refined_iou=loc['box']['mean_best_union_iou_all'],
            refined_iou_tool_only=loc['box']['mean_best_union_iou_tool_only']))
    evaluations.sort(key=lambda r:r['step']);write_csv(out/'vstar_steps.csv',evaluations)
    svg(out/'vstar.svg',evaluations,[('V* accuracy and tool-use rate',['accuracy','tool_use_rate']),
        ('V* best union IoU (no-tool cases included as zero)',['coarse_iou','refined_iou'])])
    atomic(out/'status.json',json.dumps(dict(last_committed_step=metrics[-1]['step'] if metrics else 0,
        evaluated_steps=[r['step'] for r in evaluations]),indent=2))


if __name__=='__main__':main()
