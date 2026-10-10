"""BF16 full-parameter online joint GRPO, synchronized across16GPUs.

One on-policy optimizer update per collected batch. Text old log probabilities
are recomputed on the unchanged behavior policy (new.detach()); at this single
pass the text ratio is exactly1, but its gradient is nonzero. Numeric actions
retain their sampled behavior log probability. No reference/KL model is used.
"""
import argparse
import datetime
import gc
import json
import os
import random
import time
import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import save_file
from policy import *
from rollout import sample,judge_group,public_trace
from reward import REWARD_SPEC

class CPUOffloadAdafactor:
    """Exact FP32 master weights on rank0 with BF16 replicas on every GPU."""
    def __init__(self,groups,rank):
        self.groups=groups;self.rank=rank;self.step_count=0
        self.parameters=[p for group in groups for p in group['params']]
        self.lrs=[group['lr'] for group in groups for _ in group['params']]
        if rank==0:
            self.master=[p.detach().cpu().float().contiguous() for p in self.parameters]
            self.row_vars=[None]*len(self.parameters);self.col_vars=[None]*len(self.parameters)
            self.variances=[None]*len(self.parameters)
            self.rounding_rng=torch.Generator(device='cpu').manual_seed(20261003)
        else:
            self.master=self.row_vars=self.col_vars=self.variances=None

    def zero_grad(self,set_to_none=True):
        for p in self.parameters:p.grad=None if set_to_none else p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.step_count+=1
        if self.rank==0:
            beta_weight=self.step_count**-0.8
            eps1=torch.finfo(torch.float32).eps
            for i,(p,master,lr) in enumerate(zip(self.parameters,self.master,self.lrs)):
                grad_bf16=p.grad.detach().cpu();grad=grad_bf16.float();del grad_bf16
                if grad.ndim>1:
                    if self.row_vars[i] is None:
                        row_shape=list(grad.shape);row_shape[-1]=1
                        col_shape=list(grad.shape);col_shape[-2]=1
                        self.row_vars[i]=torch.zeros(row_shape,dtype=torch.float32)
                        self.col_vars[i]=torch.zeros(col_shape,dtype=torch.float32)
                    row_var=self.row_vars[i];col_var=self.col_vars[i]
                    row_mean=torch.norm(grad,dim=-1,keepdim=True).square_().div_(grad.size(-1))
                    col_mean=torch.norm(grad,dim=-2,keepdim=True).square_().div_(grad.size(-2))
                    row_var.lerp_(row_mean,beta_weight);col_var.lerp_(col_mean,beta_weight)
                    estimate=row_var @ col_var
                    estimate.div_(row_var.mean(dim=-2,keepdim=True).clamp_(min=eps1))
                else:
                    if self.variances[i] is None:self.variances[i]=torch.zeros_like(grad)
                    self.variances[i].lerp_(grad.square(),beta_weight)
                    estimate=self.variances[i].clone()
                update=estimate.clamp_(min=eps1*eps1).rsqrt_().mul_(grad)
                rho=min(lr,1/self.step_count**.5)
                alpha=max(.001,master.norm().item()/master.numel()**.5)*rho
                denom=max(1.,update.norm().item()/(update.numel()**.5))
                master.add_(update,alpha=-alpha/denom)
                if p.dtype==torch.bfloat16:
                    # Unbiased stochastic rounding preserves sub-ULP updates from
                    # the FP32 master instead of silently quantizing them to zero.
                    noise=torch.randint(0,65536,master.shape,dtype=torch.int32,
                        generator=self.rounding_rng)
                    rounded=(master.view(torch.int32)+noise).bitwise_and_(-65536)
                    p.copy_(rounded.view(torch.float32).to(torch.bfloat16))
                    del noise,rounded
                else:p.copy_(master)
                del grad,estimate,update
                dist.broadcast(p,src=0)
        else:
            for p in self.parameters:dist.broadcast(p,src=0)

    def state_dict(self):
        assert self.rank==0
        return dict(type='CPUOffloadAdafactor',step=self.step_count,master=self.master,
            row_vars=self.row_vars,col_vars=self.col_vars,variances=self.variances,lrs=self.lrs,
            rounding_rng=self.rounding_rng.get_state())

    def load_state_dict(self,state):
        assert self.rank==0 and state['type']=='CPUOffloadAdafactor'
        assert len(state['master'])==len(self.parameters) and state['lrs']==self.lrs
        self.step_count=int(state['step']);self.master=state['master']
        self.row_vars=state['row_vars'];self.col_vars=state['col_vars'];self.variances=state['variances']
        self.rounding_rng.set_state(state['rounding_rng'])

def parameter_hash(vlm,head):
    h=hashlib.sha256()
    for prefix,module in [('vlm',vlm),('head',head)]:
        for name,p in module.named_parameters():
            raw=p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            h.update((prefix+name).encode());h.update(raw)
    return h.hexdigest()

def probes(vlm,head):
    return dict(vision=vlm.model.visual.patch_embed.proj.weight.detach().flatten()[:8192].float().clone(),
                language=vlm.model.language_model.layers[0].self_attn.q_proj.weight.detach().flatten()[:8192].float().clone(),
                head=head.output[-1].weight.detach().flatten().float().clone())

def save_checkpoint(vlm,head,processor,optimizer,sampler,step,rank,world,plan_hash):
    hashes=[None]*world;dist.all_gather_object(hashes,parameter_hash(vlm,head))
    assert len(set(hashes))==1,'Full parameter replica divergence'
    rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),sampler=sampler.get_state())
    states=[None]*world;dist.all_gather_object(states,rng)
    if rank==0:
        path=ROOT/'checkpoints'/f'step{step}';path.mkdir(parents=True,exist_ok=False)
        vlm.save_pretrained(path/'vlm',safe_serialization=True,max_shard_size='4GB')
        processor.save_pretrained(path/'vlm')
        save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(path/'visual_refiner.safetensors'))
        torch.save(dict(step=step,rank_rng=states,plan_sha256=plan_hash),path/'training_state.pt')
        torch.save(optimizer.state_dict(),path/'optimizer_state.pt')
        files={str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file()}
        atomic_json(path/'manifest.json',dict(step=step,files_sha256=files,full_parameter_sha256=hashes[0],
            replicas_equal=True,replicas=world,plan_sha256=plan_hash,vlm_frozen=False,independent_roi_encoding=False))
        print('CHECKPOINT_COMPLETE',step,hashes[0],flush=True)
    dist.barrier()

def main():
    if (ROOT/'STOPPED.json').exists():
        raise RuntimeError('Experiment stopped by user. Create a separate explicitly configured run; do not overwrite historical artifacts.')
    parser=argparse.ArgumentParser();parser.add_argument('--until',type=int,choices=(1,*range(32,129,32)),required=True)
    parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    local_rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=45))
    rank=dist.get_rank();world=dist.get_world_size();assert world==16
    plan=json.loads((ROOT/'plan.json').read_text());plan_hash=sha(ROOT/'plan.json')
    assert plan.get('reward_spec')==REWARD_SPEC,'Reward differs from frozen plan; create a new experiment plan'
    assert plan['vlm_frozen'] is False and plan['reference_model'] is None
    data=ROOT/'data/train.jsonl';assert sha(data)==plan['train_sha256']
    rows=[json.loads(x) for x in data.read_text().splitlines()]
    start=0 if a.smoke else a.until-32
    checkpoint=ROOT/'checkpoints'/f'step{start}' if start else None
    if start:
        assert json.loads((ROOT/f'eval_step{start}/summary.json').read_text())['identity_audit_passed']
        spec=json.loads((checkpoint/'manifest.json').read_text())
        assert spec['plan_sha256']==plan_hash and spec['step']==start
        for name,value in spec['files_sha256'].items():assert sha(checkpoint/name)==value,name
    torch.manual_seed(plan['seed']);random.seed(plan['seed'])
    vlm,head,processor=load_policy(checkpoint,training=True)
    assert all(p.requires_grad for p in vlm.parameters()) and all(p.requires_grad for p in head.parameters())
    parameters=list(vlm.parameters())+list(head.parameters())
    optimizer=CPUOffloadAdafactor([dict(params=list(vlm.parameters()),lr=plan['vlm_lr']),
                                   dict(params=list(head.parameters()),lr=plan['head_lr'])],rank)
    sampler=torch.Generator().manual_seed(plan['seed'])
    if start:
        state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
        assert state['step']==start and state['plan_sha256']==plan_hash
        if rank==0:
            optimizer.load_state_dict(torch.load(checkpoint/'optimizer_state.pt',map_location='cpu',weights_only=False,mmap=True))
        rng=state['rank_rng'][rank];torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
        random.setstate(rng['python']);sampler.set_state(rng['sampler'])
        del state;gc.collect()
        assert parameter_hash(vlm,head)==spec['full_parameter_sha256']
        print('RESUME_VERIFIED',rank,start,flush=True)
    else:
        torch.manual_seed(plan['seed']+rank);random.seed(plan['seed']+rank)
    logdir=ROOT/('distributed_smoke' if a.smoke else f'train_{start+1}_{a.until}')
    logdir.mkdir(exist_ok=True)
    handle=(logdir/f'rank{rank}.jsonl').open('x')
    for step in range(start+1,a.until+1):
        tick=time.monotonic();before=probes(vlm,head);trajectories=None
        for wave in range(4):
            indexes=torch.randint(len(rows),(world,),generator=sampler);row=rows[int(indexes[rank])]
            assert sha(row['image'])==row['image_sha256']
            with Image.open(row['image']) as im:image=im.convert('RGB')
            trajectories=[]
            for draw in range(plan['rollouts_per_prompt']):
                trajectory=sample(vlm,head,processor,image,row['user'],version=step-1,training=True)
                trajectories.append(trajectory)
                print('ROLLOUT',rank,step,wave,draw+1,'tokens',trajectory['output_tokens'],'tools',len(trajectory['actions']),flush=True)
            rewards=judge_group(trajectories,row);advantages=group_advantages(rewards)
            effective=int(bool((advantages!=0).any()))
            count=torch.tensor(effective,device='cuda');dist.all_reduce(count)
            public=dict(step=step,wave=wave,rank=rank,source_id=row['id'],source=row['source'],source_row=row['source_row'],
                        rewards=rewards,advantages=advantages.tolist(),effective_groups=int(count),
                        trajectories=[public_trace(t) for t in trajectories])
            handle.write(json.dumps(public)+'\n');handle.flush()
            if int(count)>0:break
            print('ZERO_VARIANCE_RECOLLECT',rank,step,wave,flush=True)
        else:raise RuntimeError('Four globally zero-variance batches: no fake optimizer step')
        optimizer.zero_grad(set_to_none=True);vlm.train();head.train()
        token_loss=0.;action_loss=0.;numeric_ratio_error=0.;updated_trajectories=0
        for trajectory,advantage in zip(trajectories,advantages):
            assert trajectory['version']==step-1
            if float(advantage)==0:continue
            updated_trajectories+=1
            total=max(1,trajectory['output_tokens'])
            for turn in trajectory['turns']:
                new=token_logprobs(vlm,turn['inputs'],turn['response'],turn['temperature'])
                old=new.detach() # Unchanged on-policy behavior; exactly one update after the entire batch.
                loss=clipped_loss(new,old,advantage.cuda())*(len(new)/total)/plan['rollouts_per_prompt']
                assert torch.isfinite(loss);loss.backward();token_loss+=float(loss.detach())
                del new,old,loss
            for action in trajectory['actions']:
                distribution=action_distribution(vlm,head,action['inputs'],action['coarse'],action['image_wh'])
                new=distribution.log_prob(action['raw'].cuda());old=action['old_logprob'].cuda()
                numeric_ratio_error=max(numeric_ratio_error,float((new.detach()-old).abs()))
                loss=clipped_loss(new,old,advantage.cuda())/len(trajectory['actions'])/plan['rollouts_per_prompt']
                assert torch.isfinite(loss);loss.backward();action_loss+=float(loss.detach())
                del distribution,new,old,loss
        # Variable numbers of tool calls are safe: no collectives occur during
        # generation or individual forward/backward passes. Synchronize once.
        for p in parameters:
            if p.grad is None:p.grad=torch.zeros_like(p)
            dist.all_reduce(p.grad);p.grad.div_(world)
        norms={}
        for name,module in [('vision',vlm.model.visual),('language',vlm.model.language_model),('head',head)]:
            value=torch.zeros((),device='cuda')
            for p in module.parameters():value+=p.grad.float().square().sum()
            norms[name]=float(value.sqrt());assert torch.isfinite(value)
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        after=probes(vlm,head)
        delta={name:float((after[name]-value).norm()) for name,value in before.items()}
        assert norms['vision']>0 and norms['language']>0,'VLM received no policy gradient'
        assert delta['vision']>0 and delta['language']>0,'VLM did not update'
        metric=dict(step=step,rank=rank,seconds=time.monotonic()-tick,effective_groups=int(count),
            token_loss=token_loss,action_loss=action_loss,gradient_norm=float(norm),component_gradients=norms,
            parameter_probe_delta=delta,numeric_behavior_logprob_max_error=numeric_ratio_error,
            updated_trajectories=updated_trajectories,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
        atomic_json(logdir/f'rank{rank}.status.json',metric)
        with (logdir/f'rank{rank}.updates.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        print('UPDATE',json.dumps(metric),flush=True)
        del trajectories;gc.collect()
        if step==a.until:
            if a.smoke:
                signatures=[None]*world
                probe_hash=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in after.values())).hexdigest()
                dist.all_gather_object(signatures,probe_hash);assert len(set(signatures))==1
                if rank==0:atomic_json(ROOT/'distributed_smoke.json',dict(passed=True,replicas=world,probe_hash=probe_hash,metrics=metric))
            else:save_checkpoint(vlm,head,processor,optimizer,sampler,step,rank,world,plan_hash)
    handle.close();dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
