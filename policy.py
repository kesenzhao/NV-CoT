"""End-to-end differentiable Qwen3 and original-image visual action policy."""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from config import ROOT, CODE
from deepeyes_protocol import normalize_protocol_response,parse_tool_call,extract_answer,ProtocolError
from visual_refiner import VisualBoxRefiner
from continuous_policy import apply_text_box_residual,sanitize_normalized_boxes

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def baseline():
    import baseline_utils
    return baseline_utils

class JointVisualRefiner(VisualBoxRefiner):
    """Same architecture/state keys as the last head, without feature detachment."""
    def forward(self,hidden,boxes,image_wh,tokens,xy,branch,valid):
        hidden,boxes,image_wh,tokens,xy=[v.float() for v in (hidden,boxes,image_wh,tokens,xy)]
        assert valid.dtype==torch.bool and valid.any(-1).all()
        center=(boxes[:,:2]+boxes[:,2:])*.5;size=boxes[:,2:]-boxes[:,:2]
        geometry=torch.cat([center,size,(image_wh[:,0]/image_wh[:,1]).log()[:,None]],-1)
        q=(self.query(hidden)+self.geometry(geometry))[:,None]
        memory=self.visual(tokens)+self.position(xy)+self.branch(branch)
        for block in self.blocks:q=block(q,memory,memory_key_padding_mask=~valid)
        return self.output(q[:,0])

def visual_context(tokens,grid,box,merge=2):
    """Differentiable equivalent of the original8x8/global,16x16/local pooling.

    Only geometry is detached: policy gradients flow into original-image tokens.
    No ROI crop is separately encoded for this feature branch.
    """
    g=grid.detach().cpu();assert g.ndim==2 and (g[:,0]==1).all()
    counts=g[:,1]*g[:,2]//merge**2
    assert len(tokens)==int(counts.sum()) and (g[:,1:]%merge==0).all()
    h,w=[int(v)//merge for v in g[0,1:]]
    features=tokens[:int(counts[0])].float().reshape(h,w,-1)
    box=box.detach().to(tokens.device).float()
    yy,xx=torch.meshgrid((torch.arange(h,device=tokens.device)+.5)/h,
                         (torch.arange(w,device=tokens.device)+.5)/w,indexing='ij')
    positions=torch.stack([xx,yy],-1)
    center=(box[:2]+box[2:])*.5;size=(box[2:]-box[:2])*1.5
    lo=(center-size*.5).clamp(0,1);hi=(center+size*.5).clamp(0,1)
    x0,y0=int(torch.floor(lo[0]*w)),int(torch.floor(lo[1]*h))
    x1,y1=max(x0+1,int(torch.ceil(hi[0]*w))),max(y0+1,int(torch.ceil(hi[1]*h)))
    ts=[];xs=[];bs=[];vs=[]
    for i,(feature,pos,side) in enumerate(((features,positions,8),(features[y0:y1,x0:x1],positions[y0:y1,x0:x1],16))):
        shape=(min(side,feature.shape[0]),min(side,feature.shape[1]))
        def pool(v):return F.adaptive_avg_pool2d(v.permute(2,0,1)[None],shape)[0].permute(1,2,0).reshape(-1,v.shape[-1])
        feat,xy=pool(feature),pool(pos);pad=side*side-len(feat)
        ts.append(F.pad(feat,(0,0,0,pad)));xs.append(F.pad(xy,(0,0,0,pad)))
        bs.append(torch.full((side*side,),i,device=tokens.device,dtype=torch.long))
        vs.append(torch.arange(side*side,device=tokens.device)<len(feat))
    return torch.cat(ts),torch.cat(xs),torch.cat(bs),torch.cat(vs)

def load_policy(checkpoint=None,training=True):
    path=Path(checkpoint)/'vlm' if checkpoint else Path(os.environ['EVAL_MODEL'])
    processor=AutoProcessor.from_pretrained(path,local_files_only=True,use_fast=True)
    vlm=Qwen3VLForConditionalGeneration.from_pretrained(path,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda()
    hidden_size=vlm.config.text_config.hidden_size
    assert hidden_size==4096
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261003)
        head=JointVisualRefiner(hidden_size,hidden_size,width=256,heads=4,layers=2).float()
        # Fresh near-identity head; tiny nonzero projection allows immediate
        # gradient flow into upstream features (no inherited supervised head).
        torch.nn.init.normal_(head.output[-1].weight,std=.001)
    head=head.cuda()
    if checkpoint:
        head.load_state_dict(load_file(str(Path(checkpoint)/'visual_refiner.safetensors')),strict=True)
    vlm.requires_grad_(training);head.requires_grad_(training)
    if training:vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    vlm.eval();head.eval()
    return vlm,head,processor

def encode(processor,messages,images,suffix='',max_pixels=1048576):
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+suffix
    return processor(text=[prompt],images=images,min_pixels=3136,max_pixels=max_pixels,return_tensors='pt')

def to_cuda(inputs):return {k:v.cuda() for k,v in inputs.items()}

def token_logprobs(vlm,inputs,response,temperature):
    """Causal assistant tokens only; prompts and tool observations are not actions."""
    data=to_cuda(inputs);prefix=data['input_ids'].shape[-1]
    response=response.cuda();data['input_ids']=torch.cat([data['input_ids'],response[None]],-1)
    data['attention_mask']=torch.ones_like(data['input_ids'])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=vlm.model(**data,use_cache=False,return_dict=True)
        hidden=output.last_hidden_state[0,prefix-1:prefix-1+len(response)]
        parts=[]
        for i in range(0,len(response),64):
            logits=vlm.lm_head(hidden[i:i+64]).float()/temperature
            parts.append(-F.cross_entropy(logits,response[i:i+64],reduction='none'))
        return torch.cat(parts)

def action_distribution(vlm,head,inputs,coarse,image_wh):
    captured={}
    def hook(_module,_args,output):captured['tokens']=output[0]
    handle=vlm.model.visual.register_forward_hook(hook)
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            data=to_cuda(inputs)
            output=vlm.model(**data,use_cache=False,return_dict=True)
            tokens,xy,branch,valid=visual_context(captured['tokens'],data['image_grid_thw'],coarse)
            mean=head(output.last_hidden_state[:,-1],coarse.cuda()[None],
                torch.tensor(image_wh,device='cuda',dtype=torch.float32)[None],
                tokens[None],xy[None],branch[None],valid[None])[0].float()
    finally:handle.remove()
    return torch.distributions.Independent(torch.distributions.Normal(mean,torch.full_like(mean,.15)),1)

def clipped_loss(new,old,advantage,clip=.2):
    ratio=(new-old.detach()).exp()
    assert torch.isfinite(ratio).all()
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()

def group_advantages(rewards):
    rewards=torch.tensor(rewards,dtype=torch.float32)
    return (rewards-rewards.mean())/rewards.std(unbiased=False).clamp_min(1e-6)
