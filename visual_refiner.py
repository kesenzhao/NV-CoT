"""Opt-in frozen-feature visual box refiner; not enabled by legacy policies.

Inputs must refer to the image addressed by the current tool call. The caller
owns image-grid decoding and coarse-box normalization. Outputs are additive
raw-action corrections, before the existing bounded residual transform.
"""
import torch
from torch import nn
from torch.nn import functional as F


def build_visual_context(image_tokens, grid_thw, image_index, box, merge_size=2,
                         global_side=8, local_side=16, expansion=1.5):
    """Pool FINAL visual outputs (after Qwen2.5 window-order restoration).

    Explicit image selection prevents accidental mixing of original/crop images.
    This helper supports static images only; video must use a separate mapping.
    Box coordinates and returned xy are normalized to the selected whole image.
    """
    grids = torch.as_tensor(grid_thw).detach().cpu()
    if grids.ndim != 2 or grids.shape[1] != 3 or not 0 <= image_index < len(grids):
        raise ValueError('Invalid image grid/index')
    if merge_size < 1 or global_side < 1 or local_side < 1 or expansion < 1:
        raise ValueError('Invalid pooling configuration')
    if not (grids > 0).all() or not (grids[:,0] == 1).all() or not (grids[:,1:] % merge_size == 0).all():
        raise ValueError('Expected static image grids divisible by merger size')
    counts = (grids[:,1] * grids[:,2] // merge_size**2).long()
    if image_tokens.ndim != 2 or len(image_tokens) != int(counts.sum()):
        raise ValueError('Visual output length does not match image grids')
    box = torch.as_tensor(box,device=image_tokens.device,dtype=torch.float32).detach()
    if box.shape != (4,) or not torch.isfinite(box).all() or not ((box >= 0)&(box <= 1)).all() or not (box[2:] > box[:2]).all():
        raise ValueError('Invalid normalized coarse box')
    start = int(counts[:image_index].sum())
    h, w = (int(v)//merge_size for v in grids[image_index,1:])
    features = image_tokens[start:start+int(counts[image_index])].detach().float().reshape(h,w,-1)
    yy,xx = torch.meshgrid((torch.arange(h,device=box.device)+.5)/h,
                           (torch.arange(w,device=box.device)+.5)/w,indexing='ij')
    positions = torch.stack([xx,yy],-1)
    center, size = (box[:2]+box[2:])*.5, (box[2:]-box[:2])*expansion
    lo, hi = (center-size*.5).clamp(0,1), (center+size*.5).clamp(0,1)
    x0,y0 = int(torch.floor(lo[0]*w)),int(torch.floor(lo[1]*h))
    x1,y1 = int(torch.ceil(hi[0]*w)),int(torch.ceil(hi[1]*h))
    x1,y1 = max(x0+1,x1),max(y0+1,y1)
    parts = []
    for feat,pos,side in ((features,positions,global_side),
                          (features[y0:y1,x0:x1],positions[y0:y1,x0:x1],local_side)):
        shape = (min(side,feat.shape[0]),min(side,feat.shape[1]))
        pool = lambda x: F.adaptive_avg_pool2d(x.permute(2,0,1)[None],shape)[0].permute(1,2,0).reshape(-1,x.shape[-1])
        parts.append((pool(feat),pool(pos)))
    tokens,xy,branches,valid = [],[],[],[]
    for idx,((feat,pos),side) in enumerate(zip(parts,(global_side,local_side))):
        pad = side**2-len(feat)
        tokens.append(F.pad(feat,(0,0,0,pad)))
        xy.append(F.pad(pos,(0,0,0,pad)))
        branches.append(torch.full((side**2,),idx,device=box.device,dtype=torch.long))
        valid.append(torch.arange(side**2,device=box.device)<len(feat))
    return dict(tokens=torch.cat(tokens),xy=torch.cat(xy),branch=torch.cat(branches),valid=torch.cat(valid))


class VisualBoxRefiner(nn.Module):
    def __init__(self, hidden_size, visual_size, width=256, heads=4, layers=2):
        super().__init__()
        if width % heads or layers < 1:
            raise ValueError('Invalid attention dimensions')
        self.query = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width))
        self.geometry = nn.Linear(5, width)
        self.visual = nn.Sequential(nn.LayerNorm(visual_size), nn.Linear(visual_size, width))
        self.position = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        self.branch = nn.Embedding(2, width)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(width, heads, width * 2, dropout=0.,
                                       batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, hidden, boxes, image_wh, tokens, xy, branch, valid):
        # Deliberate detach: even an accidentally grad-enabled encoder remains frozen.
        dtype = self.output[-1].weight.dtype
        hidden, boxes, image_wh, tokens, xy = [
            x.detach().to(dtype=dtype) for x in (hidden, boxes, image_wh, tokens, xy)]
        if hidden.ndim != 2 or tokens.ndim != 3:
            raise ValueError('Expected hidden[B,H], tokens[B,N,V]')
        b, n = tokens.shape[:2]
        if (hidden.shape[0] != b or boxes.shape != (b,4) or image_wh.shape != (b,2)
                or xy.shape != (b,n,2) or branch.shape != (b,n) or valid.shape != (b,n)):
            raise ValueError('Visual context shape mismatch')
        if valid.dtype != torch.bool or not valid.any(-1).all():
            raise ValueError('Each action needs at least one valid visual token')
        if branch.dtype != torch.long or not ((branch == 0) | (branch == 1)).all():
            raise ValueError('Branch must identify global0/local1')
        if not all(torch.isfinite(x).all() for x in (hidden, boxes, image_wh, tokens, xy)):
            raise ValueError('Nonfinite visual context')
        if not (image_wh > 0).all() or not ((boxes >= 0) & (boxes <= 1)).all():
            raise ValueError('Expected positive image dimensions and normalized boxes')
        if not ((xy >= 0) & (xy <= 1)).all() or not (boxes[:,2:] > boxes[:,:2]).all():
            raise ValueError('Invalid spatial coordinates')
        center = (boxes[:,:2] + boxes[:,2:]) * .5
        size = boxes[:,2:] - boxes[:,:2]
        geometry = torch.cat([center, size, (image_wh[:,0]/image_wh[:,1]).log()[:,None]], -1)
        q = (self.query(hidden) + self.geometry(geometry))[:,None]
        memory = self.visual(tokens) + self.position(xy) + self.branch(branch)
        for block in self.blocks:
            q = block(q, memory, memory_key_padding_mask=~valid)
        return self.output(q[:,0])
