import hashlib,json,math,re
from pathlib import Path
from PIL import Image
from transformers import StoppingCriteria
from deepeyes_protocol import ProtocolError
from config import VSTAR_DIR as DATA, CODE
OPTION=re.compile(r"^\s*([A-F])(?:\s|[.):,-]|$)",re.I)
def constants():
    return {'instruction_prompt_system': 'You are a helpful assistant.\n\n# Tools\nYou may call one or more functions to assist with the user query.\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox_2d"]}}}\n</tools>\n\n# How to call a tool\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n**Example**:  \n<tool_call>  \n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  \n</tool_call>\nAll bbox_2d coordinates are relative xyxy in the range 0 to 1000, always relative to the original full image, not a returned crop.', 'USER_PROMPT_V2': '\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> '}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def dataset():
    rows=[]
    for category in ('direct_attributes','relative_position'):
        for path in sorted((DATA/category).glob('*.jpg')):
            annotation=path.with_suffix('.json')
            obj=json.loads(annotation.read_text())
            # Deliberately discard annotation boxes; never provide them to generation.
            rows.append(dict(id=f'{category}/{path.name}',category=category,image=str(path),
                question=obj['question'],options=obj['options'],image_sha256=sha(path),annotation_sha256=sha(annotation)))
    assert len(rows)==191 and len({r['id'] for r in rows})==191
    fingerprints=json.loads((CODE/'metadata/vstar_fingerprints.json').read_text())
    assert {r['id']:{k:r[k] for k in ('image_sha256','annotation_sha256')} for r in rows}==fingerprints, 'V* benchmark differs from frozen original files'
    return rows

def pixel_box(image,bbox):
    if len(bbox)!=4 or not all(math.isfinite(v) for v in bbox):raise ProtocolError('Nonfinite box')
    if min(bbox)<0 or max(bbox)>1000:raise ProtocolError('Box outside normalized 0..1000 range')
    w,h=image.size
    return [bbox[0]*w/1000,bbox[1]*h/1000,bbox[2]*w/1000,bbox[3]*h/1000]

def resized_crop(image,bbox):
    if len(bbox)!=4 or not all(math.isfinite(v) for v in bbox):raise ProtocolError('Nonfinite box')
    left,top,right,bottom=bbox
    if right<=left or bottom<=top:raise ProtocolError('Degenerate box')
    # Resource guard only; do not silently clamp or substitute a different box.
    if max(abs(v) for v in bbox)>100000 or (right-left)*(bottom-top)>1e8:raise ProtocolError('Oversized crop')
    crop=image.crop((left,top,right,bottom))
    if min(crop.size)<1:raise ProtocolError('Subpixel empty crop')
    w,h=crop.size;factor=32;area_max=12845056;area_min=3136
    rw=max(factor,round(w/factor)*factor);rh=max(factor,round(h/factor)*factor)
    if rw*rh>area_max:
        beta=math.sqrt(w*h/area_max);rw=max(factor,math.floor(w/beta/factor)*factor);rh=max(factor,math.floor(h/beta/factor)*factor)
    elif rw*rh<area_min:
        beta=math.sqrt(area_min/(w*h));rw=math.ceil(w*beta/factor)*factor;rh=math.ceil(h*beta/factor)*factor
    return crop.resize((rw,rh),Image.Resampling.BICUBIC)

class StopTerminal(StoppingCriteria):
    def __init__(self,tokenizer,start):self.tokenizer=tokenizer;self.start=start
    def __call__(self,input_ids,scores,**kwargs):
        text=self.tokenizer.decode(input_ids[0,max(self.start,input_ids.shape[1]-64):].tolist(),skip_special_tokens=False)
        return '</tool_call>' in text or '</answer>' in text
