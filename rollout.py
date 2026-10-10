"""Current VLM generates text and the jointly trained head samples crop actions."""
import concurrent.futures
import json
import re
import time
import torch
from transformers import StoppingCriteriaList
from policy import *
from reward import reward_components

@torch.no_grad()
def sample(vlm,head,processor,image,user,version,training=True):
    protocol=baseline();constants=protocol.constants()
    messages=[dict(role='system',content=constants['instruction_prompt_system']),dict(role='user',content=[
        dict(type='image'),dict(type='text',text=user)])]
    images=[image];turns=[];actions=[];traces=[];answer='';error=None
    vlm.eval();head.eval()
    cap=1048576 if training else 12845056
    budget=768 if training else 1536
    temperature=.8 if training else 1.
    try:
        for turn in range(5):
            inputs=encode(processor,messages,images,max_pixels=cap)
            length=inputs['input_ids'].shape[-1]
            if length+budget>(12288 if training else 32768):
                raise ProtocolError('Context budget exceeded')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                generated=vlm.generate(**to_cuda(inputs),do_sample=training,
                    temperature=temperature,top_p=1.,top_k=0,repetition_penalty=1.,
                    max_new_tokens=budget,use_cache=True,
                    stopping_criteria=StoppingCriteriaList([protocol.StopTerminal(processor.tokenizer,length)]),
                    pad_token_id=processor.tokenizer.pad_token_id)[0,length:].cpu()
            text=processor.tokenizer.decode(generated.tolist(),skip_special_tokens=False)
            text=text.replace('<|im_end|>','').replace('<|endoftext|>','')
            turns.append(dict(inputs=dict(inputs),response=generated,text=text,temperature=temperature))
            normalized,kind=normalize_protocol_response(text,finish_reason=None)
            if kind=='answer':answer=extract_answer(normalized);break
            call=parse_tool_call(normalized)
            if len(actions)>=3:raise ProtocolError('Exceeded3crop budget')
            bbox=[float(v) for v in call['arguments']['bbox_2d']]
            protocol.pixel_box(image,bbox)
            if bbox[2]<=bbox[0] or bbox[3]<=bbox[1]:raise ProtocolError('Degenerate box')
            coarse=sanitize_normalized_boxes(torch.tensor(bbox)/1000,.02)
            match=re.search(r'"bbox_2d"\s*:\s*\[',normalized)
            if not match:raise ProtocolError('No canonical bbox prefix')
            query=encode(processor,messages,images,normalized[:match.end()]+'<|box_start|>',max_pixels=cap)
            dist=action_distribution(vlm,head,query,coarse,image.size)
            raw=dist.sample() if training else dist.base_dist.loc
            box=apply_text_box_residual(coarse.cuda(),.15*torch.tanh(raw),.02)
            crop=protocol.resized_crop(image,protocol.pixel_box(image,(box*1000).tolist()))
            actions.append(dict(inputs=dict(query),coarse=coarse,image_wh=image.size,
                raw=raw.cpu(),old_logprob=dist.log_prob(raw).cpu()))
            traces.append(dict(coarse=coarse.tolist(),box=box.tolist(),raw=raw.tolist(),
                               old_logprob=float(dist.log_prob(raw))))
            images.append(crop)
            messages.extend([dict(role='assistant',content=normalized),dict(role='user',content=[
                dict(type='text',text='<tool_response>'),dict(type='image'),
                dict(type='text',text=constants['USER_PROMPT_V2']),dict(type='text',text='</tool_response>')])])
        if not answer:raise ProtocolError('No final answer')
    except ProtocolError as exc:error=str(exc)
    return dict(version=version,turns=turns,actions=actions,answer=answer,error=error,
                crop_trace=traces,output_tokens=sum(len(t['response']) for t in turns))

def judge_group(trajectories,row):
    from judge import judge_answer
    answers=sorted({t['answer'].strip() for t in trajectories if t['error'] is None})
    def judge(answer):
        for attempt in range(3):
            try:return answer,judge_answer(answer,row['ground_truth'],row['question'],strict=True)
            except Exception as exc:
                print('JUDGE_RETRY',attempt+1,type(exc).__name__,flush=True)
                if attempt==2:raise RuntimeError('Scoring API failed after bounded retries; no substitute reward') from None
                time.sleep(2*(attempt+1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        scores=dict(pool.map(judge,answers))
    for t in trajectories:
        t['answer_correct']=scores[t['answer'].strip()] if t['error'] is None else 0.
        # Actions are appended only after the crop actually succeeds. Generated
        # tool tags alone do not qualify; repeated calls do not stack bonuses.
        t['reward_components']=reward_components(
            t['answer_correct'],t['error'] is not None,len(t['actions']))
        t['reward']=t['reward_components']['total']
    return [t['reward'] for t in trajectories]

def public_trace(t):
    return {**{k:v for k,v in t.items() if k not in ('turns','actions')},
            'texts':[x['text'] for x in t['turns']], 'tools':len(t['actions'])}
