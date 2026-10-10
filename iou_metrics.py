"""Post-generation localization diagnostics; never used in policy or reward."""
import math


def iou(a, b):
    def area(x): return max(0., x[2]-x[0])*max(0., x[3]-x[1])
    intersection=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
    union=area(a)+area(b)-intersection
    return intersection/union if union else 0.


def localization(annotation, image_wh, traces):
    w,h=image_wh
    targets=[]
    for x,y,bw,bh in annotation['bbox']:
        assert all(math.isfinite(v) for v in (x,y,bw,bh)) and bw>0 and bh>0
        targets.append([max(0.,x/w),max(0.,y/h),min(1.,(x+bw)/w),min(1.,(y+bh)/h)])
    assert targets and all(b[2]>b[0] and b[3]>b[1] for b in targets)
    union=[min(b[0] for b in targets),min(b[1] for b in targets),
           max(b[2] for b in targets),max(b[3] for b in targets)]
    result=dict(gt_boxes_normalized=targets,gt_union_normalized=union,tool_count=len(traces))
    for field in ('coarse','box'):
        values=[iou(t[field],union) for t in traces]
        result[field]=dict(per_call_union_iou=values,first_union_iou=values[0] if values else 0.,
            best_union_iou=max(values,default=0.),
            per_object_best_iou=[max((iou(t[field],b) for t in traces),default=0.) for b in targets])
    result['best_union_iou_delta']=result['box']['best_union_iou']-result['coarse']['best_union_iou']
    return result


def summarize_localization(rows):
    def avg(xs):return sum(xs)/len(xs) if xs else None
    called=[r for r in rows if r['localization']['tool_count']]
    result=dict(samples=len(rows),tool_samples=len(called),no_tool_samples=len(rows)-len(called),
        convention='normalized original-image xyxy; target=bounding rectangle of all official xywh boxes; no-call=0; best is max over executed calls')
    for field in ('coarse','box'):
        allvals=[r['localization'][field]['best_union_iou'] for r in rows]
        result[field]=dict(mean_first_union_iou_all=avg([r['localization'][field]['first_union_iou'] for r in rows]),
            mean_best_union_iou_all=avg(allvals),
            mean_best_union_iou_tool_only=avg([r['localization'][field]['best_union_iou'] for r in called]),
            mean_per_object_best_iou_all=avg([v for r in rows for v in r['localization'][field]['per_object_best_iou']]),
            union_iou_ge_0_5_all=avg([float(v>=.5) for v in allvals]))
    result['mean_best_union_iou_delta_all']=avg([r['localization']['best_union_iou_delta'] for r in rows])
    return result
