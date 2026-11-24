"""Label-free, absent-side-only, position/shape-separated YOLO fallback."""
import numpy as np
from common import rule,roi,points

def trigger(base,valid,pose,usable,w,h,box):
    good,scale=rule.context(pose,usable,w,h)
    if scale<=1:return {}
    boxes=rule.crop_boxes(pose,usable,[s for s in (0,1) if valid[s].any()],w,h)
    # Absent in full native view, not simply below ROI. Require current visible wrist in ROI.
    return {s:b for s,b in boxes.items() if roi(np.asarray([pose[15+ s]]),np.array([good[15+s]]),box)[1][0]}

def supplement(full,crops,base,valid,raw_hands,raw_sides,pose,usable,w,h,targets):
    good,scale=rule.context(pose,usable,w,h)
    old=list(raw_hands);sides=list(raw_sides)
    for s in (0,1):
        if valid[s].any() and s not in sides:old.append(base[s]);sides.append(s)
    strict=rule.merge(full,crops,old,sides,pose,usable,w,h)
    arms=[rule.arm(points(d),pose,good,scale) for d in full]
    result=[]
    for q in strict:
        q=dict(q);q['strict_reason']=q['reason'];q['position_only']=False
        s=q['side'];p=points(full[q['index']])
        if s not in targets or s<0 or valid[s].any():
            q.update(accepted=False,reason='not_missing_target');result.append(q);continue
        if arms.count(s)!=1:
            q.update(accepted=False,reason='multiple_full_same_arm');result.append(q);continue
        if q['accepted']:
            q['reason']='strict_full_hand_evidence';result.append(q);continue
        # Reuse identical point-agreement tolerance. No angle truth inferred from wrist evidence.
        if q['strict_reason'] not in ('image_scales_disagree_or_ambiguous','palm_outside_source'):
            result.append(q);continue
        if not rule.inside(p[:1],w,h)[0]:result.append(q);continue
        # Recheck duplicates even if strict merge exited early for missing palm points.
        relations=[rule.relation(p,b,s,rule.arm(b,pose,good,scale),scale) for b in old]
        if any(t.startswith('same_') or t=='ambiguous_overlap' for t in relations) or s in sides:
            q.update(reason='position_duplicate_or_identity_hold');result.append(q);continue
        matches=[];eligible=0
        for j,c in enumerate(crops):
            a=points(c);box=np.asarray(c['box'])
            if c['requested_side']!=s or rule.arm(a,pose,good,scale)!=s:continue
            eligible+=1
            ok=rule.inside(a[:1],w,h)[0] and rule.inside(a[:1]-box[:2],box[2]-box[0],box[3]-box[1])[0]
            if ok and np.linalg.norm(p[0]-a[0])<=rule.PARAMS['point_agreement']*scale:matches.append(j)
        if len(matches)==1 and eligible==1:
            support=[True]+[False]*20
            q.update(accepted=True,position_only=True,reason='wrist_only_two_scale_current_arm',crop_index=matches[0],joint_supported=support)
        else:q.update(reason='wrist_disagreement_missing_or_multiple_crop')
        result.append(q)
    strict_p=base.copy();strict_v=valid.copy();split_p=base.copy();split_v=valid.copy()
    for q in result:
        if not q['accepted']:continue
        s=q['side'];a=points(full[q['index']]);v=np.asarray(q['joint_supported'],bool)
        assert not valid[s].any()
        split_p[s]=np.where(v[:,None],a,np.nan);split_v[s]=v
        if not q['position_only']:strict_p[s]=split_p[s];strict_v[s]=v
    return strict_p,strict_v,split_p,split_v,result
