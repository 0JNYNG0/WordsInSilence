"""Candidate-only image-finger identity conflict guard, no word/label/time input."""
import numpy as np
PARAMS={'point_agreement_shoulder':0.08,'minimum_agreeing_fingers':4,'minimum_points_per_finger':3}
# 0.08 inherited unchanged from full-frame merge's per-point agreement.
# Four separate finger chains + corresponding tips avoid one-point contacts.
def inspect_pair(a,b,va,vb,shoulder_px):
    a=np.asarray(a,float);b=np.asarray(b,float);va=np.asarray(va,bool);vb=np.asarray(vb,bool)
    result={'status':'insufficient_finger_evidence','hold_new':False,'finger_tests':[]}
    if a.shape!=(21,2) or b.shape!=(21,2) or va.shape!=(21,) or vb.shape!=(21,) or not np.isfinite(shoulder_px) or shoulder_px<=1:
        return result
    mask=va & vb & np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    threshold=PARAMS['point_agreement_shoulder']*float(shoulder_px)
    for f in range(5):
        js=np.arange(1+4*f,5+4*f);ok=mask[js];tip=int(js[-1])
        sufficient=int(ok.sum())>=PARAMS['minimum_points_per_finger'] and bool(mask[tip])
        distances=np.linalg.norm(a[js[ok]]-b[js[ok]],axis=1)
        median=float(np.median(distances)) if len(distances) else None
        tipdist=float(np.linalg.norm(a[tip]-b[tip])) if mask[tip] else None
        agree=bool(sufficient and median<=threshold and tipdist<=threshold)
        result['finger_tests'].append({'finger':f,'supported_points':int(ok.sum()),'tip_supported':bool(mask[tip]),'median_distance_px':median,'tip_distance_px':tipdist,'agree':agree,'sufficient':sufficient})
    sufficient=sum(q['sufficient'] for q in result['finger_tests']);agree=sum(q['agree'] for q in result['finger_tests'])
    result.update(sufficient_fingers=sufficient,agreeing_fingers=agree,threshold_px=threshold)
    if sufficient<4:return result
    result.update(status='finger_occupancy_not_contradictory',hold_new=False)
    if agree>=PARAMS['minimum_agreeing_fingers']:
        # Two genuinely superimposed hands could be ambiguous. Do not merge/delete
        # baseline or assign identity: retain both hypotheses, withhold NEW use.
        result.update(status='same_image_fingers_identity_unresolved',hold_new=True)
    return result

