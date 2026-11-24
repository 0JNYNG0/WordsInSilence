"""Apply frozen generic guards; inferred geometry is never relabeled as observed."""
from pathlib import Path
import json, sys, hashlib
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_yolo_selective_wrist_20261025'))
from common import rule,roi

LINKS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
       (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]

def bones(j):return np.array([np.linalg.norm(j[b]-j[a]) for a,b in LINKS])
def rotation_delta(a,b):
    relative=a @ b.swapaxes(-1,-2)
    return float(np.max(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1))))
def temporal_check(pred,t,s,scale,policy):
    available=sorted(i for i,side in pred if side==s)
    before=[i for i in available if t-policy['context_each_side_frames']<=i<t]
    after=[i for i in available if t<i<=t+policy['context_each_side_frames']]
    if not before or not after:return False,'no_two_sided_temporal_evidence',{}
    i,j=max(before),min(after);a=pred[i,s];b=pred[t,s];c=pred[j,s]
    phase=(t-i)/(j-i);expected=a['xy_px'][0]*(1-phase)+c['xy_px'][0]*phase
    residual=float(np.linalg.norm(b['xy_px'][0]-expected)/scale)
    pose=max(rotation_delta(b['hand_pose'],a['hand_pose']),rotation_delta(b['hand_pose'],c['hand_pose']))
    lengths=bones(b['joints3d_canonical']); changes=[]
    for other in (a,c):
        ref=bones(other['joints3d_canonical']); changes.append(float(np.max(abs(lengths-ref)/np.maximum(ref,1e-8))))
    delta=max(changes)
    details={'neighbor_frames':[i,j],'wrist_residual_shoulder':residual,'max_pose_rotation_radians':pose,'max_relative_bone_change':delta}
    if residual>policy['max_wrist_residual_shoulder']:return False,'temporal_wrist_disagreement',details
    if pose>policy['max_pose_rotation_radians_per_neighbor']:return False,'pose_transition_uncertain_not_error',details
    if delta>policy['max_relative_bone_length_change']:return False,'hand_shape_inconsistent',details
    return True,'temporally_consistent_estimate',details

def main():
    f=json.loads((U/'FROZEN_PROTOCOL.json').read_text('utf-8'))
    status=json.loads((U/'INFERENCE_STATUS.json').read_text());assert status['status']=='complete'
    out=U/'comparisons';out.mkdir(exist_ok=True);summary=[];all_times=[]
    for k,m in enumerate(f['manifest']):
        with np.load(m['prepared']) as z:a={x:z[x] for x in z.files}
        pred={};records=[]
        for t,s in np.argwhere(a['requested']):
            p=U/f'raw_predictions/{k:02d}_{t:04d}_{s}.npz';info=json.loads(p.with_suffix('.json').read_text())
            assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
            with np.load(p) as z:pred[int(t),int(s)]={x:z[x] for x in z.files}
            all_times.append(info['inference_seconds'])
        hx=a['full_xy'].copy();hv=a['full_valid'].copy();tx=hx.copy();tv=hv.copy()
        added_h=np.zeros_like(hv);added_t=np.zeros_like(hv)
        for t,s in np.argwhere(a['trigger']):
            t=int(t);s=int(s);p=pred[t,s];xy=p['xy_px'];pose=a['pose'][t];good,scale=rule.context(pose,a['pose_valid'][t],1920,1080)
            rec={'frame':t,'side':s,'H':False,'HT':False,'reason':'unassessed','finger_evidence':'model_inferred_unvalidated','position_evidence':'model_estimate_with_current_arm_gate'}
            if not np.isfinite(xy).all() or p['cam'][0]<=0:rec['reason']='invalid_projection'
            elif not rule.inside(xy[rule.PALM],1920,1080).all():rec['reason']='palm_outside_full_source'
            elif rule.arm(xy,pose,good,scale)!=s:rec['reason']='current_arm_identity_not_confirmed'
            else:
                others=[b for b in (0,1) if a['full_valid'][t,b].any()]
                relations=[rule.relation(xy,a['full_xy'][t,b],s,rule.arm(a['full_xy'][t,b],pose,good,scale),scale) for b in others]
                if any(x.startswith('same_') or x=='ambiguous_overlap' for x in relations):rec['reason']='existing_hand_duplicate_or_ambiguous'
                else:
                    assert not a['full_valid'][t,s].any()
                    rec.update(H=True,reason='current_arm_gated_estimate_only')
                    hx[t,s]=xy;hv[t,s]=rule.inside(xy,1920,1080);added_h[t,s]=hv[t,s]
                    passed,why,details=temporal_check(pred,t,s,scale,f['HT_policy'])
                    rec.update(HT=passed,HT_reason=why,temporal=details)
                    if passed:tx[t,s]=xy;tv[t,s]=hv[t,s];added_t[t,s]=tv[t,s]
            records.append(rec)
        hx,hv=roi(hx,hv,a['roi_box']);tx,tv=roi(tx,tv,a['roi_box']);added_h &= hv;added_t &= tv
        base=a['baseline_xy'];valid=a['baseline_valid']
        assert np.array_equal(hv[valid],valid[valid]) and np.array_equal(tv[valid],valid[valid])
        assert np.array_equal(hx[valid],base[valid]) and np.array_equal(tx[valid],base[valid])
        # This mask means original evidence is preserved, not that MediaPipe itself is precise ground truth.
        observed_b=valid.copy();observed_h=valid.copy();observed_t=valid.copy()
        dest=out/(m['clip_id']+'.npz')
        if dest.exists():raise FileExistsError(dest)
        np.savez_compressed(dest,B_xy=base,B_valid=valid,H_xy=hx,H_valid=hv,HT_xy=tx,HT_valid=tv,
                            H_inferred=added_h,HT_inferred=added_t,B_evidence=observed_b,H_evidence=observed_h,HT_evidence=observed_t,
                            frame_id=a['frame_id'],timestamp_us=a['timestamp_us'],roi_box=a['roi_box'])
        dest.with_suffix('.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        r={'clip_id':m['clip_id'],'frames':m['frames'],'B_point_frames':int(valid.sum()),
           'H_added_estimate_hand_frames':int(added_h.any(-1).sum()),'HT_added_estimate_hand_frames':int(added_t.any(-1).sum()),
           'H_added_estimate_point_frames':int(added_h.sum()),'HT_added_estimate_point_frames':int(added_t.sum()),
           'original_points_changed':0,'original_evidence_gained':0,'attempted_missing_hands':int(a['trigger'].sum()),
           'H_reasons':dict(Counter(x['reason'] for x in records)),
           'HT_reasons':dict(Counter(x.get('HT_reason','H_held') for x in records)),
           'crossing_frames':m['crossing_frames'],'crossing_B_points':int(valid[m['crossing_frames']].sum())}
        summary.append(r)
    result={'status':'COMPARISON_COMPUTED_NOT_ACCURACY_CERTIFIED','clips':7,'frames':812,'inference_hands':len(all_times),
            'rows':summary,'median_inference_seconds_per_hand':float(np.median(all_times)),
            'mean_inference_seconds_per_hand':float(np.mean(all_times)),'total_inference_seconds':float(np.sum(all_times)),
            'observed_finger_accuracy_measured':False,'hidden_3D_accuracy_measured':False,'training':0,'user_recording_access':0,'production_change':0,
            'interpretation':'Counts are additional model estimates, not corrected hands or increased verified comparison evidence. Require image review; preserve all original slots/time even when held.'}
    (U/'COMPARISON.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
