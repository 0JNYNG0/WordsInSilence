"""Independent saved-input checks. Geometry decision replay is not independent human truth."""
from common import *
from collections import Counter
sys.path[:0]=[str(ROOT/'research/sign_model_pilot_v1/src')]
from pilot.data import build_channels
from pilot.web_input import to_native_like,active_span

def main(partial=False):
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected'])
    selective=module('verify_dev_selective',O/'word100_yolo_selective_wrist_20261025/candidate.py')
    evidence=module('verify_dev_shapes',O/'word100_finger_evidence_20261020/candidate.py')
    totals=Counter();queue=[];row_results=[];pins={}
    for row in f['rows']:
        cid=row['clip_id'];rp=U/'records'/(cid+'.json')
        if not rp.exists():assert partial;continue
        record=read(rp);check(record['outputs']);assert record['context_sha256']==sha(U/'FROZEN.json')
        rec=read(row['record']);base=load(rec['native_capture']['candidate']);raw=load(rec['native_capture']['raw']);assignment=load(Path(rec['native_capture']['raw']).parent/'assignment.npz')['sides']
        old=load(rec['views']['masked']['observation']);p=load(U/'outputs'/(cid+'.npz'));n=len(base['frame_id']);box=row['box']
        assert sha(record['source_video'])==row['source']['sha256']
        expected_frames={t['index'] for t in row['recovery_target_frames']};actual_frames={t['i'] for t in record['frames']};assert actual_frames==expected_frames
        for key in ('xy_px','observed','usable','conf','hand_score'):assert same(old[key],p[key][0])
        for key in ('frame_id','timestamp_us'):assert same(base[key],p[key])
        for k in (0,1,2):
            # Whole-source geometry applied first; every coordinate uses the identical ROI.
            xx,vv=roi(p['full_xy_px'][k],p['full_usable'][k],box)
            assert same(vv,p['usable'][k]) and same(xx[vv],p['xy_px'][k][vv])
            if k:
                assert same(p['xy_px'][k][old['usable']],old['xy_px'][old['usable']])
                assert p['usable'][k][old['usable']].all()
                assert same(p['xy_px'][k,:,:33],old['xy_px'][:,:33])
                assert same(p['usable'][k,:,:33],old['usable'][:,:33])
        nat,present=to_native_like(old['xy_px'],old['usable']);lo,hi=active_span(present)
        ch=build_channels(nat[lo:hi+1],present[lo:hi+1],old['timestamp_us'][lo:hi+1])['x']
        assert same(ch,load(rec['views']['masked']['channel_file'])['x_whole'])
        totals['original_channels_reproduced']+=ch.size;totals['original_valid_joint_frames_preserved']+=int(old['usable'].sum())
        additions=Counter()
        for rr in record['frames']:
            i=rr['i'];nn=int(raw['raw_count'][i]);w,h=int(raw['width'][i]),int(raw['height'][i]);pose=raw['pose_xy_px'][i];pu=raw['pose_usable'][i]
            hand=base['xy_px'][i,33:].reshape(2,21,2);valid=base['usable'][i,33:].reshape(2,21)
            original=list(raw['hand_xy_px'][i,:nn]);sides=assignment[i,:nn].tolist()
            for s in (0,1):
                if valid[s].any() and s not in sides:original.append(hand[s]);sides.append(s)
            decisions=rule.merge(rr['mp_full'],rr['mp_crop'],original,sides,pose,pu,w,h)
            pp=hand.copy();pv=valid.copy()
            for q in decisions:
                s=q['side']
                if q['accepted'] and (str(s) not in rr['boxes'] or valid[s].any()):q.update(accepted=False,reason='not_missing_target')
                if q['accepted']:
                    vv=np.asarray(q['joint_supported'],bool);pp[s]=np.where(vv[:,None],points(rr['mp_full'][q['index']]),np.nan);pv[s]=vv
            assert decisions==rr['mp_decisions']
            assert same(pp,p['full_xy_px'][1,i,33:].reshape(2,21,2)) and same(pv,p['full_usable'][1,i,33:].reshape(2,21))
            if rr['yolo_boxes']:
                _,_,yy,yv,yd=selective.supplement(rr['yolo_full'],rr['yolo_crop'],pp,pv,raw['hand_xy_px'][i,:nn],assignment[i,:nn].tolist(),pose,pu,w,h,{int(k):v for k,v in rr['yolo_boxes'].items()})
                assert yd==rr['yolo_decisions']
            else:yy,yv=pp,pv
            assert same(yy,p['full_xy_px'][2,i,33:].reshape(2,21,2)) and same(yv,p['full_usable'][2,i,33:].reshape(2,21))
            for stage,ds,full,crop in [('mp',rr['mp_decisions'],rr['mp_full'],rr['mp_crop']),('yolo',rr['yolo_decisions'],rr['yolo_full'],rr['yolo_crop'])]:
                k=1 if stage=='mp' else 2
                for q in ds:
                    if not q['accepted']:continue
                    s=q['side'];hs=slice(33+21*s,54+21*s);v=p['usable'][k,i,hs];prior=p['usable'][k-1,i,hs]
                    if not (v&~prior).any():continue
                    a=p['xy_px'][k,i,hs].astype(float);b=points(crop[q['crop_index']]);bb=np.asarray(q['joint_supported'],bool)
                    bvalid=roi(b,bb,box)[1];quality=evidence.compare_shape(a,v,b,bvalid)
                    # Semantic accuracy is intentionally not inferred from these geometry flags.
                    entry=dict(clip_id=cid,frame=i,side=s,stage=stage,points=int(v.sum()),position_only=bool(q.get('position_only',False)),
                        comparable_angle_count=int((quality['angle_available']&~quality['angle_unresolved']).sum()),image=str(U/'frames'/f'{cid}_{i:04d}.jpg'),
                        anatomy_verified=False,identity_ground_truth=False)
                    if entry['position_only']:assert v[0] and v.sum()==1 and entry['comparable_angle_count']==0
                    assert Path(entry['image']).is_file();queue.append(entry);additions[stage+'_hand_frames']+=1;additions[stage+'_points']+=int((v&~prior).sum())
        totals.update(clips=1,frames=n,**dict(additions));row_results.append(dict(clip_id=cid,counts=dict(additions)))
        pins[str(rp)]=sha(rp);pins.update(record['outputs'])
    assert partial or totals['clips']==300
    check(f['protected']);check(f['pins'])
    name='SMOKE_VERIFIED.json' if partial else 'INPUTS_VERIFIED.json'
    write(U/name,dict(at=now(),status='NUMERIC_PASS_VISUAL_REVIEW_PENDING',counts=dict(totals),rows=row_results,pins=pins,
        method='Exact old-channel reconstruction, separate coordinate/mask assertions, replay of frozen geometry decisions. Not human joint truth.',queue=queue))
    print(json.dumps(dict(totals)),flush=True)
if __name__=='__main__':main('--partial' in sys.argv)
