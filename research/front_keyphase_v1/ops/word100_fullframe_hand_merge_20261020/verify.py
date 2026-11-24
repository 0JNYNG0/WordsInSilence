"""Independent saved arithmetic and ROI reconstruction, no detector or candidate import."""
import json,hashlib,importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;P=[0,5,9,13,17]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def inside(a,w,h):return np.isfinite(a).all(-1)&(a[...,0]>=0)&(a[...,0]<w)&(a[...,1]>=0)&(a[...,1]<h)
def same(a,b):return np.array_equal(a,b,equal_nan=True)
def union_relation(x,y,side,other,scale):
    if scale<=1:return 'unknown'
    if side>=0 and other>=0 and side!=other:return 'different_current_arms'
    if not np.isfinite(x[P]).all() or not np.isfinite(y[P]).all():return 'unknown'
    wd=np.sqrt(((x[0]-y[0])**2).sum())/scale
    md=np.median(np.sqrt(((x[P]-y[P])**2).sum(1)))/scale
    if wd<=.20 and md<=.10:return 'same_geometry'
    ov=0
    if np.isfinite(x).all() and np.isfinite(y).all():
        denom=min(np.prod(np.ptp(x,axis=0)),np.prod(np.ptp(y,axis=0)))
        if denom>0:ov=np.prod(np.clip(np.minimum(x.max(0),y.max(0))-np.maximum(x.min(0),y.min(0)),0,None))/denom
    if side>=0 and side==other and ov>.25:return 'same_current_arm_overlap'
    if ov>=.5 and wd<=.30:return 'ambiguous_overlap'
    return 'separate'

def main():
    f=read(U/'FROZEN.json');idx=read(U/'INDEX.json');rows={r['clip_id']:r for r in f['rows']}
    assert len(idx)==36 and not (U/'RUNNING.json').exists()
    for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
    counts=Counter();reasons=Counter();adds=[];conflicts=[];controls=Counter();lat=[]
    ann=read(O/'word100_image_arm_connectivity_20261018/ANNOTATED.json')
    cross=read(O/'word100_image_arm_connectivity_20261018/CROSSING.json')
    for item in idx:
        assert sha(item['path'])==item['sha256']
        for p,h in item['outputs'].items():assert sha(p)==h
        r=rows[item['clip_id']];d=read(item['path']);z=load(r['raw']);old=load(r['native'])
        ass=load(r['assignment'])['sides'];dest=U/'outputs'/r['clip_id']
        merged=load(dest/'merged_native.npz');masked=load(dest/'masked.npz');oldmasked=load(r['old_masked'])
        for k in ['frame_id','timestamp_us']:assert same(old[k],merged[k]) and same(old[k],masked[k])
        assert same(old['xy_px'][:,:33],merged['xy_px'][:,:33])
        occupied=old['observed']
        for k in ['xy_px','usable','observed','conf']:
            assert same(old[k][occupied],merged[k][occupied]),(r['clip_id'],k)
        expected={k:a.copy() for k,a in old.items()}
        for i,e in enumerate(d['frames']):
            n=int(z['raw_count'][i]);w,h=int(z['width'][i]),int(z['height'][i]);pose=z['pose_xy_px'][i]
            assert e['baseline_sides']==ass[i,:n].tolist()
            valid=z['pose_usable'][i]&inside(pose,w,h)
            scale=np.sqrt(((pose[11]-pose[12])**2).sum()) if valid[11] and valid[12] else 0
            def side_for(p):
                if scale<=1 or not np.isfinite(p[0]).all():return -1
                ds=np.array([np.sqrt(((p[0]-pose[j])**2).sum())/scale if valid[k] and valid[j] else np.inf
                             for k,j in [(13,15),(14,16)]])
                s=int(np.argmin(ds));return s if ds[s]<=.25 and ds[1-s]-ds[s]>=.10 else -1
            raw=z['hand_xy_px'][i,:n];raw_s=[side_for(p) for p in raw];reconstructed=[]
            for j,det in enumerate(e['full']):
                p=np.array(det['xy']);s=side_for(p);why='current_arm_uncertain';cj=None;support=np.zeros(21,bool);relations=[];matches=[]
                if s>=0:
                    if not inside(p[P],w,h).all():why='palm_outside_source'
                    else:
                        relations=[union_relation(p,b,s,bs,scale) for b,bs in zip(raw,raw_s)]
                        matches=[k for k,x in enumerate(relations) if x.startswith('same_')]
                        if matches:why='existing_identity_conflict' if any(e['baseline_sides'][k]!=s for k in matches) else 'confirm_existing_hand'
                        elif 'ambiguous_overlap' in relations:why='unresolved_existing_overlap'
                        elif s in e['baseline_sides']:why='existing_side_preserved'
                        else:
                            corroboration=[]
                            for k,c in enumerate(e['crop']):
                                q=np.array(c['xy']);b=c['box']
                                assert 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h
                                if c['requested_side']!=s or side_for(q)!=s:continue
                                m=inside(p,w,h)&inside(q,w,h)&inside(q-np.array(b[:2]),b[2]-b[0],b[3]-b[1])
                                m &= np.sqrt(((p-q)**2).sum(1))<=.08*scale
                                if m[P].all():corroboration.append((k,m))
                            if len(corroboration)==1:
                                cj,support=corroboration[0];why='new_hand_current_arm_and_two_image_scales'
                            else:why='image_scales_disagree_or_ambiguous'
                reconstructed.append(dict(side=s,reason=why,crop_index=cj,joint_supported=support.tolist(),relations=relations,matched_old=matches))
            for s in (0,1):
                tied=[q for q in reconstructed if q['side']==s and q['reason']=='new_hand_current_arm_and_two_image_scales']
                if len(tied)>1:
                    for q in tied:q['reason']='multiple_new_same_side'
            for j,(q,t) in enumerate(zip(e['decisions'],reconstructed)):
                for key in ['side','reason','crop_index','joint_supported','relations','matched_old']:
                    assert q[key]==t[key],(r['clip_id'],i,j,key)
                ok=t['reason']=='new_hand_current_arm_and_two_image_scales'
                assert q['accepted']==ok
                counts['decisions_verified']+=1;reasons[q['reason']]+=1
                if q['reason']=='existing_identity_conflict':
                    conflicts.append(dict(clip_id=r['clip_id'],label=r['label'],index=i,decision=j,side=q['side'],old=q['matched_old']))
                if not ok:continue
                s=q['side'];sl=slice(33+21*s,54+21*s);m=np.array(q['joint_supported']);p=np.array(e['full'][j]['xy'],np.float32)
                assert not old['observed'][i,sl].any() and s not in e['baseline_sides']
                expected['xy_px'][i,sl]=np.where(m[:,None],p,np.nan)
                expected['observed'][i,sl]=m;expected['usable'][i,sl]=m
                expected['hand_score'][i,s]=min(e['full'][j]['score'],e['crop'][q['crop_index']]['score'])
                counts['added_native_hand_frames']+=1;counts['added_native_joint_frames']+=int(m.sum())
                retained=masked['observed'][i,sl]
                counts['added_roi_hand_frames']+=bool(retained.any())
                counts['added_roi_palm_frames']+=bool(retained[P].all())
                counts['added_roi_joint_frames']+=int(retained.sum())
                counts['added_roi_core_hand_frames']+=bool(retained.any()) and e['core']
                counts['added_native_incomplete_hands']+=not bool(m.all())
                adds.append(dict(clip_id=r['clip_id'],label=r['label'],index=i,frame_id=e['frame_id'],
                    time_s=e['timestamp_us']/1e6,side=s,decision=j,crop_index=q['crop_index'],
                    native_points=int(m.sum()),roi_points=int(retained.sum()),core=e['core']))
            counts['frames']+=1;lat.append(e['full_ms']+e['crop_ms'])
        for k in ['xy_px','observed','usable','conf','hand_score','frame_id','timestamp_us']:assert same(expected[k],merged[k]),(r['clip_id'],k)
        box=r['box'];p=merged['xy_px'];mask=merged['observed']&np.isfinite(p).all(-1)&(p[...,0]>=box[0])&(p[...,0]<box[2])&(p[...,1]>=box[1])&(p[...,1]<box[3])
        assert same(mask,masked['observed']) and same(merged['usable']&mask,masked['usable'])
        assert same(np.where(mask[...,None],p,np.nan),masked['xy_px'])
        assert not np.isfinite(masked['xy_px'][~mask]).any()
        counts['outside_points_removed']+=int((merged['observed']&~mask).sum())
        counts['baseline_roi_palm_frames']+=sum(oldmasked['usable'][:,33+21*s+np.array(P)].all(1).sum() for s in (0,1))
        counts['merged_roi_palm_frames']+=sum(masked['usable'][:,33+21*s+np.array(P)].all(1).sum() for s in (0,1))
        for a in ann:
            if a['clip_id']==r['clip_id'] and a['view']=='masked':
                controls[a['verdict']]+=1
                # Original assigned hand never rewritten. Known error is retained, not fixed.
                assert same(old['xy_px'][a['index']],merged['xy_px'][a['index']]) or not np.array_equal(old['observed'][a['index']],merged['observed'][a['index']])
        controls['crossing_frames']+=sum(c['clip_id']==r['clip_id'] and c['view']=='masked' for c in cross)
    result=dict(status='SAVED_MERGE_AND_ROI_ARITHMETIC_PASS_PENDING_VISUAL',clips=len(idx),counts={k:int(v) for k,v in counts.items()},
        decisions=dict(reasons),controls=dict(controls),latency_ms=dict(median=float(np.median(lat)),p95=float(np.quantile(lat,.95))),
        protected_files=len(f['protected']),training=0,user_data=0,production_changes=0,
        caveat='Control values preserved by construction. Two image scales use same model/pixels; agreement is not independent anatomical validation.')
    for name,obj in [('VERIFY.json',result),('ADDITIONS.json',adds),('CONFLICTS.json',conflicts)]:
        (U/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
