"""Separate recomputation without importing audit.py; TRAIN observations only."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
U=Path(__file__).resolve().parent;P=U.parent/'word100_upperbody_preflight_20261014'
sys.stdout.reconfigure(encoding='utf-8')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=read(U/'RESULT.json');g=read(U/'SOURCE_GUARD.json')
manifest=read(U.parent/'word100_training_20261013/MANIFEST.json')
train={r['clip_id']:r for r in manifest if r['role']=='train_candidate'}
assert set(train)=={r['clip_id'] for r in a['rows']}
records=Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013/records')
boundary=[];checked=0
for r in a['rows']:
    m=train[r['clip_id']];rec=read(records/(r['clip_id']+'.json'));npz=Path(rec['observation'])
    s=read(npz.with_suffix('.sidecar.json'))
    with np.load(npz,allow_pickle=False) as z:
        xy=z['xy_px'];u=z['usable'];times=z['timestamp_us'];fid=z['frame_id']
    assert np.isfinite(xy[u]).all() and np.all(np.diff(times)>0)
    good=np.logical_and.reduce([u[:,k] for k in [11,12,23,24]])
    shoulder=np.median((xy[good,11]+xy[good,12])/2,axis=0)
    hip=np.median((xy[good,23]+xy[good,24])/2,axis=0)
    bottom=round(float(shoulder[1]+.75*(hip[1]-shoulder[1])))
    width=round(bottom*s['width']/s['height']);left=max(0,min(s['width']-width,round(float(shoulder[0]-width/2))))
    assert r['box']==[left,0,left+width,bottom]
    inside=np.logical_and.reduce([xy[:,:,0]>=left,xy[:,:,0]<left+width,xy[:,:,1]>=0,xy[:,:,1]<bottom])
    start,end=m['intervals'][0];tt=times/1e6
    for phase,lo,hi in [('core',start,end),('middle_third',start+(end-start)/3,end-(end-start)/3)]:
        ix=np.where((tt>=lo)&(tt<=hi))[0];assert len(ix)==r[phase]['frames']
        for side,offset in [('L',33),('R',54)]:
            valid=u[ix,offset:offset+21];keep=inside[ix,offset:offset+21]
            assert int(valid.sum())==r[phase][side]['observed_points']
            assert int((valid&~keep).sum())==r[phase][side]['outside_points']
            assert int(((valid.sum(1)>0)&((valid&keep).sum(1)==0)).sum())==r[phase][side]['no_point_inside_frames']
        no_inside=np.any(u[:,33:],axis=1)&~np.any(u[:,33:]&inside[:,33:],axis=1)
        assert int(no_inside[ix].sum())==r[phase]['both_observed_no_points_inside']
    # Longest such geometric gap inside the annotation; frame-step duration is explicit.
    flags=no_inside&(tt>=start)&(tt<=end);runs=[];beg=None
    step=float(np.median(np.diff(tt)))
    for f,v in enumerate(np.r_[flags,False]):
        if v and beg is None:beg=f
        elif not v and beg is not None:runs.append((beg,f-1,float(tt[f-1]-tt[beg]+step)));beg=None
    if r['middle_third']['both_observed_no_points_inside']:
        longest=max(runs,key=lambda x:x[2]);boundary.append({'clip_id':r['clip_id'],'label':r['canonical_label'],'signer_id':r['signer_id'],'middle_no_inside_frames':r['middle_third']['both_observed_no_points_inside'],'longest_core_run_frames':[longest[0],longest[1]],'longest_core_run_seconds':longest[2],'middle_frame':int(np.flatnonzero(no_inside&(tt>=start+(end-start)/3)&(tt<=end-(end-start)/3))[0])})
    checked+=1
assert all(sha(p)==h for block in g.values() for p,h in block.items())
diag=[]
for cid,frame in [('NIA_SL_WORD1133_REAL01_F',56),('NIA_SL_WORD1545_REAL01_F',60),('NIA_SL_WORD0003_REAL01_F',33)]:
    item=next(x for x in read(P/'PLAN.json')['items'] if x['clip_id']==cid)
    with np.load(P/cid/'native.npz') as z:n=z['xy_px'][frame];nu=z['usable'][frame]
    with np.load(P/cid/'crop.npz') as z:c=z['xy_px'][frame]+np.array(item['box'][:2]);cu=z['usable'][frame]
    diag.append({'clip_id':cid,'frame':frame,'crop_pose_wrist_usable_L_R':cu[[15,16]].tolist(),'native_hand_wrist_xy_L_R':n[[33,54]].tolist(),'crop_hand_wrist_xy_L_R':[[float(v) if np.isfinite(v) else None for v in row] for row in c[[33,54]]],'crop_right_to_native_left_px':float(np.linalg.norm(c[54]-n[33])),'crop_right_to_native_right_px':float(np.linalg.norm(c[54]-n[54]))})
result={'status':'PASS','checked_training_clips':checked,'source_files_unchanged':sum(len(b) for b in g.values()),'review_flags_clips':len(boundary),'review_flags_classes':len({b['label'] for b in boundary}),'boundary_flags':boundary,'assignment_diagnostics':diag,'meaning':'Independent algorithmic recount by same Codex author; not independent agent/expert review. Geometric no-point-inside is not measured crop detection or semantic damage.'}
(U/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['boundary_flags','assignment_diagnostics']},ensure_ascii=False))
print(json.dumps(diag,ensure_ascii=False))
