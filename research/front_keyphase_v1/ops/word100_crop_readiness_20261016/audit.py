"""Read-only TRAIN geometry and stored crop correspondence audit; no inference."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,time
import numpy as np

U=Path(__file__).resolve().parent
LAB=U.parents[1]
ROOT=LAB.parents[1]
PRIOR=U.parent/'word100_upperbody_preflight_20261014'
OBS=Path('W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013/observations')
sys.path.insert(0,str(LAB/'src'))
from wis_lab.core.observation import Observation
sys.stdout.reconfigure(encoding='utf-8')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
    (U/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def box_for(ob):
    good=ob.usable[:,[11,12,23,24]].all(1)
    if not good.any():return None
    sh=np.median(ob.xy_px[good][:,[11,12],:].mean(1),axis=0)
    hip=np.median(ob.xy_px[good][:,[23,24],:].mean(1),axis=0)
    w,h=int(ob.sidecar['width']),int(ob.sidecar['height'])
    if hip[1]<=sh[1]:return None
    bottom=int(round(sh[1]+.75*(hip[1]-sh[1])))
    if not 0<bottom<h:return None
    cw=int(round(bottom*w/h));x0=max(0,min(w-cw,int(round(sh[0]-cw/2))))
    return [x0,0,x0+cw,bottom]
def geometry(ob,box,ids):
    xy=ob.xy_px[ids];u=ob.usable[ids];x0,y0,x1,y1=box
    inside=(xy[:,:,0]>=x0)&(xy[:,:,0]<x1)&(xy[:,:,1]>=y0)&(xy[:,:,1]<y1)
    out={'frames':int(ids.sum())}
    for side,start in [('L',33),('R',54)]:
        xyh=xy[:,start:start+21];uh=u[:,start:start+21];ih=inside[:,start:start+21]
        n=int(uh.sum());nh=int(uh.any(1).sum())
        out[side]={'observed_points':n,'outside_points':int((uh&~ih).sum()),
            'horizontal_outside_points':int((uh&((xyh[:,:,0]<x0)|(xyh[:,:,0]>=x1))).sum()),
            'bottom_outside_points':int((uh&(xyh[:,:,1]>=y1)).sum()),
            'observed_hand_frames':nh,'no_point_inside_frames':int((uh.any(1)&~(uh&ih).any(1)).sum()),
            'partially_clipped_frames':int(((uh&ih).any(1)&(uh&~ih).any(1)).sum())}
        out[side]['outside_fraction']=out[side]['outside_points']/n if n else None
        out[side]['fully_outside_frame_fraction']=out[side]['no_point_inside_frames']/nh if nh else None
    out['both_observed_no_points_inside']=int((u[:,33:].any(1)&~(u[:,33:]&inside[:,33:]).any(1)).sum())
    return out
def main():
    assert not (U/'RESULT.json').exists(),'Do not overwrite completed audit'
    frozen=read(U.parent/'word100_training_20261013/MANIFEST.json')
    rows=[r for r in frozen if r['role']=='train_candidate']
    assert len(rows)==899 and {r['signer_id'] for r in rows}==set(range(1,10))
    pins=[U/'SCOPE.json',Path(__file__),U.parent/'word100_training_20261013/MANIFEST.json',PRIOR/'PLAN.json',PRIOR/'RESULT.json',LAB/'src/wis_lab/capture/landmarker.py',LAB/'config/word100_practice.json',LAB/'runs/word100_tcn_v1/whole_context/best.pt']
    guards={str(p):sha(p) for p in pins};records=[];source_pins={};st=time.perf_counter()
    for i,r in enumerate(rows):
        rp=OBS.parent/'records'/(r['clip_id']+'.json');record=read(rp)
        assert record['clip_id']==r['clip_id'] and record['role']=='train_candidate'
        assert record['manifest_sha256']==guards[str(U.parent/'word100_training_20261013/MANIFEST.json')]
        p=Path(record['observation']);sc=p.with_suffix('.sidecar.json');ob=Observation.load(str(p))
        h=sha(p);assert h==ob.sidecar['observation_sha256']==record['observation_sha256']
        assert ob.sidecar['clip_id']==r['clip_id']
        source_pins[str(p)]=h;source_pins[str(sc)]=sha(sc);source_pins[str(rp)]=sha(rp)
        box=box_for(ob)
        rec={k:r[k] for k in ['clip_id','word_code','signer_id','canonical_label']};rec['box']=box
        if box:
            a,b=r['intervals'][0];t=ob.timestamp_us/1e6;core=(t>=a)&(t<=b)
            mid=(t>=a+(b-a)/3)&(t<=b-(b-a)/3)
            rec['core']=geometry(ob,box,core);rec['middle_third']=geometry(ob,box,mid)
        records.append(rec)
        if (i+1)%200==0:print('audited',i+1,flush=True)
    pairs=[];pplan=read(PRIOR/'PLAN.json')
    for item in pplan['items']:
        cid=item['clip_id'];obs=[]
        for mode in ['native','crop']:
            p=PRIOR/cid/(mode+'.npz');ob=Observation.load(str(p));assert sha(p)==ob.sidecar['observation_sha256']
            source_pins[str(p)]=sha(p);source_pins[str(p.with_suffix('.sidecar.json'))]=sha(p.with_suffix('.sidecar.json'));obs.append(ob)
        n,c=obs;np.testing.assert_array_equal(n.timestamp_us,c.timestamp_us)
        mapped=c.xy_px+np.array(item['box'][:2],np.float32)
        a,b=item['core_seconds'];core=(n.timestamp_us>=a*1e6)&(n.timestamp_us<=b*1e6)
        usable=n.usable[:,[33,54]].all(1)&c.usable[:,[33,54]].all(1)&core
        pair_rows=[]
        for f in np.flatnonzero(usable):
            nw=n.xy_px[f,[33,54]];cw=mapped[f,[33,54]]
            same=float(np.linalg.norm(nw-cw,axis=1).sum());cross=float(np.linalg.norm(nw-cw[::-1],axis=1).sum())
            if cross<same:pair_rows.append({'frame':int(f),'seconds':float(n.timestamp_us[f]/1e6),'same_px':same,'cross_px':cross,'separation_native_px':float(np.linalg.norm(nw[0]-nw[1]))})
        missing={}
        x0,y0,x1,y1=item['box'];inside=(n.xy_px[:,:,0]>=x0)&(n.xy_px[:,:,0]<x1)&(n.xy_px[:,:,1]>=y0)&(n.xy_px[:,:,1]<y1)
        for side,start in [('L',33),('R',54)]:
            sl=slice(start,start+21)
            # Original estimated points inside does not guarantee visible, separable hands.
            lost=np.flatnonzero(core&n.usable[:,sl].all(1)&inside[:,sl].all(1)&~c.usable[:,sl].any(1))
            missing[side]=[{'frame':int(f),'seconds':float(n.timestamp_us[f]/1e6)} for f in lost]
        pairs.append({'clip_id':cid,'label':item['label'],'comparable_core_frames':int(usable.sum()),'cross_side_closer_frames':pair_rows,'all21_native_inside_but_crop_side_absent':missing,'assignment':{mode:{k:ob.sidecar[k] for k in ['handedness_label_disagreements','handedness_label_comparisons','unassigned_hand_detections']} for mode,ob in zip(['native','crop'],obs)}})
    valid=[r for r in records if r['box']]
    summary={'train_clips':len(records),'classes':len({r['word_code'] for r in records}),'signers':sorted({r['signer_id'] for r in records}),'valid_boxes':len(valid),'missing_boxes':len(records)-len(valid)}
    for phase in ['core','middle_third']:
        summary[phase]={'any_observed_point_outside_clips':sum(any(r[phase][s]['outside_points']>0 for s in ['L','R']) for r in valid),
          'horizontal_outside_clips':sum(any(r[phase][s]['horizontal_outside_points']>0 for s in ['L','R']) for r in valid),
          'both_observed_no_points_inside_clips':sum(r[phase]['both_observed_no_points_inside']>0 for r in valid),
          'one_side_fully_outside_at_least_half_observed_frames':sum(any((r[phase][s]['fully_outside_frame_fraction'] or 0)>=.5 for s in ['L','R']) for r in valid)}
    assert all(sha(p)==h for p,h in guards.items())
    write('SOURCE_GUARD.json',{'fixed':guards,'observations':source_pins})
    write('RESULT.json',{'status':'COMPUTED_PENDING_REVIEW','summary':summary,'rows':records,'stored_pairs':pairs,'seconds':time.perf_counter()-st,'scope':'TRAIN only; geometry is not cropped MediaPipe output or semantic ground truth','training':0,'inference':0,'extraction':0})
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
