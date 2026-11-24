"""Isolated geometric alignment diagnostics, never creates certified target labels."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;P=U.parent/'word100_aihub_coordinate_quality_20261026'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(n,x):
    p=U/n;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def stats(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(n=len(x),median=float(np.median(x)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None)
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def freeze():
    previous=read(P/'FROZEN.json');verified=read(P/'VERIFIED.json');check(verified['pins']);check(previous['protected'])
    pins={str(P/n):sha(P/n) for n in ['FROZEN.json','DONE.json','VERIFIED.json','DIAGNOSTICS.json','VISUAL_REVIEW.json']}
    rows=[]
    for r in read(P/'DONE.json')['rows']:
        src=next(x for x in previous['rows'] if x['clip_id']==r['clip_id'])
        assert src['signer_id']==2 and src['role']=='train_candidate'
        assert sha(r['arrays'])==r['arrays_sha'] and sha(src['candidate'])==src['candidate_sha']
        rows.append(dict(clip_id=r['clip_id'],label=r['label'],array=r['arrays'],array_sha=r['arrays_sha'],mp=src['candidate'],mp_sha=src['candidate_sha'],frames=r['frames'],intervals=src['intervals'],missing_frames=r['missing_frames'],extra_frames=r['extra_frames']))
    ids=['NIA_SL_WORD1290_REAL02_F','NIA_SL_WORD2029_REAL02_F','NIA_SL_WORD0003_REAL02_F','NIA_SL_WORD0185_REAL02_F','NIA_SL_WORD0920_REAL02_F','NIA_SL_WORD1545_REAL02_F']
    visual=[]
    for cid in ids:
        r=next(r for r in rows if r['clip_id']==cid);a,b=r['intervals'][0]
        frames=sorted(set([0,r['frames']-1]+[int(round((a+(b-a)*q)*30)) for q in [.25,.5,.75]]))
        vp=P/'clips'/(cid+'.mp4');assert vp.exists();pins[str(vp)]=sha(vp)
        visual.append(dict(clip_id=cid,label=r['label'],frames=frames,video=str(vp)))
    write('AUTHORIZATION.json',dict(at=now(),request='Continue source quality work; test common body/forearm-based correction for coordinate offsets',training=False,personal_data_access=False,operating_change=False))
    write('FROZEN.json',dict(at=now(),rows=rows,visuals=visual,pins=pins,protected=previous['protected'],conditions={
      'A':'Provided source hand XY unchanged',
      'B':'One translation per clip; median MP minus provided body shoulders/elbows at frames 0..14; no hand points used; applied to every available hand XY',
      'C':'Per-frame per-hand rigid translation; MP pose wrist minus provided pose wrist, only if both elbows/wrists observed and confidence >=0.5. No interpolation. Not a ground-truth correction.'},body_mapping={'MP':[11,12,13,14],'AH':[5,2,6,3]},wrist_mapping={'MP':[15,16],'AH':[7,4]},elbow_mapping={'MP':[13,14],'AH':[6,3]},calibration_frames=15,confidence=.5,diagnostic_only=True,gate='No adoption from MP agreement alone. Inspect RGB at unused dynamic times, preserve all hand geometry/missing masks, report normal regressions. No numeric correctness claim without independent target points.',missing='Never create missing frame/hand points. C unavailable when arm observations fail; compare paired metrics only and report availability.',threshold_tuning=False))
    print('FROZEN',len(rows),len(visual),flush=True)

def compute(r,f):
    z=np.load(r['array']);mp=np.load(r['mp']);n=r['frames'];raw=z['official'][...,:2];valid=z['official_valid'];body=z['official_pose']
    ah=f['body_mapping']['AH'];mi=f['body_mapping']['MP'];conf=f['confidence']
    bodyok=(np.isfinite(body[:,ah]).all(-1)&(body[:,ah,2]>0)&mp['observed'][:,mi]&(mp['conf'][:,mi]>=conf)&np.isfinite(mp['xy_px'][:,mi]).all(-1))
    delta=mp['xy_px'][:,mi]-body[:,ah,:2];cal=np.arange(n)<f['calibration_frames'];fit=delta[cal][bodyok[cal]]
    assert len(fit)>=12,'insufficient body calibration points'
    shift=np.median(fit,axis=0);B=raw+shift
    wa=f['wrist_mapping']['AH'];wm=f['wrist_mapping']['MP'];ea=f['elbow_mapping']['AH'];em=f['elbow_mapping']['MP']
    cw=mp['xy_px'][:,wm]-body[:,wa,:2]
    cok=(np.isfinite(body[:,wa]).all(-1)&np.isfinite(body[:,ea]).all(-1)&(body[:,wa,2]>0)&(body[:,ea,2]>0)&mp['observed'][:,wm]&mp['observed'][:,em]&(mp['conf'][:,wm]>=conf)&(mp['conf'][:,em]>=conf)&np.isfinite(mp['xy_px'][:,wm]).all(-1)&np.isfinite(mp['xy_px'][:,em]).all(-1))
    C=raw+cw[:,:,None,:];C[~cok]=np.nan
    arr=dict(A=raw.copy(),B=B,C=C,source_valid=valid.copy(),c_available=cok,body_fit_mask=bodyok&cal[:,None],body_delta=delta,body_valid=bodyok,global_translation=shift,wrist_translation=cw,mp_xy=z['mp_xy'],mp_valid=z['mp_valid'],frame_id=z['frame_id'],timestamp_us=z['timestamp_us'])
    # All metrics compare with a noisy detector, not ground truth.
    after=np.arange(n)>=f['calibration_frames'];t=z['timestamp_us']/1e6;active=np.zeros(n,bool)
    for lo,hi in r['intervals']:active|=(t>=lo)&(t<=hi)
    metrics={}
    for scope,scopeok in [('post_calibration',after),('annotated_action',active),('all',np.ones(n,bool))]:
        common=valid&z['mp_valid']&scopeok[:,None,None]
        paired=common&cok[:,:,None]
        out={}
        for key,values in [('A',raw),('B',B),('C',C)]:
            diff=np.linalg.norm(values-z['mp_xy'],axis=-1);v=common if key!='C' else paired
            out[key]=dict(all_points=stats(diff[v]),wrist=stats(diff[:,:,0][v[:,:,0]]),paired_wrist=stats(diff[:,:,0][paired[:,:,0]]),paired_fingertips=stats(diff[:,:,[4,8,12,16,20]][paired[:,:,[4,8,12,16,20]]]))
        metrics[scope]=out
    # Per-hand translations preserve internal distances and angles mathematically.
    geometry=[]
    for key in ['B','C']:
        v=valid if key=='B' else valid&cok[:,:,None]
        orig=raw-raw[:,:,0,None,:];new=arr[key]-arr[key][:,:,0,None,:]
        dv=v&v[:,:,0,None];err=np.abs(orig-new)[np.broadcast_to(dv[...,None],orig.shape)]
        geometry.append(dict(condition=key,max_local_xy_delta=float(err.max()) if err.size else None))
    global_step=np.linalg.norm(np.diff(B,axis=0)-np.diff(raw,axis=0),axis=-1)
    validstep=valid[1:]&valid[:-1]
    local_step=np.linalg.norm(np.diff(C,axis=0),axis=-1);base_step=np.linalg.norm(np.diff(raw,axis=0),axis=-1)
    cstep=validstep&cok[1:,:,None]&cok[:-1,:,None]
    hold_body=bodyok&after[:,None]
    rr=dict(clip_id=r['clip_id'],label=r['label'],global_translation=shift.tolist(),fit_points=len(fit),body_residual_post=stats(np.linalg.norm(delta-shift,axis=-1)[hold_body]),body_delta_component_spread_p90=[float(np.quantile(np.abs(delta[...,j]-shift[j])[hold_body],.9)) for j in range(2)],c_hand_frames_available=int(cok.sum()),total_hand_frames=n*2,c_existing_hand_frames_available=int((cok&valid[:,:,0]).sum()),existing_hand_frames=int(valid[:,:,0].sum()),metrics=metrics,geometry=geometry,global_temporal_delta_max=float(global_step[validstep].max()),local_temporal_steps=dict(A=stats(base_step[cstep]),C=stats(local_step[cstep])),not_accuracy=True)
    return arr,rr

def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);results=[]
    (U/'arrays').mkdir(exist_ok=True)
    for r in f['rows']:
        assert sha(r['array'])==r['array_sha'] and sha(r['mp'])==r['mp_sha'];arr,res=compute(r,f)
        ap=U/'arrays'/(r['clip_id']+'.npz');assert not ap.exists();np.savez_compressed(ap,**arr);res['array']=str(ap);res['array_sha']=sha(ap)
        write('records/'+r['clip_id']+'.json',res);results.append(res)
    check(f['protected']);write('DONE.json',dict(at=now(),status='DIAGNOSTICS_COMPLETE_NOT_ADOPTED',rows=results,training=0,new_detection=0,personal_data_access=0))
    print('DONE',len(results),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
