import json,hashlib,shutil,time,os,sys,traceback,math
from pathlib import Path
from datetime import datetime,timezone,timedelta
import cv2,numpy as np
from candidate import Tracker,PARAMS,inside
U=Path(__file__).resolve().parent;P=U.parent.parent/'word100_image_tracking_pair_20261020';R=U.parents[4]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def clean(x):
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x
def write(p,x):Path(p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def check(x):
    for p,h in x.items():assert sha(p)==h,p
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def freeze():
    assert not (U/'FROZEN.json').exists() and read(U/'CONTRACTS.json')['status']=='PASS'
    old=read(P/'FROZEN.json');check(old['pins']);check(old['protected'])
    assert not (P/'RUNNING.json').exists()
    pins=dict(old['pins'])
    for p in [P/'FROZEN.json',P/'VERIFY.json',P/'INDEX.json',P/'EPISODES.json',P/'SYNTHETIC.json',
              P/'ANNOTATED.json',P/'CROSSINGS.json',U/'candidate.py',U/'contracts.py',U/'CONTRACTS.json',U/'run.py']:
        pins[str(p)]=sha(p)
    for r in read(P/'INDEX.json'):assert sha(r['path'])==r['sha256'];pins[r['path']]=r['sha256']
    first=read(U.parent/'FROZEN.json');assert first['params']==PARAMS
    for pp in [U.parent/'FROZEN.json',U.parent/'candidate.py',U.parent/'run.py',U.parent/'VERIFY.json',U.parent/'INDEX.json']:
        pins[str(pp)]=sha(pp)
    for item in read(U.parent/'INDEX.json'):
        assert sha(item['path'])==item['sha256'];pins[item['path']]=item['sha256']
    (U/'before').mkdir(exist_ok=True)
    for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',U.parent.parent/'STATE.json']:
        dest=U/'before'/p.name
        if dest.exists():assert sha(dest)==sha(p),p
        else:shutil.copyfile(p,dest)
    write(U/'FROZEN.json',dict(at=now(),rows=old['rows'],params=PARAMS,pins=pins,protected=old['protected'],
        authorization='User approves public video region/center separation and current-observation revalidation on rotation/reappearance; no training; C/T3F/user recordings unchanged.',
        scope='Same36 TRAIN clips and exact saved70x6 dropout targets. No new labels, DEV/holdout, webcam, user/mentor data, inference models, downloads.',
        method='Current measured C palm center kept separate from tracked texture region and similarity-warped anatomical center estimate. Similarity includes in-plane rotation and uniform scale, not 3D reconstruction. Current arm-confirmed observations reset anatomy; still-valid pixels inside current palm may be retained when fresh corners are insufficient. C sides unchanged deliberately.',
        comparison='Reuse old scores/traces. Natural missing-palm center and region support, all added centers visually reviewed. Synthetic420 support vs abstain, all-supported and same-target paired proxy errors; report regressions/coverage loss, never claim abstention as corrected coordinates. Reappearance/rotation current-priority checks. Known normal42,error12,cross54 unchanged does not prove new generalization.',
        gates='No current observation deletion, side reassignment, observed fingers creation, outside/future image, hidden dropout center leakage. Numeric gates not tuned after results. Public image position proxy not exact physical truth or sign-acceptance threshold.',
        numerical_rationale='Retain old LK/time/arm thresholds; fixed similarity motion plausibility, 70% robust inliers, feature-spread/extrapolation gates.15deg only identifies observed rotation review events,0.10 shoulder only descriptive error reporting as old study.',
        revision='After first36 result, fixed unnecessary pixel-track deletion: retain only forward/backward-valid prior points inside current palm when fresh corners insufficient. Current center reset unchanged. All numeric gates and targets unchanged. First candidate evidence preserved; not a new independent test.',
        role='Codex implementation, separate saved-arithmetic verifier and Codex visual review; no other agent/human review.',
        output_states=['observed','estimated','region_only','missing']))
    print('FROZEN',len(old['rows']),'clips',flush=True)
def run():
    c=read(U/'FROZEN.json');check(c['pins']);check(c['protected'])
    assert not (U/'RUNNING.json').exists() and not (U/'INDEX.json').exists()
    write(U/'RUNNING.json',dict(pid=os.getpid(),at=now()))
    (U/'records').mkdir(exist_ok=True);idx=[]
    for ri,r in enumerate(c['rows']):
        cid=r['clip_id'];z=load(r['views']['sensor']['raw']);old=read(P/'records'/f'{cid}.json')
        trackers=[Tracker(),Tracker()];cap=cv2.VideoCapture(r['video']);assert cap.isOpened()
        frame=0;entries=[];latency=[]
        for e in old['frames']:
            f=e['index'];fid=e['frame_id']
            while frame<=fid:ok,image=cap.read();assert ok;(frame:=frame+1)
            w,h=int(z['width'][f]),int(z['height'][f]);assert r['box']==[0,0,w,h]
            fac=PARAMS['image_width']/w;gray=cv2.cvtColor(cv2.resize(image[:h,:w],(PARAMS['image_width'],round(h*fac))),cv2.COLOR_BGR2GRAY)
            n=int(z['raw_count'][f]);hands=z['hand_xy_px'][f,:n]*fac;pose=z['pose_xy_px'][f]*fac
            usable=z['pose_usable'][f].copy()&inside(pose,gray.shape[1],gray.shape[0])
            out=[]
            for vi in (0,1):
                suppressed=[] if vi==0 else e['suppressed']
                hidden={e['baseline_sides'][i] for i in suppressed}
                measured=[None if xy is None or s in hidden else (np.array(xy)*fac).tolist() for s,xy in enumerate(e['direct'])]
                start=time.perf_counter()
                o=trackers[vi].step(gray,hands,e['baseline_sides'],pose,usable,e['time'],measured,suppressed)
                if vi==0:latency.append((time.perf_counter()-start)*1000)
                out.append(o)
            entries.append(dict(index=f,frame_id=fid,time=e['time'],core=e['core'],factor=fac,
                baseline_sides=e['baseline_sides'],direct=e['direct'],old_estimate=e['estimate'],
                old_test_estimate=e['test_estimate'],scale_px=e['scale_px'],natural=out[0],test=out[1]))
        cap.release()
        dest=U/'records'/f'{cid}.json';assert not dest.exists()
        write(dest,dict(clip_id=cid,label=r['label'],signer_id=r['signer_id'],blocks=old['blocks'],
            frames=entries,per_frame_ms=latency))
        idx.append(dict(clip_id=cid,path=str(dest),sha256=sha(dest),frames=len(entries)))
        print('COMPARE',ri+1,cid,flush=True)
    check(c['pins']);check(c['protected'])
    write(U/'INDEX.json',idx);(U/'RUNNING.json').unlink()
    print('COMPLETE',len(idx),flush=True)
if __name__=='__main__':
    cv2.setNumThreads(1)
    try:freeze() if sys.argv[1]=='freeze' else run()
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',dict(at=now(),error=traceback.format_exc()))
        raise
