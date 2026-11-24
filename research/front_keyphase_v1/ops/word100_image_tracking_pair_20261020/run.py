import json,hashlib,sys,time,os,traceback,zipfile,zlib,shutil
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import numpy as np,cv2
from tracker import Tracker,PARAMS,PALM,inside

U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/'word100_image_arm_connectivity_20261018'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p

def prepare():
    assert not (U/'FROZEN.json').exists()
    ctx=read(P/'CONTEXT.json');rows={r['clip_id']:r for r in ctx['rows']}
    ann=read(P/'ANNOTATED.json');cross=read(P/'CROSSING.json')
    nine=read(O/'word100_remaining_identity_audit_20261018/REMAINING9.json')
    selected={};reasons=defaultdict(list)
    def add(cid,why):
        if cid in rows:selected[cid]=rows[cid];reasons[cid].append(why)
    for a in nine:add(a['clip_id'],'prior_initial_identity_error')
    # Earlier public diagnostic words; select first available TRAIN performer only.
    for word in ['마음에들다','마음에 들다','냉커피','수영','믿다','감사','상처','기쁘다']:
        matches=[r for r in rows.values() if r['label']==word]
        if matches:add(sorted(matches,key=lambda r:r['clip_id'])[0]['clip_id'],'prior_gap_or_reentry_context')
    for sid in range(1,10):
        candidates=sorted({a['clip_id'] for a in cross if rows[a['clip_id']]['signer_id']==sid})
        if candidates:add(candidates[0],'crossing_control')
    # Fill to36 with greatest number of known normal identity labels, round-robin people.
    counts=Counter(a['clip_id'] for a in ann if a['verdict']=='normal_identity' and a['view']=='sensor')
    while len(selected)<36:
        changed=False
        for sid in range(1,10):
            pool=[r for r in rows.values() if r['signer_id']==sid and r['clip_id'] not in selected]
            if pool and len(selected)<36:
                r=sorted(pool,key=lambda r:(-counts[r['clip_id']],r['clip_id']))[0]
                add(r['clip_id'],'normal_identity_balanced_people');changed=True
        if not changed:break
    assert len(selected)==36
    video_lookup={x['clip_id']:x['source_video'] for x in read(P/'RESULTS.json')}
    pins={}
    for cid,r in selected.items():
        assert r['role']=='train_candidate' and 1<=r['signer_id']<=9
        v=Path(video_lookup[cid]);assert v.exists() and sha(v)==r['source']['sha256']
        r['video']=str(v);r['selection_reasons']=reasons[cid]
        pins[str(v)]=sha(v)
        for k in ['raw','assignment','observation']:
            p=r['views']['sensor'][k];pins[p]=sha(p)
        pins[r['record']]=sha(r['record'])
    for p in [P/'CONTEXT.json',P/'ANNOTATED.json',P/'CROSSING.json',
              O/'word100_remaining_identity_audit_20261018/REMAINING9.json',
              U/'tracker.py',U/'run.py',U/'contracts.py',U/'CONTRACTS.json']:
        pins[str(p)]=sha(p)
    protected=read(O/'word100_repeat_compression_guard_20261020/FROZEN.json')['protected'];check(protected)
    (U/'before').mkdir(exist_ok=True)
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
        assert not (U/'before'/p.name).exists();shutil.copyfile(p,U/'before'/p.name)
    assert read(U/'CONTRACTS.json')['status']=='PASS'
    write(U/'FROZEN.json',dict(at=now(),params=PARAMS,rows=list(selected.values()),pins=pins,protected=protected,
        authorization='User approved public problem/motion visualization and current vs image tracking comparison. No training, C/T3F change or user data.',
        method='Same cropped sensor pixels and saved C raw detections/assignment. Causal LK forward/backward features, current arm anatomical seeding; max200ms support, no finger landmark creation. Unique image+current arm may suggest side correction. Unchanged baseline protected as separate file.',
        scope='36 known diagnostic TRAIN clips selected before new results;9people. No new detector, no classification, no DEV/holdout. Existing curated normal/identity/crossing evidence reused only for matching sensor frames.',
        acceptance='All new side proposals and natural missing-position episodes visually review; normal/crossing newly wrong proposals reject adoption. Position estimates not observed fingers. Missing position support is not improvement until image review. Report abstentions, proxy error and latency; no thresholds changed after results.',
        controls='Actual sensor videos primary. Per hand first contiguous6frames with complete usable palm inside annotatedcore after >=3same-hand prior frames is a synthetic detector dropout; RGB and pose retained, hand candidate suppressed. Hidden observed palm positions are proxy targets, not independent truth. Causal forward predicted position checked also on unsuppressed current detections.',
        feedback='Visibility/recording guidance distinct from sign correction. Public interperson variation is not a validated acceptable-sign interval.',
        role='Same Codex author; separate reconstruction and visual review, not another person/agent'))
    print('FROZEN',len(selected),'videos',len(pins),'pins',flush=True)

def dropout_blocks(r,z,obs,assignment):
    blocks={}
    ts=z['timestamp_us']/1e6
    core=np.array([any(a<=t<=b for a,b in r['intervals']) for t in ts])
    for s,offset in [(0,33),(1,54)]:
        ok=obs['usable'][:,np.array(PALM)+offset].all(1)&core
        for f in range(3,len(ok)-5):
            if ok[f-3:f+6].all():
                # Require an actual detector candidate belonging to this C side.
                if all(s in assignment[g,:int(z['raw_count'][g])] for g in range(f,f+6)):
                    blocks[s]=[f,f+6];break
    return blocks

def run():
    c=read(U/'FROZEN.json');check(c['pins']);check(c['protected'])
    assert not (U/'RUNNING.json').exists() and not (U/'SUMMARY.json').exists()
    write(U/'RUNNING.json',dict(pid=os.getpid(),at=now()))
    for d in ['records','plots','evidence','videos']:(U/d).mkdir(exist_ok=True)
    results=[];counts=Counter();start=time.monotonic()
    for ri,r in enumerate(c['rows']):
        cid=r['clip_id'];z=load(r['views']['sensor']['raw']);obs=load(r['views']['sensor']['observation'])
        ass=load(r['views']['sensor']['assignment'])['sides'];blocks=dropout_blocks(r,z,obs,ass)
        trackers=[Tracker(),Tracker()];cap=cv2.VideoCapture(r['video']);assert cap.isOpened()
        fps=cap.get(cv2.CAP_PROP_FPS);frame=0;entries=[];timing=[]
        for f,fid in enumerate(z['frame_id']):
            while frame<=int(fid):
                success,image=cap.read();assert success,(cid,int(fid),frame)
                frame+=1
            # crop before any image processing; no source pixels below sensor.
            w,h=int(z['width'][f]),int(z['height'][f]);crop=image[:h,:w]
            factor=PARAMS['image_width']/w
            gray=cv2.cvtColor(cv2.resize(crop,(PARAMS['image_width'],int(round(h*factor)))),cv2.COLOR_BGR2GRAY)
            n=int(z['raw_count'][f]);hands=z['hand_xy_px'][f,:n]*factor
            base=ass[f,:n].astype(int);pose=z['pose_xy_px'][f]*factor;usable=z['pose_usable'][f].copy()
            usable &= inside(pose,gray.shape[1],gray.shape[0])
            ts=float(z['timestamp_us'][f]/1e6)
            begin=time.perf_counter()
            natural=trackers[0].step(gray,hands,base,pose,usable,ts)
            timing.append(time.perf_counter()-begin)
            suppressed=[i for i,s in enumerate(base) if int(s) in blocks and blocks[int(s)][0]<=f<blocks[int(s)][1]]
            test=trackers[1].step(gray,hands,base,pose,usable,ts,suppressed)
            # Retain only measured baseline hand coordinates as direct evidence.
            centers=[]
            for side,offset in [(0,33),(1,54)]:
                ps=np.array(PALM)+offset
                centers.append(obs['xy_px'][f,ps].mean(0).tolist() if obs['usable'][f,ps].all() else None)
            estimates=[[None if xy is None else (np.array(xy)/factor).tolist() for xy in out['estimate']] for out in [natural,test]]
            core=any(a<=ts<=b for a,b in r['intervals'])
            entries.append(dict(index=f,frame_id=int(fid),time=ts,core=core,baseline_sides=base.tolist(),
                direct=centers,estimate=estimates[0],test_estimate=estimates[1],natural=natural,
                test=test,suppressed=suppressed,scale_px=float(np.linalg.norm(z['pose_xy_px'][f,11]-z['pose_xy_px'][f,12])),
                raw_palms=[x[PALM].mean(0).tolist() for x in z['hand_xy_px'][f,:n]]))
        cap.release()
        out=dict(clip_id=cid,label=r['label'],signer_id=r['signer_id'],frames=entries,blocks=blocks,
                 elapsed_tracker_s=sum(timing),per_frame_ms=[1000*t for t in timing],fps=fps)
        path=U/'records'/f'{cid}.json';write(path,out)
        results.append(dict(clip_id=cid,label=r['label'],frames=len(entries),blocks=blocks,
                            path=str(path),sha256=sha(path),elapsed_tracker_s=sum(timing)))
        counts['frames']+=len(entries)
        print('TRACK',ri+1,'/',len(c['rows']),cid,len(entries),flush=True)
    write(U/'INDEX.json',results)
    write(U/'SUMMARY.json',dict(status='EXECUTED_PENDING_VERIFICATION_VISUAL',clips=len(results),counts=dict(counts),
        elapsed_seconds=time.monotonic()-start,training=0,classifier_inference=0,new_mediapipe=0,user_reads=0))
    check(c['pins']);check(c['protected']);(U/'RUNNING.json').unlink()
    print('DONE',dict(counts),flush=True)

if __name__=='__main__':
    cv2.setNumThreads(1)
    try:prepare() if sys.argv[1]=='prepare' else run()
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',dict(at=now(),error=traceback.format_exc()))
        raise
