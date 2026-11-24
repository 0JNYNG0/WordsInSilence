import sys,os,json,hashlib,time,shutil,traceback,importlib.util
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np,cv2,mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from candidate import PARAMS,PALM,crop_boxes,merge
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];LAB=O.parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(LAB/'src'),str(ROOT/'research/sign_model_pilot_v1/src')]
from wis_lab.core.observation import Observation
MODEL=ROOT/'Tools/models/hand_landmarker.task'
ROI=O/'word100_fullframe_roi_pair_20261016/roi_adapter.py'
spec=importlib.util.spec_from_file_location('full_merge_roi',ROI);roi=importlib.util.module_from_spec(spec);spec.loader.exec_module(roi)
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as fp:json.dump(x,fp,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def stamp():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def copy_obs(b):
    return Observation(**{k:getattr(b,k).copy() for k in ['xy_px','timestamp_us','frame_id','observed','usable','conf','hand_score']},
                       sidecar=dict(b.sidecar))

def freeze():
    assert not (U/'FROZEN.json').exists()
    old=read(O/'word100_current_image_redetection_20261020/FROZEN.json')
    check(old['protected']);rows=[];pins={}
    for oldr in old['rows']:
        assert oldr['role']=='train_candidate' and 1<=oldr['signer_id']<=9
        rec=read(oldr['record'])
        r={k:oldr[k] for k in ['clip_id','signer_id','label','word_code','role','video','box','intervals']}
        r.update(raw=oldr['views']['masked']['raw'],assignment=oldr['views']['masked']['assignment'],
                 native=rec['native_capture']['candidate'],old_masked=oldr['views']['masked']['observation'])
        rows.append(r)
        for k in ['video','raw','assignment','native','old_masked']:
            p=Path(r[k]);pins[str(p)]=sha(p)
            if k in ['native','old_masked']:
                sp=p.with_suffix('.sidecar.json');pins[str(sp)]=sha(sp)
    for p in [U/'candidate.py',U/'contracts.py',U/'CONTRACTS.json',U/'run.py',ROI,MODEL,
              O/'word100_image_arm_connectivity_20261018/ANNOTATED.json',
              O/'word100_image_arm_connectivity_20261018/CROSSING.json',
              ROOT/'research/sign_model_pilot_v1/src/pilot/data.py',
              ROOT/'research/sign_model_pilot_v1/src/pilot/web_input.py']:
        pins[str(p)]=sha(p)
    assert read(U/'CONTRACTS.json')['status']=='PASS'
    (U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
    write(U/'FROZEN.json',dict(at=stamp(),rows=rows,pins=pins,protected=old['protected'],params=PARAMS,
        authorization='User full-source existing/new hand merge then ROI, public validation, no training or production/user data.',
        pipeline='Reuse frozen native whole-video detection and identity. New IMAGE full-native frame, current-arm crop on SAME native frame for missing sides. Reconcile BEFORE fixed saved ROI. Crop is a local detector scale, not the upper-body sensor training view.',
        baseline_policy='Preserve existing native observations and side assignments. Matched redetection adds confirmation only; conflict flags never create second hand or force relabel. Only absent sides get current-arm and two-scale agreement-supported points.',
        quality_policy='Per-joint same-image scale agreement is a diagnostic proxy, not visibility/anatomical truth. Existing partial hands not overwritten; no fabricated interpolation.',
        gates='Review every added ROI-used hand-frame and every explicit identity conflict on RGB. Existing full-native labeled controls and crossing preservation, no new physical duplicates, ROI exclusion and no temporal loss. Failed visual gates prevent adoption; do not tune after outcomes.',
        scope='Same36 known TRAIN/23words/9people. Sensor prior duplicate is one separate contract regression fixture only. No sensor predictions used as native model input.',
        not_run=['training','classifier','user recordings','DEV/heldout','camera','downloads','production edit'],
        role='Codex implementation, separate saved arithmetic reconstruction and visual review by same Codex. Not independent human/agent.'))
    print('FROZEN36',flush=True)

def detect(lm,rgb,box=None):
    h,w=rgb.shape[:2];t=time.perf_counter()
    rr=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)))
    dt=(time.perf_counter()-t)*1000;out=[];x0,y0=box[:2] if box else (0,0)
    for pp,ss in zip(rr.hand_landmarks,rr.handedness):
        d=dict(xy=[[float(p.x*w+x0),float(p.y*h+y0)] for p in pp],
               label=ss[0].category_name,score=float(ss[0].score))
        if box:d['box']=box
        out.append(d)
    return out,dt

def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected'])
    assert not (U/'RUNNING.json').exists() and not (U/'INDEX.json').exists()
    write(U/'RUNNING.json',dict(pid=os.getpid(),at=stamp()))
    (U/'records').mkdir();(U/'outputs').mkdir();index=[]
    opts=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.5,
        min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for rn,r in enumerate(f['rows']):
            cid=r['clip_id'];z=load(r['raw']);ass=load(r['assignment'])['sides']
            native=Observation.load(r['native']);candidate=copy_obs(native)
            # Confirm that the reused full detection maps to the existing masked comparison.
            masked0=roi.apply_roi(native,r['box']);oldmask=Observation.load(r['old_masked'])
            for k in ['xy_px','observed','usable','frame_id','timestamp_us']:
                assert np.array_equal(getattr(masked0,k),getattr(oldmask,k),equal_nan=True),(cid,k)
            cap=cv2.VideoCapture(r['video']);assert cap.isOpened();fid=0;frames=[]
            for i,target in enumerate(z['frame_id']):
                while fid<=int(target):
                    ok,bgr=cap.read();assert ok;fid+=1
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                w,h=int(z['width'][i]),int(z['height'][i]);assert rgb.shape[:2]==(h,w)
                n=int(z['raw_count'][i]);sides=ass[i,:n].tolist()
                full,fullms=detect(lm,rgb)
                boxes=crop_boxes(z['pose_xy_px'][i],z['pose_usable'][i],sides,w,h)
                crop=[];cropms=0.
                for s,b in boxes.items():
                    ds,dt=detect(lm,rgb[b[1]:b[3],b[0]:b[2]],b);cropms+=dt
                    for d in ds:d['requested_side']=s
                    crop.extend(ds)
                decisions=merge(full,crop,z['hand_xy_px'][i,:n],sides,z['pose_xy_px'][i],z['pose_usable'][i],w,h)
                for q in decisions:
                    if not q['accepted']:continue
                    s=q['side'];sl=slice(33+21*s,54+21*s);valid=np.array(q['joint_supported'])
                    assert not native.observed[i,sl].any()
                    pts=np.asarray(full[q['index']]['xy'],np.float32)
                    candidate.xy_px[i,sl]=np.where(valid[:,None],pts,np.nan)
                    candidate.observed[i,sl]=valid;candidate.usable[i,sl]=valid
                    candidate.hand_score[i,s]=min(full[q['index']]['score'],crop[q['crop_index']]['score'])
                frames.append(dict(index=i,frame_id=int(target),timestamp_us=int(z['timestamp_us'][i]),
                    core=any(a<=z['timestamp_us'][i]/1e6<=b for a,b in r['intervals']),
                    baseline_sides=sides,full=full,crop=crop,decisions=decisions,boxes=boxes,
                    full_ms=fullms,crop_ms=cropms))
            cap.release();candidate.validate();masked=roi.apply_roi(candidate,r['box'])
            outdir=U/'outputs'/cid;outdir.mkdir()
            candidate.sidecar['research_hand_merge']='native_existing_current_image_agreement_v1_NOT_ADOPTED'
            candidate.save(str(outdir/'merged_native.npz'));masked.save(str(outdir/'masked.npz'))
            channels=roi.build_roi_channels(masked)
            np.savez_compressed(outdir/'channels.npz',**channels)
            # This is feature preparation only, never a classifier call.
            path=U/'records'/f'{cid}.json'
            write(path,dict(clip_id=cid,label=r['label'],signer_id=r['signer_id'],frames=frames))
            outputs={str(p):sha(p) for p in outdir.iterdir() if p.is_file()}
            index.append(dict(clip_id=cid,path=str(path),sha256=sha(path),outputs=outputs,frames=len(frames)))
            print('DONE',rn+1,36,cid,'add',sum(q['accepted'] for e in frames for q in e['decisions']),flush=True)
    write(U/'INDEX.json',index);check(f['pins']);check(f['protected']);(U/'RUNNING.json').unlink()
    print('COMPLETE',flush=True)

if __name__=='__main__':
    cv2.setNumThreads(1)
    try:freeze() if sys.argv[1]=='freeze' else run()
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',dict(at=stamp(),error=traceback.format_exc()))
        raise
