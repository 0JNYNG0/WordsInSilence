import sys, os, json, time, hashlib, shutil, traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from candidate import PARAMS, PALM, crops, decide

U = Path(__file__).resolve().parent
O = U.parent
ROOT = O.parents[2]
PREV = O/'word100_image_tracking_pair_20261020'
MODEL = ROOT/'Tools/models/hand_landmarker.task'


def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def write(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p, allow_pickle=False) as z: return {k:z[k] for k in z.files}
def check(pins):
    for p, h in pins.items(): assert sha(p) == h, p
def stamp(): return datetime.now(timezone(timedelta(hours=9))).isoformat()


def freeze():
    assert not (U/'FROZEN.json').exists()
    base = read(PREV/'FROZEN.json')
    check(base['protected'])
    pins = {str(PREV/'FROZEN.json'):sha(PREV/'FROZEN.json')}
    for r in base['rows']:
        assert r['role'] == 'train_candidate' and 1 <= r['signer_id'] <= 9
        for p in [r['video'], r['record']] + [
            r['views'][v][k] for v in ['masked','sensor'] for k in ['raw','assignment','observation']]:
            pins[p] = sha(p)
    for p in [U/'candidate.py', U/'run.py', U/'contracts.py', U/'CONTRACTS.json', MODEL,
              ROOT/'research/front_keyphase_v1/src/wis_lab/capture/landmarker.py',
              ROOT/'research/front_keyphase_v1/src/wis_lab/capture/video_source.py',
              O/'word100_image_arm_connectivity_20261018/ANNOTATED.json',
              O/'word100_image_arm_connectivity_20261018/CROSSING.json']:
        pins[str(p)] = sha(p)
    speed_ids = [sorted(r['clip_id'] for r in base['rows'] if r['signer_id']==s)[0] for s in [1,2,3]]
    assert read(U/'CONTRACTS.json')['status']=='PASS'
    (U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md', ROOT/'NEW_CHAT_HANDOFF.md', O/'STATE.json']:
        shutil.copyfile(p,U/'before'/p.name)
    write(U/'FROZEN.json',dict(at=stamp(), rows=base['rows'], pins=pins,
        protected=base['protected'],params=PARAMS,mediapipe=mp.__version__,
        authorization='User current-image redetection public comparison and initial/speed explanation; no training, production changes, user data, downloads.',
        conditions=['saved_C_sensor','IMAGE_full_sensor','IMAGE_current_pose_wrist_crop'],
        method='Existing local task weights; IMAGE always detects current pixels independently. Full sensor image every saved frame; crop only missing named C side with currently usable interior elbow/wrist. Baseline side and points never overwritten. No raw handedness used for acceptance. Current arm unique and duplicate guards.',
        scope='Same known36 TRAIN videos/23words/9people, no DEV or holdout/user. Engineering diagnosis, not unseen generalization.',
        speed=dict(ids=speed_ids,factors=[1,.75,.5,1],mode='VIDEO',pixels='same decoded sensor pixels once each; timestamp scaling only, no duplication/interpolation/sleep; last1 repeat-control'),
        initial='Audit existing full/native vs sensor raw first frame/first hand/known12identity errors. No human truth assumed for unreviewed first frames.',
        acceptance='Review every newly proposed natural frame on source RGB; report wrong/unclear/visible same side separately. Zero baseline modifications; preserved controls by construction are not improved identity accuracy. Any false or unresolved additions prevent automatic adoption. No threshold tuning after results.',
        role='Codex implementation plus separate saved check and visual review; not another independent reviewer'))
    print('FROZEN36',speed_ids,flush=True)


def detector(mode):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),running_mode=mode,
        num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,
        min_tracking_confidence=.5))


def serialize(result, w, h, box=None):
    out=[]
    ox,oy = box[:2] if box else [0,0]
    for pts, cats in zip(result.hand_landmarks,result.handedness):
        d=dict(xy=[[p.x*w+ox,p.y*h+oy] for p in pts],
               label=cats[0].category_name,handedness_score=float(cats[0].score))
        if box: d['box']=box
        out.append(d)
    return out


def image_detect(lm, rgb, box=None):
    h,w=rgb.shape[:2]
    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
    start=time.perf_counter();res=lm.detect(image);ms=(time.perf_counter()-start)*1000
    return serialize(res,w,h,box),ms


def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected'])
    assert not (U/'RUNNING.json').exists() and not (U/'INDEX.json').exists()
    write(U/'RUNNING.json',dict(pid=os.getpid(),at=stamp()))
    (U/'records').mkdir();index=[]
    with detector(vision.RunningMode.IMAGE) as lm:
        for n,r in enumerate(f['rows']):
            cid=r['clip_id'];z=load(r['views']['sensor']['raw']);obs=load(r['views']['sensor']['observation'])
            ass=load(r['views']['sensor']['assignment'])['sides']
            cap=cv2.VideoCapture(r['video']);assert cap.isOpened();fid=0;entries=[];speed_images=[]
            for i,target in enumerate(z['frame_id']):
                while fid<=int(target):
                    ok,bgr=cap.read();assert ok;(fid:=fid+1)
                w,h=int(z['width'][i]),int(z['height'][i]);rgb=cv2.cvtColor(bgr[:h,:w],cv2.COLOR_BGR2RGB)
                assert rgb.shape[:2]==(h,w)
                count=int(z['raw_count'][i]);hands=z['hand_xy_px'][i,:count];sides=ass[i,:count].tolist()
                pose=z['pose_xy_px'][i];good=z['pose_usable'][i];ts=int(z['timestamp_us'][i])
                dets,ms=image_detect(lm,rgb)
                full=decide(dets,hands,sides,pose,good,w,h)
                boxes=crops(pose,good,sides,w,h);cropdet=[];cropms=0
                for side,box in boxes.items():
                    x0,y0,x1,y1=box
                    dd,dt=image_detect(lm,rgb[y0:y1,x0:x1],box);cropms+=dt
                    for d in dd:d['requested_side']=side
                    cropdet.extend(dd)
                crop=decide(cropdet,hands,sides,pose,good,w,h)
                # Wrong crop-side associations remain diagnostics; no forced reassignment.
                for result in crop:
                    if result['accepted'] and result['side']!=cropdet[result['index']]['requested_side']:
                        result.update(accepted=False,reason='different_arm_than_crop_request')
                entries.append(dict(index=i,frame_id=int(target),timestamp_us=ts,
                    core=any(a<=ts/1e6<=b for a,b in r['intervals']),
                    baseline_sides=sides,baseline_palm_usable=[bool(obs['usable'][i,np.array(PALM)+o].all()) for o in [33,54]],
                    image=dets,image_checks=full,crop=cropdet,crop_checks=crop,
                    full_ms=ms,crop_ms=cropms,boxes=boxes))
                if cid in f['speed']['ids']:speed_images.append((rgb.copy(),ts))
            cap.release()
            speed=[]
            if speed_images:
                for speed_id,factor in enumerate(f['speed']['factors']):
                    states=[];last=-1;timing=[]
                    with detector(vision.RunningMode.VIDEO) as video_lm:
                        for rgb,ts in speed_images:
                            ms=max(last+1,round(ts/1000/factor));last=ms
                            im=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
                            t=time.perf_counter();rr=video_lm.detect_for_video(im,ms);timing.append((time.perf_counter()-t)*1000)
                            states.append(dict(timestamp_ms=ms,hands=serialize(rr,rgb.shape[1],rgb.shape[0])))
                    speed.append(dict(factor=factor,pass_index=speed_id,frames=states,inference_ms=timing))
            out=dict(clip_id=cid,label=r['label'],signer_id=r['signer_id'],frames=entries,speed=speed)
            path=U/'records'/f'{cid}.json';write(path,out)
            index.append(dict(clip_id=cid,path=str(path),sha256=sha(path),frames=len(entries)))
            additions=[sum(q['accepted'] for e in entries for q in e[k]) for k in ['image_checks','crop_checks']]
            print('DONE',n+1,36,cid,'frames',len(entries),'proposals',additions,flush=True)
    write(U/'INDEX.json',index);check(f['pins']);check(f['protected']);(U/'RUNNING.json').unlink()
    print('COMPLETE',flush=True)


if __name__=='__main__':
    cv2.setNumThreads(1)
    try:freeze() if sys.argv[1]=='freeze' else run()
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',dict(at=stamp(),error=traceback.format_exc()))
        raise
