"""Frozen detector-only four-way experiment on public original-size video.

Existing observations are reused for exact accuracy baselines; fresh same-device
detector calls measure component latency. No classifier, learning or webcam.
"""
from pathlib import Path
import sys,os,json,time,hashlib,importlib.util,traceback,datetime,shutil
import numpy as np,cv2
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def module(name,p):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
MERGE=O/'word100_fullframe_hand_merge_20261020/candidate.py'
rule=module('fourway_frozen_merge',MERGE)
CFG={'imgsz':640,'conf':0.25,'iou':0.7,'max_det':10,'device':'cpu','verbose':False,'save':False,'half':False,'augment':False,'batch':1}
POINT_CONF=.5
def roi(xy,valid,box):
    x0,y0,x1,y1=box
    mask=valid & np.isfinite(xy).all(-1)&(xy[...,0]>=x0)&(xy[...,0]<x1)&(xy[...,1]>=y0)&(xy[...,1]<y1)
    return np.where(mask[...,None],xy,np.nan).astype(np.float32),mask

def single_side(detections,pose,usable,w,h):
    good,scale=rule.context(pose,usable,w,h);side=[rule.arm(np.asarray(d['xy']),pose,good,scale) for d in detections]
    xy=np.full((2,21,2),np.nan,np.float32);valid=np.zeros((2,21),bool)
    reasons=[]
    for s in (0,1):
        indices=[i for i,a in enumerate(side) if a==s]
        if len(indices)==1:
            d=detections[indices[0]];p=np.asarray(d['xy']);v=np.asarray(d['valid'])&rule.inside(p,w,h)
            xy[s]=np.where(v[:,None],p,np.nan);valid[s]=v
        reasons.append('one_current_arm_candidate' if len(indices)==1 else 'no_current_arm_candidate' if not indices else 'multiple_same_arm_withheld')
    return xy,valid,side,reasons

def combine(detections,crops,z,i):
    n=int(z['raw_raw_count'][i]);base=z['native_xy_px'][i,33:].reshape(2,21,2).copy()
    valid=z['native_usable'][i,33:].reshape(2,21).copy()
    sides=z['assignment_sides'][i,:n].tolist();w=int(z['width'][i]);h=int(z['height'][i])
    q=rule.merge(detections,crops,z['raw_hand_xy_px'][i,:n],sides,z['raw_pose_xy_px'][i],z['raw_pose_usable'][i],w,h)
    for d in q:
        if d['accepted']:
            s=d['side'];assert not z['native_observed'][i,33+21*s:54+21*s].any()
            v=np.asarray(d['joint_supported']);p=np.asarray(detections[d['index']]['xy'])
            base[s]=np.where(v[:,None],p,np.nan);valid[s]=v
    return base,valid,q

def freeze():
    assert not (U/'FROZEN.json').exists()
    bundle=O/'word100_temporal_public_bundle_20261025'
    rows=[r for r in read(bundle/'MANIFEST.json') if r['signer_id']<=6]
    assert len(rows)==39 and all(r['classifier_prior_role']=='train_candidate' for r in rows)
    pins={};
    for r in rows:
        assert sha(r['rgb_path'])==r['rgb_sha256'];assert sha(r['arrays'])==r['arrays_sha256']
        pins[r['rgb_path']]=r['rgb_sha256'];pins[r['arrays']]=r['arrays_sha256']
    sources=[MERGE,U/'experiment.py',U/'model_io.py',U/'MODEL_SOURCE.json',U/'yolo_hand_best.pt',
             ROOT/'Tools/models/hand_landmarker.task',ROOT/'Tools/models/pose_landmarker_lite.task',
             O/'word100_recovery_image_benchmark_20261025/SELECTION.json',
             O/'word100_recovery_image_benchmark_20261025/BLIND_REVIEW_A.json',
             O/'word100_recovery_image_benchmark_20261025/BLIND_REVIEW_B.json',
             O/'word100_temporal_refiner_pilot_20261025/VISIBLE_SAMPLE_SELECTION.json',
             O/'word100_temporal_refiner_pilot_20261025/VISIBLE_LABELS_FROZEN.json',
             O/'word100_image_arm_connectivity_20261018/CROSSING.json']
    pins.update({str(p):sha(p) for p in sources})
    protect={r['path']:r['actual'] for r in read(O/'word100_recovery_image_benchmark_20261025/PROTECTION_CHECK.json')['rows']}
    for p,h in protect.items():assert sha(p)==h,p
    (U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
    write(U/'FROZEN.json',{'at':datetime.datetime.now().astimezone().isoformat(),'authorization':'User approved one hand YOLO and four-condition public-video accuracy/speed comparison; preserve C/T3-F and no user data.',
        'rows':rows,'frames':sum(r['frames'] for r in rows),'words':len({r['word_code'] for r in rows}),'people':sorted({r['signer_id'] for r in rows}),
        'conditions':{'M0':'Exact saved full-frame native MP+existing side assignment then fixed ROI',
                      'M1':'Exact saved MP full-frame + arm crop re-detection merged then fixed ROI',
                      'Y':'YOLO full-frame hands only, shared saved MP body for anatomical side; no MP hand output',
                      'MY':'M0 + YOLO full/crop agreement for absent sides only, same existing arm/duplicate rules, then fixed ROI'},
        'yolo':CFG,'point_confidence':POINT_CONF,'merge_rules':rule.PARAMS,'threads':4,
        'latency':'Fresh serial CPU component calls on same frames. Warmup/setup excluded. Accuracy M0/M1 exact saved outputs. Timing totals reconstruct baseline hand+pose, MP redetection, YOLO full/crop, geometry. Not end-to-end webcam FPS. First-frame costs separately present.',
        'quality':'R24 two-AI frozen coarse wrists + TRAIN01-06 prior V18 coarse controls. No expert21point truth; cannot certify finger precision. Preserve unavailable/unscorable denominators. Raw candidate localization and anatomical assignment reported separately.',
        'decision':'No tuning after scoring. Candidate useful only if newly supported wrists and no evaluated new wrong additions; original coordinates must remain exact. No automatic deployment.',
        'no_actions':['training','TCN inference','user/mentor recordings','runtime replacement','target-label-conditioned detection'],'pins':pins,'protected':protect})
    print('FROZEN',len(rows),sum(r['frames'] for r in rows),flush=True)

def ydetect(model,bgr,box=None):
    t=time.perf_counter();res=model.predict(bgr,**CFG)[0];dt=(time.perf_counter()-t)*1000
    out=[];off=np.array(box[:2] if box else [0,0]);h,w=bgr.shape[:2]
    if res.keypoints is None:return out,dt
    for pts,score in zip(res.keypoints.data.cpu().numpy(),res.boxes.conf.cpu().numpy()):
        assert pts.shape==(21,3)
        valid=(pts[:,2]>=POINT_CONF)&rule.inside(pts[:,:2],w,h)
        xy=pts[:,:2]+off
        d={'xy':np.where(valid[:,None],xy,np.nan),'valid':valid,'kp_conf':pts[:,2], 'score':float(score)}
        if box:d['box']=box
        out.append(d)
    return out,dt
def serializable(x):
    if isinstance(x,np.ndarray):return serializable(x.tolist())
    if isinstance(x,np.generic):return serializable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {str(k):serializable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serializable(v) for v in x]
    return x

def run():
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from model_io import load_model
    import torch
    f=read(U/'FROZEN.json')
    for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
    assert not (U/'RUNNING.json').exists() and not (U/'INDEX.json').exists()
    write(U/'RUNNING.json',{'pid':os.getpid(),'at':datetime.datetime.now().astimezone().isoformat()})
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    model,meta=load_model()
    black=np.zeros((1080,1920,3),np.uint8)
    for _ in range(3):ydetect(model,black)
    modelhand=str(ROOT/'Tools/models/hand_landmarker.task');modelpose=str(ROOT/'Tools/models/pose_landmarker_lite.task')
    red_opts=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=modelhand),running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    (U/'outputs').mkdir();(U/'records').mkdir();index=[]
    for rn,r in enumerate(f['rows']):
        z=load(r['arrays']);cid=r['clip_id'];n=r['frames'];h,w=1080,1920
        arrays={'y_xy':np.full((n,2,21,2),np.nan,np.float32),'y_valid':np.zeros((n,2,21),bool),
                'my_xy':np.full((n,2,21,2),np.nan,np.float32),'my_valid':np.zeros((n,2,21),bool),
                'timing_ms':np.zeros((n,7),np.float64),'frame_id':z['frame_id'],'timestamp_us':z['timestamp_us']}
        frames=[];cap=cv2.VideoCapture(r['rgb_path']);assert cap.isOpened()
        hand=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=modelhand),running_mode=vision.RunningMode.VIDEO,num_hands=2))
        pose=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=modelpose),running_mode=vision.RunningMode.VIDEO,num_poses=1))
        red=vision.HandLandmarker.create_from_options(red_opts)
        red.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=black))
        for i in range(n):
            t=time.perf_counter();ok,bgr=cap.read();assert ok
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            arrays['timing_ms'][i,0]=(time.perf_counter()-t)*1000
            ms=int(z['timestamp_us'][i]//1000)
            t=time.perf_counter();pp=pose.detect_for_video(image,ms);arrays['timing_ms'][i,1]=(time.perf_counter()-t)*1000
            t=time.perf_counter();hp=hand.detect_for_video(image,ms);arrays['timing_ms'][i,2]=(time.perf_counter()-t)*1000
            sides=z['assignment_sides'][i,:int(z['raw_raw_count'][i])].tolist()
            boxes=rule.crop_boxes(z['raw_pose_xy_px'][i],z['raw_pose_usable'][i],sides,w,h)
            t=time.perf_counter();red.detect(image)
            for box in boxes.values():
                red.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb[box[1]:box[3],box[0]:box[2]])))
            arrays['timing_ms'][i,3]=(time.perf_counter()-t)*1000
            det,dt=ydetect(model,bgr);arrays['timing_ms'][i,4]=dt
            crops=[];cropms=0
            for side,box in boxes.items():
                dd,dt=ydetect(model,bgr[box[1]:box[3],box[0]:box[2]],box);cropms+=dt
                for d in dd:d['requested_side']=side
                crops.extend(dd)
            arrays['timing_ms'][i,5]=cropms
            t=time.perf_counter()
            xy,v,assign,reasons=single_side(det,z['raw_pose_xy_px'][i],z['raw_pose_usable'][i],w,h)
            my,mv,dec=combine(det,crops,z,i)
            arrays['timing_ms'][i,6]=(time.perf_counter()-t)*1000
            arrays['y_xy'][i],arrays['y_valid'][i]=roi(xy,v,z['roi_box'])
            arrays['my_xy'][i],arrays['my_valid'][i]=roi(my,mv,z['roi_box'])
            keep=z['baseline_roi_usable'][i,33:].reshape(2,21)
            assert np.array_equal(arrays['my_xy'][i][keep],z['baseline_roi_xy_px'][i,33:].reshape(2,21,2)[keep])
            assert arrays['my_valid'][i][keep].all()
            frames.append(serializable({'i':i,'frame_id':int(z['frame_id'][i]),'full':det,'crop':crops,'y_sides':assign,'y_reasons':reasons,'merge':dec,'boxes':boxes,
                'fresh_mp_hand_count':len(hp.hand_landmarks),'saved_mp_hand_count':int(z['raw_raw_count'][i]),'body_detected':bool(pp.pose_landmarks)}))
        assert not cap.read()[0];cap.release();hand.close();pose.close();red.close()
        p=U/'outputs'/f'{cid}.npz';np.savez_compressed(p,**arrays)
        rp=U/'records'/f'{cid}.json';write(rp,{'clip_id':cid,'frames':frames})
        index.append({'clip_id':cid,'frames':n,'arrays':str(p),'arrays_sha256':sha(p),'records':str(rp),'records_sha256':sha(rp)})
        write(U/f'progress_{rn+1:02d}.json',{'complete_clips':rn+1,'frames':sum(t['frames'] for t in index),'at':datetime.datetime.now().astimezone().isoformat()})
        print('DONE',rn+1,len(f['rows']),cid,'new_hands',sum(q['accepted'] for a in frames for q in a['merge']),flush=True)
    for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
    write(U/'INDEX.json',index);write(U/'RUNTIME.json',{'environment':meta,'timing_columns':['decode_rgb','pose_VIDEO','hand_VIDEO','MP_IMAGE_full_plus_crops','YOLO_full','YOLO_crops','Y_and_MY_geometry'],'threads':4,'training':0,'classifier_inference':0,'user_data':0})
    (U/'RUNNING.json').unlink();print('COMPLETE',len(index),flush=True)

if __name__=='__main__':
    try:freeze() if sys.argv[1]=='freeze' else run()
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'error':traceback.format_exc(),'at':datetime.datetime.now().astimezone().isoformat()})
        raise
