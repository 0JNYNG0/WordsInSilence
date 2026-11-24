"""Frozen component inference and visible/ambiguous-separated 2D scoring."""
from pathlib import Path
import json, hashlib, time, sys, contextlib, io
import numpy as np
U=Path(__file__).resolve().parent
ROOT=U.parents[3]
OLD=U.parent/'word100_mano_threeway_20261025'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def freeze_code():
    assert not (U/'CODE_PINS.json').exists()
    write(U/'CODE_PINS.json',{str(p):sha(p) for p in [U/'FROZEN.json',U/'PREINFERENCE_VISUAL_REVIEW.json',U/'benchmark.py',U/'runtime_support.py',ROOT/'Tools/models/hand_landmarker.task']})

def infer():
    for p,h in read(U/'CODE_PINS.json').items():assert sha(p)==h
    f=read(U/'FROZEN.json')
    assert not (U/'RUNNING.json').exists() and not (U/'INFERENCE_COMPLETE.json').exists()
    import os
    write(U/'RUNNING.json',{'pid':os.getpid(),'status':'inference_only'})
    import cv2,mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from runtime_support import load_model,configure
    configure()
    import torch
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from torch.utils.data._utils.collate import default_collate
    model,cfg=load_model()
    opt=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(ROOT/'Tools/models/hand_landmarker.task')),running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    out=U/'predictions';out.mkdir(exist_ok=True)
    index=[]
    with vision.HandLandmarker.create_from_options(opt) as hand:
        for row in f['records']:
            i=row['id'];dest=out/f'{i:05d}.npz';assert not dest.exists()
            assert sha(row['path'])==row['sha256']
            bgr=cv2.imread(row['path']);h,w=bgr.shape[:2];rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            t=time.perf_counter();res=hand.detect(image);mp_secs=time.perf_counter()-t
            predmp=np.full((21,2),np.nan,np.float32);label=None;confidence=None
            if res.hand_landmarks:
                j=int(np.argmax([a[0].score for a in res.handedness]))
                predmp=np.array([[v.x*w,v.y*h] for v in res.hand_landmarks[j]],np.float32)
                label=res.handedness[j][0].category_name;confidence=float(res.handedness[j][0].score)
            t=time.perf_counter()
            dataset=ViTDetDataset(cfg,bgr,np.array([[0,0,w,h]],np.float32),np.array([1]),rescale_factor=1.0)
            with contextlib.redirect_stdout(io.StringIO()):item=dataset[0]
            batch=default_collate([item]);prep=time.perf_counter()-t;t=time.perf_counter()
            with torch.inference_mode():outm=model(batch)
            hsecs=time.perf_counter()-t
            ph=outm['pred_keypoints_2d'][0].cpu().numpy()*float(item['box_size'])+np.asarray(item['box_center'])
            assert np.isfinite(ph).all()
            fallback=predmp.copy() if res.hand_landmarks else ph.copy()
            np.savez_compressed(dest,mp_xy=predmp,hamer_xy=ph,fallback_xy=fallback)
            index.append({'id':i,'prediction':str(dest),'sha256':sha(dest),'mp_count':len(res.hand_landmarks),'mp_label':label,'mp_confidence':confidence,'mp_seconds':mp_secs,'hamer_preprocess_seconds':prep,'hamer_seconds':hsecs,'fallback_used':not bool(res.hand_landmarks)})
            write(U/'PROGRESS.json',{'completed':len(index),'total':32})
            if len(index)%8==0:print('INFERRED',len(index),'/32',flush=True)
    write(U/'INFERENCE_COMPLETE.json',{'index':index,'model_fitting':False,'no_GT_arrays_loaded_by_inference':True,'new_training':0})
    (U/'RUNNING.json').unlink()

def score():
    f=read(U/'FROZEN.json');v=read(U/'PREINFERENCE_VISUAL_REVIEW.json');r=read(U/'INFERENCE_COMPLETE.json')
    assert sha(U/'truth.npz')==f['truth_sha256']
    with np.load(U/'truth.npz') as z:gt=z['xy_px'];ids=z['ids']
    predictions={k:[] for k in ['mp','hamer','fallback']}
    for row in r['index']:
        assert sha(row['prediction'])==row['sha256']
        with np.load(row['prediction']) as z:
            for k in predictions:predictions[k].append(z[k+'_xy'])
    results={};errors={}
    vis=np.zeros((32,21),bool)
    for j,i in enumerate(ids):vis[j,v['clearly_visible_fingertips'][str(i)]]=True
    tips=np.zeros_like(vis);tips[:,[4,8,12,16,20]]=True
    paired=np.isfinite(np.array(predictions['mp'])).all((1,2))
    for k,arr in predictions.items():
        arr=np.array(arr);err=np.linalg.norm(arr-gt,axis=-1);errors[k]=err;avail=np.isfinite(err).all(1)
        def mean(x):return float(np.nanmean(x)) if np.isfinite(x).any() else None
        results[k]={'available_images':int(avail.sum()),'total_images':32,'wrist_mean_px':mean(err[:,0]),'finger20_mean_px':mean(err[:,1:]),'tip5_mean_px':mean(err[tips]),'visible_tip_mean_px':mean(err[vis]),'visible_tip_comparisons':int(np.isfinite(err[vis]).sum()),'visible_tip_reference_total':int(vis.sum()),'uncertain_tip_mean_px':mean(err[tips&~vis]),'all21_mean_px':mean(err),'paired_all21_mean_px':mean(err[paired]),'PCK_0p05_diag_pct':float(((err<=.05*np.hypot(256,256))&np.isfinite(err)).sum()/err.size*100),'inference_median_seconds':float(np.median([x['mp_seconds'] if k=='mp' else x['hamer_seconds'] if k=='hamer' else x['mp_seconds']+(x['hamer_seconds'] if x['fallback_used'] else 0) for x in r['index']]))}
    mp_e=errors['mp'].mean(1);h_e=errors['hamer'].mean(1)
    summary={'results':results,'paired_images':int(paired.sum()),'paired_HaMeR_lower_mean_error':int((h_e[paired]<mp_e[paired]).sum()),'fallback_images':[x['id'] for x in r['index'] if x['fallback_used']],
             'per_image':[{'id':int(i),'mp_all21_px':float(mp_e[j]) if paired[j] else None,'hamer_all21_px':float(h_e[j]),'visible_tip_count':int(vis[j].sum())} for j,i in enumerate(ids)],
             'interpretation':'2D component benchmark on provided hand crops; no TCN/DTW/user accuracy, no temporal or two-hand crossing validation. Unknown tips not automatically correct or invisible.',
             'decision':'Research-only; keep prior C/T3-F and project DTW unchanged.'}
    write(U/'SCORES.json',summary);np.savez_compressed(U/'errors.npz',**errors,visible_tip_mask=vis,tip_mask=tips)
    print(json.dumps({k:x for k,x in summary.items() if k!='per_image'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:{'freeze':freeze_code,'infer':infer,'score':score}[sys.argv[1]]()
    except BaseException:
        import traceback
        if not (U/'FIRST_EXECUTION_ERROR.txt').exists():(U/'FIRST_EXECUTION_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
