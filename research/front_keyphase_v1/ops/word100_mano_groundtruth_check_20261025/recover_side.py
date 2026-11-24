"""Correct an anatomical-side input error, not a model or threshold search.
Initial predictions/scores remain untouched. MP outputs are reused byte-for-byte.
"""
from pathlib import Path
import sys, json, hashlib, time, contextlib,io
import numpy as np
U=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def freeze():
    assert not (U/'SIDE_CORRECTION.json').exists()
    r=read(U/'INFERENCE_COMPLETE.json');assert all(x['mp_label']=='Left' for x in r['index'])
    x={'reason':'Initial manual review incorrectly called the target a right hand. Reinspection of palmar anatomy and all 32 current MP side outputs identifies left. No per-image best-score flip choice.',
       'corrected_side_all':0,'side_policy':'Use anatomical Left from current MP labels (all32) corroborated by source palm/thumb view; no GT keypoints used to choose side.',
       'old_frozen_side':1,'original_scores_status':'INVALID_SIDE_INPUT_DO_NOT_USE_FOR_MODEL_COMPARISON',
       'recovery_scope':'HaMeR only rerun once with correct left crop and inverse flip; same 32 images, model, image box, rescale1.0, evaluation reference, metrics. MP results reused.',
       'source_scene_pilot_affected':False,'pilot_note':'Earlier pilot uses its existing arm-assigned side, not this benchmark manual constant. No old results overwritten.',
       'pins':{str(p):sha(p) for p in [U/'recover_side.py',U/'FROZEN.json',U/'CODE_PINS.json',U/'INFERENCE_COMPLETE.json',U/'SCORES.json',U/'runtime_support.py']}}
    write(U/'SIDE_CORRECTION.json',x)
def run():
    x=read(U/'SIDE_CORRECTION.json')
    for p,h in x['pins'].items():assert sha(p)==h
    assert not (U/'CORRECTED_INFERENCE.json').exists()
    from runtime_support import configure,load_model
    configure()
    import torch,cv2
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from torch.utils.data._utils.collate import default_collate
    model,cfg=load_model();f=read(U/'FROZEN.json');r=read(U/'INFERENCE_COMPLETE.json')
    out=U/'corrected_predictions';out.mkdir(exist_ok=True);idx=[]
    for row,old in zip(f['records'],r['index']):
        assert row['id']==old['id'] and sha(row['path'])==row['sha256'] and sha(old['prediction'])==old['sha256']
        dest=out/f"{row['id']:05d}.npz";assert not dest.exists()
        bgr=cv2.imread(row['path']);h,w=bgr.shape[:2]
        with contextlib.redirect_stdout(io.StringIO()):item=ViTDetDataset(cfg,bgr,np.array([[0,0,w,h]],np.float32),np.array([0]),rescale_factor=1.0)[0]
        batch=default_collate([item]);t=time.perf_counter()
        with torch.inference_mode():pred=model(batch)
        elapsed=time.perf_counter()-t;xy=pred['pred_keypoints_2d'][0].cpu().numpy();xy[:,0]*=-1
        xy=xy*float(item['box_size'])+np.asarray(item['box_center']);assert np.isfinite(xy).all()
        with np.load(old['prediction']) as z:mp=z['mp_xy'].copy()
        fallback=mp.copy() if old['mp_count'] else xy.copy()
        np.savez_compressed(dest,mp_xy=mp,hamer_xy=xy,fallback_xy=fallback)
        idx.append({**old,'prediction':str(dest),'sha256':sha(dest),'hamer_seconds':elapsed,'side':0})
        if len(idx)%8==0:print('SIDE_CORRECTED',len(idx),'/32',flush=True)
    write(U/'CORRECTED_INFERENCE.json',{'index':idx,'correction':str(U/'SIDE_CORRECTION.json'),'training':False})
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
