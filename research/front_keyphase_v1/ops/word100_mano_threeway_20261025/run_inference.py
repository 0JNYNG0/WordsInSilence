"""Frozen pretrained HaMeR inference only. No optimizer, word label, or reference pose."""
import argparse, json, time, hashlib, contextlib, io
from pathlib import Path
import numpy as np
from runtime_support import UNIT as U, load_model, configure

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()

def coordinate_check():
    import cv2
    from hamer.datasets.utils import generate_image_patch_cv2
    rng=np.random.default_rng(3); errors=[]
    image=np.zeros((1080,1920,3),np.uint8); center=np.array([1032.,426.]);size=234.
    for right in (0,1):
        _, affine=generate_image_patch_cv2(image,*center,size,size,256,256,right==0,1.,0)
        points=center+rng.uniform(-size/3,size/3,(40,2))
        flipped=points.copy()
        if not right:flipped[:,0]=1919-flipped[:,0]
        patch=cv2.transform(flipped[None].astype(np.float32),affine)[0]
        norm=patch/256.-.5
        restored=norm.copy();restored[:,0]*=2*right-1;restored=restored*size+center
        errors.append(float(np.max(np.abs(restored-points))))
    assert max(errors)<.001,errors
    return {'inverse_crop_both_hands_max_px':errors,'same_MP21_chain_order':'wrist, thumb4, index4, middle4, ring4, little4; exact anatomical centers are not identical detector ground truth','depth_used_for_TCN':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=100000);args=ap.parse_args()
    configure()
    import cv2, torch
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from torch.utils.data._utils.collate import default_collate
    protocol=json.loads((U/'FROZEN_PROTOCOL.json').read_text('utf-8'))
    outdir=U/'raw_predictions';outdir.mkdir(exist_ok=True)
    model,cfg=load_model()
    (U/'COORDINATE_CONTRACT.json').write_text(json.dumps(coordinate_check(),indent=2),encoding='utf-8')
    jobs=[]
    for k,m in enumerate(protocol['manifest']):
        assert digest(m['prepared'])==m['prepared_sha256']
        with np.load(m['prepared']) as z:
            for t,s in np.argwhere(z['requested']):
                # Existing normal observation control first for the minimal coordinate smoke test.
                priority=0 if m['role']=='existing_observation_control' and t in m['anchors'] else 1 if t in m['anchors'] else 2
                jobs.append((priority,k,int(t),int(s)))
    jobs.sort()
    completed=0;start=time.perf_counter(); timings=[]
    for _,k,t,s in jobs:
        m=protocol['manifest'][k];dest=outdir/f'{k:02d}_{t:04d}_{s}.npz';meta=dest.with_suffix('.json')
        if dest.exists():
            if not meta.exists():raise RuntimeError('Partial prior inference, preserve before recovery')
            info=json.loads(meta.read_text());assert digest(dest)==info['sha256']
            continue
        if completed>=args.limit:break
        with np.load(m['prepared']) as z:box=z['boxes'][t,s].copy()
        cap=cv2.VideoCapture(m['rgb_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,bgr=cap.read();cap.release()
        if not ok:raise RuntimeError('public frame unavailable')
        t0=time.perf_counter()
        dataset=ViTDetDataset(cfg,bgr,np.array([box]),np.array([s]),rescale_factor=2.0)
        with contextlib.redirect_stdout(io.StringIO()):item=dataset[0]
        batch=default_collate([item]);t1=time.perf_counter()
        with torch.inference_mode():output=model(batch)
        elapsed=time.perf_counter()-t1
        points=output['pred_keypoints_2d'][0].cpu().numpy();points[:,0]*=2*s-1
        points=points*float(item['box_size'])+np.asarray(item['box_center'])
        params=output['pred_mano_params']
        values={'xy_px':points,'joints3d_canonical':output['pred_keypoints_3d'][0].cpu().numpy(),
                'vertices_canonical':output['pred_vertices'][0].cpu().numpy(),
                'cam':output['pred_cam'][0].cpu().numpy(),
                'global_orient':params['global_orient'][0].cpu().numpy(),
                'hand_pose':params['hand_pose'][0].cpu().numpy(),'betas':params['betas'][0].cpu().numpy(),
                'box_center':item['box_center'],'box_size':np.asarray(item['box_size']),
                'side':np.asarray(s),'frame_index':np.asarray(t)}
        assert all(np.isfinite(v).all() for v in values.values())
        np.savez_compressed(dest,**values)
        info={'public_manifest_index':k,'frame':t,'side':s,'sha256':digest(dest),
              'preprocess_seconds':t1-t0,'inference_seconds':elapsed,'model_estimate_not_ground_truth':True}
        meta.write_text(json.dumps(info,indent=2),encoding='utf-8');timings.append(elapsed);completed+=1
        status={'status':'running','total_jobs':len(jobs),'saved_jobs':len(list(outdir.glob('*.npz'))),
                'new_this_run':completed,'elapsed_this_run':time.perf_counter()-start,
                'recent_seconds_per_hand':float(np.mean(timings[-10:]))}
        (U/'INFERENCE_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
        if completed<=2 or completed%10==0:print(json.dumps(status),flush=True)
    status['status']='complete' if status['saved_jobs']==len(jobs) else 'bounded_coordinate_smoke_complete'
    (U/'INFERENCE_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('DONE',json.dumps(status),flush=True)
if __name__=='__main__':main()
