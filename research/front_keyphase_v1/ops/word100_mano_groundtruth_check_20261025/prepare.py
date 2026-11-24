"""Small public, author-published component benchmark. No user data or training."""
from pathlib import Path
import json, hashlib, urllib.request, datetime
import numpy as np
from scipy.io import loadmat

U=Path(__file__).resolve().parent
ROOT=U.parents[3]
OLD=U.parent/'word100_mano_threeway_20261025'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    assert not (U/'FROZEN.json').exists()
    rev=json.loads((U/'alternate_commit.json').read_text())['sha']
    tree={x['path']:x for x in json.loads((U/'alternate_tree.json').read_text())['tree']}
    ids=np.linspace(0,582,32,dtype=int).tolist()
    (U/'images').mkdir(exist_ok=True)
    records=[]
    for i in ids:
        rel=f'data/real_world_testset/images/{i:05d}.jpg'
        p=U/'images'/f'{i:05d}.jpg'
        if not p.exists():
            with urllib.request.urlopen(f'https://raw.githubusercontent.com/3d-hand-shape/hand-graph-cnn/{rev}/{rel}',timeout=30) as r: p.write_bytes(r.read())
        data=p.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==tree[rel]['sha']
        records.append({'id':i,'path':str(p),'sha256':sha(p),'official_git_blob':tree[rel]['sha']})
    for name in ['pose_gt.mat','params.mat']:
        data=(U/name).read_bytes();rel='data/real_world_testset/'+name
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==tree[rel]['sha']
    params=loadmat(U/'params.mat');xyz=loadmat(U/'pose_gt.mat')['pose_gt'][ids].astype(np.float64)
    import cv2
    uv=[]
    for i,q,r in zip(ids,xyz,records):
        im=cv2.imread(r['path']);h,w=im.shape[:2];assert h==w==256
        fx,fy,cx,cy=params['cam_param'][i];x,y,bw,bh=params['bbox'][i];assert bw==bh
        a=q[:,:2]/q[:,2,None]*[fx,fy]+[cx,cy]
        a=(a-[x,y])*[w/bw,h/bh];uv.append(a)
    uv=np.stack(uv);assert np.isfinite(uv).all()
    np.savez_compressed(U/'truth.npz',ids=ids,xy_px=uv,xyz_source=xyz)
    # Preserve all previous evidence and the established protected project assets.
    pre=json.loads((U.parent/'word100_3d_hand_pilot_preflight_20261025/PREPARED.json').read_text(encoding='utf-8'))
    protected={}
    for k,v in pre.items():
        if 'protect' in k: protected[k]=v
    oldpins={str(p):sha(p) for p in OLD.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.html','.txt','.py') and not any(s in p.parts for s in ['private_assets','external','runtime_env','downloads','cache','__pycache__'])}
    f={'at':datetime.datetime.now().astimezone().isoformat(),'scope':'author-published static hand component accuracy, not end-to-end KSL/HT/TCN accuracy',
       'source_repo':'https://github.com/3d-hand-shape/hand-graph-cnn','source_commit':rev,'paper':'Ge et al., CVPR 2019, 3D Hand Shape and Pose Estimation from a Single RGB Image',
       'FreiHAND_attempt':'Official evaluation URL HTTP 404; official lmbweb server connection failed. No FreiHAND file obtained.',
       'selection':'32 equally spaced integer indices across 583 official test images, before predictions; no exclusions based on outcome',
       'records':records,'truth_sha256':sha(U/'truth.npz'),'joint_order':'wrist, thumb 1..4, index 5..8, middle 9..12, ring 13..16, little 17..20; anatomical centers may differ between models',
       'visibility':'No per-joint visibility field supplied. All-joint benchmark kept separate from pre-inference visible fingertip annotation; no hidden-hand accuracy guarantee.',
       'conditions':['MP IMAGE, 2 hands, confidence .5; if multiple outputs choose highest handedness confidence (no GT selection)','HaMeR frozen, full 256x256 supplied image box, rescale 1.0; supplied image handedness only','fallback: retain every MP result; use HaMeR only when MP has no hand'],
       'metrics':['available images/32','mean 2D wrist pixel error','mean 20 finger-point pixel error','mean 5 fingertip pixel error','paired images mean error difference; missing not zero error','PCK at 0.05 image diagonal over all 32*21 points, unavailable counted failure'],
       'inference_GT_policy':'Only image identity and pre-reviewed handedness; no xyz/uv/visibility/target label enters either predictor. GT only for scoring.',
       'temporal_test':False,'bilateral_crossing_test':False,'training':False,'model_selection_or_threshold_tuning':False,'user_data':False,'deployment':False,
       'pretraining_overlap':'Unknown for this external set; not claimed unbiased unseen-subject generalization. No fitting in this project.',
       'protected':protected,'prior_result_pins':oldpins}
    write(U/'FROZEN.json',f)
    # Original images with projected author truth, for coordinate and visible-tip review before model output.
    links=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
    for page in range(4):
        canvas=np.zeros((8*290,512,3),np.uint8)
        for row in range(8):
            j=page*8+row;im=cv2.imread(records[j]['path']);ov=im.copy()
            for a,b in links:cv2.line(ov,tuple(np.rint(uv[j,a]).astype(int)),tuple(np.rint(uv[j,b]).astype(int)),(0,235,255),1)
            for k in range(21):
                pt=tuple(np.rint(uv[j,k]).astype(int));cv2.circle(ov,pt,2,(0,235,255),-1)
                if k in [0,4,8,12,16,20]:cv2.putText(ov,str(k),pt,cv2.FONT_HERSHEY_SIMPLEX,.33,(50,0,255),1)
            canvas[row*290+30:row*290+286,:256]=im;canvas[row*290+30:row*290+286,256:]=ov
            cv2.putText(canvas,f'ID {ids[j]:05d}  Original | Author reference',(5,row*290+22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.imwrite(str(U/f'precheck_{page}.jpg'),canvas)
    print('FROZEN 32 source images, no inference',flush=True)
if __name__=='__main__':main()
