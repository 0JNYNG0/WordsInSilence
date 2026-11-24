"""Input/coordinate preflight only. No 3D fitting or substitute reconstruction."""
from pathlib import Path
import json, hashlib, importlib.util, sys, datetime, ast, html
import numpy as np
import cv2

U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
OLD=O/'word100_yolo_hand_fourway_20261025'; SEL=O/'word100_yolo_selective_wrist_20261025'
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def adapter_contract():
    # Execute the real historical conversion against deterministic fake API output.
    # This validates pixel offsets and masks, not detector accuracy or real letterbox internals.
    exp=module('preflight_old_detector_adapter',OLD/'experiment.py')
    class Array:
        def __init__(self,x): self.x=x
        def cpu(self): return self
        def numpy(self): return self.x
    class Dummy: pass
    pts=np.zeros((1,21,3),np.float32); pts[0,:,0]=np.arange(21)+20;pts[0,:,1]=np.arange(21)+30;pts[0,:,2]=.9
    pts[0,19,2]=.1;pts[0,20,0]=-1
    class Fake:
        def predict(self,bgr,**kwargs):
            q=Dummy();q.keypoints=Dummy();q.boxes=Dummy();q.keypoints.data=Array(pts);q.boxes.conf=Array(np.array([.8]));return [q]
    model=Fake();box=[128,64,640,448]
    full,_=exp.ydetect(model,np.zeros((1080,1920,3),np.uint8))
    cropped,_=exp.ydetect(model,np.zeros((384,512,3),np.uint8),box)
    p=np.asarray(full[0]['xy']);c=np.asarray(cropped[0]['xy']);v=np.asarray(full[0]['valid'])
    assert v.sum()==19 and not v[19] and not v[20]
    assert np.array_equal(v,cropped[0]['valid'])
    assert np.allclose(c[v],p[v]+[128,64],rtol=0,atol=0)
    assert np.isnan(c[~v]).all()
    view=O/'word100_yolo_five_scene_view_20261025/build.py'
    tree=ast.parse(view.read_text('utf8'))
    links=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LINKS' for t in n.targets))
    chains=[[0,1,2,3,4],[0,5,6,7,8],[5,9,10,11,12],[9,13,14,15,16],[13,17,18,19,20]]
    assert all((a,b) in links for chain in chains for a,b in zip(chain[:-1],chain[1:]))
    return {'status':'PASS','crop_offset_applied_once':True,'invalid_points_preserved':True,'rendered_connections_match_expected_MP21_topology':True,
            'limits':['Fake API output validates adapter math, not fresh model inference.','Custom YOLO checkpoint per-index semantic training labels remain unverified; kpt_shape21 alone cannot prove ordering.','Depth/anatomical accuracy and real multi-model registration not tested.']}

def main():
    assert not (U/'PREPARED.json').exists()
    f=read(SEL/'FROZEN.json'); allrows={r['clip_id']:r for r in f['old_rows']}
    cases=read(SEL/'FIVE_SCENE_AUDIT.json')
    ids=[c['clip_id'] for c in cases]+['NIA_SL_WORD0003_REAL01_F','NIA_SL_WORD0185_REAL03_F']
    assert len(set(ids))==7
    sr={r['clip_id']:r for r in read(SEL/'INDEX.json')}; yr={r['clip_id']:r for r in read(OLD/'INDEX.json')}
    crossing=read(OLD/'CROSSING_PRESERVATION.json')
    pins={str(p):sha(p) for p in [OLD/'experiment.py',SEL/'candidate.py',SEL/'common.py',SEL/'FIVE_SCENE_AUDIT.json',U/'PROTOCOL_KO.txt',U/'prepare.py']}
    for p,h in f['protected'].items(): assert sha(p)==h,p
    manifest=[]; audits=[]; assets=U/'assets';assets.mkdir(exist_ok=False)
    for cid in ids:
        r=allrows[cid]; assert r['signer_id']<=6 and r['classifier_prior_role']=='train_candidate'
        for key,hkey in [('rgb_path','rgb_sha256'),('arrays','arrays_sha256')]:
            assert sha(r[key])==r[hkey];pins[r[key]]=r[hkey]
        for row in [sr[cid],yr[cid]]:
            for key in ['arrays','records']:
                assert sha(row[key])==row[key+'_sha256'];pins[row[key]]=row[key+'_sha256']
        with np.load(r['arrays'],allow_pickle=False) as inp:
            z={k:inp[k] for k in ['frame_id','timestamp_us','width','height','roi_box','native_xy_px','native_usable','redetected_roi_xy_px','redetected_roi_usable']}
        n=r['frames']; assert len(z['frame_id'])==n and np.all(np.diff(z['timestamp_us'])>0)
        assert np.all(z['width']==1920) and np.all(z['height']==1080)
        x0,y0,x1,y1=z['roi_box']; pts=z['redetected_roi_xy_px'];v=z['redetected_roi_usable'].astype(bool)
        # The upper-body mask applies to hands; pose channels have a separate contract.
        hp=pts[:,33:];hv=v[:,33:]
        assert np.isfinite(hp[hv]).all()
        assert ((hp[hv,0]>=x0)&(hp[hv,0]<x1)&(hp[hv,1]>=y0)&(hp[hv,1]<y1)).all()
        rec=read(yr[cid]['records']); finite_crop_points=0;crop_candidates=0
        for frame in rec['frames']:
            for d in frame['crop']:
                a=np.asarray(d['xy'],float);good=np.asarray(d['valid'],bool);b=np.asarray(d['box']); local=a[good]-b[:2]
                assert (local[:,0]>=0).all() and (local[:,0]<b[2]-b[0]).all()
                assert (local[:,1]>=0).all() and (local[:,1]<b[3]-b[1]).all()
                finite_crop_points+=int(good.sum());crop_candidates+=1
        anchors=[c['frame_id'] for c in cases if c['clip_id']==cid]
        crossframes=sorted({q['frame_id'] for q in crossing if q['clip_id']==cid})
        if not anchors:anchors=[crossframes[0] if crossframes else n//2]
        wanted={max(0,min(n-1,a+d)) for a in anchors for d in [-3,0,3]}
        cap=cv2.VideoCapture(r['rgb_path']);assert cap.isOpened();count=0;images=[]
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape[:2]==(1080,1920)
            if count in wanted:
                target=assets/f'{cid}_{count:04d}.jpg';assert cv2.imwrite(str(target),im,[cv2.IMWRITE_JPEG_QUALITY,94]);images.append({'frame':count,'path':str(target)})
            count+=1
        cap.release();assert count==n
        manifest.append({'clip_id':cid,'word_label_for_display_only':r['display_label'],'signer_id':r['signer_id'],'role':'known_problem' if cid in [c['clip_id'] for c in cases] else 'crossing_control' if crossframes else 'existing_observation_control','rgb_path':r['rgb_path'],'rgb_sha256':r['rgb_sha256'],'frames':n,'frame_size':[1920,1080],'arrays':r['arrays'],'arrays_sha256':r['arrays_sha256'],'fixed_roi':list(map(int,z['roi_box'])),'anchors':anchors,'crossing_frames':crossframes,'original_samples':images,'baseline_M1':r['arrays'],'baseline_selective_YOLO':sr[cid]['arrays']})
        audits.append({'clip_id':cid,'frames_decoded':count,'timestamps_strictly_increasing':True,'retained_hand_roi_points':int(hv.sum()),'stored_crop_candidates':crop_candidates,'crop_points_in_original_coordinate_box':finite_crop_points})
    contract=adapter_contract()
    import torch,importlib.util
    environment={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda_available_in_this_runtime':torch.cuda.is_available(),'GPU_hardware_inventory':'not established; CIM query unavailable','smplx_installed':importlib.util.find_spec('smplx') is not None,'MANO_official_asset':'absent; user confirmed not acquired','HaMeR_installed':'not found in project','note':'Existing torch is CPU build. No model install/download in preflight.'}
    assert all(sha(p)==h for p,h in pins.items())
    assert all(sha(p)==h for p,h in f['protected'].items())
    result={'at_kst':datetime.datetime.now().astimezone().isoformat(),'status':'INPUTS_PREPARED_3D_EXECUTION_NOT_STARTED','clips':len(manifest),'frames':sum(q['frames'] for q in manifest),'people':sorted({q['signer_id'] for q in manifest}),'public_words':len({q['clip_id'].split('_')[2] for q in manifest}),'manifest':manifest,'input_audit':audits,'adapter_contract':contract,'environment':environment,'protected_files_verified':len(f['protected']),'pins':pins,'protected':f['protected'],'remaining':['Official MANO model acquisition by user.','HaMeR environment/weight setup and model-keypoint mapping validation.','Independent precise visible-joint labels and explicit 3D biomechanical/temporal optimization design before fitting.'],'training':0,'new_model_inference':0,'3d_reconstruction':0,'geometry_only_substitution':False,'user_recording_content_access':0}
    save(U/'PREPARED.json',result)
    print(json.dumps({k:result[k] for k in ['status','clips','frames','people','protected_files_verified','training','3d_reconstruction']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
