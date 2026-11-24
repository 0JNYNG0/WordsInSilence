"""Frozen-C public sampling diagnostic. No training or user-data access."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,importlib.util,os,time,traceback,argparse,copy
from datetime import datetime,timezone
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;LAB=U.parents[1];ROOT=LAB.parents[1]
P=U.parent/'word100_body_boundary_20261017'
sys.path[:0]=[str(LAB/'src'),str(ROOT/'research/sign_model_pilot_v1/src')]
from wis_lab.core.observation import Observation,empty_frames
from pilot.web_input import to_native_like
from pilot.data import build_channels
from wis_lab.service import word100_candidate as service
ARRAYS=('xy_px','timestamp_us','frame_id','observed','usable','conf','hand_score')
CONDITIONS=[('fps30',30,0),('fps20_p0',20,0),('fps15_p0',15,0),('fps20_p5',20,1),('fps15_p5',15,1)]
WORDS=['WORD0500','WORD0003','WORD1642','WORD1884']
PIN=service.PIN
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def status(stage,**kw):
    v=dict(stage=stage,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw)
    (U/'STATUS.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(v,ensure_ascii=False),flush=True)
def take(obs,idx):
    return Observation(**{k:getattr(obs,k)[idx].copy() for k in ARRAYS},sidecar=copy.deepcopy(obs.sidecar))
def indices(ts,fps,phase=0):
    """Nearest source frame, earlier on ties; retain both endpoints. No timestamps rewritten."""
    ts=np.asarray(ts,dtype=np.int64)
    assert len(ts)>=2 and (np.diff(ts)>0).all()
    if fps==30 and phase==0:return np.arange(len(ts),dtype=np.int64)
    # Scaled integer targets avoid floating rounding/tie ambiguity.
    offset=ts-ts[0]; scaled=offset*2*fps
    target=(np.arange(0,int(offset[-1]*fps//1000000)+2,dtype=np.int64)*2+phase)*1000000
    target=target[(target>0)&(target<scaled[-1])]
    right=np.searchsorted(scaled,target);left=np.maximum(0,right-1)
    choice=np.where(target-scaled[left]<=scaled[right]-target,left,right)
    return np.unique(np.r_[0,choice,len(ts)-1]).astype(np.int64)
def features(obs):
    xy,mask=to_native_like(obs.xy_px,obs.usable)
    return build_channels(xy,mask,obs.timestamp_us)
def freeze():
    fp=U/'FROZEN_DIAGNOSTIC.json'
    if fp.exists():return read(fp)
    rows=read(U.parent/'word100_training_20261013/MANIFEST.json')
    assert Counter(r['role'] for r in rows)=={'train_candidate':899,'development_only':300}
    assert all(r['signer_id'] in (range(1,10) if r['role']=='train_candidate' else (11,12,13)) for r in rows)
    assert sha(P/'training_C_20261013_best.pt')==PIN
    sources=[Path(__file__),P/'training_CONTEXT.json',P/'training_C_20261013_RESULT.json',
             U.parent/'word100_training_20261013/MANIFEST.json',P/'training_C_20261013_best.pt',
             LAB/'config/contract.json',LAB/'src/wis_lab/service/word100_candidate.py',
             ROOT/'research/sign_model_pilot_v1/src/pilot/data.py',
             ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py']
    frozen={'status':'FROZEN_BEFORE_NEW_SCORES','authorized':'Public camera-interval diagnosis only, report before training.',
      'model':dict(key=service.KEY,sha256=PIN,seed=20261013,epoch=77),
      'rows':[{k:r[k] for k in ['clip_id','role','signer_id','label_index','word_code','canonical_label']} for r in rows],
      'views':['masked','sensor'],'conditions':CONDITIONS,
      'primary':['fps20_p0','fps15_p0'],'robustness':['fps20_p5','fps15_p5'],
      'scope':'Existing 30fps observations selected without rerunning detector for full100; rebuild same280 features on selected observations.',
      'window':'Fixed original whole_span, endpoints retained; same elapsed time, no active-span retuning after selection.',
      'time':'Original microsecond timestamps and frame IDs preserved; dt rebuilt from timestamps.',
      'interpolation':'NONE; no fill, repetition, synthetic hands, speed change, mirror or reverse.',
      'phase':'Primary anchored at first frame; secondary grid shifted half period; report both, never choose best.',
      'statistics':'TRAIN899 separate from repeated DEV300; DEV30 saved scores reused, no re-inference. Same clips repeated, not independent new samples.',
      'rgb_check':{'public_train_ids':[f'NIA_SL_{w}_REAL01_F' for w in WORDS],
                   'rates':[30,20,15],'phase':0,'pipeline':'fresh frozen CandidateLandmarker per clip/rate, same frozen crop, full video before active_span',
                   'purpose':'12 observations, 4 unique TRAIN videos; qualitative detector effect, not a population accuracy estimate'},
      'decision':'Descriptive paired Top1/Top5, by-signer and old missingness groups; corrected/regressed counts. No coefficient/candidate selection or improvement claim.',
      'no':['training','user recording reads','heldout14-18','camera','downloads','model/DTW/web changes','Git'],
      'pins':{str(p.resolve()):sha(p) for p in sources}}
    write(fp,frozen);return frozen
def check_context(c):
    for p,h in c['pins'].items():assert sha(p)==h,p
    assert sha(P/'training_C_20261013_best.pt')==PIN
def prep():
    c=freeze();check_context(c);done=U/'PREPARED.json'
    assert not done.exists(),'Prepared result already exists; reuse it.'
    (U/'inputs').mkdir(exist_ok=True)
    synthetic=[]
    t=np.arange(61,dtype=np.int64)*33333
    for rate in [20,15]:
        for phase in [0,1]:
            idx=indices(t,rate,phase);assert idx[0]==0 and idx[-1]==60 and (np.diff(idx)>0).all()
            assert len(idx)<len(t) and np.array_equal(t[idx],np.take(t,idx))
            synthetic.append(dict(rate=rate,phase=phase,frames=len(idx),duration_us=int(t[idx[-1]]-t[idx[0]])))
    rows=[];pins={};totalvalues=0
    for j,row in enumerate(c['rows']):
        rp=P/'data/records'/f"{row['clip_id']}.json";r=read(rp);pins[str(rp)]=sha(rp)
        assert r['role']==row['role'] and r['signer_id']==row['signer_id']
        for view in c['views']:
            info=r['views'][view];op=Path(info['observation']);cp=Path(info['channel_file'])
            for f in [op,cp]:
                h=sha(f);assert h==r['outputs'][str(f)];pins[str(f)]=h
            side=op.with_suffix('.sidecar.json');pins[str(side)]=sha(side)
            ob=Observation.load(str(op));lo,hi=info['feature_info']['whole_span']
            whole=take(ob,np.arange(lo,hi+1))
            original=features(whole)
            with np.load(cp) as z:stored=z['x_whole']
            assert np.array_equal(original['x'],stored),f'Original280 identity {row["clip_id"]}/{view}'
            totalvalues+=stored.size
            arrays={};summaries={}
            for name,fps,phase in CONDITIONS:
                idx=indices(whole.timestamp_us,fps,phase);small=take(whole,idx);small.validate()
                f=original if name=='fps30' else features(small);x=f['x']
                assert np.array_equal(small.usable,whole.usable[idx])
                assert np.array_equal(small.observed,whole.observed[idx])
                assert np.array_equal(small.xy_px,whole.xy_px[idx],equal_nan=True)
                assert small.timestamp_us[-1]-small.timestamp_us[0]==whole.timestamp_us[-1]-whole.timestamp_us[0]
                assert not ((x[:,186:279]!=0)&(x[:,186:279]!=1)).any()
                assert np.array_equal(x[1:,-1],np.diff(small.timestamp_us).astype(np.float32)/1e6)
                arrays[name]=x;arrays[name+'_index']=idx
                arrays[name+'_timestamp_us']=small.timestamp_us;arrays[name+'_frame_id']=small.frame_id
                summaries[name]={'frames':len(x),'elapsed_us':int(small.timestamp_us[-1]-small.timestamp_us[0]),
                      'indices':idx.tolist(),'dt_min_ms':float(np.diff(small.timestamp_us).min()/1000),
                      'dt_max_ms':float(np.diff(small.timestamp_us).max()/1000),
                      'body_scale':float(f['body_scale']),
                      'hand_scales':{k:float(v) if np.isfinite(v) else None for k,v in f['hand_scales'].items()},
                      'source_missing_point_frames':int((~small.usable[:,33:]).sum())}
            p=U/'inputs'/f"{row['clip_id']}__{view}.npz"
            assert not p.exists(),'Partial evidence exists; inspect before recovery'
            np.savez_compressed(p,**arrays)
            rows.append(dict(row,view=view,path=str(p),sha256=sha(p),observation=str(op),
                             original_whole_span=[lo,hi],conditions=summaries))
        if (j+1)%100==0:status('PREPARING',completed_clips=j+1,total=1199)
    result={'status':'PREPARED','context_sha256':sha(U/'FROZEN_DIAGNOSTIC.json'),'rows':rows,
         'pins':pins,'baseline280_exact_values':totalvalues,'baseline_views':len(rows),
         'synthetic_sampling_checks':synthetic,'all_duration_masks_coordinates_preserved':True,'training':0}
    write(done,result);status('PREPARED',views=len(rows),exact280values=totalvalues)
def model():
    import torch
    from pilot.model_masked_v1 import MaskedSignTCN
    torch.set_num_threads(4)
    ck=torch.load(P/'training_C_20261013_best.pt',map_location='cpu',weights_only=True)
    m=MaskedSignTCN(n_classes=100,in_channels=280,**ck['config'])
    m.load_state_dict(ck['state_dict'],strict=True);m.eval()
    assert sum(p.numel() for p in m.parameters())==37636
    return m
def statehash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()
def forward(m,xs):
    import torch
    if not xs:return []
    n=max(len(x) for x in xs);x=np.zeros((len(xs),n,280),np.float32);mask=np.zeros((len(xs),n),np.float32)
    for i,z in enumerate(xs):x[i,:len(z)]=z;mask[i,:len(z)]=1
    with torch.inference_mode():out=m(torch.from_numpy(x),torch.from_numpy(mask)).numpy()
    assert out.shape==(len(xs),100) and np.isfinite(out).all()
    return out.tolist()
def pred(row,logits,origin):
    order=sorted(range(100),key=lambda i:(-logits[i],i));y=row['label_index']
    return {k:row[k] for k in ['clip_id','role','signer_id','label_index','word_code','canonical_label']}|dict(
       logits=logits,target_rank=order.index(y)+1,top1=order[0],top5=order[:5],origin=origin)
def infer():
    c=read(U/'FROZEN_DIAGNOSTIC.json');check_context(c);prepared=read(U/'PREPARED.json')
    assert prepared['context_sha256']==sha(U/'FROZEN_DIAGNOSTIC.json')
    m=model();before=statehash(m)
    saved=read(P/'training_C_20261013_RESULT.json')['predictions']
    saved={v:{r['clip_id']:r for r in saved[v]} for v in c['views']}
    for view in c['views']:
        rows=[r for r in prepared['rows'] if r['view']==view]
        for name,_,_ in CONDITIONS:
            dest=U/f'PRED_{view}_{name}.json'
            if dest.exists():
                assert read(dest)['context_sha256']==prepared['context_sha256']
                continue
            scores=[];batch=[];xs=[]
            def flush():
                if not batch:return
                logits=forward(m,xs)
                scores.extend(pred(r,z,'new_frozen_inference') for r,z in zip(batch,logits))
                batch.clear();xs.clear()
            for r in rows:
                if name=='fps30' and r['role']=='development_only':
                    s=saved[view][r['clip_id']]
                    assert s['label_index']==r['label_index'] and s['signer_id']==r['signer_id']
                    scores.append(pred(r,s['logits'],'stored_dev_baseline'))
                else:
                    assert sha(r['path'])==r['sha256']
                    with np.load(r['path']) as z:xs.append(z[name])
                    batch.append(r)
                    if len(batch)==32:flush()
            flush()
            assert len(scores)==1199 and statehash(m)==before
            scores.sort(key=lambda r:r['clip_id'])
            write(dest,{'context_sha256':prepared['context_sha256'],'condition':name,'view':view,
                        'model_sha256':PIN,'frozen_state_sha256':before,'rows':scores})
            status('INFERENCE_GROUP_COMPLETE',view=view,condition=name,rows=len(scores))
    assert statehash(m)==before;status('CACHED_INPUT_INFERENCE_COMPLETE')
def capture():
    import cv2
    c=read(U/'FROZEN_DIAGNOSTIC.json');check_context(c);m=model();mh=statehash(m)
    old=read(U.parent/'word100_mind_public_audit_20261017/AUDIT.json')
    contract=read(LAB/'config/contract.json')
    outdir=U/'rgb';outdir.mkdir(exist_ok=True)
    for cid in c['rgb_check']['public_train_ids']:
        source=next(r for r in old['records'] if r['clip_id']==cid)
        assert sha(source['video'])==source['source']['sha256']
        rec=read(P/'data/records'/f'{cid}.json')
        original=Observation.load(rec['views']['sensor']['observation'])
        # Decode once, retaining original source timestamps/frame identity for sampling.
        cap=cv2.VideoCapture(source['video']);frames=[]
        while True:
            ok,bgr=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        cap.release();assert len(frames)==original.n_frames
        x0,y0,x1,y1=rec['calibration']['box_source_pixels'];assert x0==0 and y0==0
        row=next(r for r in c['rows'] if r['clip_id']==cid)
        for fps in [30,20,15]:
            d=outdir/f'{cid}_{fps}';d.mkdir(exist_ok=True);dest=d/'RESULT.json'
            if dest.exists():continue
            idx=indices(original.timestamp_us,fps,0);arrays=empty_frames(len(idx))
            start=time.perf_counter()
            with service.CandidateLandmarker(contract) as detector:
                for k,i in enumerate(idx):
                    detector.process(np.ascontiguousarray(frames[i][y0:y1,x0:x1]),int(original.timestamp_us[i]),arrays,k)
                    arrays['frame_id'][k]=original.frame_id[i]
                ob=detector.finalize_observation(Observation(**arrays,sidecar={'clip_id':cid}))
            op=d/'observation.npz';ob.save(str(op))
            clip,reason,support=service.build_candidate_channels(ob)
            pr=pred(row,forward(m,[clip['x']])[0],'new_rgb_pipeline') if clip is not None else None
            # Compare newly obtained detection with original30fps cached sensor at same source frames.
            orig=take(original,idx)
            both=ob.observed&orig.observed
            diff=float(np.nanmax(np.abs(ob.xy_px[both]-orig.xy_px[both]))) if both.any() else None
            v={'context_sha256':sha(U/'FROZEN_DIAGNOSTIC.json'),'clip_id':cid,'fps':fps,
               'video':source['video'],'source_video_sha256':source['source']['sha256'],
               'source_indices':idx.tolist(),'source_timestamps_us':original.timestamp_us[idx].tolist(),
               'crop':[x0,y0,x1,y1],'observation':str(op),'observation_sha256':sha(op),'seconds':time.perf_counter()-start,
               'support':support,'reason':reason,'prediction':pr,
               'old30_sameframe_observed_equal':bool(np.array_equal(ob.observed,orig.observed)),
               'old30_sameframe_usable_equal':bool(np.array_equal(ob.usable,orig.usable)),
               'old30_sameframe_max_joint_delta_pixels':diff,
               'observed_hand_point_frames':int(ob.observed[:,33:].sum()),
               'old30_selected_observed_hand_point_frames':int(orig.observed[:,33:].sum())}
            write(dest,v);status('RGB_COMPLETE',clip_id=cid,fps=fps,rank=pr['target_rank'] if pr else None)
    assert statehash(m)==mh;status('RGB_COMPLETE_ALL')
if __name__=='__main__':
    mode=argparse.ArgumentParser();mode.add_argument('stage',choices=['prepare','infer','capture'])
    args=mode.parse_args();start=time.time()
    try:
        {'prepare':prep,'infer':infer,'capture':capture}[args.stage]()
    except Exception:
        path=U/f'FIRST_ERROR_{args.stage}.txt'
        if not path.exists():path.write_text(traceback.format_exc(),'utf-8')
        status('FAILED',stage_failed=args.stage,error=traceback.format_exc());raise

