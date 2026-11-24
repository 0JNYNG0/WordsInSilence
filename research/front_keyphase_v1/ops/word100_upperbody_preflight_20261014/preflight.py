"""Public TRAIN-only actual RGB crop before unchanged MediaPipe. No classifier/training."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

U=Path(__file__).resolve().parent
LAB=U.parents[1]
ROOT=LAB.parents[1]
sys.path[:0]=[str(LAB/'src'),str(ROOT/'research/sign_model_pilot_v1/src')]
from wis_lab.capture.landmarker import Landmarker
from wis_lab.capture.video_source import extract
from wis_lab.core.observation import Observation,sha256_file
from pilot.web_input import to_native_like,active_span
from pilot.data import build_channels

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d):
    Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')

def plan():
    assert not (U/'PLAN.json').exists(), 'Plan is frozen; do not replace'
    cfg=read(LAB/'config/word100_practice.json')
    manifest={r['clip_id']:r for r in read(LAB/'ops/word100_training_20261013/MANIFEST.json')}
    items=[]
    for item in cfg['items']:
        assert item['signer_id']==1 and item['source_role']=='train_candidate'
        assert sha256_file(item['video_path'])==item['source_sha256']
        ob=Observation.load(item['observation'])
        good=ob.usable[:,[11,12,23,24]].all(1)
        assert good.any()
        sh=np.median(ob.xy_px[good][:,[11,12],:].mean(1),axis=0)
        hip=np.median(ob.xy_px[good][:,[23,24],:].mean(1),axis=0)
        w,h=int(ob.sidecar['width']),int(ob.sidecar['height'])
        assert hip[1]>sh[1]
        # A single predeclared product approximation, NOT anatomical navel estimation.
        bottom=int(round(sh[1]+.75*(hip[1]-sh[1])))
        assert 0<bottom<h
        cw=int(round(bottom*w/h))
        x0=max(0,min(w-cw,int(round(sh[0]-cw/2))))
        box=[x0,0,x0+cw,bottom]
        row=manifest[item['clip_id']]
        assert row['role']=='train_candidate' and len(row['intervals'])==1
        items.append(dict(item,box=box,original_size=[w,h],core_seconds=row['intervals'][0],
                          observation_sha256=sha256_file(item['observation']),
                          reference_shoulder=sh.tolist(),reference_hip=hip.tolist()))
    assert len(items)==12
    pins=[Path(__file__),LAB/'config/contract.json',LAB/'src/wis_lab/capture/landmarker.py',
          LAB/'src/wis_lab/capture/video_source.py',LAB/'src/wis_lab/core/observation.py',
          ROOT/'research/sign_model_pilot_v1/src/pilot/data.py',ROOT/'research/sign_model_pilot_v1/src/pilot/web_input.py',
          ROOT/'Tools/models/hand_landmarker.task',ROOT/'Tools/models/pose_landmarker_lite.task',Path(cfg['checkpoint'])]
    write(U/'PLAN.json',dict(status='FROZEN_BEFORE_EXTRACTION',scope='12 TRAIN REAL01 videos, native vs actual RGB crop; input preflight only',
          authorization='2026-10-14 user: 작업하면서 내 말은 참고해; 상반신 부분만 MediaPipe로 추출 후 학습. This unit only checks actual extraction before any model training.',
          roi_rule='Clip-fixed box. Top=0, bottom=median shoulder center y + 0.75*(median hip center y - shoulder y); width maintains source aspect ratio, centered on shoulder x and clamped inside source. Native pixels, no resizing/encoding. Approximate upper-torso product view, not measured navel or selected optimal crop.',
          conditions=['native','crop'],all_decoded_frames_preserved=True,
          semantics='No classification score, no training eligibility verdict; core hand loss flagged for review. Static support hands may be required.',
          prohibited=['User recordings or screenshot pixel fitting','New classifier inference/training','Production changes','Camera','Downloads','Heldout','Parameter search after outputs'],
          pins={str(p):sha256_file(p) for p in pins},items=items))
    print('Frozen 12-clip actual RGB crop preflight',flush=True)

class CropProxy:
    def __init__(self,base,box):self.base,self.box=base,box
    def __getattr__(self,key):return getattr(self.base,key)
    def process(self,rgb,timestamp_us,out,row):
        x0,y0,x1,y1=self.box
        self.base.process(rgb[y0:y1,x0:x1],timestamp_us,out,row)

def rates(ob,ids):
    if len(ids)==0:return None
    w,h=ob.sidecar['width'],ob.sidecar['height']
    xy=ob.xy_px[ids];u=ob.usable[ids];o=ob.observed[ids]
    strict=o&np.isfinite(xy).all(2)&(xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<h)
    out={'frames':len(ids),'both_shoulders_usable':float(u[:,[11,12]].all(1).mean()),
         'hips_usable':u[:,[23,24]].mean(0).tolist(),'pose_wrists_usable':u[:,[15,16]].mean(0).tolist()}
    for side,start in [('L',33),('R',54)]:
        sl=slice(start,start+21)
        out[side]=dict(any_observed=float(o[:,sl].any(1).mean()),any_usable=float(u[:,sl].any(1).mean()),
           all21_usable=float(u[:,sl].all(1).mean()),point_usable=float(u[:,sl].mean()),
           wrist_usable=float(u[:,start].mean()),mcp_usable=float(u[:,[start+5,start+9,start+13,start+17]].mean()),
           strict_inside_point=float(strict[:,sl].mean()),usable_but_outside_strict=int((u[:,sl]&~strict[:,sl]).sum()))
    return out

def summary(ob,core):
    ob.validate()
    t=ob.timestamp_us/1e6;a,b=core
    nat,pr=to_native_like(ob.xy_px,ob.usable)
    span=active_span(pr)
    ch=None
    if span:
        lo,hi=span;c=build_channels(nat[lo:hi+1],pr[lo:hi+1],ob.timestamp_us[lo:hi+1]);x=c['x']
        assert x.shape[1]==280 and np.isfinite(x).all()
        assert (x[:,:102].reshape(-1,51,2)[~x[:,186:237].astype(bool)]==0).all()
        assert (x[:,102:186].reshape(-1,42,2)[~x[:,237:279].astype(bool)]==0).all()
        ch=dict(shape=list(x.shape),finite=True,masked_coordinates_zero=True)
    return dict(full=rates(ob,np.arange(ob.n_frames)),core=rates(ob,np.flatnonzero((t>=a)&(t<=b))),
       before_core=rates(ob,np.flatnonzero(t<a)),after_core=rates(ob,np.flatnonzero(t>b)),
       active_span=list(span) if span else None,channels=ch,
       assignment={k:ob.sidecar[k] for k in ['handedness_label_disagreements','handedness_label_comparisons','unassigned_hand_detections']})

def geometry(native,box,core):
    x0,y0,x1,y1=box;t=native.timestamp_us/1e6;a,b=core;ids=(t>=a)&(t<=b)
    xy=native.xy_px[ids];u=native.usable[ids]
    inside=(xy[:,:,0]>=x0)&(xy[:,:,0]<x1)&(xy[:,:,1]>=y0)&(xy[:,:,1]<y1)
    out={}
    for side,start in [('L',33),('R',54)]:
        sl=slice(start,start+21);n=int(u[:,sl].sum())
        out[side]=dict(original_usable_points=n,outside_crop_points=int((u[:,sl]&~inside[:,sl]).sum()),
          outside_crop_fraction=float((u[:,sl]&~inside[:,sl]).sum()/n) if n else None,
          frames_with_no_original_usable_hand_point_inside=int((u[:,sl].any(1)&~(u[:,sl]&inside[:,sl]).any(1)).sum()))
    return out

def run():
    p=read(U/'PLAN.json');plan_hash=sha256_file(U/'PLAN.json')
    assert all(sha256_file(f)==v for f,v in p['pins'].items())
    assert not (U/'RESULT.json').exists(),'Completed outputs are frozen'
    contract=read(LAB/'config/contract.json');rows=[];started=time.perf_counter()
    for item in p['items']:
        cid=item['clip_id'];d=U/cid;d.mkdir(exist_ok=True)
        assert sha256_file(item['video_path'])==item['source_sha256']
        assert sha256_file(item['observation'])==item['observation_sha256']
        obs={};secs={}
        for mode in p['conditions']:
            dest=d/f'{mode}.npz'
            if dest.exists():
                ob=Observation.load(str(dest));assert ob.sidecar['preflight_plan_sha256']==plan_hash
                secs[mode]=ob.sidecar['extraction_seconds']
            else:
                st=time.perf_counter()
                with Landmarker(contract) as lm:
                    ob=extract(item['video_path'],CropProxy(lm,item['box']) if mode=='crop' else lm,cid)
                secs[mode]=time.perf_counter()-st
                if mode=='crop':
                    x0,y0,x1,y1=item['box'];ob.sidecar.update(width=x1-x0,height=y1-y0)
                ob.sidecar.update(preflight_plan_sha256=plan_hash,preflight_mode=mode,
                    source_bbox_px=item['box'] if mode=='crop' else [0,0,*item['original_size']],
                    original_size=item['original_size'],extraction_seconds=secs[mode])
                ob.sidecar['quality']=ob.quality_summary();ob.save(str(dest))
            obs[mode]=ob
        np.testing.assert_array_equal(obs['native'].timestamp_us,obs['crop'].timestamp_us)
        np.testing.assert_array_equal(obs['native'].frame_id,obs['crop'].frame_id)
        assert obs['native'].sidecar['extractor_id']==obs['crop'].sidecar['extractor_id']
        row=dict(clip_id=cid,label=item['label'],box=item['box'],core_seconds=item['core_seconds'],seconds=secs,
          summaries={k:summary(v,item['core_seconds']) for k,v in obs.items()},
          original_core_geometry=geometry(obs['native'],item['box'],item['core_seconds']),
          output_sha256={k:sha256_file(d/f'{k}.npz') for k in obs},timestamps_identical=True)
        write(d/'comparison.json',row);rows.append(row)
        print(json.dumps(dict(completed=len(rows),total=12,label=item['label'],seconds=secs),ensure_ascii=False),flush=True)
    assert all(sha256_file(f)==v for f,v in p['pins'].items())
    write(U/'RESULT.json',dict(status='EXTRACTION_COMPLETE_PENDING_INDEPENDENT_REVIEW',plan_sha256=plan_hash,
        clips=len(rows),extractions=24,seconds=time.perf_counter()-started,rows=rows,training=0,classifier_inference=0))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);args=parser.parse_args()
    plan() if args.action=='plan' else run()
