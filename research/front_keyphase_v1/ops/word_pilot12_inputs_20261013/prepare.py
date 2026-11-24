"""Authorized fixed public training pilot; no camera, network or sign-classifier calls."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

UNIT = Path(__file__).resolve().parent
BASE = UNIT.parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE/'src'), str(ROOT/'research/sign_model_pilot_v1/src'),
               str(BASE/'ops/rgb_pose_readiness_20261010'),
               str(BASE/'ops/rgb_tensor_adapter_20261010')]
import av
import numpy as np
from PIL import Image, ImageDraw
from wis_lab.capture.landmarker import Landmarker, DEFAULT_HAND_MODEL, DEFAULT_POSE_MODEL
from wis_lab.capture.video_source import extract
from pilot.web_input import to_native_like, active_span
from pilot.data import build_channels
from wis_lab.service.neural import _channels
from prepare_rgb_pairs import make_crop
from rgb_input import crop_original, validate

PILOT = BASE/'ops/word_acquisition_inventory_20261011/PILOT_RESULT.json'
OUT = Path('W:/WordsInSilence/data/AIHub_103/pilot12_inputs_20261013')
HANDS = (slice(33,54), slice(54,75))

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def preflight():
    assert not (UNIT/'SCOPE.json').exists(), 'Reuse existing scope, do not silently overwrite'
    pilot=read(PILOT)
    assert pilot['status']=='PASS' and len(pilot['clips'])==12
    files=[PILOT, BASE/'ops/word_acquisition_inventory_20261011/LOCAL_PILOT_PLAN.json',
           BASE/'config/contract.json', BASE/'src/wis_lab/capture/landmarker.py',
           BASE/'src/wis_lab/capture/video_source.py', BASE/'src/wis_lab/core/observation.py',
           BASE/'src/wis_lab/service/neural.py', ROOT/'research/sign_model_pilot_v1/src/pilot/data.py',
           ROOT/'research/sign_model_pilot_v1/src/pilot/web_input.py',
           BASE/'ops/rgb_pose_readiness_20261010/prepare_rgb_pairs.py',
           BASE/'ops/rgb_tensor_adapter_20261010/rgb_input.py',
           Path(DEFAULT_HAND_MODEL),Path(DEFAULT_POSE_MODEL),Path(__file__),UNIT/'verify.py']
    pins={str(p.resolve()):digest(p) for p in files}
    for c in pilot['clips']:
        assert '_REAL01_F' in c['clip_id'] and Path(c['path']).is_file()
        assert digest(c['path'])==c['sha256'], c['clip_id']
        pins[c['path']]=c['sha256']
    assert not OUT.exists(), 'Existing run must be inspected, not duplicated'
    scope={'at_kst':datetime.now().astimezone().isoformat(),
        'authorization':'User: 알겠어 일단 첫번째 작업부터 진행해줘. Fixed pilot input preparation only.',
        'source_pilot':str(PILOT),'output':str(OUT),'clips':[c['clip_id'] for c in pilot['clips']],
        'source_sha256':pins,'source_count':len(pins),'signer_ids':['01'],'frames_expected':1444,
        'role':'existing training-person technical input pilot, not generalization test',
        'observation':'Fresh production Landmarker per clip; all decoded PTS retained; no resize/mirror/coordinate interpolation',
        'pose_input':'Full-timeline 280 channels plus production _channels active-span parity. Labels never passed to feature functions.',
        'rgb':'Reuse fixed observed-only square1.5 bounding geometry; direct224 RGB and padding mask, every source frame. No region filling/hand synthesis.',
        'checks':['unchanged source hashes','decoded frame IDs and PTS equal observations','75-point missing NaNs and masks valid',
                  '280-channel hand/body masks and invalid zeros correct','web active-span channels equal prepared channels',
                  'RGB pixels and padding equal independent original-frame recrop','all crop corners and observed21 inclusion valid',
                  'source frame RGB hashes and timestamps exact','classifiers and camera never called'],
        'quality_policy':'Report observed and missing coverage in full and annotation intervals; do not assert all hands should be visible or tune filters.',
        'timing_policy':'Separate model creation, extraction incl existing double decode, channels, RGB decode/crop and cache writes. No speed optimization in this unit.',
        'limits':['12 clips from one known training signer','No sign recognition or handshape correctness accuracy','Actual runtime not an SLA for whole dataset'],
        'forbidden':['training','sign model inference','vocabulary selection','downloads','camera','source/model mutation','Git changes'],
        'runtime':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'av':av.__version__,'numpy':np.__version__}}
    write(UNIT/'SCOPE.json',scope)
    print(json.dumps({'preflight':'PASS','clips':12,'pins':len(pins),'output':str(OUT)}),flush=True)

def ratio_stats(obs,mask):
    n=int(mask.sum())
    lh=obs.usable[mask,33:54];rh=obs.usable[mask,54:75]
    return {'frames':n,'left_any':int(lh.any(1).sum()),'left_all21':int(lh.all(1).sum()),
            'right_any':int(rh.any(1).sum()),'right_all21':int(rh.all(1).sum()),
            'any_hand':int((lh.any(1)|rh.any(1)).sum()),'both_hands':int((lh.any(1)&rh.any(1)).sum()),
            'shoulders_both':int(obs.usable[mask][:,[11,12]].all(1).sum())}

def make_visual(path,cid,obs,panels):
    canvas=Image.new('RGB',(1200,460*len(panels)), 'white'); d=ImageDraw.Draw(canvas)
    for row,(fid,rgb,crops) in enumerate(panels):
        y=row*460
        thumb=Image.fromarray(rgb);thumb.thumbnail((760,428));td=ImageDraw.Draw(thumb)
        sx,sy=thumb.width/rgb.shape[1],thumb.height/rgb.shape[0]
        for side,sl in enumerate(HANDS):
            color=('red','darkorange')[side]
            for pt in obs.xy_px[fid,sl][obs.usable[fid,sl]]:
                x,z=pt[0]*sx,pt[1]*sy;td.ellipse((x-2,z-2,x+2,z+2),fill=color)
            box=crops[side].get('box')
            if box is not None: td.rectangle(tuple(np.asarray(box)*np.array([sx,sy,sx,sy])),outline=color,width=2)
        d.text((8,y+3),f'{cid} frame={fid} t={obs.timestamp_us[fid]}us',fill='black')
        canvas.paste(thumb,(8,y+25))
        for side,cr in enumerate(crops):
            xx=780+side*204;d.text((xx,y+36),('LEFT','RIGHT')[side]+': '+cr['reason'],fill='black')
            canvas.paste(Image.fromarray(cr['rgb']).resize((200,200)),(xx,y+65))
    canvas.save(path)

def run():
    scope=read(UNIT/'SCOPE.json')
    assert str(OUT)==scope['output']
    assert all(digest(p)==h for p,h in scope['source_sha256'].items())
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ('observations','channels','rgb','visual'): (OUT/name).mkdir()
    write(OUT/'SCOPE.json',scope)
    contract=read(BASE/'config/contract.json'); pilot=read(PILOT)
    start=time.perf_counter();reports=[]
    try:
        for idx,c in enumerate(pilot['clips']):
            cid=c['clip_id']; timing={};t=time.perf_counter()
            lm=Landmarker(contract);timing['model_creation_seconds']=time.perf_counter()-t
            try:
                t=time.perf_counter();obs=extract(c['path'],lm,cid);timing['extract_seconds']=time.perf_counter()-t
            finally: lm.close()
            obs.validate();n=obs.n_frames
            assert n==c['decoded_frames'] and np.array_equal(obs.frame_id,np.arange(n))
            assert obs.sidecar['dropped_non_monotonic_pts']==0 and obs.sidecar['source_sha256']==c['sha256']
            obspath=OUT/'observations'/f'{cid}.npz';t=time.perf_counter();obs.save(str(obspath));timing['observation_write_seconds']=time.perf_counter()-t
            t=time.perf_counter();nat,present=to_native_like(obs.xy_px,obs.usable)
            full=build_channels(nat,present,obs.timestamp_us)
            span=active_span(present);assert span is not None
            lo,hi=span;web,reason,web_stats=_channels(obs);assert web is not None,reason
            active=build_channels(nat[lo:hi+1],present[lo:hi+1],obs.timestamp_us[lo:hi+1])
            np.testing.assert_array_equal(active['x'],web['x'])
            channelpath=OUT/'channels'/f'{cid}.npz'
            np.savez_compressed(channelpath,x_full=full['x'],x_web=web['x'],
                frame_id=obs.frame_id,timestamp_us=obs.timestamp_us,web_frame_id=obs.frame_id[lo:hi+1],
                web_timestamp_us=obs.timestamp_us[lo:hi+1],span=np.array([lo,hi],np.int64),
                body_scale_full=np.array(full['body_scale']),hand_scales_full=np.array(list(full['hand_scales'].values())))
            timing['channels_and_write_seconds']=time.perf_counter()-t
            arrays={'pose_x':full['x'],'frame_id':obs.frame_id.copy(),'timestamp_us':obs.timestamp_us.copy(),
                'rgb':np.zeros((n,2,224,224,3),np.uint8),'pixel_valid':np.zeros((n,2,224,224),bool),
                'available':np.zeros((n,2),bool),'observed_rgb':np.zeros((n,2),bool),'region_estimated':np.zeros((n,2),bool),
                'landmark_observed_count':np.stack([obs.usable[:,sl].sum(1) for sl in HANDS],axis=1).astype(np.int32),
                'bbox_xyxy':np.full((n,2,4),-1,np.int32),'reason':np.full((n,2),'',dtype='U16'),
                'shoulder_relative_size':np.full((n,2),np.nan),'frame_relative_size':np.full((n,2),np.nan)}
            counts=Counter();frames=[];panels=[]
            core=np.zeros(n,bool)
            for a,b in c['intervals']: core|=(obs.timestamp_us>=a*1e6)&(obs.timestamp_us<=b*1e6)
            selected={lo,(lo+hi)//2,hi}
            t=time.perf_counter()
            with av.open(c['path'],mode='r') as con:
                stream=con.streams.video[0];assert float(stream.metadata.get('rotate','0'))%360==0
                assert stream.sample_aspect_ratio is None or stream.sample_aspect_ratio==1
                first=None;got=0
                for fid,frame in enumerate(con.decode(stream)):
                    assert frame.pts is not None
                    first=frame.pts if first is None else first
                    ts=int(round(float((frame.pts-first)*stream.time_base)*1e6))
                    assert ts==obs.timestamp_us[fid]
                    rgb=frame.to_ndarray(format='rgb24');assert rgb.shape==(c['height'],c['width'],3)
                    shoulder=obs.xy_px[fid,[11,12]].copy();shoulder[~obs.usable[fid,[11,12]]]=np.nan
                    shown=[]
                    for side,sl in enumerate(HANDS):
                        cr=make_crop(rgb,obs.xy_px[fid,sl],obs.usable[fid,sl],shoulder)
                        reason=cr['crop_reason'];counts[reason]+=1;arrays['reason'][fid,side]=reason
                        if cr['crop_valid']:
                            box=cr['bbox_xyxy'];im,pm=crop_original(rgb,box)
                            arrays['rgb'][fid,side]=im;arrays['pixel_valid'][fid,side]=pm
                            arrays['available'][fid,side]=arrays['observed_rgb'][fid,side]=True
                            arrays['bbox_xyxy'][fid,side]=box
                            for k in ('shoulder_relative_size','frame_relative_size'):arrays[k][fid,side]=cr[k]
                        shown.append({'reason':reason,'rgb':arrays['rgb'][fid,side],
                                      'box':cr['bbox_xyxy'] if cr['crop_valid'] else None})
                    frames.append({'frame_id':fid,'timestamp_us':ts,'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
                    if fid in selected: panels.append((fid,rgb.copy(),shown))
                    got+=1
                assert got==n
            timing['rgb_decode_crop_seconds']=time.perf_counter()-t
            validate(arrays)
            rgbpath=OUT/'rgb'/f'{cid}.npz';t=time.perf_counter();np.savez_compressed(rgbpath,**arrays)
            timing['rgb_write_seconds']=time.perf_counter()-t
            write(OUT/'rgb'/f'{cid}.frames.json',frames)
            make_visual(OUT/'visual'/f'{cid}.jpg',cid,obs,panels)
            report={'clip_id':cid,'label':c['label'],'source':c['path'],'source_sha256':c['sha256'],
                'observation':str(obspath),'observation_sha256':digest(obspath),'channels':str(channelpath),
                'channels_sha256':digest(channelpath),'rgb_cache':str(rgbpath),'rgb_sha256':digest(rgbpath),
                'frames':n,'duration_seconds':c['actual_duration_s'],'full':ratio_stats(obs,np.ones(n,bool)),
                'annotation_core':ratio_stats(obs,core),'web_span':[lo,hi],'web_support':web_stats,
                'crop_reasons':dict(counts),'timing':timing,'extractor_id':obs.sidecar['extractor_id'],
                'cache_bytes':{'observation':obspath.stat().st_size,'channels':channelpath.stat().st_size,'rgb':rgbpath.stat().st_size},
                'handedness_disagreements':obs.sidecar['handedness_label_disagreements'],
                'handedness_comparisons':obs.sidecar['handedness_label_comparisons'],'intervals':c['intervals']}
            reports.append(report);write(OUT/'progress.json',{'pid':os.getpid(),'completed':idx+1,'total':12,'last':cid,'elapsed_seconds':time.perf_counter()-start})
            print(json.dumps({'completed':idx+1,'frames':n,'extract_s':round(timing['extract_seconds'],2),'crop_counts':dict(counts)}),flush=True)
        assert sum(r['frames'] for r in reports)==1444
        unchanged=all(digest(p)==h for p,h in scope['source_sha256'].items());assert unchanged
        totals={name:dict(sum((Counter(r[name]) for r in reports),Counter())) for name in ('full','annotation_core','crop_reasons','cache_bytes','timing')}
        result={'status':'PRODUCER_COMPLETE_PENDING_INDEPENDENT_CHECK','clips':reports,'totals':totals,
            'elapsed_seconds':time.perf_counter()-start,'source_files_unchanged':len(scope['source_sha256']),
            'classification_inference':0,'training':0,'camera':0,'new_video_extractions_from_zip':0,
            'note':'MediaPipe detector inference only; no sign recognition scores. Full and active features have separately computed normalization scales.'}
        write(OUT/'RESULT.json',result);write(UNIT/'EXECUTION_RESULT.json',{'result':str(OUT/'RESULT.json'),'status':result['status']})
    except Exception as exc:
        write(OUT/'FIRST_FAILURE.json',{'type':type(exc).__name__,'message':str(exc),'completed_clips':len(reports)})
        raise

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    preflight() if args.preflight else run()
