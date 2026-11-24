"""Separate saved-output checks; no inference, no training, no human-truth claim."""
from pathlib import Path
import hashlib,json,numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=read(U/'FROZEN.json');m=read(U/'METRICS.json');assert read(U/'RUN_COMPLETE.json')['clips']==24
checks=[];total=0;hands=0;main=0;raw_count=0;output_pins={}
for r in f['rows']:
 assert sha(r['source'])==r['source_sha256'];cid=r['clip_id'];d=U/'runs'/cid
 a=np.load(d/'mediapipe/observations.npz');b=np.load(d/'rtmw_gpu/rtmw.npz');raw=np.load(d/'mediapipe/raw.npz');n=len(a['frame_id'])
 assert a['xy_px'].shape==(n,75,2) and b['xy_px'].shape==(n,133,2)
 assert np.array_equal(a['frame_id'],np.arange(n));assert np.array_equal(a['frame_id'],b['frame_id'])
 assert abs(a['timestamp_us']-b['timestamp_us']).max()<=1;assert (np.diff(a['timestamp_us'])>0).all()
 assert np.isfinite(b['xy_px']).all();assert len(b['frame_sha256'])==n
 total+=n;raw_count+=int(raw['raw_count'].sum())
 h=a['usable'][:,33:].reshape(n,2,21).sum(2)>=16;hands+=int(h.sum())
 mask=np.zeros(n,bool)
 for lo,hi in r['intervals'] or []:mask|=(a['timestamp_us']>=lo*1e6)&(a['timestamp_us']<=hi*1e6)
 main+=int(h[mask].sum());checks.append(dict(clip=cid,frames=n,same_timestamps=True,source_sha256_verified=True))
 for p in [d/'mediapipe/observations.npz',d/'mediapipe/raw.npz',d/'rtmw_gpu/rtmw.npz']:output_pins[str(p)]=sha(p)
assert total==m['totals']['frames']==2738;assert hands==m['totals']['mp_all_hand_slots']==5230;assert main==m['totals']['mp_main_hands_at_least_16']==1906
assert len(set(r['source_sha256'] for r in f['rows']))==24
for p,h in f['protected'].items():assert sha(p)==h
for p,h in f['models'].items():assert sha(U/p)==h
si=read(U/'VISUAL_INDEX.json');assert len(si)==49
for x in si:
 assert (U/x['image']).is_file();assert x['clip_id'] in [r['clip_id'] for r in f['rows']]
ss=read(U/'SPEED_SUMMARY.json')
for x in ss['methods']:
 vals=[read(p)['wall_seconds'] for p in (U/'speed').glob('*/?/'+x['method']+'/RUN.json')]
 assert len(vals)==9 and abs(float(np.median(vals))-x['median'])<1e-9
v=dict(status='PASS_SAVED_ARTIFACT_AND_ARITHMETIC_CHECKS',verifier='same Codex separate verification script; not independent human labels',clips=24,unique_sources=24,words=8,recorded_ids=[1,2,5,6],native_frames=total,protected_files_unchanged=len(f['protected']),model_files_pinned=len(f['models']),raw_mp_detections=raw_count,current_mp_hand_slots=hands,all_hand_slots=total*2,visual_centers=49,speed_methods_each_n=9,training=0,user_mentor_inputs=0,production_replacement=False,checks=checks,output_pins=output_pins,limitations=['RTMW raw scores are not visibility probabilities; no detection accuracy from output coverage.','49 qualitative original RGB center reviews are not independent landmark truth.','Current MP final output includes identity assignment; RTMW has no temporal assignment/smoothing.','Timing scopes intentionally separated in report; no end-to-end app speed claim.','No new user webcam or held-out sign-feedback evaluation performed.'])
(U/'VERIFICATION.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');print(v['status'],total)
