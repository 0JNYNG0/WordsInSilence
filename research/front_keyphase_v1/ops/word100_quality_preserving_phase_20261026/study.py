from pathlib import Path
import sys,json,hashlib,datetime,traceback
from collections import defaultdict
import numpy as np
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_broad_phase_alignment_20261026'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def check(pins):
 for p,h in pins.items():assert sha(p)==h,p
NORMAL={'fast075','slow15','pace','pause'};GROUPS={'angle':(0,15),'palm':(15,17),'elbow':(17,19),'pose_wrist':(19,21),'hand_wrist':(21,23)}
def freeze():
 f=read(B/'FROZEN.json');r=read(B/'RESULT.json');v=read(B/'VERIFIED.json');q=read(B/'QUALITY_CORRECTION.json');check(r['pins']);check(q['pins']);check(f['protected']);assert v['gate']['passed']
 pins={**r['pins'],**q['pins'],**{str(p):sha(p) for p in [Path(__file__),B/'FROZEN.json',B/'RESULT.json',B/'VERIFIED.json',B/'QUALITY_CORRECTION.json']}}
 write(U/'FROZEN.json',dict(at=now(),pins=pins,protected=f['protected'],rows=r['rows'],authorization='Autonomous common feedback-improvement goal. No training/inference/operating modification.',
  rule='After existing full descriptor availability fallback, if original descriptor had corroborated quality but candidate descriptor no longer does, retain original whole5component descriptor for that channel. Coordinate-only evidence stays coordinate-only; no upgrade from missing to observed. Selection by evidence flags only, never error/word/ground truth time. Retain original instantaneous discrepancy separately.',
  evaluation='Reuse all4500 frozen public100word developed DEV conditions, no new timewarp/detector/DTW/TCN. Preserved expected grid, instantaneous evidence and old sequence coverage plus old quality. Compare known normal effective channel-weighted time, per-group residuals on same support, deleted-window before/mid/after no positive discrepancy erased to numerical zero1e-6. No semantic classification or new human test.',
  gate='All preservation invariants; normal effective time lower; each normal residual group lower; no deleted-region positive channel becomes numericzero. Research diagnostic only. Earlier3word cross-person primary loss remains unresolved and is not overwritten by this new metric.',
  limits='Per-channel source may differ, not coherent whole pose. Body quality correction sidecars are required. No user/mentor/extra199/200. Existing models and operating untouched.'))
 print('FROZEN_QUALITY_ONLY_SELECTION',flush=True)
def run():
 f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);(U/'cases').mkdir();rows=[];pins={}
 for row in f['rows']:
  z=load(B/'cases'/(row['name']+'.npz'));q=load(B/'quality_correction'/(row['name']+'.npz')) if row['kind'] in ['wholeL','wholeR','gapL','gapR'] else z
  oldq=q['sequence_quality'];newq=q['sequence_new_quality'];fallback=oldq&~newq
  delta=np.where(fallback[...,None],z['sequence_old_delta'],z['sequence_new_delta']);qual=oldq|newq;source=np.where(fallback,2,z['phase_source']);effective=np.where(fallback,z['old_error'][:,None,None],z['effective_error'])
  p=U/'cases'/(row['name']+'.npz');np.savez_compressed(p,sequence_delta=delta,sequence_quality=qual,source=source,quality_fallback=fallback,effective_error=effective)
  pins[str(p)]=sha(p);rows.append(dict(name=row['name'],clip_id=row['clip_id'],kind=row['kind'],word_code=row['word_code'],fallback=int(fallback.sum())))
 check(f['pins']);check(f['protected']);write(U/'RESULT.json',dict(at=now(),status='SAVED_PENDING_VERIFY',rows=rows,pins=pins,training=0,production_changed=False));print('SAVED4500',sum(x['fallback'] for x in rows),flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=U/'FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
