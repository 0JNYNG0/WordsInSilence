import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import sys,json,shutil,traceback,importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];P=O/'word100_rest_aware_alignment_20261021'
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
prev=module('motion_previous',P/'study.py');C=module('motion_candidate',U/'candidate.py')
read,load,write,sha,now=prev.read,prev.load,prev.write,prev.sha,prev.now

def freeze():
 f=read(P/'FROZEN.json');prev.prev.pipe.check(f['pins']);prev.prev.pipe.check(f['protected'])
 assert read(P/'SYNC_CHECK.json')['status']=='PASS' and not (P/'RUNNING.json').exists()
 pins=dict(f['pins'])
 for folder in ('pairs','queries','stress'):
  for p in (P/folder).glob('*'):pins[str(p)]=sha(p)
 for p in [U/'candidate.py',U/'study.py',P/'FINAL_AUDIT.json',P/'VISUAL_REVIEW.json',O/'word100_repeat_compression_guard_20261020/RESULT_KO.txt']:
  pins[str(p)]=sha(p)
 (U/'before').mkdir()
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
 write(U/'FROZEN.json',dict(at=now(),rows=f['rows'],roles=f['roles'],protected=f['protected'],pins=pins,
 authorization='User: 우선 그렇게 진행해줘; motion vs visibility and local alignment substitution diagnostics, no training or production change.',
 method=['Location observed separately from geometric position/angle change. No point/state deletion.',
 'Position envelope: midpoint of existing and rechecked palm centres, radius half their distance plus1pixel, normalized by existing body frame. Adjacent original frames only; no interpolation over absence.',
 'Angle envelope: min/max of two current estimates for state4angles. Non-overlap (epsilon1e-6) gives geometric change evidence; no semantic activity claim or calibrated statistical confidence.',
 'Positive full-source stable lowering reused as separate axis, not overwritten by ROI presence; lowering/change conflict shown separately. Absence never creates lowering.',
 'DTW plateau: several observed changing samples to one sample, both directions, perhand/angle. Warnings only. No path/score/scope/threshold changes and no wrong-sign verdict.',
 'Keep source core gap timeline independent of new path. Unknown/absent inside public annotated core recorded unless positive source lowering; does not shrink reference denominator.',
 'Aggregate flags >=5 distinct reference people at same query hand/angle. This is review agreement, not sign-error ground truth.'],
 controls=['27original clips/216pairs for all3savedconditions; 8changedpair RGB counterexamples.',
 '54whole-hand/27short3frame saved-trajectory controls,432+216 saved paths; query evidence removed consistently. Reference expected identical.',
 'New duration0.75/1.5 uniform resampling and0.5s midpoint pause for27clips each=81controls/648alignments. Finite interpolation only; masks AND. Synthetic, noRGBextract.',
 '2/3repeats for27clips each using known construction matched repeated references and tiled saved paths=54controls/432diagnostics. Does not infer repeatcount.'],
 scope='TRAIN3words9people27clips; old80normal/cross preservation evidence reused, not new100word alignment. All no training/classifier/user/mentor/DEV/heldout/camera/download/external/Git/web/modelchange.',
 decision='Diagnostic only even if warnings appear on normal speed/repetition. Report rates, missed known failures, gaps and unknowns honestly. No post-result tuning.'))
 print('FROZEN',flush=True)

def data(row):
 q=prev.sample(row);src=load(prev.R/'schedules'/(row['clip_id']+'__source.npz'))
 native=prev.prev.Obs.load(row['native']);origin,scale,_=prev.prev.body_frame(native)
 e=C.primitives(q,src['rest'],origin,scale)
 return q,e

def summary(e,path,expected,q,r,required):
 x=C.inspect(q,r,path,expected,required)
 e.update(events=x['events'],warning_angle_frames=int(x['angle_warning'].sum()),warning_position_frames=int(x['position_warning'].sum()),
   gap_angle_frames=int(x['insufficient_timeline'].sum()),missing_angle_frames=int(x['missing_timeline'].sum()),wrong_action=None)
 return x

def save_pair(folder,pid,meta,x):
 np.savez_compressed(U/folder/(pid+'.npz'),**{k:x[k] for k in ('angle_warning','position_warning','insufficient_timeline','missing_timeline','expected','path')})
 write(U/folder/(pid+'.json'),meta)

def original():
 f=read(U/'FROZEN.json');prev.prev.pipe.check(f['pins']);prev.prev.pipe.check(f['protected'])
 for sub in ('evidence','pairs','queries'): (U/sub).mkdir()
 ds={r['clip_id']:data(r) for r in f['rows']};counts=Counter();allrows=[]
 for qr in f['rows']:
  cid=qr['clip_id'];q,e=ds[cid];st=C.states(e);np.savez_compressed(U/'evidence'/(cid+'.npz'),**e,**st)
  counts.update(clip=1,location_seen=int(e['observed'].sum()),position_change=int(st['position_change'].sum()),angle_change=int(st['angle_change'].sum()),
    lowered=int(e['source_rest'].sum()),seen_and_lowered=int((e['observed']&e['source_rest']).sum()),rest_change_conflict=int(st['rest_change_conflict'].sum()))
  required=f['roles'].get(qr['word_code'],{}).get('required_sides',['L','R'])
  refs=[r for r in f['rows'] if r['word_code']==qr['word_code'] and r['signer_id']!=qr['signer_id']]
  for c in prev.CONDITIONS:
   votes=np.zeros_like(e['angle_valid'],np.int16);pvs=np.zeros_like(e['observed'],np.int16)
   for rr in refs:
    rid=rr['clip_id'];pid=c+'__'+cid+'__'+rid;stored=load(P/'pairs'/(pid+'.npz'))
    meta=dict(query=cid,reference=rid,condition=c);x=summary(meta,stored['path'],stored['expected'],e,ds[rid][1],required)
    save_pair('pairs',pid,meta,x);votes+=x['angle_warning'];pvs+=x['position_warning']
    allrows.append({k:v for k,v in meta.items() if k!='events'})
   meta=dict(query=cid,condition=c,angle_consensus_frames=int((votes>=5).sum()),position_consensus_frames=int((pvs>=5).sum()),
     references=8,wrong_action=None)
   np.savez_compressed(U/'queries'/(c+'__'+cid+'.npz'),angle_votes=votes,position_votes=pvs)
   write(U/'queries'/(c+'__'+cid+'.json'),meta)
  print('ORIGINAL',cid,flush=True)
 write(U/'ORIGINAL.json',dict(counts=dict(counts),pairs=allrows))

def damaged(q,e,side,indices):
 # Current query evidence only: original visible/fullview witnesses cannot leak into missing controls.
 z={k:v.copy() for k,v in e.items()};h=('L','R').index(side)
 for k in ('observed','palm_valid','source_rest'):z[k][indices,h]=False
 for k in ('angle_valid',):z[k][indices,h]=False
 for k in ('centre','radius','angle_low','angle_high'):z[k][indices,h]=np.nan
 z['missing'][indices,h]=True;z['unconfirmed'][indices,h]=True
 return z

def missing():
 f=read(U/'FROZEN.json');(U/'missing').mkdir();ds={r['clip_id']:data(r) for r in f['rows']};out=[]
 for qr in f['rows']:
  cid=qr['clip_id'];q,e=ds[cid];required=f['roles'].get(qr['word_code'],{}).get('required_sides',['L','R'])
  refs=[r for r in f['rows'] if r['word_code']==qr['word_code'] and r['signer_id']!=qr['signer_id']]
  for typ,side in [('whole','L'),('whole','R'),('short',required[-1])]:
   for rr in refs:
    rid=rr['clip_id'];name=typ+'_'+side+'__'+cid+'__'+rid;saved=load(P/'stress'/(name+'.npz'));indices=saved['removed_query_rows']
    z=damaged(q,e,side,indices);meta=dict(type=typ,side=side,query=cid,reference=rid)
    x=summary(meta,saved['path'],saved['expected'],z,ds[rid][1],required)
    h=('L','R').index(side);expected_gap=int((z['core'][indices]).sum())*15 if side in required else 0
    assert int(x['missing_timeline'][indices,h].sum())==expected_gap
    meta.update(deleted_cells_marked=expected_gap,range_preserved=bool(np.array_equal(x['expected'],saved['expected'])))
    save_pair('missing',name,meta,x);out.append({k:v for k,v in meta.items() if k!='events'})
  print('MISSING',cid,flush=True)
 write(U/'MISSING.json',dict(rows=out,statement='All required-side deleted core cells explicitly shown; optional hand remains optional. No wrong_action verdict.'))

def transform(q,e,kind):
 n=len(q['core'])
 if kind.startswith('speed'):
  factor=float(kind[5:]);idx=np.linspace(0,n-1,round((n-1)*factor)+1)
 elif kind=='pause':
  mid=int(np.flatnonzero(q['core'])[len(np.flatnonzero(q['core']))//2]);idx=np.r_[np.arange(mid+1),np.full(15,mid),np.arange(mid+1,n)]
 else:raise ValueError(kind)
 a=np.floor(idx).astype(int);b=np.ceil(idx).astype(int);t=idx-a
 def change(v):
  if v.dtype.kind=='b':return v[a]&v[b]
  if v.dtype.kind in 'iu':return v[a].copy()
  w=t.reshape((-1,)+(1,)*(v.ndim-1));valid=np.isfinite(v[a])&np.isfinite(v[b]);return np.where(valid,v[a]*(1-w)+v[b]*w,np.nan)
 zz={k:change(v) for k,v in e.items()};qq={k:change(q[k]) for k in ('features','present','core')}
 zz['frame_id']=np.arange(len(idx));zz['timestamp_us']=np.rint(np.arange(len(idx))*1e6/30).astype(np.int64)
 # Invalid endpoints mean unknown, never reconstructed evidence.
 zz['unconfirmed']=~zz['angle_valid'];zz['missing']=e['missing'][a]|e['missing'][b]
 return qq,zz,idx

def repeated(z,k):
 out={a:np.concatenate([v]*k) for a,v in z.items()};out['frame_id']=np.arange(len(out['core']))
 out['timestamp_us']=np.rint(np.arange(len(out['core']))*1e6/30).astype(np.int64);return out

def controls():
 f=read(U/'FROZEN.json');(U/'controls').mkdir();ds={r['clip_id']:data(r) for r in f['rows']};out=[]
 for qr in f['rows']:
  cid=qr['clip_id'];q,e=ds[cid];required=f['roles'].get(qr['word_code'],{}).get('required_sides',['L','R'])
  refs=[r for r in f['rows'] if r['word_code']==qr['word_code'] and r['signer_id']!=qr['signer_id']]
  for kind in ['speed0.75','speed1.5','pause','repeat2','repeat3']:
   if kind.startswith('repeat'):k=int(kind[-1]);zz=repeated(e,k)
   else:qq,zz,source_idx=transform(q,e,kind)
   votes=np.zeros_like(zz['angle_valid'],np.int16)
   for rr in refs:
    rid=rr['clip_id'];rf,re=ds[rid];pid=cid+'__'+rid;old=load(P/'pairs'/('baseline__'+pid+'.npz'))
    if kind.startswith('repeat'):
     ref=repeated(re,k);path=np.concatenate([old['path']+np.array([c*len(e['core']),c*len(re['core'])]) for c in range(k)])
     expected=np.concatenate([old['expected']]*k)
    else:
     ref=re;weights=prev.C.weights_for(required,(False,False),prev.F.GROUPS)
     cost,_=prev.legacy(qq,rf,group_weights=weights,hand_participation=False,one_sided_mode='flat')
     path=np.asarray(prev.dp(cost,.2),int);expected=old['expected']
    meta=dict(query=cid,reference=rid,kind=kind);x=summary(meta,path,expected,zz,ref,required);votes+=x['angle_warning']
    save_pair('controls',kind+'__'+pid,meta,x)
   out.append(dict(query=cid,kind=kind,consensus_angle_frames=int((votes>=5).sum()),has_consensus=bool((votes>=5).any()),wrong_action=None))
  print('CONTROLS',cid,flush=True)
 write(U/'CONTROLS.json',dict(rows=out,repeats='Known exact repeated sequences with repeated reference and tiled saved path; not inferred count, not independent repeat alignment.'))

if __name__=='__main__':
 try:
  stage=sys.argv[1]
  if stage=='freeze':freeze()
  else:
   f=read(U/'FROZEN.json');prev.prev.pipe.check(f['pins']);prev.prev.pipe.check(f['protected'])
   lock=U/(stage+'_RUNNING.json');write(lock,dict(pid=os.getpid(),at=now()));globals()[stage]()
   prev.prev.pipe.check(f['protected']);lock.unlink()
 except BaseException:
  p=U/'FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
