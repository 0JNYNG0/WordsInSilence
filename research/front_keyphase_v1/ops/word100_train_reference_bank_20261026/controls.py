"""Deterministic observation-level controls; no new RGB detector or classifier calls."""
import prepare as R
import numpy as np,sys,os,time,traceback
from collections import defaultdict,Counter
from pathlib import Path
G=R.mod('fixed_empirical_range',R.U/'range.py');M=R.mod('fixed_public_pair_controls',R.O/'word100_multipeople_feedback_integration_20261026/run.py')
KINDS=['speed075','speed125','delete3','missing_L','missing_R','gap3_L','gap3_R']

def transformed(q,kind):
 n=len(q['core']);ix=np.arange(n);core=np.flatnonzero(q['core']);assert len(core)>=3;middle=int(core[len(core)//2]);cut=np.arange(max(0,min(n-3,middle-1)),max(0,min(n-3,middle-1))+3)
 if kind.startswith('speed'):
  speed=.75 if kind=='speed075' else 1.25;ix=np.rint(np.linspace(0,n-1,int(np.ceil((n-1)/speed))+1)).astype(int)
 elif kind=='delete3':ix=np.delete(ix,cut)
 z={k:(v[ix].copy() if v.ndim>0 and v.shape[0]==n else v.copy()) for k,v in q.items()};origtime=q['timestamp_us'];step=float(np.median(np.diff(origtime)))
 if kind.startswith('speed'):z['timestamp_us']=np.rint(np.linspace(origtime[0],origtime[0]+(origtime[-1]-origtime[0])/speed,len(ix))).astype(np.int64)
 elif kind=='delete3':z['timestamp_us']=np.rint(origtime[0]+np.arange(len(ix))*step).astype(np.int64)
 h=-1;affected=np.zeros(len(ix),bool)
 if kind.startswith(('missing','gap3')):
  h=0 if kind.endswith('L') else 1;affected[:]=True if kind.startswith('missing') else False
  if kind.startswith('gap3'):affected[cut]=True
  hand=slice(33+21*h,54+21*h);z['xy_px'][affected,hand]=np.nan;z['usable'][affected,hand]=False
  for key in ['available','quality_before','quality_after','quality_eligible']:
   z[key][affected,h,:17]=False;z[key][affected,h,21:]=False
  z['values'][affected,h,:17]=np.nan;z['values'][affected,h,21:]=np.nan
  for key in ['state_before','state_after']:z[key][affected,h]=0
  # Match removal of hand observations in all dependent static feature groups, preserving arm/body.
  side='L' if h==0 else 'R';cursor=0
  for gi,g in enumerate(M.F.GROUP_ORDER):
   width=M.F.GROUPS[g][0]
   if g=='bimanual' or (g.endswith('_'+side) and not g.startswith('arm_')):z['features'][affected,cursor:cursor+width]=np.nan;z['present'][affected,gi]=False
   cursor+=width
  assert not z['available'][affected,h,:17].any();np.testing.assert_array_equal(z['values'][:,1-h],q['values'][:,1-h]);np.testing.assert_array_equal(z['values'][:,h,17:21],q['values'][:,h,17:21])
 assert (np.diff(z['timestamp_us'])>0).all();return z,ix,cut,h,affected

def protocol():
 assert M.F.GROUPS['motion_L'][1]==0 and M.F.GROUPS['motion_R'][1]==0
 f=R.read(R.U/'FROZEN.json');by=defaultdict(list)
 for row in f['catalog']:by[row['word_code']].append(row)
 rows=[]
 for word,rr in sorted(by.items()):
  q=min(rr,key=lambda x:x['signer_id']);rows.append(dict(query=q['clip_id'],word_code=word,heldout_person=q['signer_id'],references=[r['clip_id'] for r in rr if r['signer_id']!=q['signer_id']]))
 assert len(rows)==100
 R.write(R.U/'CONTROLS_PROTOCOL.json',dict(at=R.now(),rows=rows,kinds=KINDS,expected_pairs=sum(len(x['references'])*len(KINDS) for x in rows),pins={str(p):R.sha(p) for p in [Path(__file__),R.U/'RANGE_PROTOCOL.json',R.U/'FROZEN.json',Path(M.__file__),Path(M.F.__file__),Path(M.D.__file__)]},
  interpretation='Observation-level input stress, not fresh RGB/MediaPipe trials or a true erroneous-sign dataset. Known original source indices and timestamps. Freeze original perreference range without mutation. Never retune rule or choose model from control score.',
  transformations='Same public original rows only. Slow.75/fast1.25 nearest stored input sample on linear source-index grid with explicit scaled timestamps. Three inputframes removed at core temporal midpoint and time gap closed. Wholehand/3frame absent L/R independently removes all hand-dependent coordinate/feature groups, keeps body/otherhand; dynamic feature neighboring values not recomputed and its frozen cost weight0. No side swap or coordinate interpolation.',
  checks='Normal speed across same public reference retains domain. Compare new matched source-index with original path, reporting drift rather than groundtruth across people. Count old supported to new insufficient separately; never improved. For deletion target reference neighborhoods mapped to removed source frame interval plus2, report missed/changed/uncertain, not automatic linguistic-error annotation. Wholehand absence must yield no within-band on that hand, rawarm/otherhand/sourcecoordinates unchanged. Gap3 must not be hidden by realignment: report cases escaping targeted gap; do not declare pass from fixed-mask checks alone.',
  manual_limits='No new human sign phase labels; original crossings and resting hand coordinates unchanged. This is preservation/structural audit, not proof all normal crossings or onehand semantics are correct.',training=0,classifier_inference=0,new_detector=0))
 print('CONTROLS_PROTOCOL_FROZEN',len(rows),sum(len(x['references'])*len(KINDS) for x in rows),flush=True)

def freeze():
 p=R.read(R.U/'CONTROLS_PROTOCOL.json');R.check(p['pins']);assert R.read(R.U/'RANGE_VERIFIED.json')['counts']['pairs']==7184;f=R.read(R.U/'RANGE_FROZEN.json');r=R.read(R.U/'RANGE_DONE.json');R.check(f['pins']);R.check(r['pins']);R.write(R.U/'CONTROLS_FROZEN.json',dict(at=R.now(),rows=p['rows'],kinds=p['kinds'],expected_pairs=p['expected_pairs'],inputs=f['rows'],original_ranges=r['rows'],protected=f['protected'],pins={**f['pins'],**r['pins'],**{str(x):R.sha(x) for x in [R.U/'CONTROLS_PROTOCOL.json',R.U/'RANGE_VERIFIED.json',R.U/'RANGE_DONE.json',Path(__file__)]}}))
 for d in ['control_inputs','control_pairs','control_ranges','control_records','control_starts']:(R.U/d).mkdir()
 print('CONTROLS_FROZEN',p['expected_pairs'],flush=True)

def run():
 f=R.read(R.U/'CONTROLS_FROZEN.json');R.check(f['pins']);R.check(f['protected']);R.write(R.U/'CONTROLS_RUNNING.json',dict(at=R.now(),pid=os.getpid()));samples={x['clip_id']:R.load(x['output']) for x in f['inputs']};ranges={(x['query'],x['reference']):x for x in f['original_ranges']};rows=[];pins={};counts=Counter();beg=time.perf_counter()
 for selected in f['rows']:
  qid=selected['query'];q=samples[qid]
  for kind in f['kinds']:
   changed,ix,cut,h,affected=transformed(q,kind);inputpath=R.U/'control_inputs'/(qid+'__'+kind+'.npz')
   if inputpath.exists():
    old=R.load(inputpath)
    for k,v in changed.items():np.testing.assert_array_equal(old[k],v)
   else:np.savez_compressed(inputpath,**changed,source_index=ix,deleted_candidate_indices=cut,affected_frames=affected)
   pins[str(inputpath)]=R.sha(inputpath)
   for rid in selected['references']:
    name=qid+'__'+rid+'__'+kind;rec=R.U/'control_records'/(name+'.json')
    if rec.exists():
     rr=R.read(rec);assert rr['context_sha256']==R.sha(R.U/'CONTROLS_FROZEN.json');R.check(rr['outputs']);rows.append(rr['row']);pins.update(rr['outputs']);counts.update(rr['counts']);continue
    sp=R.U/'control_starts'/(name+'.json');assert not sp.exists();R.write(sp,dict(at=R.now()));oldmeta=ranges[qid,rid];b=R.load(oldmeta['output']);oldpair=R.load(oldmeta['pair']);r=samples[rid];pair=M.pair(changed,r);pp=R.U/'control_pairs'/(name+'.npz');np.savez_compressed(pp,**pair);po,se,_,_=G.support(pair,changed,r);ps=G.classify(pair['domain'],pair['point_ok'],po,pair['point_delta'],b['point_bound'],b['point_peer_count']);ss=G.classify(pair['domain'],pair['sequence_ok'],se,pair['sequence_delta'],b['sequence_bound'],b['sequence_peer_count']);rp=R.U/'control_ranges'/(name+'.npz');np.savez_compressed(rp,point_status=ps,sequence_status=ss,point_evidence=po,sequence_evidence=se)
    np.testing.assert_array_equal(pair['domain'],b['domain']);mapped_old=np.interp(pair['phase_old'],np.arange(len(ix)),ix);drift=np.abs(mapped_old-oldpair['phase_old']);target=(oldpair['phase_old']>=cut.min()-2)&(oldpair['phase_old']<=cut.max()+2)&r['core'];c=Counter(pairs=1,domain=int(pair['domain'].sum()))
    for mode,status in [('point',ps),('sequence',ss)]:
     before=b[mode+'_status'];old_supported=(before==4)|(before==5);new_supported=(status==4)|(status==5)
     for group,sl in [('angle',slice(0,15)),('position',slice(15,23))]:
      for code in range(1,6):c[f'{mode}_{group}_state{code}']=int((status[:,:,sl]==code).sum())
     c[mode+'_old_supported_to_unknown']=int((old_supported&~new_supported).sum());c[mode+'_old_beyond_to_within']=int(((before==5)&(status==4)).sum());c[mode+'_new_beyond']=int(((before!=5)&(status==5)).sum());c[mode+'_target_new_beyond']=int(((before!=5)&(status==5))[target].sum());c[mode+'_target_lost_support']=int((old_supported&~new_supported)[target].sum())
     if kind.startswith('missing'):
      assert not (status[:,h,:17]>=4).any() and not(status[:,h,21:]>=4).any();c[mode+'_wholehand_no_false_normal']=1
    summary=dict(kind=kind,query=qid,reference=rid,input=str(inputpath),pair=str(pp),output=str(rp),original_range=oldmeta['output'],original_pair=oldmeta['pair'],original_frame_mapping=ix.tolist(),target_original_indices=cut.tolist(),alignment_drift_mean=float(drift[r['core']].mean()) if r['core'].any() else None,alignment_drift_max=float(drift[r['core']].max()) if r['core'].any() else None,target_reference_frames=int(target.sum()),counts=dict(c));out={str(pp):R.sha(pp),str(rp):R.sha(rp),str(inputpath):R.sha(inputpath)};R.write(rec,dict(at=R.now(),context_sha256=R.sha(R.U/'CONTROLS_FROZEN.json'),row=summary,counts=dict(c),outputs=out));rows.append(summary);pins.update(out);counts.update(c)
  print('EMPIRICAL_CONTROLS',len(rows),f['expected_pairs'],'seconds',round(time.perf_counter()-beg,1),flush=True)
 assert len(rows)==f['expected_pairs'];R.check(f['protected']);R.write(R.U/'CONTROLS_DONE.json',dict(at=R.now(),rows=rows,pins=pins,counts=dict(counts),seconds=time.perf_counter()-beg,verified=False,linguistic_error_groundtruth=False));(R.U/'CONTROLS_RUNNING.json').unlink();print('EMPIRICAL_CONTROLS_DONE',len(rows),flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=R.U/('CONTROLS_'+sys.argv[1].upper()+'_FIRST_ERROR.txt')
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
