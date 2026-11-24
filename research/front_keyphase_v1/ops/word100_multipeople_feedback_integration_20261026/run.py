import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
from pathlib import Path
import sys,json,datetime,hashlib,importlib.util,time,traceback
from collections import defaultdict,Counter
import numpy as np
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;O=U.parent;BP=O/'word100_broad_phase_preflight_20261026';B=O/'word100_broad_phase_alignment_20261026';CROOT=O/'word100_feedback_evidence_consumer_20261026';sys.path.insert(0,str(O.parent/'src'))
from wis_lab.core import dtw as D,features as F
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
C=module('research_consumer',CROOT/'consumer.py');P=module('frozen_fractional',O/'word100_fractional_phase_alignment_20261026/candidate.py')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False)
def check(pins):
 for p,h in pins.items():assert sha(p)==h,p

def map_values(z,index,first,last):
 a=np.floor(index).astype(int);b=np.ceil(index).astype(int);w=(index-a)[:,None,None]
 valid=z['available'][a]&z['available'][b];quality=z['quality_eligible'][a]&z['quality_eligible'][b]
 return dict(values=np.where(valid,z['values'][a]*(1-w)+z['values'][b]*w,np.nan),valid=valid,quality=quality,first=first,last=last,source_valid=z['available'],source_quality=z['quality_eligible'])
def describe(m,domain):
 n=len(domain);v=np.full(domain.shape+(5,),np.nan);ok=np.zeros(domain.shape,bool);quality=ok.copy()
 for j in range(2,n-2):
  idx=[j-2,j,j+2];start=int(m['first'][j-2]);end=int(m['last'][j+2]);valid=domain[j-2:j+3].all(0)&m['source_valid'][start:end+1].all(0)&m['valid'][idx].all(0)
  p=np.moveaxis(m['values'][idx],0,-1);v[j]=np.concatenate([p,np.diff(p,axis=-1)],-1);ok[j]=valid;quality[j]=valid&m['source_quality'][start:end+1].all(0)&m['quality'][idx].all(0)
 return np.where(ok[...,None],v,np.nan),ok,quality
def pair(q,r):
 cost,evidence=D.frame_cost_matrix(q,r,one_sided_mode='flat',hand_participation=False);path=np.asarray(D._optimal_path(cost,.2),int);n=len(r['core']);assert len(path) and (path[0]==[0,0]).all() and (path[-1]==[len(q['core'])-1,n-1]).all()
 ids=[path[path[:,1]==j,0] for j in range(n)];base=np.array([np.median(x) for x in ids]);first=np.array([x.min() for x in ids]);last=np.array([x.max() for x in ids]);new=P.phase(base);domain=np.broadcast_to(r['core'][:,None,None],(n,2,23)).copy()
 om=map_values(q,base,first,last);nm=map_values(q,new,np.minimum(first,np.floor(new).astype(int)),np.maximum(last,np.ceil(new).astype(int)));rm=map_values(r,np.arange(n,dtype=float),np.arange(n),np.arange(n))
 od,ov,oq=describe(om,domain);nd,nv,nq=describe(nm,domain);rd,rv,rq=describe(rm,domain)
 source=np.where(nv,1,np.where(ov,2,0)).astype(np.uint8);rawquality=np.where(source==1,nq,oq)&rq;oldquality=oq&rq;keep=oldquality&~rawquality;source=np.where(keep,2,source).astype(np.uint8)
 values=np.where((source==1)[...,None],nd,od);ok=(source>0)&rv;quality=np.where(source==1,nq,oq)&rq&ok;point=domain&om['valid']&r['available']
 return dict(domain=domain,path=path,phase_old=base,phase_new=new,span_first=first,span_last=last,point_ok=point,point_delta=np.where(point,abs(om['values']-r['values']),np.nan).astype(np.float32),
  sequence_old_ok=ov&rv,sequence_ok=ok,sequence_old_delta=np.where((ov&rv)[...,None],abs(od-rd),np.nan).astype(np.float32),sequence_delta=np.where(ok[...,None],abs(values-rd),np.nan).astype(np.float32),
  sequence_old_quality=oldquality,sequence_quality=quality,source=source,quality_fallback=keep,reference_missing=domain&~r['available'],reference_times_us=r['timestamp_us'],query_times_us=q['timestamp_us'],mean_path_evidence=np.array(evidence[tuple(path.T)].mean()))

def freeze():
 v=read(CROOT/'VERIFIED.json');assert v['status']=='PASS_ALL4500_CONSUMER_RECEIPTS';f=read(B/'FROZEN.json');check(f['protected']);prepared=read(BP/'PREPARED.json');check(prepared['output_pins']);rows=prepared['rows'];by=defaultdict(list)
 for r in rows:by[r['word_code']].append(r)
 assert len(rows)==300 and len(by)==100 and all(len(x)==3 for x in by.values()) and {x['signer_id'] for x in rows}=={11,12,13}
 pairs=[]
 for word,rr in sorted(by.items()):
  coords=[]
  for row in rr:
   z=load(row['output']);coords.append(hashlib.sha256(z['xy_px'].tobytes()+z['usable'].tobytes()).hexdigest())
  assert len(set(coords))==3,('duplicate public observation',word)
  for q in sorted(rr,key=lambda x:x['clip_id']):
   for r in sorted(rr,key=lambda x:x['clip_id']):
    if q['signer_id']!=r['signer_id']:pairs.append(dict(query=q['clip_id'],reference=r['clip_id'],word_code=word,label=q['label']))
 paths=[Path(__file__),CROOT/'consumer.py',CROOT/'VERIFIED.json',BP/'PREPARED.json',B/'study.py',Path(P.__file__),Path(D.__file__),Path(F.__file__)];pins={**prepared['output_pins'],**{str(p):sha(p) for p in paths}}
 cf=read(CROOT/'FROZEN.json');scores=cf['scores'];pins.update({p:sha(p) for p in scores.values()})
 write(U/'FROZEN.json',dict(at=now(),rows=rows,pairs=pairs,pins=pins,protected=f['protected'],scores=scores,
  authorization='Active autonomous common recognition/feedback goal. Research integration on already-developed public inputs, operating unchanged.',
  scope='100public words,3recordedIDs11/12/13,300originalDEV clips,all600directed different-ID same-word pairs. Each query retains2named references. No synthetic manipulation and no new detector/training/classifier inference. No assertion these IDs independently certify identity.',
  method='Existing fullframe→ROI originalC grouped features,flat missing1.5,band.2,frozen weights. Original pointwise plus bounded1frame fractional temporal descriptor with coordinate-availability and corroborated-quality fallback; no min-score reference selection. All original inputs/left-right kept. Saved C/T3F original masked predictions attach to query,never control alignment or feedback.',
  gates='Every reference domain retained regardless queryvisibility; no original sequence or quality lost; no coordinate/identity/operating mutations; exactly2named refs perquery; all600 source andresult arithmetic checked. No accuracy or improvement gate from lower interperson residual alone. Unknown timings across people,so no synthetic oracle MAE. Failed/empty evidence retained.',
  evidence='Reference annotation core x2x23 is an audit grid,not semantic requiredhand schedule. Existing per-angle corroboration preserved; geometry/shape kept separate. No normalthreshold estimation from only2references; no sign-correctness or correction-direction verdict.',
  comparison='Compare old/refined descriptive differences andcoverage bygroup,including increases. Keep reference-specific values,ranges,support,source andquery/reference times. No averaging hands/people into invented motion and no bestrefselection.',
  tests='Consumer14fixtures+4500validation reused.600new realdifferentID pairs independently check paths,selection,gaps andpublic ranks; inspect worst disagreement andlowcoverage using savedobservations. Not anatomical/semantic goldstandard.',training=0,user_data=0,operating_change=False))
 print('FROZEN_MULTIPEOPLE600',flush=True)
def run():
 f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()));(U/'pairs').mkdir();(U/'receipts').mkdir();(U/'queries').mkdir()
 samples={r['clip_id']:load(r['output']) for r in f['rows']};meta={r['clip_id']:r for r in f['rows']};scores={k:{x['clip_id']:{a:x[a] for a in ('status','prediction','top5','correct','top5_correct')} for x in read(p)['predictions']['masked']} for k,p in f['scores'].items()}
 rows=[];pins={};by=defaultdict(list);start=time.perf_counter()
 for i,m in enumerate(f['pairs']):
  qid,rid=m['query'],m['reference'];z=pair(samples[qid],samples[rid]);name=qid+'__'+rid;p=U/'pairs'/(name+'.npz');np.savez_compressed(p,**z);pins[str(p)]=sha(p)
  keys=['domain','point_ok','point_delta','sequence_ok','sequence_delta','sequence_quality','source','reference_missing','reference_times_us'];receipt=C.build(**{k:z[k] for k in keys},reference_id=rid,query_id=qid,kind='different_public_id',recognition={k:v[qid] for k,v in scores.items()})
  receipt['research_source']=dict(word_code=m['word_code'],label=m['label'],query_record_id=meta[qid]['signer_id'],reference_record_id=meta[rid]['signer_id'],reference_type='different_public_id_same_word',split='development_only',anatomical_truth=False,semantic_verdict=False)
  receipt['time_correspondence']=dict(reference_seconds=(z['reference_times_us']/1e6).tolist(),original_query_seconds=np.interp(z['phase_old'],np.arange(len(z['query_times_us'])),z['query_times_us']/1e6).tolist(),refined_query_seconds=np.interp(z['phase_new'],np.arange(len(z['query_times_us'])),z['query_times_us']/1e6).tolist(),meaning='Proposed alignment,not proof of hand visibility;per-channel source chooses original/refined,whole-pose timing may differ.')
  rp=U/'receipts'/(name+'.json');write(rp,receipt);pins[str(rp)]=sha(rp)
  stats=[]
  for h in (0,1):
   for group,a,b,unit in C.GROUPS:
    ok=z['sequence_old_ok'][:,h,a:b]&z['sequence_ok'][:,h,a:b];od=z['sequence_old_delta'][:,h,a:b];nd=z['sequence_delta'][:,h,a:b]
    stats.append(dict(side=h,group=group,unit=unit,old_supported=int(z['sequence_old_ok'][:,h,a:b].sum()),new_supported=int(z['sequence_ok'][:,h,a:b].sum()),old_quality=int(z['sequence_old_quality'][:,h,a:b].sum()),new_quality=int(z['sequence_quality'][:,h,a:b].sum()),common_cells=int(ok.sum()),old_mean=C.finite_mean(od[ok]),new_mean=C.finite_mean(nd[ok])))
  row=dict(name=name,**m,pair=str(p),receipt=str(rp),groups=stats,alignment_support=float(z['mean_path_evidence']),quality_fallback=int(z['quality_fallback'].sum()));rows.append(row);by[qid].append(row)
  if (i+1)%100==0:print('PAIRS',i+1,600,flush=True)
 for qid,refs in by.items():
  assert len(refs)==2;p=U/'queries'/(qid+'.json');write(p,dict(query_id=qid,word_code=meta[qid]['word_code'],label=meta[qid]['label'],recognition={k:v[qid] for k,v in scores.items()},references=refs,reference_aggregation='none;each preserved',whole_gesture_correct=None,normal_tolerance=None));pins[str(p)]=sha(p)
 assert len(rows)==600 and len(by)==300;check(f['pins']);check(f['protected']);write(U/'RESULT.json',dict(at=now(),status='COMPUTED_PENDING_SEPARATE_VERIFICATION',rows=rows,pins=pins,query_count=300,reference_count_per_query=2,seconds=time.perf_counter()-start,training=0,inference=0,operating_change=False));(U/'RUNNING.json').unlink();print('MULTIPEOPLE_COMPLETE600',flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=U/'FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
