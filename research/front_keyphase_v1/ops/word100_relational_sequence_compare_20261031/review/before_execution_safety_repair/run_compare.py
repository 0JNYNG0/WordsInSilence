from pathlib import Path
from collections import Counter
import json,hashlib,time,sys,copy,numpy as np
from relational import GROUPS,features,compare,free_order
U=Path(__file__).resolve().parent;OPS=U.parent;OLD=OPS/'word100_core_stage_evidence_20261031';EP=OPS/'word100_semantic_episode_pilot_20261031'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def prepare():
 rows=read(U/'AUTHORIZATION.json')['rows'];ph=read(OLD/'PHASE_DATA.json');clips={c['alias']:c for c in ph['clips']};arrays={}
 for r in rows:
  assert sha(r['input'])==r['input_sha256'];arrays[r['alias']]={k:v.copy() for k,v in np.load(r['input'],allow_pickle=False).items() if k in ['x','mask','timestamp_us','frame_id']}
 return rows,clips,arrays
def freeze():
 rows,clips,arrays=prepare()
 pins={str(p.resolve()):sha(p) for p in [U/'AUTHORIZATION.json',U/'relational.py',U/'run_compare.py',U/'SOURCE_FEATURE_CONTRACT.json',U/'review/CONTRACT_REVIEW.json',U/'review/CONTRACT_REVIEW_ADDENDUM.json',OLD/'PHASE_DATA.json',OLD/'SOURCE_PHASE_CONTRACTS.json',EP/'EMPIRICAL_RANGE.json']}
 write(U/'FROZEN.json',dict(status='FROZEN_BEFORE_NEW_RESULTS',groups=GROUPS,weights='equal 1/13 by group, equal features within group',mask_cost=1,reference_unavailable_alignment_cost=.5,DTW_normalized_band=.25,range='per individual reference/core/group, maximum observed mean difference of the two other fit persons; requires both sufficient including 2 distinct observed query frames; no DEV calibration',range_epsilon=1e-6,free_order='five original frames, unconstrained search, ties 1e-9; no semantic validity claim; ambiguous/reverse never direct error',controls=['identity','speed_075','speed_125','short_shape_change_5f','left_short_shape_change_5f','omit_short_core_5f','right_hand_unobserved','swap_adjacent_core_blocks'],model_training=False,test_claim='feature-space probes, not actual incorrect KSL or reshot speed/video rendering; speed_125 drops samples so cannot guarantee short actions preserved; no interpolation across missing masks',candidate_adoption='diagnostic prototype only; no operational replacement regardless of result',pins=pins))
 print('FROZEN',flush=True)
def check():
 f=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in f['pins'].items()),'frozen hash mismatch'
def pair_run(a,b,kind,data,clips,path=None):
 p=U/'pairs'/f'{kind}_{a}_{b}.json';p.parent.mkdir(exist_ok=True)
 if p.exists():return read(p)
 st=time.perf_counter();result=compare(data[a],data[b],clips[b]['segments'],path);result.update(query=a,reference=b,condition=kind,seconds=time.perf_counter()-st);write(p,result);return result
def cal_ranges(results):
 groups={}
 for d in results:
  for s in d['stages']:
   for g,v in s['groups'].items():groups.setdefault((d['reference'],s['segment_id'],g),[]).append(v)
 out=[]
 for (ref,seg,g),vals in groups.items():
  eligible=len(vals)==2 and all(v['sufficient'] and v['mean_difference'] is not None for v in vals)
  out.append(dict(reference=ref,segment_id=seg,group=g,fit_pairs=len(vals),supported=eligible,max_difference=max(v['mean_difference'] for v in vals) if eligible else None))
 return out
def decisions(d,ranges):
 lookup={(r['reference'],r['segment_id'],r['group']):r for r in ranges};out=[]
 for s in d['stages']:
  for g,v in s['groups'].items():
   cal=lookup.get((d['reference'],s['segment_id'],g));threshold=cal['max_difference'] if cal else None
   state='reference_range_unavailable' if threshold is None else 'observation_insufficient' if not v['sufficient'] else 'larger_than_fit_reference_range' if v['mean_difference']>threshold+1e-6 else 'within_small_reference_range'
   out.append(dict(segment_id=s['segment_id'],reference_core=s['reference_core'],query_frames=s['query_frames'],group=g,state=state,mean_difference=v['mean_difference'],peak_difference=v['peak_difference'],range_max=threshold,coverage=v['coverage'],path_missing_cells=v['path_missing_cells'],not_educational_verdict=True))
 return out
def run():
 check();rows,clips,arrays=prepare();data={a:features(v) for a,v in arrays.items()};emp=read(EP/'EMPIRICAL_RANGE.json');fit=[];dev=[];saved={}
 for w in emp['calibration']:
  for q in w['lopo']:
   for p in q['pairs']:saved[(q['query'],p['reference'])]=p
 for q in emp['development']:
  for p in q['pairs']:saved[(q['alias'],p['reference'])]=p
 for a in rows:
  for b in rows:
   if a['alias']==b['alias'] or b['role']!='fit' or a['word_id']!=b['word_id']:continue
   aa,bb=a['alias'],b['alias'];path=np.asarray(saved[(aa,bb)]['path_original_frames'],int)
   for kind,pp in [('B_same_path',path),('C_relational_path',None)]:
    d=pair_run(aa,bb,kind,data,clips,pp);(fit if a['role']=='fit' else dev).append(d)
   print(aa,bb,'done',flush=True)
 range_by={kind:cal_ranges([d for d in fit if d['condition']==kind]) for kind in ['B_same_path','C_relational_path']}
 write(U/'REFERENCE_RANGES.json',dict(ranges=range_by,calibration_pairs_per_condition=24,normality_claim=False))
 results=[]
 for d in dev:
  dec=decisions(d,range_by[d['condition']]);baseline=saved[(d['query'],d['reference'])]
  results.append(dict(query=d['query'],reference=d['reference'],condition=d['condition'],decisions=dec,counts=dict(Counter(x['state'] for x in dec)),seconds=d['seconds'],old_saved_mean=baseline['distance'],query_frames=d['query_frames'],max_query_reuse=d['max_query_reuse'],max_reference_reuse=d['max_reference_reuse'],free_order=d['free_order']))
 write(U/'DEVELOPMENT.json',dict(status='COMPLETE_DIAGNOSTIC',results=results,all_references_retained=True,word_recognition_accuracy_not_measured=True))
 print('comparison complete',flush=True)
def controls():
 check();rows,clips,arrays=prepare();out=[];folder=U/'controls';folder.mkdir(exist_ok=True)
 for row in rows:
  a=row['alias'];base=arrays[a];n=len(base['x']);r=features(base);cores=[s for s in clips[a]['segments'] if s['role']=='core_candidate'];base_ids=np.arange(n)
  first=cores[0]['core'];center=(first[0]+first[1])//2;changed=list(range(max(0,center-2),min(n,center+3)))
  variants={}
  for name,speed in [('identity',1),('speed_075',.75),('speed_125',1.25)]:
   ids=np.round(np.linspace(0,n-1,round(n/speed))).astype(int);v={k:base[k][ids].copy() for k in base};v['frame_id']=np.arange(len(ids));v['timestamp_us']=np.arange(len(ids))*33333;variants[name]=(v,dict(original_indices=ids.tolist(),kind='resampled saved features; not MediaPipe re-extraction',kept_original_ids=len(set(ids)),original_frame_count=n))
  for cname,cols in [('short_shape_change_5f',list(range(39,53))),('left_short_shape_change_5f',list(range(11,25)))]:
   v={k:x.copy() for k,x in base.items()};v['x'][np.ix_(changed,cols)]=np.where(v['x'][np.ix_(changed,cols)]<.5,1.,0.);variants[cname]=(v,dict(changed_frames=changed,actual_observed_changed_cells=int(v['mask'][np.ix_(changed,cols)].sum()),kind='synthetic projected-angle perturbation only'))
  ids=np.setdiff1d(base_ids,changed);v={k:base[k][ids].copy() for k in base};v['frame_id']=np.arange(len(ids));v['timestamp_us']=np.arange(len(ids))*33333;variants['omit_short_core_5f']=(v,dict(omitted_original_frames=changed,original_indices=ids.tolist(),kind='known five-frame omission; not proof that those frames are linguistically mandatory'))
  v={k:x.copy() for k,x in base.items()};cols=[28,29,34,35,36,37]+list(range(39,59))+[60,65,66,67,68];v['mask'][:,cols]=False;variants['right_hand_unobserved']=(v,dict(kind='missing actual right hand; pose wrist/arm estimates remain separately available'))
  if len(cores)>=2:
   s0,e0=cores[0]['core'];s1,e1=cores[1]['core'];ids=np.r_[base_ids[:s0],base_ids[s1:e1+1],base_ids[e0+1:s1],base_ids[s0:e0+1],base_ids[e1+1:]];assert len(ids)==n and len(set(ids))==n
   v={k:base[k][ids].copy() for k in base};v['frame_id']=base['frame_id'].copy();v['timestamp_us']=base['timestamp_us'].copy();variants['swap_adjacent_core_blocks']=(v,dict(original_indices=ids.tolist(),swapped_cores=[cores[0]['segment_id'],cores[1]['segment_id']],kind='known feature-order edit, not a real erroneous performance'))
  for name,(v,meta) in variants.items():
   p=folder/f'{a}_{name}.json'
   if p.exists():
    assert (folder/f'{a}_{name}.npz').exists(),'Partial control output preserved; inspect before recovery'
    out.append(read(p));continue
   d=compare(features(v),r,clips[a]['segments']);d.update(alias=a,control=name,metadata=meta);write(p,d);np.savez_compressed(folder/f'{a}_{name}.npz',**v);out.append(d)
  print('controls',a,flush=True)
 summary=[]
 for name in sorted(set(x['control'] for x in out)):
  ds=[d for d in out if d['control']==name]
  summary.append(dict(control=name,cases=len(ds),positive_shape_peak=sum((d['groups']['R_shape']['peak_difference'] or 0)>1e-6 for d in ds),shape_insufficient=sum(not d['groups']['R_shape']['sufficient'] for d in ds),order_reverse=sum(any(x['state']=='reverse' for x in d['free_order']['relations']) for d in ds),order_ambiguous=sum(any(x['state'] in ['ambiguous','insufficient'] for x in d['free_order']['relations']) for d in ds),scope_retained=all(d['query_scope_retained']==d['query_frames'] and d['reference_scope_retained']==d['reference_frames'] for d in ds)))
 write(U/'CONTROL_SUMMARY.json',dict(status='COMPLETE_NOT_REAL_KSL_ERROR_TEST',summary=summary,rows=[dict(alias=d['alias'],control=d['control'],file=str(folder/f"{d['alias']}_{d['control']}.json")) for d in out]))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':{'freeze':freeze,'run':run,'controls':controls}[sys.argv[1]]()
