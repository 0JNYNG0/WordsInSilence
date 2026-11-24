"""Independent saved-path arithmetic audit; no producer imports/DTW/forward."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib,json
import numpy as np

U=Path(__file__).resolve().parents[1]
F=['R_shape','L_shape','R_angles','L_angles','hand_relation','arm_path','hand_axis']
ST=['SIMILAR_REFERENCE_RANGE','DIFFERENCE_CANDIDATE','INSUFFICIENT_EVIDENCE','NOT_REQUIRED_REFERENCE_REST']
N={'same_expression':12,'different_target_expression':36,'speed_075':12,'speed_125':12,'main_R_hand_missing':12,'short_R_shape_axis':12}
checks=0;failures=[];categories=Counter()
def check(category,name,value):
 global checks
 checks+=1;categories[category]+=1
 if not bool(value):failures.append(category+':'+name)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def nz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def equal(x,y):
 if x is None or y is None:return x is y
 if isinstance(x,(list,tuple,np.ndarray)) or isinstance(y,(list,tuple,np.ndarray)):
  return np.asarray(x).shape==np.asarray(y).shape and np.allclose(x,y,atol=1e-12,rtol=1e-12,equal_nan=True)
 if isinstance(x,(bool,str)) or isinstance(y,(bool,str)):return x==y
 return bool(np.isclose(x,y,atol=1e-12,rtol=1e-12))

def channels(a,names):
 x=np.asarray(a['x'],float);m=np.asarray(a['mask'],bool)&np.isfinite(x);p=np.asarray(a['probabilities'],float)
 cols=lambda names_: [names.index(k) for k in names_]
 out={}
 for side,pc in [('R',[0,1,3]),('L',[2,4])]:
  angles=[j for j,name in enumerate(names) if name.startswith(side+'_') and '_angle' in name and 'elbow' not in name]
  wrist=cols([side+'_hand_wrist_x',side+'_hand_wrist_y'])
  support=m[:,wrist].all(1)&(m[:,angles].sum(1)>=10); bridged=support.copy()
  indices=np.flatnonzero(support)
  for left,right in zip(indices[:-1],indices[1:]):
   if a['timestamp_us'][right]/1e6-a['timestamp_us'][left]/1e6<=.2:bridged[left:right+1]=True
  pm=np.isfinite(p[:,pc])&bridged[:,None];pm[np.sum(p[:,pc]>=.5,axis=1)>1]=False
  out[side+'_shape']=(np.nan_to_num(p[:,pc]),pm)
  out[side+'_angles']=(x[:,angles],m[:,angles])
 for family,labels in [
  ('hand_relation',['hands_wrist_delta_x','hands_wrist_delta_y','hands_wrist_distance']),
  ('arm_path',[s+'_'+point+'_'+d for s in ['L','R'] for point in ['pose_wrist','elbow'] for d in ['x','y']]),
  ('hand_axis',[s+'_palm_axis_'+d for s in ['L','R'] for d in ['x','y']])]:
  ix=cols(labels);out[family]=(np.where(m[:,ix],x[:,ix],0),m[:,ix])
 return out

def scopes(row,ch):
 scope={f:np.ones_like(ch[f][1],bool) for f in F}
 for s in row['stages']:
  lo,hi=s['frames'];sides=s.get('confirmed_rest_sides',[])
  for side in sides:
   scope[side+'_shape'][lo:hi+1]=False;scope[side+'_angles'][lo:hi+1]=False
   scope['hand_axis'][lo:hi+1,slice(0,2) if side=='L' else slice(2,4)]=False
  if sides:scope['hand_relation'][lo:hi+1]=False
 return scope

def measure(qx,qm,rx,rm,path,lo,hi,shape):
 selected=path[(path[:,1]>=lo)&(path[:,1]<=hi)];qs,rs=selected.T
 need=rm[lo:hi+1];observed=np.zeros_like(need);errors=[];peak=None;peak_pair=None
 for j in range(lo,hi+1):
  q=qs[rs==j];shared=qm[q]&rm[j]; observed[j-lo]=shared.any(0)
  for dimension in np.flatnonzero(observed[j-lo]):
   values=np.abs(qx[q,dimension]-rx[j,dimension])[shared[:,dimension]]
   errors.append(float(values.max()))
  if shared.any():
   values=np.abs(qx[q]-rx[j]);values[~shared]=-np.inf
   v=float(values.max());flat=int(values.argmax());ii,dd=np.unravel_index(flat,values.shape)
   if peak is None or v>=peak:peak=v;peak_pair=(int(q[ii]),j,int(dd))
 uq=np.unique(qs);qneed=np.stack([rm[rs[qs==i]].any(0) for i in uq]);qobs=qneed&qm[uq]
 e=int(need.sum());o=int(observed.sum());qe=int(qneed.sum());qo=int(qobs.sum())
 dc=np.minimum(observed.sum(0)/np.maximum(need.sum(0),1),qobs.sum(0)/np.maximum(qneed.sum(0),1))
 coverage=(min(o/e,qo/qe) if e and qe else None)
 if shape:coverage=float(dc.min())
 return dict(expected=e,observed=o,query_expected=qe,query_observed=qo,
  reference_coverage=o/e if e else None,query_coverage=qo/qe if qe else None,coverage=coverage,
  dimension_reference_expected=need.sum(0).tolist(),dimension_reference_observed=observed.sum(0).tolist(),
  dimension_query_expected=qneed.sum(0).tolist(),dimension_query_observed=qobs.sum(0).tolist(),dimension_coverage=dc.tolist(),
  distance=float(np.quantile(errors,.9)) if errors else None,peak_distance=peak),peak_pair

def verify_pair(c,a,r,row,names,tag):
 p=np.asarray(c['path'],int);qi,ri=p.T;nq,nr=len(a['x']),len(r['x'])
 check('path',tag,np.array_equal(p[0],[0,0]) and np.array_equal(p[-1],[nq-1,nr-1])
  and set(qi)==set(range(nq)) and set(ri)==set(range(nr))
  and all(tuple(v) in [(1,0),(0,1),(1,1)] for v in np.diff(p,axis=0)))
 qc,rc=channels(a,names),channels(r,names);scope=scopes(row,rc)
 rt=r['timestamp_us']/1e6;qt=a['timestamp_us']/1e6
 check('time',tag,equal(c['query_frame_times'],qt) and equal(c['reference_frame_times'],rt))
 main=np.zeros(nr,bool)
 for lo,hi in row['intervals']:main|=(rt>=lo)&(rt<=hi)
 total=np.zeros(len(p));weight=np.zeros(len(p))
 for f in F:
  qx,qm=qc[f];rx,rm=rc[f];shared=qm[qi]&rm[ri]&scope[f][ri]
  d=np.sqrt(np.where(shared,(qx[qi]-rx[ri])**2,0).sum(1)/np.maximum(shared.sum(1),1));d[~shared.any(1)]=1
  w=(np.ones(len(p)) if f=='arm_path' else main[ri].astype(float))*scope[f][ri].any(1)
  total+=np.minimum(d,3)*w;weight+=w
 expected=float(np.mean((total/np.maximum(weight,1))[main[ri]]))
 check('path_cost',tag,equal(expected,c['distance']))
 check('stage_scope',tag,[s['canonical_stage_id'] for s in c['stages']]==[s['canonical_stage_id'] for s in row['stages']])
 for j,(stage,definition) in enumerate(zip(c['stages'],row['stages'])):
  lo,hi=definition['frames'];part=p[(ri>=lo)&(ri<=hi)];uq=np.unique(part[:,0])
  for f in F:
   qx,qm=qc[f];rx,rm=rc[f];v=stage['families'][f];keep=scope[f]
   want,peak=measure(qx,qm,rx,rm&keep,p,lo,hi,f.endswith('_shape'))
   original,_=measure(qx,qm,rx,rm,p,lo,hi,f.endswith('_shape'))
   for k,x in want.items():check('metric',tag+f+str(j)+k,equal(x,v[k]))
   expected_scope={'comparison_required':bool(keep[lo:hi+1].any()),'original_expected':original['expected'],
    'scope_excluded':original['expected']-want['expected'],'original_possible':(hi-lo+1)*rm.shape[1],
    'scoped_possible':int(keep[lo:hi+1].sum()),'original_query_expected':original['query_expected'],
    'scope_excluded_query_expected':original['query_expected']-want['query_expected'],
    'original_query_possible':len(uq)*rm.shape[1],
    'scoped_query_possible':sum(int(keep[part[part[:,0]==i,1]].any(0).sum()) for i in uq),
    'dimension_original_reference_expected':original['dimension_reference_expected'],
    'dimension_scope_excluded':(np.asarray(original['dimension_reference_expected'])-want['dimension_reference_expected']).tolist()}
   expected_scope['scope_excluded_possible']=expected_scope['original_possible']-expected_scope['scoped_possible']
   for k,x in expected_scope.items():check('scope',tag+f+str(j)+k,equal(x,v[k]))
   if not keep[lo:hi+1].all():
    for k,x in original.items():check('original_evidence',tag+f+str(j)+k,equal(x,v['original_evidence'][k]))
   if peak is None:check('peak',tag+f+str(j),v['peak_at'] is None)
   else:
    qindex,rindex,dim=peak;actual=v['peak_at']
    check('peak',tag+f+str(j),actual['query_frame']==qindex and actual['reference_frame']==rindex
      and actual['coordinate']==dim and actual['query_seconds']==qt[qindex] and actual['reference_seconds']==rt[rindex])

def classify(vs,limit):
 if all(not v['comparison_required'] for v in vs):return ST[3]
 if any(not v['comparison_required'] for v in vs):return ST[2]
 if not any(v['expected'] for v in vs) or limit['p90'] is None:return ST[2]
 if not all(v['coverage'] is not None and v['coverage']>=.65 and v['distance'] is not None for v in vs):return ST[2]
 fits=[v['distance']<=limit['p90'] and (limit['peak'] is None or v['peak_distance'] is not None and v['peak_distance']<=limit['peak']) for v in vs]
 return ST[0] if all(fits) else ST[1] if not any(fits) else ST[2]

def counts(stages):
 sc=Counter(s['status'] for s in stages);fc=Counter(f['status'] for s in stages for f in s['features'])
 return dict(stage_opportunities=len(stages),feature_opportunities=sum(len(s['features']) for s in stages),
             stages={k:sc[k] for k in ST},features={k:fc[k] for k in ST})

def run():
 outpath=U/'review/FEEDBACK_EVALUATION_VERIFICATION.json'
 if outpath.exists():raise FileExistsError(outpath)
 report=read(U/'FEEDBACK_EVALUATION.json');contract=read(U/'EVALUATION_FROZEN.json');bank=read(U/'REFERENCE_BANK.json')
 predictions=read(U/'BANK_PREDICTIONS.json');names=bank['feature_names'];rows={r['clip_id']:r for r in bank['rows']}
 dev={r['clip_id']:r for r in predictions['rows'] if r['role']=='development'}
 source_pins={str(U/n):sha(U/n) for n in ['FEEDBACK_EVALUATION.json','EVALUATION_FROZEN.json','REFERENCE_BANK.json',
  'BANK_PREDICTIONS.json','NORMAL_RANGES.json','THREE_REFERENCE_RANGES.json','evaluate_feedback.py','engine.py']}
 source_pins[str(Path(__file__))]=sha(__file__)
 for p,h in contract['pins'].items():check('pin',p,sha(p)==h)
 arrays={}
 for row in list(rows.values())+list(dev.values()):
  a=nz(row['input']);p=nz(row['probability_file']);a['probabilities']=p['probabilities'];arrays[row['clip_id']]=a
  check('input_axes',row['clip_id'],np.array_equal(a['frame_id'],p['frame_id']) and np.array_equal(a['timestamp_us'],p['timestamp_us']))
 normal=read(U/'NORMAL_RANGES.json');subset=read(U/'THREE_REFERENCE_RANGES.json')
 cal={}
 for item in normal['pair_index']:
  check('calibration_pin',item['path'],sha(item['path'])==item['sha256']);c=read(item['path'])
  check('calibration_split',item['path'],rows[item['query']]['person']!=rows[item['reference']]['person']
   and rows[item['query']]['person']<=9 and rows[item['reference']]['person']<=9)
  cal[(item['query'],item['reference'])]=c
 check('calibration_scope','288 unique FIT pairs',len(cal)==288)
 for mode,cutoff,people in [('all9',normal,range(1,10)),('subset3',subset,range(1,4))]:
  samples={f:[] for f in F};counter=Counter();nqueries=0
  for q,row in rows.items():
   if row['person'] not in people:continue
   nqueries+=1
   candidates=[c for (qid,rid),c in cal.items() if qid==q and rows[rid]['person'] in people]
   selected=sorted(candidates,key=lambda c:c['distance'])[:2]
   check('calibration_selection',mode+q,len(candidates)==len(people)-1 and len(selected)==2)
   for c in selected:
    for stage in c['stages']:
     for f,v in stage['families'].items():
      counter[(f,'possible')]+=1
      usable=v['comparison_required'] and v['coverage'] is not None and v['coverage']>=.65 and v['distance'] is not None and v['peak_distance'] is not None
      counter[(f,'supported' if usable else 'not_required_reference_rest' if not v['comparison_required'] else 'insufficient')]+=1
      for a,b in [('expected','expected_cells'),('observed','observed_cells'),('query_expected','query_expected_cells'),('query_observed','query_observed_cells')]:counter[(f,b)]+=v[a]
      if usable:samples[f].append([v['distance'],v['peak_distance']])
  for f in F:
   vv=np.asarray(samples[f]).reshape(-1,2);want=[float(np.quantile(vv[:,i],.95)) if len(vv) else None for i in range(2)]
   check('cutoff',mode+f,equal(want[0],cutoff['ranges'][f]['p90']) and equal(want[1],cutoff['ranges'][f]['peak']))
   for key,val in cutoff['counts'][f].items():check('cutoff_counts',mode+f+key,counter[(f,key)]==val)
 tasks=read(U/'EVALUATION_TASKS.json')['tasks'];indexed={c['id']:c for c in report['case_index']}
 check('scope','96 exact kinds',len(tasks)==96 and Counter(t['kind'] for t in tasks)==N and set(indexed)=={t['id'] for t in tasks})
 aggregate=defaultdict(lambda:dict(cases=0,no_effective_edit=0,modes={m:dict(stage_opportunities=0,feature_opportunities=0,stages=Counter(),features=Counter()) for m in ['all9','subset3']}))
 pair_count=stage_count=0;native={};controls=Counter()
 for task in tasks:
  entry=indexed[task['id']];check('case_pin',task['id'],sha(entry['path'])==entry['sha256']);case=read(entry['path'])
  check('task_identity',task['id'],case['task']==task)
  original=arrays[task['query']];a=original;edit=case['edit_ledger'];kind=task['kind'];control=case['modified_input']
  if control:
   check('control_pin',task['id'],sha(control['path'])==control['sha256']);a=nz(control['path'])
  mapping=np.asarray(edit['source_indices']);check('control_map',task['id'],len(mapping)==len(a['x']) and np.all(mapping>=0) and np.all(mapping<len(original['x'])))
  if kind.startswith('speed_'):
   want=np.round(np.linspace(0,len(original['x'])-1,round(len(original['x'])/task['speed_factor']))).astype(int)
   check('speed',task['id'],np.array_equal(want,mapping))
   for k in ['x','mask','rgb','probabilities']:check('speed_values',task['id']+k,np.array_equal(a[k],original[k][mapping],equal_nan=True))
   check('speed_time',task['id'],np.array_equal(a['frame_id'],np.arange(len(mapping))) and np.array_equal(a['timestamp_us'],original['timestamp_us'][0]+np.arange(len(mapping))*int(np.median(np.diff(original['timestamp_us'])))))
  elif kind=='main_R_hand_missing':
   x=original['mask'].copy();p=original['probabilities'].copy();ix=task['main_indices'];cols=task['masked_feature_columns'];pi=task['masked_posterior_columns']
   x[np.ix_(ix,cols)]=False;p[np.ix_(ix,pi)]=np.nan
   check('missing',task['id'],np.array_equal(a['mask'],x) and np.array_equal(a['probabilities'],p,equal_nan=True) and np.array_equal(a['x'],original['x'],equal_nan=True))
   check('missing_ledger',task['id'],edit['newly_removed_feature_cells']==int(original['mask'][np.ix_(ix,cols)].sum()) and edit['newly_removed_posterior_cells']==int(np.isfinite(original['probabilities'][np.ix_(ix,pi)]).sum()))
   controls['missing_removed_geometry']+=edit['newly_removed_feature_cells'];controls['missing_removed_posterior']+=edit['newly_removed_posterior_cells']
  elif kind=='short_R_shape_axis':
   ix=task['short_indices'];pi=[0,1,3];axis=[names.index('R_palm_axis_'+d) for d in ['x','y']]
   p=original['probabilities'].copy();x=original['x'].copy();v=p[np.ix_(ix,pi)];ok=np.isfinite(v);v[ok]=1-v[ok];p[np.ix_(ix,pi)]=v
   for i in ix:
    if original['mask'][i,axis].all() and np.isfinite(x[i,axis]).all():x[i,axis]=[-x[i,axis[1]],x[i,axis[0]]]
   check('short',task['id'],np.array_equal(a['x'],x,equal_nan=True) and np.array_equal(a['probabilities'],p,equal_nan=True) and np.array_equal(a['mask'],original['mask']))
   sc=int((np.isfinite(p[:,pi]) & (p[:,pi]!=original['probabilities'][:,pi])).sum())
   ac=int(((x[:,axis]!=original['x'][:,axis])&original['mask'][:,axis]&np.isfinite(x[:,axis])).sum())
   check('short_ledger',task['id'],edit['finite_shape_cells_actually_changed']==sc and edit['supported_axis_cells_actually_changed']==ac and edit['no_effective_edit']==(sc+ac==0))
   controls['short_actual_shape_cells']+=edit['finite_shape_cells_actually_changed'];controls['short_actual_axis_cells']+=edit['supported_axis_cells_actually_changed'];controls['short_no_effect']+=edit['no_effective_edit']
  comparisons=[]
  for item in case['comparisons']:
   check('pair_pin',task['id']+item['reference'],sha(item['path'])==item['sha256']);c=read(item['path']);rid=item['reference']
   check('pair_identity',task['id']+rid,c['query_id']==task['query'] and c['id']==rid and c['task_id']==task['id'] and c['kind']==kind)
   verify_pair(c,a,arrays[rid],rows[rid],names,task['id']+rid);pair_count+=1;stage_count+=len(c['stages']);comparisons.append(c)
   if kind=='same_expression':native[(task['query'],rid)]=c
   elif control:
    b=native[(task['query'],rid)]
    for s,t in zip(c['stages'],b['stages']):
     for f in F:
      for k in ['expected','dimension_reference_expected','comparison_required','original_expected','original_possible','scoped_possible','scope_excluded','scope_excluded_possible','dimension_original_reference_expected','dimension_scope_excluded']:
       check('control_denominator',task['id']+rid+f+k,equal(s['families'][f][k],t['families'][f][k]))
  check('case_scope',task['id'],len(comparisons)==9 and {c['id'] for c in comparisons}==set(task['references']))
  g=aggregate[kind];g['cases']+=1;g['no_effective_edit']+=case['edit_ledger'].get('no_effective_edit',False)
  for mode,limits in [('all9',normal['ranges']),('subset3',subset['ranges'])]:
   selected=sorted([c for c in comparisons if mode=='all9' or c['id'] in task['subset_references']],key=lambda c:c['distance'])[:2]
   saved=case['modes'][mode];check('selection',task['id']+mode,[c['id'] for c in selected]==saved['selected_whole_references'])
   for j,stage in enumerate(saved['stages']):
    statuses=[]
    for f,feature in zip(F,stage['features']):
     vs=[c['stages'][j]['families'][f] for c in selected];status=classify(vs,limits[f]);statuses.append(status)
     check('classification',task['id']+mode+f+str(j),feature['status']==status and all(equal(x,v['distance']) for x,v in zip(feature['distances'],vs)))
    want=ST[1] if ST[1] in statuses else ST[2] if ST[2] in statuses else ST[0]
    check('stage_status',task['id']+mode+str(j),stage['status']==want)
   count=counts(saved['stages']);check('counts',task['id']+mode,count==saved['counts']==entry['counts'][mode])
   for k in ['stage_opportunities','feature_opportunities']:g['modes'][mode][k]+=count[k]
   for k in ['stages','features']:g['modes'][mode][k].update(count[k])
  if len(indexed)%12==0 and task['id'].endswith('2'):print(task['id'],pair_count,flush=True)
 for kind,want in aggregate.items():
  saved=report['per_kind'][kind];check('summary',kind,want['cases']==saved['cases'] and want['no_effective_edit']==saved['no_effective_edit'] and saved['errors']==0)
  for mode,v in want['modes'].items():
   check('summary',kind+mode,dict(v,stages=dict(v['stages']),features=dict(v['features']))==saved['modes'][mode])
 check('completion','all96/864/2808',len(indexed)==96 and pair_count==report['new_pair_paths']==864 and stage_count==report['pair_stage_opportunities']==2808)
 for mode in ['all9','subset3']:
  check('completion',mode,sum(g['modes'][mode]['stage_opportunities'] for g in aggregate.values())==312 and sum(g['modes'][mode]['feature_opportunities'] for g in aggregate.values())==2184)
 check('completion','zero pipeline errors',report['errors']==0 and report['subset_new_paths']==0 and report['observer_forward']==0 and report['RGB_decode']==0 and report['training']==0 and not (U/'EVALUATION_FIRST_FAILURE.json').exists())
 for p,h in contract['pins'].items():check('end_pin',p,sha(p)==h)
 out=dict(status='PASS_SAVED_ARITHMETIC_WITH_LIMITS' if not failures else 'FAIL',checks=checks,failure_count=len(failures),failures=failures[:100],categories=dict(categories),source_pins=source_pins,
  evaluated_cases=96,saved_pair_paths_checked=pair_count,pair_stage_opportunities=stage_count,aggregate_stages_per_mode=312,aggregate_features_per_mode=2184,
  independent_arithmetic='Saved-path costs, masks/coverage, original/scoped/excluded denominators, residual90th percentile/peak source links, reference selection, pooled95th percentile cutoffs and all status/count aggregates. No path optimization.',
  control_counts=dict(controls),per_kind=report['per_kind'],
  interpretation_limits=['Same-expression warnings are proxies, not verified educational errors.','Different-target contrasts have no per-feature linguistic error truth.','Speed controls resample stored features/posteriors and test downstream behavior; they do not re-extract or test neural invariance to real camera-speed changes.','Short posterior inversion can create mutually-exclusive positive conflicts and valid abstention; this is not an important-shape-error detection rate.','Both3/9-bank panels use the same selected observer; fewer reference people also changes FIT calibration samples, so the difference is not a pure bank-size causal effect.','Reference-rest exclusions preserve original evidence and are empirical posture scope, not proof of linguistic irrelevance.'],
  producer_imports=0,new_DTWs=0,new_model_forwards=0,new_video_decodes=0)
 with outpath.open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps(dict(status=out['status'],checks=checks,failures=len(failures),path=str(outpath),sha256=sha(outpath))))
 if failures:raise AssertionError(failures[:20])

if __name__=='__main__':run()
