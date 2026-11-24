"""Post-output fixed-scope correspondence audit. Never imported by aligner."""
from pathlib import Path
import sys,json,hashlib,statistics,collections
sys.dont_write_bytecode=True
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent
PREV=OPS/'word100_target_known_feedback_20261101'
PHYS=OPS/'word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def phase_audit(q,ref,path_by_parent):
 lookup={i:None for i in range(q['native_frame_count'])};boundary=set()
 for r in q['regions']:
  if r['region_type']=='observed_physical_core':
   for i in range(r['core'][0],r['core'][1]+1):lookup[i]=r['kind']
  else:boundary.update(range(r['outer'][0],r['outer'][1]+1))
 events=[];states=collections.Counter()
 for event in ref['regions']:
  if event['region_type']!='observed_physical_core':continue
  path=path_by_parent.get(event['source_segment_id'],path_by_parent.get('*',[]));by=collections.defaultdict(list)
  for qi,ri in path:
   if event['core'][0]<=ri<=event['core'][1]:by[ri].append(qi)
  counter=collections.Counter();trace=[]
  for ri in range(event['core'][0],event['core'][1]+1):
   if not by[ri]:counter['unmapped']+=1;trace.append([ri,None,'unmapped',1]);continue
   for qi in by[ri]:
    kind=lookup[qi];label=('same_observed_kind' if kind==event['kind'] else 'different_observed_kind') if kind is not None else 'boundary_or_uncertain' if qi in boundary else 'unreviewed_or_uncertain'
    w=1/len(by[ri]);counter[label]+=w;trace.append([ri,qi,label,w])
  expected=event['core'][1]-event['core'][0]+1;assert np.isclose(sum(counter.values()),expected)
  states.update(counter);events.append(dict(event=event['id'],parent=event['source_segment_id'],reference_kind=event['kind'],reference_core=event['core'],expected=expected,weighted_states=dict(counter),trace=trace))
 return dict(reference_full_frames=ref['native_frame_count'],expected_certain_frames=sum(e['expected'] for e in events),states=dict(states),events=events,query_labels_only_for_audit=True)
def path_for_file(d,method):
 if method=='core_first':return {a['segment_id']:a['selected']['core_path'] if a.get('selected') else [] for a in d['anchors']}
 return {'*':d['path']}
def control_audit(d,task):
 path=d['path'];n=d['query_frames'];m=d['reference_frames']
 indices=np.asarray(task['metadata'].get('original_indices',list(range(n))),int);by=collections.defaultdict(list)
 for qi,ri in path:by[ri].append(qi)
 missing=set(range(m))-set(indices.tolist());error=exact=present_error=present_den=0.;missing_errors=[]
 for ri in range(m):
  if not by[ri]:continue
  values=[abs(int(indices[qi])-ri) for qi in by[ri]];v=float(np.mean(values));error+=v;exact+=sum(x==0 for x in values)/len(values)
  if ri not in missing:present_error+=v;present_den+=1
  else:missing_errors.append(v)
 if path:
  assert {p[0] for p in path}==set(range(n)) and {p[1] for p in path}==set(range(m))
 out=dict(reference_frames=m,query_frames=n,expected_reference_frames=m,present_original_ids=len(set(indices.tolist())),deleted_original_ids=sorted(missing),
   mean_original_frame_error=error/m if path else None,exact_original_frame_fraction=exact/m if path else 0.,present_original_frame_error=present_error/present_den if present_den else None,
   deleted_frames_mean_error=float(np.mean(missing_errors)) if missing_errors else None,full_path=bool(path),query_scope_retained=n if path else 0,
   forced_monotonic_order=True,actual_erroneous_sign_video=False)
 changed=task['metadata'].get('changed_frames',[])
 if changed:
  hits=[sum(int(indices[qi])==ri for qi in by[ri])/len(by[ri]) if by[ri] else 0 for ri in changed]
  out.update(changed_frames=changed,changed_original_identity_retention=float(np.mean(hits)),changed_cells=task['metadata']['actual_observed_changed_cells'])
 return out
def main():
 tasks=read(U/'TASKS.json');physical={d['alias']:d for d in read(PHYS)['clips']};rows=[];controls=[]
 # Existing paths are read on this shared audit scope, never rerun or overwritten.
 for p in tasks['pairs']:
  for method,filename in p['baselines'].items():
   d=read(filename);a=phase_audit(physical[p['query']],physical[p['reference']],path_for_file(d,method))
   rows.append(dict(method=method,seed=None,query=p['query'],reference=p['reference'],role=p['role'],word_id=p['word_id'],source=filename,source_sha256=sha(filename),audit=a,seconds=None))
 for c in tasks['controls']:
  b=read(c['baseline']);baseline_task=c
  if c['control']=='small_common_coordinate_drift':continue
  ca=control_audit(b,baseline_task);controls.append(dict(method='saved_relational_global',seed=None,alias=c['alias'],role=c['role'],control=c['control'],name=c['name'],audit=ca))
 manifest=read(U/'RESULT_MANIFEST.json')
 for rec in manifest['results']:
  assert sha(rec['path'])==rec['sha256'];d=read(rec['path']);p=d['task']
  if d['kind']=='pair':
   a=phase_audit(physical[p['query']],physical[p['reference']],{'*':d['path']})
   full_scope=({x[0] for x in d['path']}==set(range(d['query_frames'])) and {x[1] for x in d['path']}==set(range(d['reference_frames'])))
   rows.append(dict(method='joint',seed=d['seed'],query=p['query'],reference=p['reference'],role=p['role'],word_id=p['word_id'],source=rec['path'],source_sha256=rec['sha256'],audit=a,seconds=d['elapsed_seconds'],full_scope=full_scope,ambiguous=d['optimal_path_ambiguous'],family_costs=d['family_costs'],group_support=d['group_support']))
  else:
   controls.append(dict(method='joint',seed=d['seed'],alias=p['alias'],role=p['role'],control=p['control'],name=p['name'],audit=control_audit(d,p),ambiguous=d['optimal_path_ambiguous'],family_costs=d['family_costs'],group_support=d['group_support'],seconds=d['elapsed_seconds']))
 summary={};byword={}
 for row in rows:
  for target,key in [(summary,f"{row['role']}/{row['method']}/{row['seed']}"),(byword,f"{row['role']}/{row['word_id']}/{row['method']}/{row['seed']}")]:
   a=target.setdefault(key,dict(pairs=0,full_reference_frames=0,certain_reference_frames=0,states=collections.Counter(),ambiguous_paths=0))
   a['pairs']+=1;a['full_reference_frames']+=row['audit']['reference_full_frames'];a['certain_reference_frames']+=row['audit']['expected_certain_frames'];a['states'].update(row['audit']['states']);a['ambiguous_paths']+=bool(row.get('ambiguous',False))
 for g in (summary,byword):
  for row in g.values():row['same_kind_percent']=100*row['states']['same_observed_kind']/row['certain_reference_frames'];row['states']=dict(row['states'])
 control_summary={}
 for c in controls:
  key=f"{c['control']}/{c['method']}/{c['seed']}";a=control_summary.setdefault(key,dict(cases=0,errors=[],exact=[],fullpaths=0,changed_identity=[],deleted_expected=0))
  a['cases']+=1;a['errors'].append(c['audit']['mean_original_frame_error']);a['exact'].append(c['audit']['exact_original_frame_fraction']);a['fullpaths']+=c['audit']['full_path'];a['deleted_expected']+=len(c['audit']['deleted_original_ids'])
  if 'changed_original_identity_retention' in c['audit']:a['changed_identity'].append(c['audit']['changed_original_identity_retention'])
 for a in control_summary.values():
  a['mean_original_frame_error']=statistics.mean(x for x in a['errors'] if x is not None) if any(x is not None for x in a['errors']) else None
  a['exact_original_frame_percent']=statistics.mean(a['exact'])*100
 # Prospective fixed gate, no choosing winner seed/method or removing a word.
 scores=[summary[f'development/joint/{s}']['same_kind_percent'] for s in (42,43,44)]
 b=[summary[f'development/{m}/None']['same_kind_percent'] for m in ('global_dtw','relational_global')]
 gates=dict(mean_gain_at_least5pp_vs_both=all(statistics.mean(scores)>=v+5 for v in b),every_seed_above_both=all(v>max(b) for v in scores))
 regress=[]
 for word in sorted({p['word_id'] for p in tasks['pairs']}):
  base=max(byword[f'development/{word}/{m}/None']['same_kind_percent'] for m in ('global_dtw','relational_global'))
  for s in (42,43,44):
   value=byword[f'development/{word}/joint/{s}']['same_kind_percent'];regress.append(dict(word=word,seed=s,pp=value-base))
 gates['no_word_regression_over5pp']=all(x['pp']>=-5 for x in regress)
 speed=[]
 for case in ('speed_075','speed_125'):
  base=control_summary[f'{case}/saved_relational_global/None']['mean_original_frame_error']
  for s in (42,43,44):speed.append(dict(case=case,seed=s,increase=control_summary[f'{case}/joint/{s}']['mean_original_frame_error']-base))
 gates['normal_speed_error_increase_at_most1frame']=all(x['increase']<=1 for x in speed)
 gates['all_source_scope_preserved']=all(c['audit']['full_path'] for c in controls) and all(r['full_scope'] for r in rows if r['method']=='joint')
 gates['identity_error_zero']=all(control_summary[f'identity/joint/{s}']['mean_original_frame_error']<1e-8 for s in (42,43,44))
 out=dict(status='FROZEN_JOINT_ALIGNMENT_EVALUATED',summary=summary,byword=byword,control_summary=control_summary,
  decision=dict(clear_development_improvement=all(gates.values()),gates=gates,dev_scores=scores,baselines=b,word_regressions=regress,speed_changes=speed,adoption=False,educational_accuracy=None),
  pairs=rows,controls=controls,new_training=0,new_model_inference=0,query_annotations_runtime=False,
  limits=['Historical4word development only','Certain same-kind is not repeated occurrence/educational correctness','Controls modify stored features, not newly observed videos','Both-negative handshape not actual matching shape','No threshold/ref/seed tuning after results'])
 new(U/'EVALUATION.json',out)
 print(json.dumps({'decision':out['decision'],'summary':summary},ensure_ascii=False))
if __name__=='__main__':main()
