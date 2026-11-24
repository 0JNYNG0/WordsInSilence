"""Research-only structured feature/sequence comparison. No educational verdicts."""
from learning import *
import copy
LAB=OPS.parent
sys.path.insert(0,str(LAB/'src'))
from wis_lab.core.observation import Observation
from wis_lab.core import features as F
from wis_lab.core.dtw import frame_cost_matrix,_optimal_path

CONDITIONS=['legacy_dtw','pose_capacity_matched','pose_plus_signrep']
def roi_observation(path,box):
 o=Observation.load(str(path));xy=o.xy_px;keep=(xy[:,:,0]>=box[0])&(xy[:,:,0]<box[2])&(xy[:,:,1]>=box[1])&(xy[:,:,1]<box[3]);o.usable&=keep;o.observed&=keep;o.xy_px[~o.observed]=np.nan;o.conf[~o.observed]=np.nan
 for si,off in enumerate([33,54]):o.hand_score[~o.observed[:,off:off+21].any(1),si]=np.nan
 o.validate();return o
def estimate_support(r,arrays,targets):
 # Usability is an explicit detector-evidence proxy, never image-confirmed truth.
 a=arrays;names=r['feature_names'];mask=a['mask'];out=np.zeros((len(mask),len(targets)),bool)
 for j,f in enumerate(targets):
  sides=['R','L'] if f=='hands_overlap' else [f[0]];ok=np.ones(len(mask),bool)
  for s in sides:
   wrist=mask[:,names.index(s+'_hand_wrist_x')]&mask[:,names.index(s+'_hand_wrist_y')]
   if '_motion_' in f:
    cols=[names.index(s+'_wrist_velocity_'+d) for d in ['x','y']];ok&=wrist&mask[:,cols].all(1)
   elif f.endswith(('index_only','open_four','palm_front')):
    cols=[i for i,n in enumerate(names) if n.startswith(s+'_') and '_angle' in n and not 'elbow' in n]
    ok&=wrist&(mask[:,cols].sum(1)>=10)
   else:ok&=wrist
  out[:,j]=ok
 return out
def prepare_case(row,array,prob=None,obs=None):
 o=roi_observation(row['observation'],row['box']) if obs is None else obs
 full=F.sequence_features(o);seq=F.uniform_time_sample(full,96);lookup={int(f):i for i,f in enumerate(array['frame_id'])};ix=np.array([lookup[int(f)] for f in seq['frame_id']])
 targets=read(U/'LEARNING_INPUTS.json')['targets'];support=estimate_support(row,array,targets)
 # Broad projected face overlap needs face anchors too; this is not contact
 # or nose/cheek/forehead classification and does not use expression analysis.
 face_anchor=o.usable[:,[0,2,5]].all(1)&np.isfinite(o.xy_px[:,[0,2,5]]).all((1,2))
 for j,f in enumerate(targets):
  if f.endswith('face_overlap'):support[:,j]&=face_anchor
 return dict(seq=seq,indices=ix,prob=None if prob is None else prob[ix],support=support[ix],full_support=support,full_frame_ids=array['frame_id'],full_times=array['timestamp_us'],targets=targets,role_evidence=full['feedback_hand_roles'])
def pair(q,r,condition):
 cost,geom_evidence=frame_cost_matrix(q['seq'],r['seq'],hand_participation=False,one_sided_mode='flat')
 expected=r['support'].copy()
 # Only reference-confirmed low resting hand, not query absence, limits added shape terms.
 resting={s:F.observed_resting_hand(r['role_evidence'],s) for s in ['L','R']}
 for j,f in enumerate(r['targets']):
  if f[0] in resting and resting[f[0]]:expected[:,j]=False
 if condition!='legacy_dtw':
  shared=q['support'][:,None,:]&expected[None,:,:];missing=expected[None,:,:]&~q['support'][:,None,:]
  delta=np.abs(q['prob'][:,None,:]-r['prob'][None,:,:]);den=expected.sum(1)[None,:]
  extra=np.divide(np.where(shared,delta,0).sum(-1)+missing.sum(-1),den,out=np.zeros(cost.shape),where=den>0)
  cost=cost+.25*extra
 path=_optimal_path(cost,.2)
 if not path:return dict(status='NO_PATH',distance=None,events=[],reference_frames=len(r['seq']['frame_id']))
 path=np.asarray(path);qi,ri=path.T;nq,nr=cost.shape
 qmap=np.array([int(np.median(ri[qi==i])) for i in range(nq)]);rmap=np.array([int(np.median(qi[ri==j])) for j in range(nr)])
 common=expected&q['support'][rmap];expected_count=int(expected.sum());compared=int(common.sum());missing=expected&~q['support'][rmap]
 events=[]
 if condition!='legacy_dtw':
  for j,f in enumerate(q['targets']):
   a=q['prob'][:,j];b=r['prob'][qmap,j];re=expected[qmap,j];available=q['support'][:,j]&re
   confident=(np.maximum(a,1-a)>=.8)&(np.maximum(b,1-b)>=.8)
   difference=available&confident&(np.abs(a-b)>=.6)
   for idx in np.flatnonzero(difference):events.append(dict(query_frame=int(q['seq']['frame_id'][idx]),query_seconds=float(q['seq']['timestamp_us'][idx]/1e6),reference_frame=int(r['seq']['frame_id'][qmap[idx]]),reference_seconds=float(r['seq']['timestamp_us'][qmap[idx]]/1e6),feature=f,query_probability=float(a[idx]),reference_probability=float(b[idx]),kind='ESTIMATED_FEATURE_DIFFERENCE_NOT_SIGN_ERROR'))
 # Keep reference denominator and original query gaps independent of warping.
 required_any=expected.any(0);query_direct_missing=int((~q['full_support'][:,required_any]).sum()) if required_any.any() else 0
 return dict(status='NO_COMPARISON_EVIDENCE' if compared==0 else 'PARTIAL_EVIDENCE' if missing.any() or query_direct_missing else 'ESTIMATED_COMPARISON',distance=float(cost[qi,ri].mean()),geometry_evidence=float(geom_evidence[qi,ri].mean()),path=path.tolist(),query_to_reference=qmap.tolist(),reference_to_query=rmap.tolist(),reference_frames=nr,query_frames=nq,reference_scope_retained=bool(len(set(ri))==nr),reference_expected_cells=expected_count,compared_cells=compared,missing_cells=int(missing.sum()),coverage=compared/expected_count if expected_count else None,query_direct_unobserved_cells=query_direct_missing,reference_resting_evidence=resting,missing_reference_seconds=(r['seq']['timestamp_us'][missing.any(1)]/1e6).tolist(),events=events,educational_verdict=None,mandatory_omission_verdict=None)
def aggregate(query,pairs,condition):
 good=[p for p in pairs if p['distance'] is not None and p.get('compared_cells',0)>0]
 excluded=len(pairs)-len(good)
 coverage=min((p['coverage'] for p in pairs if p.get('coverage') is not None),default=None)
 if not good:return dict(query=query,condition=condition,status='NO_COMPARISON_EVIDENCE',distance=None,minimum_reference_coverage=coverage,compared_reference_count=0,excluded_reference_count=excluded,events=[],pairs=pairs,comparison_is_not_correctness=True)
 # Each reference remains complete. Consensus is not a stitched reference trajectory.
 votes={}
 for p in good:
  for e in p['events']:
   key=(e['query_frame'],e['feature']);votes.setdefault(key,[]).append(dict(reference=p['reference'],**e))
 events=[]
 for key,es in votes.items():
  if len({e['reference'] for e in es})>=2:events.append(dict(query_frame=key[0],feature=key[1],query_seconds=es[0]['query_seconds'],supporting_references=[e['reference'] for e in es],evidence=es,feedback_kind='OBSERVED_MODEL_DIFFERENCE_CANDIDATE',educational_verdict=None))
 return dict(query=query,condition=condition,status='PARTIAL_EVIDENCE' if excluded or any(p['status']!='ESTIMATED_COMPARISON' for p in good) else 'ESTIMATED_COMPARISON',distance=float(np.median([p['distance'] for p in good])),minimum_reference_coverage=coverage,compared_reference_count=len(good),excluded_reference_count=excluded,events=events,pairs=pairs,comparison_is_not_correctness=True)
def load_cases():
 info=read(U/'LEARNING_INPUTS.json');arrays=[dict(np.load(r['input'],allow_pickle=False)) for r in info['rows']]
 probs={c:np.load(U/'runs'/f'{c}_42/probabilities.npy',allow_pickle=False) for c in CONDITIONS[1:]};cases={c:{} for c in CONDITIONS}
 for i,(r,a) in enumerate(zip(info['rows'],arrays)):
  for c in CONDITIONS:cases[c][r['alias']]=prepare_case(r,a,None if c=='legacy_dtw' else probs[c][i,:len(a['x'])])
 return info,arrays,cases
def calibrate():
 info,arrays,cases=load_cases();rows=info['rows'];out=[]
 for q in rows:
  if q['role']!='development':continue
  refs=[r for r in rows if r['role']=='fit' and r['word_id']==q['word_id']];assert len(refs)==3
  for c in CONDITIONS:
   pairs=[dict(reference=r['alias'],**pair(cases[c][q['alias']],cases[c][r['alias']],c)) for r in refs]
   out.append(dict(label=q['label'],word_id=q['word_id'],**aggregate(q['alias'],pairs,c)))
 # Fixed finite-sample order statistic, for a descriptive distance alert only.
 thresholds={};calibration_valid_queries={}
 for c in CONDITIONS:
  distances=[r['distance'] for r in out if r['condition']==c and r.get('distance') is not None]
  calibration_valid_queries[c]=len(distances)
  thresholds[c]=float(np.quantile(distances,.95,method='higher')) if distances else None
 save('DEVELOPMENT_COMPARISON.json',dict(rows=out))
 save('FEEDBACK_FREEZE.json',dict(status='FROZEN_BEFORE_LOCKED_REPLAY',thresholds=thresholds,calibration_queries=sum(r['role']=='development' for r in rows),calibration_valid_queries=calibration_valid_queries,threshold_meaning='95th development query median-reference distance, not KSL acceptable boundary; <=12 samples cannot establish5pct false correction; null threshold means no calibration evidence',comparison_code_sha256=sha(__file__),protocol_sha256=sha(U/'PROTOCOL.json'),selected_seed=42,selection='Fixed seed by protocol; no test-driven model/threshold changes',mandatory_requirements_from_unconfirmed_guide=False))
def evaluate():
 frozen=read(U/'FEEDBACK_FREEZE.json');assert sha(__file__)==frozen['comparison_code_sha256'];info,arrays,cases=load_cases();rows=info['rows'];out=[];exports=[]
 for q in rows:
  if q['role']!='held_person':continue
  refs=[r for r in rows if r['role']=='fit' and r['word_id']==q['word_id']];assert len(refs)==3
  for c in CONDITIONS:
   pairs=[dict(reference=r['alias'],**pair(cases[c][q['alias']],cases[c][r['alias']],c)) for r in refs]
   result=aggregate(q['alias'],pairs,c);threshold=frozen['thresholds'][c];distance=result.get('distance')
   alert=distance>threshold if distance is not None and threshold is not None else None
   result.update(word_id=q['word_id'],label=q['label'],distance_threshold=threshold,descriptive_distance_alert=alert);out.append(result)
  exports.append(dict(alias=q['alias'],label=q['label'],word_id=q['word_id'],person=q['person'],source=q['source'],references=[dict(alias=r['alias'],person=r['person'],source=r['source']) for r in refs],time_seconds=(cases['legacy_dtw'][q['alias']]['seq']['timestamp_us']/1e6).tolist(),frame_id=cases['legacy_dtw'][q['alias']]['seq']['frame_id'].tolist()))
 summary=[]
 for c in CONDITIONS:
  rs=[r for r in out if r['condition']==c]
  summary.append(dict(condition=c,queries=len(rs),reference_pairs=sum(len(r['pairs']) for r in rs),descriptive_distance_alerts=sum(r['descriptive_distance_alert'] is True for r in rs),descriptive_distance_evaluable_queries=sum(r['descriptive_distance_alert'] is not None for r in rs),descriptive_distance_withheld_queries=sum(r['descriptive_distance_alert'] is None for r in rs),feature_difference_candidate_clips=sum(bool(r['events']) for r in rs),candidate_cells=sum(len(r['events']) for r in rs),all_reference_scopes_retained=all(p.get('reference_scope_retained',False) for r in rs for p in r['pairs']),minimum_reference_coverage=min((r['minimum_reference_coverage'] for r in rs if r.get('minimum_reference_coverage') is not None),default=None),educational_false_correction_rate=None,reason='No independent KSL correctness labels; physical reference differences are not sign errors. Legacy has no trained feature outputs, so candidate-cell count is not directly comparable.'))
 save('PAIRED_FEEDBACK.json',dict(rows=out,summary=summary,queries=exports,guide_map=str(U/'review/guide_stage_map.json'),semantic_match_verified=False))
 print(json.dumps(summary,ensure_ascii=False))
def subset_case(case,ids):
 q=copy.deepcopy(case);n=len(q['seq']['frame_id'])
 for k in ['features','present','timestamp_us','frame_id']:
  q['seq'][k]=q['seq'][k][ids].copy()
 q['seq']['timestamp_us']=np.arange(len(ids),dtype=np.int64)*33333
 q['indices']=q['indices'][ids];q['support']=q['support'][ids]
 if q['prob'] is not None:q['prob']=q['prob'][ids]
 return q
def controls():
 info,arrays,cases=load_cases();rows=info['rows'];checks=[]
 # Fixed all24 held-person cases, no handpicked successes.
 for row in rows:
  if row['role']!='held_person':continue
  a=row['alias']
  for c in CONDITIONS:
   original=cases[c][a];n=len(original['seq']['frame_id'])
   for speed in [.75,1.25]:
    mapping=np.rint(np.linspace(0,n-1,max(2,round(n/speed)))**1.0).astype(int);q=subset_case(original,mapping);r=pair(q,original,c)
    path=np.asarray(r['path']);err=np.abs(mapping[path[:,0]]-path[:,1]);seconds=np.abs(original['seq']['timestamp_us'][mapping[path[:,0]]]-original['seq']['timestamp_us'][path[:,1]])/1e6
    checks.append(dict(alias=a,condition=c,control=f'known_time_map_{speed}',mean_source_frame_error=float(err.mean()),mean_time_error_seconds=float(seconds.mean()),reference_scope_retained=r['reference_scope_retained'],scope='Resampled saved observed features and learned probabilities; not raw-video fast-motion accuracy'))
   for side in ['R','L']:
    q=copy.deepcopy(original);cols=[i for i,f in enumerate(q['targets']) if f.startswith(side+'_') or f=='hands_overlap'];q['support'][:,cols]=False;q['full_support'][:,cols]=False
    # Geometry disappearance must agree with learned-evidence disappearance.
    groups=[i for i,g in enumerate(F.GROUP_ORDER) if g.endswith('_'+side) or g=='bimanual'];q['seq']['present'][:,groups]=False
    r=pair(q,original,c);base=pair(original,original,c)
    checks.append(dict(alias=a,condition=c,control=side+'_sensor_missing',pass_contract=bool(r['reference_scope_retained'] and r['reference_expected_cells']==base['reference_expected_cells'] and r['educational_verdict'] is None),missing_cells=r['missing_cells'],expected_cells=r['reference_expected_cells'],same_denominator=base['reference_expected_cells']==r['reference_expected_cells'],status=r['status']))
   lo=max(1,int(n*.44));hi=max(lo+1,int(n*.56));keep=np.r_[np.arange(lo),np.arange(hi,n)];q=subset_case(original,keep);r=pair(q,original,c);path=np.asarray(r['path']);mapping=np.asarray(r['reference_to_query']);deleted=np.arange(lo,hi)
   checks.append(dict(alias=a,condition=c,control='short_middle_deleted',deleted_frames=hi-lo,reference_scope_retained=r['reference_scope_retained'],minimum_omitted_reference_time_preserved=float((original['seq']['timestamp_us'][hi-1]-original['seq']['timestamp_us'][lo])/1e6),max_reference_reuse=int(max(np.bincount(path[:,0]))),distance=r['distance'],mandatory_omission_accuracy=None,scope='Technical deletion retains reference interval. Does not establish linguistic necessity or real learner-error detection.'))
 save('CONTROLS.json',dict(rows=checks,semantic_error_accuracy=None,synthetic_not_real_errors=True))
 print(json.dumps(dict(cases=len(checks),contract_failures=sum(not r.get('pass_contract',True) for r in checks))))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=U/f'FEEDBACK_{sys.argv[1].upper()}_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
