"""Independent saved-array/path arithmetic only. No model, producer, DTW or video."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,traceback
import numpy as np

R=Path(__file__).resolve().parent;U=R.parent;O=U.parent
PINS={};CHECKS=Counter();FAIL=[];FACTS={}
FAMILIES=['R_shape','L_shape','R_angles','L_angles','hand_relation','arm_path','hand_axis']
ENGINE='68a94384350353db6c73025cba7c99ee406be593186c198f371d41b26fd75c11'


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()


def check(g,b,msg):
 CHECKS[g]+=1
 if not bool(b):FAIL.append(dict(group=g,message=msg))


def equal(g,a,b,msg):check(g,a==b,msg)


def pin(p):
 p=str(Path(p).resolve())
 if Path(p).suffix.lower() in ['.mp4','.webm','.mov','.avi','.zip']:
  raise ValueError('Video/archive bytes are outside this verification scope')
 h=sha(p)
 if p in PINS:equal('unchanged_during',h,PINS[p],p)
 else:PINS[p]=h
 return h


def read(p):pin(p);return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def arr(p,keys=None):
 pin(p)
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (keys or z.files)}


def close(g,a,b,msg):
 if a is None or b is None:equal(g,a,b,msg);return
 aa=np.asarray(a);bb=np.asarray(b)
 check(g,aa.shape==bb.shape and np.allclose(aa,bb,atol=2e-12,rtol=2e-12,equal_nan=True),msg)


def pointer(d,p):
 for k in p.strip('/').split('/'):d=d[int(k)] if isinstance(d,list) else d[k]
 return d


def channels(a):
 x=a['x'].astype(float);m=a['mask']&np.isfinite(x);x=np.where(m,x,0);p=a['probabilities'].astype(float)
 out={};evidence={};ts=a['timestamp_us']/1e6
 for side,off,cols in [('L',0,[2,4]),('R',28,[0,1,3])]:
  native=m[:,off:off+2].all(1)&(m[:,off+11:off+25].sum(1)>=10)
  context=native.copy();observed=np.flatnonzero(native)
  for left,right in zip(observed[:-1],observed[1:]):
   if float(ts[right]-ts[left])<=.2:context[left:right+1]=True
  good=np.isfinite(p[:,cols])&context[:,None]
  good[np.count_nonzero(p[:,cols]>=.5,axis=1)>1]=False
  out[side+'_shape']=(np.where(np.isfinite(p[:,cols]),p[:,cols],0),good)
  out[side+'_angles']=(x[:,off+11:off+25],m[:,off+11:off+25])
  evidence[side]=(native,context&~native)
 for key,cols in [('hand_relation',[56,57,58]),('arm_path',[2,3,4,5,30,31,32,33]),('hand_axis',[25,26,53,54])]:out[key]=(x[:,cols],m[:,cols])
 return out,evidence


def scope(row,ch):
 out={k:np.ones_like(v[1],bool) for k,v in ch.items()}
 for stage in row['stages']:
  lo,hi=stage['frames'];rest=stage['confirmed_rest_sides']
  for side in rest:
   out[side+'_shape'][lo:hi+1]=False;out[side+'_angles'][lo:hi+1]=False
   if side=='L':out['hand_axis'][lo:hi+1,0:2]=False
   else:out['hand_axis'][lo:hi+1,2:4]=False
  if rest:out['hand_relation'][lo:hi+1]=False
 return out


def measure(qx,qm,rx,rm,qs,rs,ref_indices,qt,rt,shape):
 valid=qm[qs]&rm[rs];difference=np.abs(qx[qs]-rx[rs]);difference[~valid]=np.nan
 expected=rm[ref_indices].sum(0);observed=np.zeros(len(expected),int);errors=[];missing=[]
 for ri in ref_indices:
  ix=np.flatnonzero(rs==ri);supported=valid[ix].any(0);observed+=supported
  if rm[ri].any() and not supported.any():missing.append(float(rt[ri]))
  if supported.any():errors.extend(np.nanmax(difference[ix][:,supported],axis=0).tolist())
 qe=np.zeros_like(expected);qo=np.zeros_like(expected);qmissing=[]
 for qi in np.unique(qs):
  need=np.any(rm[rs[qs==qi]],axis=0);got=need&qm[qi];qe+=need;qo+=got
  if need.any() and not got.any():qmissing.append(float(qt[qi]))
 rc=float(observed.sum()/expected.sum()) if expected.sum() else None
 qc=float(qo.sum()/qe.sum()) if qe.sum() else None
 dc=np.minimum(observed/np.maximum(expected,1),qo/np.maximum(qe,1))
 coverage=float(dc.min()) if shape else min(rc,qc) if rc is not None and qc is not None else None
 peak=float(np.nanmax(difference)) if valid.any() else None;at=None
 if peak is not None:
  hits=np.argwhere(valid&(difference==peak));last_ref=int(max(rs[hits[:,0]]))
  hits=hits[rs[hits[:,0]]==last_ref];pi,dimension=hits[0]
  at=dict(query_frame=int(qs[pi]),reference_frame=last_ref,query_seconds=float(qt[qs[pi]]),reference_seconds=float(rt[last_ref]),coordinate=int(dimension),representative_tie='last reference, first linked query/coordinate')
 return dict(distance=float(np.quantile(errors,.9)) if errors else None,peak_distance=peak,peak_at=at,
  coverage=coverage,reference_coverage=rc,query_coverage=qc,expected=int(expected.sum()),observed=int(observed.sum()),query_expected=int(qe.sum()),query_observed=int(qo.sum()),
  dimension_reference_expected=expected.tolist(),dimension_reference_observed=observed.tolist(),dimension_query_expected=qe.tolist(),dimension_query_observed=qo.tolist(),dimension_coverage=dc.tolist(),missing_reference_seconds=missing,missing_query_seconds=qmissing)


def check_measure(saved,expected,context):
 for k,v in expected.items():
  if isinstance(v,dict) or v is None:equal('pair_measure',saved[k],v,context+':'+k)
  else:close('pair_measure',saved[k],v,context+':'+k)


def main():
 bank=read(U/'REFERENCE_BANK.json');normal=read(U/'NORMAL_RANGES.json');pred=read(U/'BANK_PREDICTIONS.json')
 started=read(U/'CALIBRATION_STARTED.json');frozen=read(U/'FROZEN.json');public=read(U/'PUBLIC_INPUTS.json');training=read(U/'TRAINING_REPORT.json')
 for p,h in bank['source_pins'].items():equal('source_pins',pin(p),h,p)
 equal('source_pins',pin(U/'engine.py'),ENGINE,'final engine')
 for d in [normal,started]:equal('source_pins',d['engine_sha256'],ENGINE,'executed engine');equal('source_pins',d['reference_bank_sha256'],pin(U/'REFERENCE_BANK.json'),'bank pin')
 equal('source_pins',normal['builder_sha256'],pin(U/'build_bank.py'),'builder')
 equal('source_pins',bank['probability_manifest_sha256'],pin(U/'BANK_PREDICTIONS.json'),'prediction manifest')
 rest_audit=read(U/'REST_REFERENCE_AUDIT.json');equal('source_pins',bank['reference_rest_audit_sha256'],pin(U/'REST_REFERENCE_AUDIT.json'),'rest manifest')
 equal('bank',bank['observer'],training['adoption']['chosen_condition'],'fixed training decision')
 equal('bank',bank['observer'],'protected_baseline','unadopted expansion not substituted')
 rows={r['clip_id']:r for r in bank['rows']};public_rows={r['clip_id']:r for r in public['rows']}
 equal('bank',len(rows),36,'unique FIT clips');equal('bank',len(bank['rows']),36,'no duplicate rows')
 equal('bank',Counter((r['word_id'],r['person']) for r in rows.values()),Counter({(w,p):1 for w in frozen['word_ids'] for p in range(1,10)}),'four words nine recorded IDs')
 check('bank',all(r['role']=='fit' for r in rows.values()),'FIT only')
 equal('bank',sum(len(r['stages']) for r in rows.values()),117,'117 retained stages')
 check('bank',not bank['description_understanding_learned'] and not bank['educational_correctness_validated'],'no education/learned-description claim')
 arrays={};chs={};evidences={};scopes={};routes=Counter()
 for row in pred['rows']:
  cid=row['clip_id'];a=arr(row['input'],['x','mask','frame_id','timestamp_us']);p=arr(row['probability_file'])
  equal('probabilities',pin(row['input']),row['input_sha256'],'input digest');equal('probabilities',pin(row['probability_file']),row['probability_sha256'],'probability digest')
  for k in ['frame_id','timestamp_us']:close('probabilities',p[k],a[k],'exact native '+k)
  n=len(a['frame_id']);equal('probabilities',p['probabilities'].shape,(n,7),'native seven axis')
  check('probabilities',np.isfinite(p['probabilities'][:,:3]).all(),'old3 full native')
  valid=np.zeros(n,bool)
  for i in range(6,n-6):valid[i]=np.array_equal(a['frame_id'][i-6:i+7],np.arange(a['frame_id'][i]-6,a['frame_id'][i]+7))
  check('probabilities',np.array_equal(np.isfinite(p['probabilities'][:,3:]),np.repeat(valid[:,None],4,axis=1)),'new4 only real 13-frame centers')
  check('probabilities',np.array_equal(p['posterior_available'],np.isfinite(p['probabilities'])),'availability exact')
  route=row['prediction_route'];routes[route['route']]+=1
  if route['route'].startswith('reused_'):
   source=route['sources'][0];equal('probability_reuse',pin(source['path']),source['sha256'],'reuse pin')
   old=arr(source['path'],['probabilities']);check('probability_reuse',np.array_equal(old['probabilities'],p['probabilities'],equal_nan=True),'exact completed full7 reuse')
  for source in route.get('sources',[]):
   if source.get('kind')=='original_fullclip_old3':
    pin(source['path']);native=np.load(source['path'],allow_pickle=False)
    info=read(O/'word100_stage_feedback_cycle_20261031/LEARNING_INPUTS.json');columns=[info['targets'].index(t) for t in bank['targets'][:3]]
    check('probability_reuse',np.array_equal(p['probabilities'][:,:3],native[source['row'],:n][:,columns]),'exact saved native old3')
  if cid in rows:
   equal('bank',row['person'],rows[cid]['person'],'prediction bank identity')
   a['probabilities']=p['probabilities'];arrays[cid]=a;chs[cid],evidences[cid]=channels(a);scopes[cid]=scope(rows[cid],chs[cid])
 equal('bank',len(pred['rows']),48,'12 development predictions separate from36 FIT bank')
 equal('bank',len({r['clip_id'] for r in pred['rows']}),48,'no repeated prediction clip')
 descriptions=read(O/'word100_description_evidence_feedback_20261101/REPORT_DATA.json');rest_side_counts=Counter();direct_stages=inferred_stages=0
 for cid,row in rows.items():
  for k in ['source','source_sha256','input','input_sha256','observation','observation_sha256','word_id','person','role']:
   equal('bank_lineage',row[k],public_rows[cid][k],'public row '+k)
  mapping=read(row['stage_mapping']);equal('stage_projection',pin(row['stage_mapping']),row['stage_mapping_sha256'],'mapping pin')
  equal('stage_projection',mapping['engine_sha256'],ENGINE,'mapping engine');check('stage_projection',not mapping['query_stage_truth_used'],'no query label path')
  equal('stage_projection',mapping['stages'],row['stages'],'mapping stages exact')
  a=arrays[cid];canonical=arrays[row['canonical_clip_id']];path=np.asarray(mapping['path'])
  check('stage_projection',np.array_equal(path[0],[0,0]) and np.array_equal(path[-1],[len(a['frame_id'])-1,len(canonical['frame_id'])-1]),'mapping endpoints')
  check('stage_projection',all(tuple(d) in {(1,0),(0,1),(1,1)} for d in np.diff(path,axis=0)),'native monotone mapping')
  for key,expected in [('reference_frame_id',a['frame_id']),('canonical_frame_id',canonical['frame_id']),('reference_timestamp_us',a['timestamp_us']),('canonical_timestamp_us',canonical['timestamp_us'])]:close('stage_projection',mapping[key],expected,key)
  for stage in row['stages']:
   source=pointer(descriptions,stage['saved_stage_pointer']);equal('description',stage['description'],source['reference_rgb_description'],'source description')
   equal('description',stage['canonical_core_frame_ids'],source['reference_core'],'canonical frame span')
   lo,hi=[int(np.flatnonzero(canonical['frame_id']==f)[0]) for f in source['reference_core']]
   picks=path[(path[:,1]>=lo)&(path[:,1]<=hi),0];bounds=[int(picks.min()),int(picks.max())]
   equal('stage_projection',stage['frames'],bounds,'projected full occurrence range')
   equal('stage_projection',stage['projected_unique_reference_frames'],len(np.unique(picks)),'projection support count')
   expected='canonical01_saved_reviewed_span' if row['person']==1 else 'inferred_from_whole_sequence_DTW_not_truth'
   equal('stage_projection',stage['boundary_provenance'],expected,'inferred not ground truth')
   direct_stages+=row['person']==1;inferred_stages+=row['person']!=1
   check('description',stage['stage_semantic_correctness'] is None and not stage['mandatory_repetition_verified'] and not stage['source_cards_are_hard_rules'],'no semantic rule upgrade')
   for card in stage['source_cards']:
    oldcard=pointer(descriptions,card['saved_card_pointer']);check('description',card['context_only'],'card context only')
    for key in ['text','url','scope','restrictions','limitations','named_shape_verified','source_action_verified','full_span_semantic_verified']:equal('description',card[key],oldcard.get(key),'card '+key)
   ev=stage['reference_rest_evidence'];sources=ev['sources'];ob=arr(sources['native']['path'],['xy_px','observed','usable','frame_id','timestamp_us']);record=read(sources['calibration']['path']);cal=record['calibration']
   for spec in sources.values():equal('rest_sources',pin(spec['path']),spec['sha256'],'rest evidence pin')
   equal('rest_sources',record['views']['native']['observation'],sources['native']['path'],'native rather than ROI view')
   close('rest_sources',ob['frame_id'],a['frame_id'],'rest input frame match');close('rest_sources',ob['timestamp_us'],a['timestamp_us'],'rest input PTS match')
   ids=np.arange(bounds[0],bounds[1]+1);xy=ob['xy_px'].astype(float);confirmed=[]
   calibrated=cal.get('status')=='CALIBRATED' and cal.get('reason')=='STABLE_ARMS_DOWN_WINDOW'
   for side,wrist,anchorname in [('L',33,'left'),('R',54,'right')]:
    item=ev['sides'][side];positions=[11,12,wrist]
    valid=(ob['observed'][ids][:,positions]&ob['usable'][ids][:,positions]&np.isfinite(xy[ids][:,positions]).all(-1)).all(1)
    width=np.linalg.norm(xy[ids,11]-xy[ids,12],axis=1);valid&=np.isfinite(width)&(width>0)
    distances=np.full(len(ids),np.nan)
    if calibrated:
     anchor=np.asarray(cal['wrist_anchors_normalized'][anchorname]);positions=(xy[ids[valid],wrist]-(xy[ids[valid],11]+xy[ids[valid],12])/2)/width[valid,None]
     distances[valid]=np.linalg.norm(positions-anchor,axis=1)
    near=np.isfinite(distances)&(distances<=.08)
    if calibrated and near.sum()/len(ids)>=.8:confirmed.append(side);rest_side_counts[side]+=1
    for key,value in [('possible_stage_frames',len(ids)),('raw_usable_same_site_frames',int(valid.sum())),('raw_unavailable_frames',int((~valid).sum())),('near_anchor_frames',int(near.sum())),('comparable_anchor_frames',int(np.isfinite(distances).sum())),('native_wrist_landmark',wrist),('pose_wrist_fallback_used',False)]:equal('rest_math',item[key],value,key)
    equal('rest_math',item['near_anchor'],near.tolist(),'rest original native-frame predicate');equal('rest_math',item['raw_usable'],valid.tolist(),'rest original quality mask')
    close('rest_math',[np.nan if v is None else v for v in item['anchor_distance_normalized']],distances,'same-site normalized distances')
    equal('rest_math',item['fraction_of_whole_stage'],float(near.sum()/len(ids)) if calibrated else None,'missing stays denominator')
   equal('rest_math',stage['confirmed_rest_sides'],confirmed,'only reference-supported rest')
   check('rest_claims',not ev['query_values_or_missing_used'] and not ev['linguistic_irrelevance_verified'] and not ev['finger_motion_or_whole_hand_rest_verified'],'rest not semantic truth')
 equal('rest_math',dict(rest_side_counts),rest_audit['confirmed_side_stage_counts'],'all234side-stage counts')
 expected_pairs={(q,r) for q in rows for r in rows if rows[q]['word_id']==rows[r]['word_id'] and rows[q]['person']!=rows[r]['person']}
 equal('calibration',len(expected_pairs),288,'all leave-person-out opportunities');equal('calibration',{(p['query'],p['reference']) for p in normal['pair_index']},expected_pairs,'all pairs');equal('calibration',len(normal['pair_index']),288,'no duplicate pair')
 pairs={};byquery=defaultdict(list);all_scope=Counter()
 for item in normal['pair_index']:
  p=read(item['path']);equal('pair_pins',pin(item['path']),item['sha256'],'pair hash');qid,rid=item['query'],item['reference'];pairs[(qid,rid)]=p;byquery[qid].append((item,p))
  equal('calibration',p['query_person'],rows[qid]['person'],'query identity');equal('calibration',p['person'],rows[rid]['person'],'ref identity');equal('calibration',p['id'],rid,'reference ID')
  path=np.asarray(p['path']);qi,ri=path.T;qa,ra=arrays[qid],arrays[rid];qt=qa['timestamp_us']/1e6;rt=ra['timestamp_us']/1e6
  check('path',np.array_equal(path[0],[0,0]) and np.array_equal(path[-1],[len(qt)-1,len(rt)-1]) and all(tuple(d) in {(1,0),(0,1),(1,1)} for d in np.diff(path,axis=0)),'saved full native monotone path')
  close('path',p['query_frame_times'],qt,'query PTS');close('path',p['reference_frame_times'],rt,'reference PTS')
  numerator=np.zeros(len(path));denominator=np.zeros(len(path));main=np.zeros(len(rt),bool)
  for lo,hi in rows[rid]['intervals']:main|=(rt>=lo)&(rt<=hi)
  for name in FAMILIES:
   qx,qm=chs[qid][name];rx,rm=chs[rid][name];keep=scopes[rid][name]
   usable=qm[qi]&rm[ri]&keep[ri];count=usable.sum(1);difference=np.abs(qx[qi]-rx[ri]);distance=np.sqrt(((difference**2)*usable).sum(1)/np.maximum(count,1));distance[count==0]=1
   weight=np.ones(len(path)) if name=='arm_path' else main[ri].astype(float);weight*=keep[ri].any(1)
   numerator+=np.minimum(distance,3)*weight;denominator+=weight
  close('path_cost',p['distance'],float(np.mean((numerator/np.maximum(denominator,1))[main[ri]])),'saved path contribution, no DTW optimization')
  equal('pair_stage',len(p['stages']),len(rows[rid]['stages']),'all reference stage occurrences')
  for saved,stage in zip(p['stages'],rows[rid]['stages']):
   lo,hi=stage['frames'];sel=(ri>=lo)&(ri<=hi);qs,rs=qi[sel],ri[sel];ref=np.arange(lo,hi+1);uq=np.unique(qs)
   equal('pair_stage',saved['canonical_stage_id'],stage['canonical_stage_id'],'canonical occurrence ID');equal('pair_stage',saved['boundary_provenance'],stage['boundary_provenance'],'boundary provenance')
   for name in FAMILIES:
    qx,qm=chs[qid][name];rx,rm=chs[rid][name];keep=scopes[rid][name];v=saved['families'][name]
    original=measure(qx,qm,rx,rm,qs,rs,ref,qt,rt,name.endswith('_shape'));scoped=measure(qx,qm,rx,rm&keep,qs,rs,ref,qt,rt,name.endswith('_shape'))
    check_measure(v,scoped,name)
    expected=dict(original_expected=original['expected'],scope_excluded=original['expected']-scoped['expected'],original_possible=len(ref)*rm.shape[1],scoped_possible=int(keep[ref].sum()),scope_excluded_possible=int((~keep[ref]).sum()),original_query_expected=original['query_expected'],scope_excluded_query_expected=original['query_expected']-scoped['query_expected'],original_query_possible=len(uq)*rm.shape[1],scoped_query_possible=sum(int(keep[rs[qs==i]].any(0).sum()) for i in uq),comparison_required=bool(keep[ref].any()))
    for key,value in expected.items():equal('scope_counts',v[key],value,name+':'+key)
    check('scope_counts',v['original_expected']==v['expected']+v['scope_excluded'],'expected denominator closure')
    all_scope.update(original_expected=v['original_expected'],scoped_expected=v['expected'],scope_excluded=v['scope_excluded'])
    if not keep[ref].all():check_measure(v['original_evidence'],original,name+' original')
    else:equal('scope_counts',v['original_evidence'],None,'unchanged raw evidence need not duplicate')
    check('scope_claims',v['linguistic_requiredness'] is None,'no rest semantic requiredness')
 ledger=[];values=defaultdict(list);chosen_pairs=[]
 for qid in rows:
  chosen=sorted(byquery[qid],key=lambda item:item[1]['distance'])[:2]
  for rank,(item,p) in enumerate(chosen,1):
   chosen_pairs.append((qid,item['reference']))
   for si,stage in enumerate(p['stages']):
    for name in FAMILIES:
     v=stage['families'][name];usable=v['comparison_required'] and v['coverage'] is not None and v['coverage']>=.65 and v['distance'] is not None and v['peak_distance'] is not None
     if usable:values[name].append((v['distance'],v['peak_distance']))
     ledger.append(dict(query=qid,person=rows[qid]['person'],word_id=rows[qid]['word_id'],selected_rank=rank,reference=p['id'],canonical_stage_id=stage['canonical_stage_id'],stage_index=si,family=name,usable_for_range=usable,distance=v['distance'],comparison_required=v['comparison_required'],peak_distance=v['peak_distance'],coverage=v['coverage'],expected=v['expected'],observed=v['observed'],query_expected=v['query_expected'],query_observed=v['query_observed'],pair_json_pointer=f'/stages/{si}/families/{name}',pair_path=item['path']))
 equal('calibration_ledger',len(ledger),1638,'117stage×2whole references×7families')
 for actual,expected in zip(normal['selected_stage_family_ledger'],ledger):equal('calibration_ledger',actual,expected,'exact saved source selection and denominator')
 equal('calibration_ledger',len(normal['selected_stage_family_ledger']),len(ledger),'no missing ledger rows')
 for name in FAMILIES:
  selected=[v for v in ledger if v['family']==name];vv=np.asarray(values[name],float)
  for col,key in enumerate(['p90','peak']):close('ranges',normal['ranges'][name][key],float(np.quantile(vv[:,col],.95)) if len(vv) else None,name+':'+key)
  counts=dict(possible=len(selected),supported=len(vv),not_required_reference_rest=sum(not v['comparison_required'] for v in selected),insufficient=sum(v['comparison_required'] and not v['usable_for_range'] for v in selected),expected_cells=sum(v['expected'] for v in selected),observed_cells=sum(v['observed'] for v in selected),query_expected_cells=sum(v['query_expected'] for v in selected),query_observed_cells=sum(v['query_observed'] for v in selected))
  equal('ranges',normal['counts'][name],counts,'family counts closure '+name)
 check('claims',normal['development_used'] is False and normal['user_data_used'] is False and normal['independent_sample_accuracy_claim'] is False,'FIT only no accuracy claim')
 equal('claims',normal['query_people'],list(range(1,10)),'recorded IDs calibration');equal('claims',normal['complete_pair_opportunities'],288,'full pair count');equal('claims',normal['selected_pair_opportunities'],72,'selected whole pairs')
 FACTS.update(reference_clips=36,recorded_ids_per_expression=9,development_clips_excluded=12,prediction_clips=48,stage_occurrences=117,canonical_direct_stage_occurrences=direct_stages,inferred_stage_occurrences=inferred_stages,reference_rest_side_counts=dict(rest_side_counts),full_calibration_pairs=288,selected_whole_pairs=72,stage_family_opportunities=1638,all_pair_original_scoped_counts=dict(all_scope),prediction_routes=dict(routes),normal_family_counts=normal['counts'])


if __name__=='__main__':
 dest=R/'BANK_CALIBRATION_VERIFICATION.json'
 if dest.exists():raise SystemExit('Preserve existing result')
 try:main()
 except BaseException:FAIL.append(dict(group='verifier_exception',message=traceback.format_exc()))
 for p,h in list(PINS.items()):equal('preservation_after',sha(p),h,p)
 result=dict(status='PASS' if not FAIL else 'FAIL',checks=sum(CHECKS.values()),check_groups=dict(CHECKS),failures=FAIL,facts=FACTS,source_pins=PINS,verifier_sha256=sha(__file__),
  method='Independent numpy/stdlib reconstruction of channels, reference-rest evidence/scope, arithmetic along already-saved paths, all source projection/selection ledgers and FIT quantiles. No producer import or path optimization.',
  limitations=['This reviewer implemented the preceding narrow scope repair; these stored-result checks use separate arithmetic but are not a separate human or KSL expert review.','No new forward pass or DTW was executed. New prediction numerical values are not independently re-inferred.','Source video/archive bytes were not opened; source video path/hash metadata identity was compared with the frozen public manifest.','Recorded IDs are not independently proven biological signer identities. Prior development exposure remains historical; this is descriptive FIT calibration, not accuracy.','Only canonical01 spans were previously reviewed; other-reference boundaries are inferred by saved DTW, not new stage truth.','Reference rest is same-site wrist proximity with whole-stage support, not a claim of stationary fingers or linguistic irrelevance.','All quantities are projected image/body-normalized features, not metric3D rotation or educational correctness.'],actions=dict(model_inference=0,training=0,dtw_runs=0,video_or_archive_opens=0,user_data=0,old_result_changes=0))
 with dest.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({'status':result['status'],'checks':result['checks'],'failures':len(FAIL),'output':str(dest)}))
 if FAIL:print(json.dumps(FAIL[:6],ensure_ascii=False,indent=2));raise SystemExit(1)
