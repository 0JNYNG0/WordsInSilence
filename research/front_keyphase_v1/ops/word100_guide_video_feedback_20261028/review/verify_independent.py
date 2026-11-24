"""Separate arithmetic verification; no training, inference or source writes."""
from pathlib import Path
import json, hashlib, math, inspect, sys
from collections import Counter
import numpy as np

U = Path(__file__).resolve().parents[1]
O = U.parent
checks, failures = [], []

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name, value):
    (checks if bool(value) else failures).append(name)
def close(a,b,tol=1e-8):
    if a is None or b is None:return a is None and b is None
    return math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol)

frozen=read(U/'FROZEN.json')
previous_result=read(U/'review/INDEPENDENT_VERIFY.json') if (U/'review/INDEPENDENT_VERIFY.json').exists() else None
audited_files=['motion_observer.py','compare_pipeline.py','build_guides.py','FROZEN.json','LEARNING_CONTRACT.json','OBSERVATIONS.json','PAIRED_REPLAY.json','CONTROLS.json','SUMMARY.json','GUIDE_CARDS.json']
run_hashes={name:sha(U/name) for name in audited_files}
for group in ('input_pins','protected'):
    for p,h in frozen[group].items():check(group+':'+Path(p).name,sha(p)==h)
exports=read(U/'OBSERVATIONS.json')
clips={v['alias']:v for v in exports['clips']}
rawrows=read(O/'word100_shared_feature_bundle_20261027/visual_inputs/INDEX.json')['rows']
records={v['alias']:v for v in read(O/'word100_shared_feature_bundle_20261027/FROZEN.json')['rows']}
rows={v['alias']:v for v in rawrows}
all_raw={}
full_count,annotation_count=0,0
max_feature_error=0.

def scalar_angle(points, valid, ids):
    n=len(points);out=np.full(n,np.nan);a,b,c=ids
    for i in range(n):
        if not valid[i,[a,b,c]].all():continue
        u=points[i,a]-points[i,b];v=points[i,c]-points[i,b]
        product=math.hypot(*u)*math.hypot(*v)
        if not np.isfinite(product) or product<=1e-12:continue
        out[i]=math.atan2(abs(u[0]*v[1]-u[1]*v[0]),u[0]*v[0]+u[1]*v[1])/math.pi
    return out

def expected_features(raw):
    xy=raw['xy_px'].astype(np.float64)
    valid=raw['observed'].astype(bool)&raw['usable'].astype(bool)&np.isfinite(xy).all(2)
    scale=np.hypot(*(xy[:,11]-xy[:,12]).T)
    bv=valid[:,11]&valid[:,12]&np.isfinite(scale)&(scale>1e-8)
    nv=valid&bv[:,None]
    norm=np.full_like(xy,np.nan)
    for i in np.flatnonzero(bv):norm[i]=(xy[i]-(xy[i,11]+xy[i,12])/2)/scale[i]
    out={}
    def put(name,val,good):out[name]=np.where(good&np.isfinite(val),val,np.nan)
    for side,offset,arm in [('L',33,(11,13,15)),('R',54,(12,14,16))]:
        for name,j in [('hand_wrist',offset),('pose_wrist',arm[2]),('elbow',arm[1]),('index_tip',offset+8)]:
            for d,axis in enumerate('xy'):put(f'{side}_{name}_{axis}',norm[:,j,d],nv[:,j])
        palm=[offset+j for j in [0,5,9,13,17]]
        for d,axis in enumerate('xy'):put(f'{side}_palm_center_{axis}',norm[:,palm,d].mean(1),nv[:,palm].all(1))
        out[side+'_elbow_angle']=scalar_angle(xy,valid,arm)
        for name,triplet in [('thumb_mcp',(1,2,3)),('thumb_ip',(2,3,4))]+[(finger+'_'+joint,triplet) for finger,start in [('index',5),('middle',9),('ring',13),('little',17)] for joint,triplet in [('mcp',(0,start,start+1)),('pip',(start,start+1,start+2)),('dip',(start+1,start+2,start+3))]]:
            out[side+'_'+name+'_angle']=scalar_angle(xy,valid,tuple(offset+j for j in triplet))
        delta=xy[:,offset+9]-xy[:,offset];length=np.hypot(delta[:,0],delta[:,1]);ok=valid[:,offset]&valid[:,offset+9]&(length>1e-8)&np.isfinite(length)
        for d,axis in enumerate('xy'):
            vec=np.full(len(xy),np.nan);np.divide(delta[:,d],length,out=vec,where=ok);put(side+'_palm_axis_'+axis,vec,ok)
        put(side+'_palm_width',np.hypot(*(norm[:,offset+5]-norm[:,offset+17]).T),nv[:,offset+5]&nv[:,offset+17])
        put(side+'_wrist_estimator_distance',np.hypot(*(norm[:,offset]-norm[:,arm[2]]).T),nv[:,offset]&nv[:,arm[2]])
        vel=np.full((len(xy),2),np.nan)
        for t in range(1,len(xy)):
            dt=(int(raw['timestamp_us'][t])-int(raw['timestamp_us'][t-1]))/1e6
            if raw['frame_id'][t]!=raw['frame_id'][t-1]+1 or not 0<dt<=.15 or not nv[t,offset] or not nv[t-1,offset]:continue
            vel[t]=(norm[t,offset]-norm[t-1,offset])/dt
        speed=np.hypot(vel[:,0],vel[:,1]);motion=np.isfinite(speed)&(speed>1e-8)
        for d,axis in enumerate('xy'):
            out[side+'_wrist_velocity_'+axis]=vel[:,d]
            vec=np.full(len(xy),np.nan);np.divide(vel[:,d],speed,out=vec,where=motion);out[side+'_wrist_motion_direction_'+axis]=vec
    delta=norm[:,54]-norm[:,33];ok=nv[:,54]&nv[:,33]
    for d,axis in enumerate('xy'):put('hands_wrist_delta_'+axis,delta[:,d],ok)
    put('hands_wrist_distance',np.hypot(delta[:,0],delta[:,1]),ok)
    return out

check('36_public_clips',len(clips)==36 and set(clips)==set(rows)==set(records))
check('12_expressions_3_record_ids',len({v['word_id'] for v in clips.values()})==12 and {v['person'] for v in clips.values()}=={1,2,3})
for alias,c in clips.items():
    rec=read(records[alias]['record'])
    with np.load(rec['views']['masked']['observation'],allow_pickle=False) as z:
        raw={k:z[k].copy() for k in ['xy_px','observed','usable','timestamp_us','frame_id']}
    all_raw[alias]=raw
    sel=np.zeros(len(raw['frame_id']),bool)
    for start,end in rows[alias]['annotation_intervals']:sel|=(raw['timestamp_us']>=round(start*1e6))&(raw['timestamp_us']<=round(end*1e6))
    full_count+=len(sel);annotation_count+=int(sel.sum())
    check(alias+':full_frame_count',c['full_video_frames_retained']==len(sel))
    check(alias+':all_annotation_frames',c['comparison_frames']==sel.sum() and np.array_equal(c['frame_id'],raw['frame_id'][sel]))
    check(alias+':actual_pts',np.array_equal(np.rint(np.array(c['timestamp_seconds'])*1e6).astype(np.int64),raw['timestamp_us'][sel]))
    check(alias+':identity_and_role',c['word_id']==rows[alias]['word_id'] and c['person']==rows[alias]['recorded_person_id'] and c['role']==rows[alias]['role'])
    x=np.array(c['x'],float);mask=np.array(c['mask'],bool)
    check(alias+':69_features',x.shape==(int(sel.sum()),69) and len(set(c['feature_names']))==69)
    check(alias+':null_mask_exact',np.array_equal(np.isfinite(x),mask))
    expected=expected_features(raw)
    ex=np.stack([expected[name][sel] for name in c['feature_names']],axis=1)
    check(alias+':independent_feature_mask',np.array_equal(np.isfinite(ex),mask))
    check(alias+':independent_feature_arithmetic',np.allclose(ex,x,atol=1e-8,rtol=1e-8,equal_nan=True))
    if np.isfinite(ex-x).any():max_feature_error=max(max_feature_error,float(np.nanmax(abs(ex-x))))
    check(alias+':unknown_semantics',c['motion_summary']['correctness'] is None and c['motion_summary']['semantic_stage_boundaries'] is None and c['motion_summary']['repeat_count'] is None)
check('4351_source_and1500_annotation_frames',full_count==4351 and annotation_count==1500)

paired=read(U/'PAIRED_REPLAY.json')['pairs']
expected_pairs={(q['alias'],r['alias']) for q in rawrows for r in rawrows if q['recorded_person_id']==3 and r['recorded_person_id'] in (1,2) and q['word_id']==r['word_id']}
check('24_unique_person03_vs01_02_pairs',len(paired)==24 and {(v['query'],v['reference']) for v in paired}==expected_pairs)
max_cost_error=0.;pair_coverage=[]
for p in paired:
    tag=p['query']+'_'+p['reference'];q=clips[p['query']];r=clips[p['reference']]
    qx=np.array(q['x'],float);rx=np.array(r['x'],float);qm=np.array(q['mask'],bool);rm=np.array(r['mask'],bool)
    path=np.array(p['path']);n,m=len(qx),len(rx)
    check(tag+':full_scope',set(path[:,0])==set(range(n)) and set(path[:,1])==set(range(m)) and p['reference_scope_retained'])
    check(tag+':legal_monotone_path',np.array_equal(path[0],[0,0]) and np.array_equal(path[-1],[n-1,m-1]) and all(tuple(z) in [(1,0),(0,1),(1,1)] for z in np.diff(path,axis=0)))
    selected={k:v for k,v in q['groups'].items() if not k.endswith(('_speed','_diagnostic'))}
    check(tag+':no_speed_cost',set(p['groups'])==set(selected))
    mapping=np.array(p['reference_to_query']);expected_mapping=[]
    for j in range(m):
        opts=[int(i) for i,jj in path if jj==j]
        expected_mapping.append(min(opts,key=lambda i:abs(i/n-j/m)))
    check(tag+':reference_representative',np.array_equal(mapping,expected_mapping))
    total_known=total_expected=total_missing=total_unknown=total_query_unknown=0
    cost=np.zeros((n,m))
    for name,cols in selected.items():
        g=p['groups'][name];expect=rm[:,cols];actual=qm[mapping][:,cols];common=expect&actual
        common_n=int(common.sum());expected_n=int(expect.sum());missing_n=expected_n-common_n;total=expect.size
        total_known+=common_n;total_expected+=expected_n;total_missing+=missing_n;total_unknown+=total-expected_n
        direct_unknown=int((~qm[:,cols]).sum());total_query_unknown+=direct_unknown
        diff=np.abs(qx[mapping][:,cols]-rx[:,cols]);vals=diff[common]
        check(tag+':'+name+':denominators',g['common_cells']==common_n and g['reference_observed_cells']==expected_n and g['reference_total_cells']==total and g['query_unobserved_at_reference_cells']==missing_n and g['reference_unobserved_cells']==total-expected_n)
        check(tag+':'+name+':difference',close(g['mean_absolute_difference'],float(vals.mean()) if len(vals) else None) and close(g['max_absolute_difference'],float(vals.max()) if len(vals) else None))
        check(tag+':'+name+':coverage',close(g['coverage_of_observed_reference'],common_n/expected_n if expected_n else None))
        missed_times=np.array(r['timestamp_seconds'])[np.any(expect&~actual,axis=1)]
        check(tag+':'+name+':missing_time',np.array_equal(missed_times,g['missing_reference_times']))
        check(tag+':'+name+':query_direct_gap',g['query_direct_unobserved_cells']==direct_unknown and g['query_direct_total_cells']==qm[:,cols].size and np.array_equal(g['query_direct_unobserved_times'],np.array(q['timestamp_seconds'])[~qm[:,cols].all(1)]))
        # Independent corrected cost: reference-unobserved features have no
        # measured difference. Observed reference + query missing costs 1.
        group_cost=np.zeros((n,m))
        for col in cols:
            good=qm[:,col,None]&rm[None,:,col]
            distance=np.minimum(abs(qx[:,col,None]-rx[None,:,col]),2.)
            group_cost+=np.where(good,distance,np.where(rm[None,:,col],1.,0.))
        den=rm[:,cols].sum(1)
        cost+=np.divide(group_cost,den[None,:],out=np.zeros_like(group_cost),where=den[None,:]>0)/len(selected)
    check(tag+':sum_counts',p['common_cells']==total_known and p['reference_observed_cells']==total_expected and p['query_missing_cells']==total_missing and p['reference_unknown_cells']==total_unknown)
    wanted='NO_COMPARISON_EVIDENCE' if total_known==0 else 'PARTIAL_EVIDENCE' if total_missing or total_unknown or total_query_unknown else 'OBSERVED_DIFFERENCE_ONLY'
    check(tag+':status',p['status']==wanted and p['educational_verdict'] is None and p['important_omission_verdict'] is None)
    pathcost=sum(float(cost[i,j]) for i,j in path)/len(path)
    grid=np.full((n+1,m+1),np.inf);grid[0,0]=0
    for i in range(1,n+1):
        for j in range(1,m+1):grid[i,j]=float(cost[i-1,j-1])+min(grid[i-1,j-1],grid[i-1,j],grid[i,j-1])
    max_cost_error=max(max_cost_error,abs(pathcost-p['alignment_cost']))
    check(tag+':dtw_saved_cost',close(pathcost,p['alignment_cost']) and close(grid[n,m]/len(path),p['alignment_cost']))
    pair_coverage.append({'query':p['query'],'reference':p['reference'],'common':total_known,'expected':total_expected,'missing':total_missing,'reference_unknown':total_unknown})

guides=read(U/'GUIDE_CARDS.json')['cards'];claims=[v for g in guides for c in g['candidates'] for v in c['claims']]
check('12_guides_no_verified_educational_rule',len(guides)==12 and all(g['educational_correction_enabled'] is False and g['full_guide_match_verified'] is False for g in guides))
check('guide_uncertainty_preserved',all(v['requiredness'] is None and v['tolerance'] is None and v['video_match_verified'] is False for v in claims))
forms=read(O/'word100_source_token_alignment_20261027/SOURCE_FORM_LINKS.json')['links']
for f in forms:
    card=next(c for c in guides if c['word_id']==f['word_id'])
    check(f['word_id']+':form_state_preserved',card['link']==f)
    for rejected in f['rejected_ids']:
        check(f['word_id']+':rejected_source',any(c['entry_id']==rejected and c['status']=='REJECTED_FOR_CURRENT_VIDEO_FORM' for c in card['candidates']))
control=read(U/'CONTROLS.json')['cases'];summary=read(U/'SUMMARY.json')
check('72invariant24contract_controls',len([c for c in control if 'pass'in c])==72 and len([c for c in control if 'contract_pass'in c])==24)
check('control_saved_status',all(c.get('pass',c.get('contract_pass')) for c in control))
check('no_omission_or_order_accuracy',summary['important_omission_accuracy'] is None and summary['normal_false_correction_rate'] is None and all(c.get('important_omission_detected') is None and c.get('important_order_error_detected') is None for c in control))
check('model_training_not_claimed',summary['new_model_training_runs']==0 and summary['new_vlm_inference_runs']==0)

# Interface/negative controls invoke the implementation only as system under test.
# Arithmetic and frozen-output validation above do not import it.
sys.path.insert(0,str(U))
import motion_observer as observer
import compare_pipeline as system
check('observer_no_target_parameters',set(inspect.signature(observer.extract_features).parameters)=={'arrays','max_motion_gap_seconds'})
raw=next(iter(all_raw.values()));before={k:v.copy() for k,v in raw.items()}
f=observer.extract_features(raw)
check('observer_input_unchanged',all(np.array_equal(raw[k],v,equal_nan=True) for k,v in before.items()))
try:observer.extract_features(raw,guide='wrong expected hand shape');reject=False
except TypeError:reject=True
check('guide_injection_rejected',reject)
forbidden=False
try:system.compare(f,f,guide='expected');forbidden=True
except TypeError:pass
check('matcher_does_not_condition_observations_on_guide',not forbidden)
f=system.subset(f,np.arange(min(8,len(f['x']))));g=dict(f);g['x']=np.full_like(f['x'],np.nan);g['mask']=np.zeros_like(f['mask'])
empty=system.compare(g,f)
check('all_missing_not_good',empty['status']=='NO_COMPARISON_EVIDENCE' and empty['educational_verdict'] is None and empty['common_cells']==0 and all(c['mean_absolute_difference'] is None for c in empty['groups'].values()))
check('all_missing_against_observed_nonzero_alignment_cost',empty['alignment_cost']>0 if empty['reference_observed_cells'] else empty['alignment_cost']==0)
check('all_missing_full_denominator',empty['reference_frames']==len(f['x']) and empty['reference_scope_retained'])
empty_ref=system.compare(f,g)
check('reference_all_missing_zero_not_good',empty_ref['alignment_cost']==0 and empty_ref['status']=='NO_COMPARISON_EVIDENCE' and empty_ref['educational_verdict'] is None and empty_ref['common_cells']==0)
tinyq={'x':np.array([[0.],[np.nan],[0.]]),'mask':np.array([[1],[0],[1]],bool),'timestamp_us':np.array([0,1000,2000]),'groups':{'R_position':[0]}}
tinyr={'x':np.array([[0.],[0.]]),'mask':np.ones((2,1),bool),'timestamp_us':np.array([0,2000]),'groups':{'R_position':[0]}}
partial=system.compare(tinyq,tinyr)
check('query_gap_not_hidden_by_reference_representatives',partial['status']=='PARTIAL_EVIDENCE' and partial['groups']['R_position']['query_direct_unobserved_cells']==1)

for group in ('input_pins','protected'):
    for p,h in frozen[group].items():check('after_'+group+':'+Path(p).name,sha(p)==h)
check('audited_files_stable_during_verification',all(sha(U/name)==h for name,h in run_hashes.items()))
history=(previous_result or {}).get('first_attempts',[])
if previous_result and previous_result.get('failed'):
    item={k:previous_result.get(k) for k in ['status','passed','failed','failures']}
    if item not in history:history.append(item)
report={'status':'PASS' if not failures else 'FAIL','author':'feedback_eval_review separate Codex arithmetic/code review; not independent human labels',
    'passed':len(checks),'failed':len(failures),'failures':failures,
    'checked_input_hashes':len(frozen['input_pins']),'protected_hashes':len(frozen['protected']),
    'clips':len(clips),'full_original_frames':full_count,'annotation_frames':annotation_count,'pairs':len(paired),
    'independent_feature_max_abs_error':max_feature_error,'independent_dtw_cost_max_abs_error':max_cost_error,
    'coverage_recount':pair_coverage,'checks':checks,'first_attempts':history,'audited_file_sha256':run_hashes,
    'limits':['No semantic guide-to-phase matcher exists in this implementation.','No KSL correctness, normal-variation acceptance or real omitted-stage ground truth.','No new pretrained VLM inference or training.','All frames retained for numerical DTW does not certify every important phase was matched.','Cached numeric replay time is not end-to-end feedback latency.']}
(U/'review/INDEPENDENT_VERIFY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['checks','coverage_recount']},ensure_ascii=False,indent=2))
if failures:raise SystemExit(1)
