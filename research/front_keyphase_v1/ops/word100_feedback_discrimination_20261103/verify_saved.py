"""Independent arithmetic/identity checks of saved values. No model imports/calls."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import sys
import time
import traceback
import numpy as np

U=Path(__file__).resolve().parent;O=U.parent
AJ=O/'word100_local_event_controls_20261103'
AA=O/'word100_partial_order_live_20261102'
M=O/'word100_learned_motion_order_20261102'
R=O/'word100_learned_rgb_motion_20261102'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

class Audit:
    def __init__(self):self.checks=[]
    def test(self,name,ok):
        if not bool(ok):raise AssertionError(name)
        self.checks.append(name)
    def equal(self,name,left,right):self.test(name,left==right)
    def pins(self,values):
        for p,h in values.items():self.equal('sha:'+p,sha(p),h)

def rebuild(raw,valid_mask,geometry_stats,rgb_stats,geometry_valid):
    """Assemble the specified915 fields directly, without importing any producer."""
    n=len(raw['frame_id']);v=raw['x'];mask=raw['mask'];image=raw['image_available']
    axis=v[:,2:4]-v[:,4:6];relative=v[:,36:38]-v[:,4:6]
    length_squared=np.einsum('ij,ij->i',axis,axis)
    relation_valid=np.logical_and.reduce(mask[:,[2,3,4,5,36,37]],axis=1)&np.isfinite(length_squared)&(length_squared>.0025)
    denominator=np.maximum(length_squared,1e-6)
    along=(relative[:,0]*axis[:,0]+relative[:,1]*axis[:,1])/denominator
    side=(relative[:,0]*axis[:,1]-relative[:,1]*axis[:,0])/denominator
    physical=np.concatenate((v[:,:61],along[:,None],side[:,None],raw['retained_probability'],raw['relation_probability']),axis=1)
    observed=np.concatenate((mask[:,:61]&np.isfinite(v[:,:61]),relation_valid[:,None],relation_valid[:,None],
                             np.isfinite(raw['retained_probability']),np.isfinite(raw['relation_probability'])),axis=1)
    observed &= image[:,None]
    physical=np.where(observed,physical,0).astype(np.float32)
    observed &= np.isfinite(physical)
    physical=np.where(observed,physical,0)
    geo_values=np.where(observed,(physical-np.array(geometry_stats['mean']))/np.array(geometry_stats['std']),0)
    geo=np.concatenate((geo_values,observed),axis=1).astype(np.float32)
    rgb_available=image&np.isfinite(raw['rgb']).all(axis=1)
    rgb_values=np.where(rgb_available[:,None],(raw['rgb']-np.array(rgb_stats['mean']))/np.array(rgb_stats['std']),0)
    rgb=np.concatenate((rgb_values,rgb_available[:,None]),axis=1).astype(np.float32)
    x=np.concatenate((geo,rgb),axis=1)
    # Geometry observation from the actual mask half consumed by the model.
    have=rgb_available|((x[:,73:146]*np.asarray(geometry_valid)[73:])>.5).any(axis=1)
    fragments=[];start=None
    for i,yes in enumerate(have.tolist()+[False]):
        if yes and start is None:start=i
        if not yes and start is not None:fragments.append([start,i-1]);start=None
    anchor=np.zeros(n,dtype=bool)
    for lo,hi in fragments:
        if hi-lo>=1:anchor[lo:hi+1]=True
    return x,have,anchor,rgb_available,fragments

def independent_rgb_rows(times,available):
    """Enumerate positive center interpolation support using searchsorted."""
    n=len(times);starts=np.unique(np.r_[np.arange(0,n-15,2),n-16]).astype(int)
    centers=np.asarray([int(times[s])+int(times[s+15]) for s in starts],dtype=np.int64)
    rows=[]
    for i,t in enumerate(times):
        z=2*int(t)
        if z<=centers[0]:chosen=[0]
        elif z>=centers[-1]:chosen=[len(centers)-1]
        else:
            right=int(np.searchsorted(centers,z,side='left'))
            chosen=[right] if centers[right]==z else [right-1,right]
        support=set()
        for c in chosen:support.update(range(int(starts[c]),int(starts[c])+16))
        rows.append(sorted(support) if available[i] else [])
    return starts,centers,rows

def check_readout(audit,label,row,indices,target,prob,anchors,ids,ts,consumed,supported):
    lo,hi=indices;range_ids=list(range(lo,hi+1));eligible=[i for i in range_ids if anchors[i]]
    complete=all(anchors[i] for i in range_ids)
    audit.equal(label+':scope',row['indices'],[lo,hi])
    audit.equal(label+':frames',row['frames'],[int(ids[lo]),int(ids[hi])])
    audit.equal(label+':times',row['timestamp_us'],[int(ts[lo]),int(ts[hi])])
    audit.equal(label+':label_not_created',row['label'],None)
    audit.equal(label+':coverage',row['prediction_coverage'],len(eligible)/len(range_ids))
    audit.equal(label+':gaps',row['unavailable_native_frames'],[int(ids[i]) for i in range_ids if not anchors[i]])
    audit.equal(label+':supported',row['supported'],supported)
    if not supported or not eligible:
        audit.equal(label+':no_evidence_probability',row['probability'],None)
        audit.equal(label+':no_evidence_status',row['status'],'UNSUPPORTED_QUERY' if not supported else 'INSUFFICIENT_OBSERVATION')
        return
    at=max(eligible,key=lambda i:float(prob[i,target]));value=float(prob[at,target])
    expected_status=('MOVEMENT_EVIDENCE' if complete else 'MOVEMENT_EVIDENCE_WITH_GAPS') if value>=.8 else ('INSUFFICIENT_OBSERVATION' if not complete else 'LOW_MOVEMENT_SCORE' if value<=.2 else 'UNCONFIRMED')
    for key,expected in [('probability',value),('anchor',at),('anchor_frame',int(ids[at])),('anchor_timestamp_us',int(ts[at])),
                         ('raw_positive',value>=.5),('high_confirmation',value>=.8),('status',expected_status),
                         ('absence_assessable',complete),('labelled_evidence_complete',complete)]:
        audit.equal(label+':'+key,row[key],expected)
    fragment=[at]
    for i in range(at-1,-1,-1):
        if not anchors[i]:break
        fragment.append(i)
    for i in range(at+1,len(ids)):
        if not anchors[i]:break
        fragment.append(i)
    temporal=sorted(i for i in fragment if abs(i-at)<=7)
    rgb=sorted({j for i in temporal for j in consumed[i]})
    support=row['context_support']
    audit.equal(label+':temporal_context',support['temporal_native_rows'],temporal)
    audit.equal(label+':rgb_context',support['rgb_native_rows'],rgb)
    audit.equal(label+':rgb_context_time',support['rgb_native_timestamps_us'],[int(ts[i]) for i in rgb])
    audit.test(label+':context_not_onset',support['structural_support_only'] and not support['exact_onset_or_observed_event_boundary'])
    for key in ('whole_action_complete','whole_action_error','educational_error','occurrence_truth','whole_sign_correctness','complete_action'):
        audit.equal(label+':unknown_'+key,row[key],None)
    audit.equal(label+':no_correction',row['correction_enabled'],False)

def verify_reverse():
    destination=U/'SAVED_REVERSE_VERIFIED.json'
    if destination.exists():
        old=read(destination)
        if old['result_sha256']!=sha(U/'REVERSE_RESULT.json'):raise ValueError('Verified result changed')
        print('Existing reverse verification retained; no repeat')
        return
    a=Audit();manifest=read(U/'reverse_inputs.json');result=read(U/'REVERSE_RESULT.json');execution=read(U/'REVERSE_EXECUTION_FROZEN.json')
    a.pins(execution['pins']);a.pins(execution['input_pins']);a.equal('input pins unchanged',manifest['pins'],execution['input_pins'])
    a.equal('finished result hash',read(U/'REVERSE_RUN_FINISHED.json')['result_sha256'],sha(U/'REVERSE_RESULT.json'))
    a.equal('thresholds',execution['thresholds'],{'high':.8,'low':.2,'raw':.5})
    a.equal('model',result['model_sha256'],manifest['model']['sha256'])
    a.equal('four forward artifacts declared',result['model_forwards'],4)
    a.test('not current AM scope',result['current_AM_fidelity_claim'] is False and manifest['current_AM_fidelity_claim'] is False)
    a.equal('legacy scope',result['scope_family'],'legacy_R_canonical_motion_DTW')
    a.test('no training/DTW/extraction/AJrerun',all(result[k]==0 for k in ['training','new_extraction','new_DTW','AJ24_reruns']))
    a.equal('old outputs unchanged claim',result['original_results_changed'],False)
    a.equal('whole goal incomplete',result['whole_goal_complete'],False)
    originals={r['case_id']:r for r in manifest['rows']};rows=result['rows']
    a.equal('exact4unique identities',sorted(r['case_id'] for r in rows),sorted(originals))
    a.equal('prediction artifact count',len(list((U/'reverse_predictions').glob('*.npz'))),4)
    geo=read(M/'NORMALIZATION.json');rgb=read(R/'RGB_NORMALIZATION.json');oldAA=read(AA/'SAVED_CASES_MANIFEST.json')
    all_obs=[];all_full=[];frames=0;maximum_probability_error=0.;timings=[]
    for row in rows:
        cid=row['case_id'];source=originals[cid];label=cid
        a.test(label+':ID12_exposed_DEV',row['person']==12 and row['role']=='previously_exposed_development_controls')
        for key,source_key in [('cache_binding','analysis'),('saved_aa_result','saved_aa_result'),('source_ledger','source_ledger')]:a.equal(label+':'+key,row[key],source[source_key])
        a.equal(label+':prediction_sha',sha(row['prediction']['path']),row['prediction']['sha256'])
        with np.load(source['analysis']['path'],allow_pickle=False) as f:raw={k:f[k].copy() for k in f.files}
        with np.load(row['prediction']['path'],allow_pickle=False) as f:pred={k:f[k].copy() for k in f.files}
        ids=raw['frame_id'];ts=raw['timestamp_us'];n=len(ids);frames+=n
        a.equal(label+':native_shape_count',n,row['native_frames']);a.equal(label+':native_frame_manifest',ids.tolist(),source['native_frame_ids']);a.equal(label+':native_pts_manifest',ts.tolist(),source['timestamp_us'])
        for key in ('frame_id','timestamp_us'):a.test(label+':saved_axis_'+key,np.array_equal(raw[key],pred[key]) and raw[key].dtype==pred[key].dtype)
        x,observed,anchors,available,fragments=rebuild(raw,None,geo,rgb,manifest['geometry_valid'])
        a.equal(label+':915_hash',hashlib.sha256(x.tobytes()).hexdigest(),source['normalized_915_sha256'])
        a.test(label+':915_finite_shape',x.shape==(n,915) and np.isfinite(x).all())
        for key,expected in [('motion_anchor',anchors),('evidence_available',observed),('rgb_available',available)]:
            a.test(label+':'+key,pred[key].dtype==np.dtype('bool') and np.array_equal(pred[key],expected))
        a.equal(label+':manifest_fragments',source['observed_fragments'],fragments)
        a.equal(label+':image_gaps',row['image_unavailable_frames'],ids[~raw['image_available']].tolist())
        a.equal(label+':unobserved_gaps',row['unobserved_frames'],ids[~observed].tolist())
        logits=pred['logits'];prob=pred['probabilities']
        a.test(label+':saved_probability_shape_finite',logits.shape==prob.shape==(n,5) and np.isfinite(logits).all() and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all())
        z=logits.astype(np.float64);expected=np.exp(-np.logaddexp(0.,-z));error=float(np.max(np.abs(expected-prob)))
        maximum_probability_error=max(maximum_probability_error,error)
        a.test(label+':sigmoid_saved_logits',error<2e-7)
        starts,centers,consumed=independent_rgb_rows(ts,available)
        structural=source['rgb_lineage']['structural_support']
        a.equal(label+':structural_centers',structural['center_twice_us'],centers.tolist())
        a.equal(label+':structural_windows',[w['slot_rows'] for w in structural['windows']],[list(range(int(s),int(s)+16)) for s in starts])
        a.equal(label+':structural_support',structural['row_consumed_support'],consumed)
        a.test(label+':recipe_not_savedvectors',source['rgb_lineage']['window_vectors_saved'] is False and source['rgb_lineage']['numerical_reinterpolation_verified'] is False)
        a.equal(label+':same_AA_source',oldAA['cases'][row['word_id']]['order']['video_sha256'],source['video']['sha256'])
        aa=read(source['saved_aa_result']['path']);a.equal(label+':oldAA_pinned',sha(source['saved_aa_result']['path']),source['saved_aa_result']['sha256'])
        a.equal(label+':AA_prototype_path',aa['path'],source['prototype_path'])
        a.equal(label+':scope_count',len(row['observations']),len(source['scopes']))
        a.equal(label+':diagnostic_count',len(row['full_native_diagnostics']),len(source['scopes']))
        mapping=np.arange(n);lo,hi=source['source_ledger']['reversed_source_frames'];mapping[lo:hi+1]=mapping[lo:hi+1][::-1]
        a.equal(label+':actual_source_mapping',mapping.tolist(),source['source_ledger']['source_frame_indices'])
        for obs,full,scope in zip(row['observations'],row['full_native_diagnostics'],source['scopes']):
            target=scope['target_index'];a.equal(label+':fixed_scope_'+str(target),obs['fixed_scope'],scope)
            a.equal(label+':fixed_target_'+str(target),obs['target_index'],target)
            check_readout(a,label+':scope'+str(target),obs,scope['indices'],target,prob,anchors,ids,ts,consumed,target in manifest['supported_targets'])
            check_readout(a,label+':full'+str(target),full,[0,n-1],target,prob,anchors,ids,ts,consumed,target in manifest['supported_targets'])
            a.equal(label+':full_not_selection'+str(target),full['scope_selection_use'],False)
        all_obs.extend(row['observations']);all_full.extend(row['full_native_diagnostics']);timings.append(row['head_seconds'])
        a.test(label+':head_time_finite_positive',math.isfinite(row['head_seconds']) and row['head_seconds']>0)
    summary=dict(clips=4,fixed_scopes=len(all_obs),registered_people=1,registered_id=12,
                 high_occurrence=sum(bool(b.get('high_confirmation')) for b in all_obs),
                 insufficient=sum(b['status']=='INSUFFICIENT_OBSERVATION' for b in all_obs),
                 full_native_high_diagnostic=sum(bool(b.get('high_confirmation')) for b in all_full),
                 known_occurrence_gold=0,occurrence_accuracy=None,whole_sign_error_accuracy=None)
    a.equal('independent summary',result['summary'],summary)
    a.test('no native imports',not any(k in sys.modules for k in ['torch','av','cv2','mediapipe','openvino']))
    report=dict(status='PASS_SAVED_REVERSE_ARITHMETIC_AND_INPUT_LINEAGE',checks_count=len(a.checks),checks=a.checks,failed_checks=[],
        result_sha256=sha(U/'REVERSE_RESULT.json'),input_sha256=sha(U/'reverse_inputs.json'),verifier_sha256=sha(Path(__file__)),
        summary=summary,native_rows=frames,maximum_sigmoid_absolute_error=maximum_probability_error,
        head_seconds=timings,head_seconds_total=sum(timings),latency_scope='Cached-input tiny head only; excludes capture, MediaPipe, RGB, DTW and web',
        new_model_imports=0,new_model_forwards=0,new_training=0,new_video_decodes=0,old_result_rescoring=0,
        independent_human_or_KSL_expert_validation=False,
        limitations=['Same Codex system performing a separate saved-value verification, not independent sign correctness gold.',
                    'Four edited clips from one previously exposed registered ID12; five occurrence queries have no new gold labels.',
                    'All five high observations are not five correct educational decisions.',
                    'RGB support is structural recipe provenance; original window vectors were not saved.',
                    'Legacy R fixed scopes differ from current AM target0 source scope.'],whole_goal_complete=False)
    write(destination,report);print(json.dumps({k:report[k] for k in ['status','checks_count','summary','native_rows','head_seconds_total']},ensure_ascii=False))

def verify_composition():
    result_path=U/'COMPOSITION_RESULT.json';destination=U/'SAVED_COMPOSITION_VERIFIED.json'
    if not result_path.exists():print('Composition not complete; no premature verification');return
    if destination.exists():
        if read(destination)['result_sha256']!=sha(result_path):raise ValueError('Verified composition changed')
        print('Existing composition verification retained');return
    a=Audit();result=read(result_path);frozen=read(U/'COMPOSITION_FROZEN.json');a.pins(frozen['pins'])
    recovery=read(U/'COMPOSITION_RECOVERY.json');resumed=read(U/'COMPOSITION_RESUME_FROZEN.json')
    a.pins(recovery['preserved_output_pins']);a.pins(resumed['pins'])
    a.equal('resume keeps original sixteen frozen inputs',resumed['cases'],frozen['cases'])
    a.equal('exact seven prior outputs preserved',len(recovery['preserved_output_pins']),7)
    a.equal('nine pending outputs only',recovery['pending_only'],9)
    a.test('recovery did not change scores or run models',recovery['model_reruns']==0 and recovery['old_scores_changed'] is False)
    a.test('first failure remains recorded',all(Path(p).is_file() for p in recovery['first_failure_files']))
    a.equal('prior reverse verifier implementation retained',sha(U/'verify_saved_before_no_observation.py'),read(U/'SAVED_REVERSE_VERIFIED.json')['verifier_sha256'])
    a.equal('composition finish hash',read(U/'COMPOSITION_FINISHED.json')['result_sha256'],sha(result_path))
    aa=read(AA/'SAVED_CASES_MANIFEST.json');aj=read(AJ/'RESULT.json');rev=read(U/'REVERSE_RESULT.json')
    locals_={r['case_id']:r for r in aj['rows']+rev['rows']};frozen_cases={r['case_id']:r for r in frozen['cases']}
    a.equal('fixed16 unique cases',len(result['rows']),16);a.equal('exact case set',sorted(r['case_id'] for r in result['rows']),sorted(frozen_cases))
    derived=[];no_observation=[]
    def high(obs,cid):
        if 'high_confirmation' in obs:
            a.test(cid+':explicit high bool',type(obs['high_confirmation']) is bool)
            return obs['high_confirmation']
        a.test(cid+':missing high only explicit no-observation',
            obs['status']=='INSUFFICIENT_OBSERVATION' and obs['probability'] is None and
            obs['prediction_coverage']==0 and obs['anchor'] is None and
            obs['unavailable_native_frames']==obs['fixed_scope']['frame_ids'])
        return False
    for row in result['rows']:
        cid=row['case_id'];a.equal(cid+':resultsha',sha(row['result']['path']),row['result']['sha256'])
        out=read(row['result']['path']);identity=frozen_cases[cid]['identity'];base=read(identity['base_file']['path']);local=locals_[cid]
        preserved=str(Path(row['result']['path']).resolve()) in recovery['preserved_output_pins']
        if preserved:
            a.equal(cid+':preserved output has no fabricated new timing',row['composition_seconds'],None)
        else:
            a.test(cid+':new composition timing finite',math.isfinite(row['composition_seconds']) and row['composition_seconds']>=0)
        original=read(aa['cases'][row['word_id']][row['kind']]['result']);payload=out['combined_evidence_feedback']
        a.equal(cid+':whole old base preserved',{k:v for k,v in out.items() if k!='combined_evidence_feedback'},base)
        a.equal(cid+':old AA field preservation',{k:v for k,v in base.items() if k!='guide_feature_comparison'},original)
        a.equal(cid+':identity',payload['identity'],identity)
        a.equal(cid+':base value hash',digest(base),identity['base_value_sha256']);a.equal(cid+':local value hash',digest(local),identity['local_value_sha256'])
        a.equal(cid+':all source comparisons exact',payload['source_comparisons'],base['guide_feature_comparison']['items'])
        a.equal(cid+':all order items exact',payload['partial_order_items'],base['partial_order_feedback']['items'])
        a.equal(cid+':all local scopes exact',[e['motion'] for e in payload['motion_events']],local['observations'])
        a.equal(cid+':global image gaps',payload['global_image_unavailable_frames'],local['image_unavailable_frames'])
        ids=identity['native_frame_ids'];ts=identity['timestamp_us'];position={v:i for i,v in enumerate(ids)}
        a.equal(cid+':denominator',payload['native_denominator'],len(ids));a.equal(cid+':source hash',out['source_sha256'],identity['video_sha256'])
        a.equal(cid+':local timeline',local['source_ledger'][identity['local_timestamp_key']],ts)
        with np.load(identity['analysis_path'],allow_pickle=False) as arrays:
            a.equal(cid+':source native axis',arrays['frame_id'].tolist(),ids);a.equal(cid+':source PTS',arrays['timestamp_us'].tolist(),ts)
        for item in payload['source_comparisons']:
            for point in item.get('native_points',[]):
                a.test(cid+':source point original time',point['frame'] in position and abs(point['seconds']-ts[position[point['frame']]]/1e6)<1e-9)
        orders=payload['partial_order_items']
        for item in orders:
            for key in ('first_evidence','second_evidence'):
                evidence=item[key];a.equal(cid+':order evidence time',evidence['native_seconds'],[ts[i]/1e6 for i in evidence['input_frames']])
        for event,obs in zip(payload['motion_events'],local['observations']):
            linked=[i for i,order in enumerate(orders) if obs['guide_id'] in order['first']['guide_ids']+order['second']['guide_ids']]
            a.equal(cid+':guide identity link only',event['related_order_item_indices'],linked)
            flags=[]
            if high(obs,cid):flags.append('MOVEMENT_EVIDENCE_PRESENT')
            if 'high_confirmation' not in obs:
                no_observation.append(dict(case_id=cid,target_index=obs['target_index'],native_frames=obs['fixed_scope']['frame_ids'],
                    interpretation='No high-evidence contribution, not a negative movement or educational truth'))
                a.test(cid+':missing-high original schema preserved','high_confirmation' not in event['motion'])
            if obs['unavailable_native_frames']:flags.append('OBSERVATION_GAP_RETAINED')
            statuses={orders[i]['observation_status'] for i in linked}
            if 'OBSERVED_B_BEFORE_A' in statuses:flags.append('RELATED_PAIR_OBSERVED_IN_REVERSE_ORDER')
            if statuses-{'OBSERVED_A_BEFORE_B','OBSERVED_B_BEFORE_A'}:flags.append('RELATED_PAIR_ORDER_UNRESOLVED')
            if not linked:flags.append('NO_DISTINCT_PAIR_QUERY')
            a.equal(cid+':factor flags',event['flags'],flags)
            for key in ('whole_action_complete','missing_action','educational_error'):a.equal(cid+':event unknown '+key,event[key],None)
            a.test(cid+':no event correction or exact boundaries',event['correction_enabled'] is False and event['exact_event_boundary'] is False)
            ff=obs['fixed_scope']['frame_ids'];a.equal(cid+':motion times',obs['fixed_scope']['timestamp_us'],[ts[position[f]] for f in ff])
            a.equal(cid+':no scope trim',ff,ids[position[ff[0]]:position[ff[-1]]+1])
            a.test(cid+':association not same occurrence','distinct scopes' in event['link_semantics'])
        for key in ('normality','whole_sign_correctness'):a.equal(cid+':no new truth '+key,payload[key],None)
        for key in ('all_requirements_verified','wrong_handshape_in_applicable_context_validated','correction_enabled'):a.equal(cid+':no promotion '+key,payload[key],False)
        counts=dict(source_point_states=dict(Counter(p['observation'] for item in payload['source_comparisons'] for p in item.get('native_points',[]))),
                    pair_observations=dict(Counter(i['observation_status'] for i in orders)),
                    local_high=sum(high(e['motion'],cid) for e in payload['motion_events']),
                    local_scopes=len(payload['motion_events']),local_scopes_with_gaps=sum(bool(e['motion']['unavailable_native_frames']) for e in payload['motion_events']),
                    global_image_unavailable_frames=len(payload['global_image_unavailable_frames']),
                    factor_flags=dict(Counter(f for e in payload['motion_events'] for f in e['flags'])))
        for key,value in counts.items():a.equal(cid+':saved summary '+key,row[key],value)
        a.equal(cid+':accuracy unclaimed',row['accuracy_gain'],None);a.equal(cid+':whole truth unknown',row['whole_sign_correctness'],None)
        derived.append(dict(kind=row['kind'],**counts))
    summaries=[]
    for kind in ('normal','cut','gap','order'):
        chosen=[r for r in derived if r['kind']==kind];orders=Counter()
        for r in chosen:orders.update(r['pair_observations'])
        summaries.append(dict(kind=kind,clips=len(chosen),local_high=sum(r['local_high'] for r in chosen),local_scopes=sum(r['local_scopes'] for r in chosen),
            order_counts=dict(orders),local_scopes_with_gaps=sum(r['local_scopes_with_gaps'] for r in chosen),whole_sign_error_truth=None))
    a.equal('independent composition summaries',result['summary'],summaries)
    a.equal('AJ speed summary unmodified',result['unchanged_AJ_speed_summary'],[r for r in aj['summary'] if r['kind'] in ['speed_0.8','speed_1.25','trim_initial_rest']])
    a.test('zero new composition inference',all(result[k]==0 for k in ['model_forwards','new_training','video_decodes']) and result['old_scores_recomputed'] is False)
    a.test('educational improvement not claimed',result['educational_accuracy_gain_established'] is False and result['current_AM_scope_fidelity'] is False)
    write(destination,dict(status='PASS_SAVED_COMPOSITION_PRESERVATION_AND_TIME_BINDING',checks_count=len(a.checks),checks=a.checks,failed_checks=[],
        result_sha256=sha(result_path),verifier_sha256=sha(Path(__file__)),summary=summaries,new_educational_gold=0,
        first_outputs_byte_preserved=7,pending_outputs_completed=9,explicit_no_observation_rows=no_observation,
        new_model_imports=0,new_model_forwards=0,new_training=0,independent_human_or_KSL_expert_validation=False,
        limitations=['Same-system separate saved-value checks; no independent sign correctness validation.',
                    'Sixteen edited displays are four source clips from one exposed public ID12, not sixteen independent users.',
                    'All old source, order, handshape and local movement observations retain their own scopes; shared guide IDs do not establish identical occurrences.'],whole_goal_complete=False))
    print(json.dumps({'status':'PASS_SAVED_COMPOSITION','checks_count':len(a.checks),'summary':summaries},ensure_ascii=False))

if __name__=='__main__':
    try:
        if '--composition-only' not in sys.argv:verify_reverse()
        verify_composition()
    except BaseException:
        write(U/('VERIFIER_FIRST_OR_LATER_FAILURE_'+str(time.time_ns())+'.json'),dict(error=traceback.format_exc(),verifier_sha256=sha(Path(__file__)),model_forwards=0))
        raise
