"""Independent arithmetic over saved local-event outputs; never import a model."""
from pathlib import Path
from bisect import bisect_left
from collections import Counter
import hashlib, json, math
import numpy as np

U=Path(__file__).resolve().parent
O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def rgb_support(ts,available):
    n=len(ts)
    starts=list(range(0,n-15,2))
    if starts[-1]!=n-16:starts.append(n-16)
    centers=[int(ts[s])+int(ts[s+15]) for s in starts]
    result=[]
    for i,t in enumerate(ts):
        z=2*int(t)
        if z<=centers[0]:used=[0]
        elif z>=centers[-1]:used=[len(centers)-1]
        else:
            j=bisect_left(centers,z)
            used=[j] if centers[j]==z else [j-1,j]
        result.append(sorted({k for j in used for k in range(starts[j],starts[j]+16)}) if available[i] else [])
    return result

def main():
    if (U/'SAVED_VERIFIED.json').exists():raise FileExistsError('Saved verification must not be overwritten')
    d=read(U/'DATASET_FROZEN.json');e=read(U/'EVAL_RESULT.json');tr=read(U/'TRAIN_RESULT.json')
    ex=read(U/'EXECUTION_FROZEN.json');summary=read(U/'RESULT.json')
    checks=[];pins={};max_sigmoid_error=0.0
    def ck(name,ok):checks.append({'name':name,'passed':bool(ok)})
    def equal(name,a,b):
        if isinstance(a,float) or isinstance(b,float):ck(name,a is not None and b is not None and math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12))
        else:ck(name,a==b)
    for name in ['DATASET_FROZEN.json','EVAL_RESULT.json','TRAIN_RESULT.json','EXECUTION_FROZEN.json','RESULT.json','RESULT_KO.txt','TRAIN_STARTED.json','TRAIN_FINISHED.json','EVAL_STARTED.json','EVAL_FINISHED.json','REVIEW_CODE_v2.json','DATA_REVIEW.json']:
        pins[str(U/name)]=sha(U/name)
    for path,h in ex['pins'].items():ck('execution_pin:'+path,sha(path)==h)
    for path,h in summary['pins'].items():ck('summary_pin:'+path,sha(path)==h)
    tf=read(U/'TRAIN_FINISHED.json');ef=read(U/'EVAL_FINISHED.json');es=read(U/'EVAL_STARTED.json')
    ck('training_finished_hash',tf['status']=='COMPLETE' and tf['result_sha256']==sha(U/'TRAIN_RESULT.json'))
    ck('evaluation_finished_hash',ef['status']=='COMPLETE' and ef['result_sha256']==sha(U/'EVAL_RESULT.json'))
    ck('model_sha_lineage',sha(tr['model'])==tr['model_sha256']==es['model_sha256'])
    pins[tr['model']]=sha(tr['model'])
    ck('original_checkpoint_preserved',sha(ex['initialization'])==ex['initialization_sha256']==d['pretrained_checkpoint']['sha256'] and tr['original_initialization_unchanged'] is True)
    ck('old_backbone_dev_user_flags',not tr['backbone_updated'] and not tr['dev_used_in_optimizer'] and not tr['user_or_mentor_data'] and not e['backbone_recomputed'] and e['reference_video_forwards']==0)
    ck('fixed_budget',tr['seed']==ex['seed']==42 and tr['epochs']==ex['epochs']==80 and tr['steps']==ex['steps']==d['training_steps']==320)
    ck('history80x4',len(tr['history'])==80 and all(r['epoch']==i+1 and r['steps']==4*(i+1) and math.isfinite(r['mean_loss']) for i,r in enumerate(tr['history'])))
    ck('finite_training_seconds',math.isfinite(tr['seconds']) and tr['seconds']>0)
    ck('parameter_count_from_shapes',sum(math.prod(s) for s in ex['trainable_parameters'].values())==ex['parameter_count']==39013)
    ck('optimizer_contract',ex['optimizer']=='AdamW' and ex['learning_rate']==.0005 and ex['weight_decay']==.01 and ex['clipnorm']==2 and ex['group_batch']==12)
    gs=[g for g in d['groups'] if g['group_id'] in d['training_group_ids']]
    ck('fit45groups_only',len(gs)==45 and all(g['role']=='fit' and 1<=g['person']<=9 for g in gs) and ex['training_group_ids']==[g['group_id'] for g in gs])
    counts={f'{t}:{y}':sum(any(b['train_eligible'] and b['label']==y for b in g['bags']) for g in gs if g['target_index']==t) for t in d['supported_targets'] for y in [0,1]}
    equal('class_group_counts',ex['class_group_counts'],counts)
    for key,n in counts.items():equal('class_weight:'+key,ex['class_weights'][key],len(gs)/(2*len(d['supported_targets'])*n))
    ck('fixed_thresholds',e['raw']==ex['raw_threshold']==.5 and e['confirmed']==ex['confirmed_threshold']==.8 and e['low']==ex['low_threshold']==.2)
    ck('supported_masks',ex['supported_targets']==d['supported_targets']==[0,1,2,3,4])
    rows={r['clip_id']:r for r in d['rows']};groups={g['group_id']:g for g in d['groups']}
    ck('48sources60groups',len(rows)==48 and len(e['rows'])==len(groups)==60 and len({r['group_id'] for r in e['rows']})==60)
    saved={};support={};prediction_paths={}
    for r in e['rows']:
        cid=r['clip_id'];src=rows[cid];g=groups[r['group_id']]
        ck(r['group_id']+':identity',all(r[k]==g[k] for k in ['clip_id','role','person','target_index','source_sha256']) and r['source_sha256']==src['source_sha256'])
        path=Path(r['prediction']['path']);h=r['prediction']['sha256']
        ck(r['group_id']+':prediction_file_binding',path.resolve()==(U/'predictions'/(cid+'.npz')).resolve() and sha(path)==h)
        if cid in saved:
            ck(r['group_id']+':shared_one_prediction',prediction_paths[cid]==(str(path),h));continue
        with np.load(path,allow_pickle=False) as z:saved[cid]={k:z[k].copy() for k in z.files}
        a=saved[cid];prediction_paths[cid]=(str(path),h);pins[str(path)]=h;n=len(src['native_frame_ids'])
        ck(cid+':native_axes',a['frame_id'].tolist()==src['native_frame_ids'] and a['timestamp_us'].tolist()==src['timestamp_us'])
        ck(cid+':shapes_finite',a['logits'].shape==a['probabilities'].shape==(n,5) and np.isfinite(a['logits']).all() and np.isfinite(a['probabilities']).all() and ((a['probabilities']>=0)&(a['probabilities']<=1)).all())
        expected=np.exp(-np.logaddexp(0,-a['logits'].astype(np.float64)))
        err=float(np.max(np.abs(expected-a['probabilities'])))
        max_sigmoid_error=max(max_sigmoid_error,err);ck(cid+':sigmoid',err<=2e-7)
        ck(cid+':availability_masks',a['motion_anchor'].dtype.kind=='b' and a['evidence_available'].dtype.kind=='b' and a['rgb_available'].dtype.kind=='b' and a['motion_anchor'].tolist()==src['motion_anchor'] and a['evidence_available'].tolist()==src['evidence_available'] and bool(np.all(~a['rgb_available']|a['evidence_available'])))
        ck(cid+':masked_logits',bool((a['logits'][~a['motion_anchor']]==0).all()) and bool((a['probabilities'][~a['motion_anchor']]==.5).all()))
        support[cid]=rgb_support(a['timestamp_us'],a['rgb_available'])
    ck('unique_prediction_clips48',len(saved)==48 and e['whole_sequence_forwards']==len(e['timing'])==es['clips']==48)
    ck('exact_prediction_file_set',{p.resolve() for p in (U/'predictions').glob('*.npz')}=={Path(p).resolve() for p,h in prediction_paths.values()})
    regraded={};support_details=[]
    def verify_bag(cid,target,b,out,is_broad=False):
        a=saved[cid];lo,hi=b['indices'];scope=np.arange(lo,hi+1);anchors=a['motion_anchor'];use=scope[anchors[scope]];complete=bool(anchors[scope].all());supported=target in d['supported_targets']
        pref=out['bag_id']; equal(pref+':kind',out['kind'],'broad_consistency' if is_broad else b['kind'])
        equal(pref+':label',out['label'],b['label']);equal(pref+':frames',out['frames'],b['frames']);equal(pref+':indices',out['indices'],b['indices'])
        equal(pref+':timestamps',out['timestamp_us'],[int(a['timestamp_us'][lo]),int(a['timestamp_us'][hi])]);equal(pref+':supported',out['supported'],supported)
        equal(pref+':labelled_evidence',out['labelled_evidence_complete'],b['evidence_complete'])
        equal(pref+':coverage',out['prediction_coverage'],float(anchors[scope].mean()))
        equal(pref+':unavailable',out['unavailable_native_frames'],a['frame_id'][scope[~anchors[scope]]].tolist())
        if not supported or not len(use):
            ck(pref+':no_evidence_result',out['probability'] is None and out['anchor'] is None and out['status']==('UNSUPPORTED_QUERY' if not supported else 'INSUFFICIENT_OBSERVATION'))
            return {'probability':None,'high':False,'raw':False,'anchor':None}
        at=int(use[np.argmax(a['probabilities'][use,target])]);p=float(a['probabilities'][at,target]);high=p>=.8
        status=('MOVEMENT_EVIDENCE' if complete else 'MOVEMENT_EVIDENCE_WITH_GAPS') if high else ('INSUFFICIENT_OBSERVATION' if not complete else 'LOW_MOVEMENT_SCORE' if p<=.2 else 'UNCONFIRMED')
        for key,value in [('probability',p),('anchor',at),('anchor_frame',int(a['frame_id'][at])),('anchor_timestamp_us',int(a['timestamp_us'][at])),('raw_positive',p>=.5),('high_confirmation',high),('status',status),('absence_assessable',complete)]:equal(pref+':'+key,out[key],value)
        ck(pref+':no_whole_correctness',all(out[k] is None for k in ['whole_action_complete','whole_action_error','educational_error']))
        left=right=at
        while left>0 and anchors[left-1]:left-=1
        while right+1<len(anchors) and anchors[right+1]:right+=1
        native=list(range(max(left,at-7),min(right,at+7)+1));rgb=sorted({j for i in native for j in support[cid][i]})
        ctx=out['context_support']
        ck(pref+':exact_composed_support',ctx['temporal_native_rows']==native and ctx['rgb_native_rows']==rgb and ctx['rgb_native_timestamps_us']==[int(a['timestamp_us'][i]) for i in rgb] and ctx['structural_support_only'] is True and ctx['exact_onset_or_observed_event_boundary'] is False)
        return {'probability':p,'high':high,'raw':p>=.5,'anchor':at}
    for r in e['rows']:
        g=groups[r['group_id']];cid=r['clip_id'];t=r['target_index'];a=saved[cid]
        ck(g['group_id']+':bag_count',len(r['bags'])==len(g['bags'])==2)
        vals=[];results=[]
        for b,out in zip(g['bags'],r['bags']):
            equal(g['group_id']+':bag_id',b['bag_id'],out['bag_id']);v=verify_bag(cid,t,b,out);results.append(v)
            if v['probability'] is not None:vals.append(v['probability'])
        n=len(a['frame_id']);broad={'indices':[0,n-1],'frames':[int(a['frame_id'][0]),int(a['frame_id'][-1])],'label':1,'evidence_complete':bool(a['motion_anchor'].all())}
        full=verify_bag(cid,t,broad,r['broad'],True)
        positives=[b['indices'] for b in g['bags'] if b['label']==1 and b['evidence_complete']]
        inside=None if full['anchor'] is None else any(lo<=full['anchor']<=hi for lo,hi in positives)
        monotonic=full['probability'] is not None and all(full['probability']>=p for p in vals)
        equal(g['group_id']+':broad_positive_union',r['broad']['positive_bracket_union'],positives)
        equal(g['group_id']+':broad_anchor_in_positive',r['broad']['anchor_inside_reviewed_positive_bracket'],inside)
        equal(g['group_id']+':same_output_monotonic',r['broad']['same_score_broad_monotonic'],monotonic)
        ck(g['group_id']+':no_group_semantics',r['whole_clause_correctness'] is None and r['educational_error'] is None)
        regraded[g['group_id']]={'bags':results,'broad':full,'inside':inside,'monotonic':monotonic}
    calculated=[];full_role_counts={}
    for role in ['fit','development']:
        items=[(g,b,regraded[g['group_id']]['bags'][i]) for g in d['groups'] if g['role']==role for i,b in enumerate(g['bags'])]
        full_role_counts[role]={}
        for y in [1,0,None]:
            ii=[v for g,b,v in items if b['label']==y]
            full_role_counts[role][str(y)]={'denominator':len(ii),'raw':sum(v['raw'] for v in ii),'high':sum(v['high'] for v in ii),'unavailable':sum(v['probability'] is None for v in ii)}
    for t in d['supported_targets']:
        selected=[g for g in d['groups'] if g['role']=='development' and g['target_index']==t]
        native=[regraded[g['group_id']]['bags'][i] for g in selected for i,b in enumerate(g['bags']) if b['kind']=='weak_positive_bracket']
        local=[(b,regraded[g['group_id']]['bags'][i]) for g in selected for i,b in enumerate(g['bags']) if b['kind']=='reviewed_local']
        pos=[v for b,v in local if b['label']==1];neg=[v for b,v in local if b['label']==0];unknown=[v for b,v in local if b['label'] is None]
        metric={'target_index':t,'native_positive_denominator':len(native),'native_positive_high':sum(v['high'] for v in native),'native_positive_raw':sum(v['raw'] for v in native),'local_positive_denominator':len(pos),'local_positive_high':sum(v['high'] for v in pos),'local_negative_denominator':len(neg),'local_negative_high_false_positives':sum(v['high'] for v in neg),'local_unknown_denominator':len(unknown),'unavailable_bag_denominator':sum(v['probability'] is None for v in native+[v for b,v in local]),'broad_anchor_in_reviewed_positive':sum(regraded[g['group_id']]['inside'] is True for g in selected),'broad_anchor_denominator':len(selected),'broad_monotonic':sum(regraded[g['group_id']]['monotonic'] for g in selected),'prototype_research_gate':len(native)==3 and sum(v['high'] for v in native)>=2 and len(neg)>0 and sum(v['high'] for v in neg)==0}
        calculated.append(metric)
    equal('summary_per_target',summary['development']['per_target'],calculated)
    equal('summary_gate',summary['gate_pass'],all(m['prototype_research_gate'] for m in calculated))
    ck('dev_ids_and_scoped_claim',summary['development']['registered_ids']==[11,12,13] and summary['development']['previously_exposed'] is True and summary['development']['clips']==12 and not summary['live_demo_changed'] and not summary['whole_goal_complete'])
    ck('training_summary',summary['training']['steps']==tr['steps'] and summary['training']['epochs']==80 and summary['training']['parameters']==39013 and summary['training']['fit_clips']==36 and summary['training']['registered_fit_ids']==list(range(1,10)) and summary['training']['geometry_projection_finetuned']==tr['geometry_projection_changed'] and summary['training']['rgb_projection_finetuned']==tr['rgb_projection_changed'] and not summary['training']['backbone_finetuned'])
    times=np.array([r['seconds'] for r in e['timing']],float)
    ck('timing48_valid_axes',len(times)==48 and np.isfinite(times).all() and bool((times>=0).all()) and {r['clip_id'] for r in e['timing']}==set(rows) and all(r['frames']==len(rows[r['clip_id']]['native_frame_ids']) and r['role']==rows[r['clip_id']]['role'] for r in e['timing']))
    for key,v in {'median':np.median(times),'p95':np.percentile(times,95),'first':times[0],'total':times.sum()}.items():equal('head_latency:'+key,summary['cached_head_latency_seconds'][key],float(v))
    ck('latency_scope',summary['latency_scope']==e['latency_scope'] and 'excludes' in e['latency_scope'] and 'SignRep' in e['latency_scope'])
    agpath=O/'word100_context_episode_query_20261103/RESULT.json';ag=read(agpath)
    ck('AG_baseline_pin',e['baseline']['sha256']==sha(agpath) and 'net-direction' in e['baseline']['task'])
    prior=summary['saved_prior_scope_diagnostic'];ck('six_prior_scopes',len(prior)==len(ag['rows'])==6)
    for old,new in zip(ag['rows'],prior):
        a=saved[old['clip_id']];lo,hi=old['frames'];ix=np.flatnonzero((a['frame_id']>=lo)&(a['frame_id']<=hi)&a['motion_anchor']);at=int(ix[np.argmax(a['probabilities'][ix,0])])
        ck('AG_scope:'+old['clip_id']+old['scope'],new['clip_id']==old['clip_id'] and new['role']=='fit' and new['scope']==old['scope'] and new['frames']==old['frames'] and new['previous_net_direction_probability']==old['observation']['probability'] and new['new_local_occurrence_probability']==float(a['probabilities'][at,0]) and new['new_anchor_frame']==int(a['frame_id'][at]) and new['same_old_weights_rerun'] is False and 'different trained question' in new['limitation'] and 'not a fair new-person accuracy comparison' in new['limitation'])
    failure=U/'TRAIN_FAILURE_1793636906846130500.txt';initial=U/'train_local_initial.py'
    err=failure.read_text(encoding='utf-8');oldcode=initial.read_text(encoding='utf-8')
    guard=oldcode.index("dr['approved_for_one_training']")
    ck('first_failure_before_model_load_optimizer',"KeyError: 'approved_for_one_training'" in err and guard<oldcode.index('inputs=load_inputs')<oldcode.index('model=LocalEventNet()')<oldcode.index('pretrained=torch.load')<oldcode.index("write(U/'TRAIN_STARTED.json'")<oldcode.index('opt=torch.optim.AdamW'))
    first_failure={'path':str(failure),'sha256':sha(failure),'classification':'pre-training approval-field schema KeyError, before input preparation, checkpoint load, TRAIN_STARTED and optimizer creation','model_loads_at_failure':0,'optimizer_steps_at_failure':0,'not_second_training_candidate':True}
    failures=[c for c in checks if not c['passed']]
    out={'status':'PASS_SAVED_ARITHMETIC_AND_LINEAGE' if not failures else 'FAIL_SAVED_ARITHMETIC_OR_LINEAGE','checks_count':len(checks),'failed_checks':failures,'per_target_development':calculated,'fit_development_all_bag_counts':full_role_counts,'native_forward_clips':len(saved),'source_query_groups':len(e['rows']),'fixed_bag_queries':sum(len(g['bags']) for g in d['groups']),'full_native_queries':len(d['groups']),'max_sigmoid_absolute_error':max_sigmoid_error,'gate_pass':all(m['prototype_research_gate'] for m in calculated),'first_failure':first_failure,'original_checkpoint_hash_preserved':sha(ex['initialization'])==ex['initialization_sha256'],'cached_head_latency_seconds':summary['cached_head_latency_seconds'],'claims_review':{'AG_comparison_different_task_FIT_only':True,'broad_argmax_inside_positive_is_not_exact_timing':True,'whole_clause_or_education_truth_available':False,'live_demo_changed':False,'whole_goal_complete':False},'limitations':['Independent arithmetic over saved outputs, not independent human/KSL correctness or newly measured generalization.','RGB availability flags are from the saved output; input provenance and actual availability were independently checked in DATA_REVIEW. This verifier reconstructs exact structural TCN15 plus RGB16/stride2/interpolation support from native time and saved flags.','Checkpoint files are hashed only, never deserialized. Projection-changed flags are trainer-reported values; this verifier does not independently inspect weight tensors.','One completed80-epoch/320-step history and frozen executable lineage verified; no rerun, threshold selection, model import or forward was performed.','High score in an unknown labelled scope remains an unlabelled model observation, never recoded as true or false.'],'execution':{'model_imports':0,'checkpoint_loads':0,'model_forwards':0,'training':0,'video_decodes':0,'visual_reviews':0,'label_changes':0},'pins':{**pins,str(agpath):sha(agpath),str(failure):sha(failure),str(initial):sha(initial),str(Path(__file__)):sha(__file__)}}
    with (U/'SAVED_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':out['status'],'checks':len(checks),'failures':failures,'per_target':calculated,'all_bag_counts':full_role_counts,'sha256':sha(U/'SAVED_VERIFIED.json')},ensure_ascii=False))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
