"""Independent saved-value arithmetic and lineage verification; zero inference.

Does not import torch, process_head, the evaluator, or any target compiler. Run
only once the frozen candidate's PROCESS_EVALUATION.json is complete.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,math,traceback
import numpy as np

A=Path(__file__).resolve().parent
DEST=A/'PROCESS_EVALUATION_SAVED_VERIFIED.json'
CHECKS=Counter();HASHES={}
QUERIES=('right_hand_toward_left_wrist_along_arm','hands_move_apart',
         'right_hand_rises_relative_left','right_hand_descends_face_to_torso','right_hand_falls_relative_left')

def check(test,message,category='schema'):
    CHECKS[category]+=1
    if not test:raise AssertionError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    p=Path(path).resolve()
    if str(p) not in HASHES:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        HASHES[str(p)]=h.hexdigest()
    return HASHES[str(p)]

def display(p):
    if p<=.2:return 0
    if p>=.8:return 1
    return None

def independent_summary(rows):
    count=Counter(str(r['truth']) for r in rows)
    cells={(truth,pred):0 for truth in (0,1) for pred in (0,1)}
    confidence={(truth,pred):0 for truth in (0,1) for pred in (0,1,None)}
    unknown_confident=0
    for r in rows:
        y,p=r['truth'],r['probability'];d=display(p)
        if y is None:
            unknown_confident+=d is not None;continue
        cells[(y,int(p>=.5))]+=1;confidence[(y,d)]+=1
    per={}
    for y in (0,1):
        total=cells[(y,0)]+cells[(y,1)]
        per[str(y)]={'total':total,'raw_correct':cells[(y,y)],
          'confident_correct':confidence[(y,y)],'confident_wrong':confidence[(y,1-y)],'abstained':confidence[(y,None)]}
    known=sum(v['total'] for v in per.values());correct=cells[(0,0)]+cells[(1,1)]
    ba=(per['0']['raw_correct']/per['0']['total']+per['1']['raw_correct']/per['1']['total'])/2 if all(per[str(y)]['total'] for y in (0,1)) else None
    return {'counts':dict(count),'known':known,'raw_correct':correct,'raw_accuracy':correct/known if known else None,
      'class_counts':per,'raw_balanced_accuracy':ba,'unknown_raw_confident':unknown_confident}

def same(expected,actual,path):
    if isinstance(expected,dict):
        check(isinstance(actual,dict),path+' type','arithmetic')
        for k,v in expected.items():
            check(k in actual,path+'/'+k+' exists','arithmetic');same(v,actual[k],path+'/'+k)
    elif isinstance(expected,float):
        check(isinstance(actual,(float,int)) and math.isfinite(actual) and abs(actual-expected)<=1e-12,path,'arithmetic')
    else:check(actual==expected,path,'arithmetic')

def group_macro(rows):
    by={qid:independent_summary([r for r in rows if r['query_id']==qid]) for qid in sorted({r['query_id'] for r in rows})}
    ba=[d['raw_balanced_accuracy'] for d in by.values() if d['raw_balanced_accuracy'] is not None]
    acc=[d['raw_accuracy'] for d in by.values() if d['raw_accuracy'] is not None]
    cc={str(y):[] for y in (0,1)};coverage={str(y):[] for y in (0,1)}
    for d in by.values():
        for y in ('0','1'):
            c=d['class_counts'][y]
            if c['total']:
                cc[y].append(c['confident_correct']/c['total'])
                coverage[y].append((c['confident_correct']+c['confident_wrong'])/c['total'])
    return {'source_queries':len(by),'source_queries_with_both_known_classes':len(ba),
      'macro_source_query_raw_balanced_accuracy':sum(ba)/len(ba) if ba else None,
      'macro_source_query_raw_accuracy':sum(acc)/len(acc) if acc else None,
      'macro_source_query_confident_correct_recall':{y:sum(v)/len(v) if v else None for y,v in cc.items()},
      'macro_source_query_confident_coverage':{y:sum(v)/len(v) if v else None for y,v in coverage.items()},
      'per_source_query':by}

def probability(p,where):
    check(type(p) in (int,float) and math.isfinite(p) and 0<=p<=1,where,'probability')

def main():
    check(not DEST.exists(),'Preserve completed saved-value verification')
    evaluation_path=A/'PROCESS_EVALUATION.json'
    check(evaluation_path.exists(),'Evaluation must exist before this verifier runs')
    ev=read(evaluation_path);contract_path=A/'PROCESS_COMPARISON_FROZEN.json'
    contract=read(contract_path);train=read(A/'PROCESS_TRAIN_RESULT.json')
    check(ev['status']=='ONE_FIXED_RAW_EVALUATION_SAVED_NOT_INTEGRATION_APPROVAL','Saved evaluation status')
    check(ev['contract_sha256']==sha(contract_path)==train['contract_sha256'],'Frozen contract lineage','hash')
    check(ev['checkpoint_sha256']==train['checkpoint_sha256']==sha(train['checkpoint']),'Checkpoint hash','hash')
    check(sha(train['normalization'])==train['normalization_sha256'],'Normalization hash','hash')
    for p,h in contract['pins'].items():check(sha(p)==h,p,'hash')
    check(train['epochs']==80 and train['seed']==42 and train['candidates']==1,'Frozen one-candidate training')
    check(train['development_used_for_optimization'] is False and train['unknown_as_negative'] is False,'Saved training scope')
    check(len(train['history'])==80 and [h['epoch'] for h in train['history']]==list(range(1,81)),'Training history coverage')
    check(all(math.isfinite(h['training_loss']) for h in train['history']),'Finite saved training loss')

    truth={};inputs={};role_targets={}
    for role,tn,mn,expected,people in [('fit','PROCESS_VIEW_TARGETS.json','PROCESS_INPUT_MANIFEST.json',217,set(range(1,10))),
          ('development','DEV_PROCESS_VIEW_TARGETS.json','DEV_PROCESS_INPUT_MANIFEST.json',80,{11,12,13})]:
        target=read(A/tn)['rows'];im=read(A/mn)['rows']
        check(len(target)==expected,'Target coverage '+role)
        role_targets[role]=target
        for r in target:
            check(r['role']==role and r['recorded_person_id'] in people,'Target split')
            check(r['view_id'] not in truth,'Unique target view');truth[r['view_id']]=r
        for i in im:
            check(i['role']==role and i['person'] in people,'Input split')
            check((role,i['pixel_view_key']) not in inputs,'Unique input key')
            inputs[(role,i['pixel_view_key'])]=i
    check(len(ev['rows'])==len(truth)==297 and {r['view_id'] for r in ev['rows']}==set(truth),'Evaluation denominator 297')
    check(len({r['view_id'] for r in ev['rows']})==297,'No duplicate stored evaluation rows')
    axis_cache={};rgb_normalization=[]
    for (role,key),record in inputs.items():
        check(sha(record['input'])==record['input_sha256'],'Input NPZ hash','hash')
        with np.load(record['input'],allow_pickle=False) as data:
            axis_cache[(role,key)]={k:data[k].copy() for k in ['frame_id','timestamp_us','image_available']}
            if role=='fit':rgb_normalization.append(data['rgb'][data['image_available']].copy())
    # Reconstruct normalization from FIT unique actual input views only.
    rgb=np.concatenate(rgb_normalization,axis=0);norm=read(train['normalization'])
    check(norm['frames']==len(rgb),'Normalization population','normalization')
    check(np.allclose(np.asarray(norm['mean']),rgb.mean(0,dtype=np.float64),atol=1e-12,rtol=1e-12),'FIT-only RGB mean','normalization')
    check(np.allclose(np.asarray(norm['std']),np.maximum(rgb.std(0,dtype=np.float64),.001),atol=1e-12,rtol=1e-12),'FIT-only RGB std','normalization')
    del rgb,rgb_normalization

    point_records={'setup':[],'terminal':[]};interval_records=[]
    for r in ev['rows']:
        target=truth[r['view_id']];fm=target['ordered_frame_map'];key=(r['role'],r['pixel_view_key'])
        inp=inputs[key];a=axis_cache[key]
        for k in ['query_id','clip_id','role','physical_query','pixel_view_key']:
            check(r[k]==target[k],'Stored identity '+k,'lineage')
        check(r['person']==target['recorded_person_id']==inp['person'] and inp['clip_id']==r['clip_id'],'Input person/clip','lineage')
        check(inp['source_sha256']==target['source_sha256'] and r['view_id'] in inp['aliases'],'Source/alias','lineage')
        check(r['truth']==target['observed_process_evidence'] and r['kind']==target['construction_kind_audit_only'],'No changed process truth','truth')
        check(r['frames']==len(fm)==len(a['frame_id']),'Input frame count','axis')
        check(a['frame_id'].tolist()==list(range(len(fm))),'Dense real output index','axis')
        check(r['source_frames']==[f['source_native_frame'] for f in fm]==inp['source_frame_indices'],'Source frames','axis')
        check(r['source_timestamps_us']==[f['source_native_timestamp_us'] for f in fm]==inp['source_timestamps_us'],'Source native PTS','axis')
        check(r['view_timestamps_us']==[f['view_timestamp_us'] for f in fm]==a['timestamp_us'].tolist(),'View PTS','axis')
        check(r['image_available']==[f['image_available'] for f in fm]==a['image_available'].tolist(),'Actual image mask','axis')
        probability(r['probability'],r['view_id'])
        check(r['raw_display']==display(r['probability']),'Confidence thresholds .8/.2','arithmetic')
        check(r['whole_word_correctness'] is None and r['learner_error'] is None,'No educational truth','scope')
        points=r['point_probabilities']
        check(len(points)==len(fm) and all(len(p)==2 for p in points),'Point probability shape')
        for i,pair in enumerate(points):
            for p in pair:probability(p,r['view_id']+'/point/'+str(i))
        check(len(r['sparse_point_scores'])==len(target['sparse_relation_targets']),'Sparse point denominator')
        for scored,t in zip(r['sparse_point_scores'],target['sparse_relation_targets']):
            check(scored['output_index']==t['output_index'] and scored['source_frame']==t['source_frame'],'Sparse point indices','truth')
            for dim,name in enumerate(['setup','terminal']):
                check(scored[name+'_truth']==t[name+'_relation'],'Unknown point truth preserved','truth')
                same(points[t['output_index']][dim],scored[name+'_probability'],'Point score equals saved dense output')
                point_records[name].append({'truth':scored[name+'_truth'],'probability':scored[name+'_probability'],
                  'query_id':r['query_id'],'role':r['role']})
        check(len(r['interval_scores'])==len(target['fully_retained_motion_interval_targets']),'Interval denominator')
        for scored,t in zip(r['interval_scores'],target['fully_retained_motion_interval_targets']):
            check(scored['source_frames']==t['source_frames'] and scored['output_indices']==t['output_indices']
                  and scored['truth']==t['label'],'Original interval scope/unknown preserved','truth')
            ii=t['output_indices'];sf=t['source_frames']
            check(sf==list(range(sf[0],sf[-1]+1)) and ii==list(range(ii[0],ii[-1]+1)),'Interval contiguity','truth')
            check(all(r['image_available'][i] for i in ii),'Interval does not label missing pixels','truth')
            probability(scored['probability'],'interval probability')
            interval_records.append({'truth':scored['truth'],'probability':scored['probability'],'query_id':r['query_id'],'role':r['role']})
        check(math.isfinite(r['inference_seconds']) and r['inference_seconds']>=0,'Finite diagnostic timing','timing')
        check(all(math.isfinite(r[k]) and r[k]>=0 for k in ['process_forward_seconds','annotation_interval_diagnostic_seconds']),
              'Finite separate forward timing','timing')
        check(r['process_forward_seconds']+r['annotation_interval_diagnostic_seconds']<=r['inference_seconds']+1e-9,
              'Separate timings fit total diagnostic duration','timing')
        check(r['timing_scope']=='Diagnostic head forwards including annotation-selected intervals; not runtime or end-to-end latency',
              'Explicit diagnostic-only timing scope','timing')

    summaries={};clusters={}
    for role in ('fit','development'):
        rows=[r for r in ev['rows'] if r['role']==role];stored=ev['summaries'][role]
        expected=independent_summary(rows);same(expected,stored['overall'],role+'/overall')
        query_summaries={q:independent_summary([r for r in rows if r['physical_query']==q]) for q in QUERIES}
        same(query_summaries,stored['per_query'],role+'/per_query')
        bas=[d['raw_balanced_accuracy'] for d in query_summaries.values() if d['raw_balanced_accuracy'] is not None]
        same(sum(bas)/len(bas) if bas else None,stored['macro_query_balanced_accuracy'],role+'/macro_query')
        for k in sorted({r['kind'] for r in rows}):
            same(independent_summary([r for r in rows if r['kind']==k]),stored['per_kind'][k],role+'/'+k)
        for p in sorted({r['person'] for r in rows}):
            same(independent_summary([r for r in rows if r['person']==p]),stored['per_person'][str(p)],role+'/person'+str(p))
        same(independent_summary([r for r in rows if r['frames']<16]),stored['short'],role+'/short')
        same(independent_summary([r for r in rows if r['frames']>=16]),stored['long'],role+'/long')
        clusters[role]=group_macro(rows)
        same(clusters[role]['per_source_query'],stored['per_source_query'],role+'/per_source_query')
        same(clusters[role]['macro_source_query_raw_balanced_accuracy'],stored['macro_source_query_balanced_accuracy'],role+'/macro_source_query')
        check(stored['source_query_count']==clusters[role]['source_queries']==(45 if role=='fit' else 15),
              'Source-query denominator','arithmetic')
        check(stored['independent_case_count_claim'] is False,'No independent-view count claim','scope')
        summaries[role]={'process':expected,'sparse_setup':independent_summary([r for r in point_records['setup'] if r['role']==role]),
          'sparse_terminal':independent_summary([r for r in point_records['terminal'] if r['role']==role]),
          'annotated_interval':independent_summary([r for r in interval_records if r['role']==role])}

    report={'at':datetime.now(timezone.utc).isoformat(),'status':'PASS_SAVED_VALUE_ARITHMETIC_AND_LINEAGE_NO_INFERENCE',
      'evaluation_sha256':sha(evaluation_path),'contract_sha256':sha(contract_path),'checkpoint_sha256':train['checkpoint_sha256'],
      'code_sha256':sha(__file__),'rows':297,'fit_views':217,'development_views':80,'fit_source_queries':45,'development_source_queries':15,
      'summaries_independently_recomputed':summaries,'source_query_clustered_diagnostics':clusters,
      'unknown_labels_preserved_and_excluded_from_known_accuracy':True,'unknown_still_in_total_and_raw_confidence_diagnostics':True,
      'actual_input_axes_masks_and_truth_exact':True,'normalization_reconstructed_from_fit_only':True,
      'runtime_latency_evaluated':False,'timing_scope':'Saved inference timing may include annotation-selected interval forwards and excludes feature extraction; never full application latency.',
      'automatic_path_qualification_not_validated_by_this_report':True,'raw_confidence_is_not_observation_qualification':True,
      'dependent_views_not_independent_people':True,'repeated_development':True,'independent_final_test':False,
      'independent_human_expert':False,'same_codex_separate_arithmetic_algorithm':True,
      'no_compiler_or_model_modules_imported':True,'new_model_inference':0,'new_training':0,
      'checks':dict(CHECKS),'checks_total':sum(CHECKS.values()),'unique_hashed_files':len(HASHES),'whole_goal_complete':False}
    with DEST.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':report['status'],'rows':297,'checks':report['checks_total'],'sha256':sha(DEST)},ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        error=A/('PROCESS_EVAL_VERIFY_FAILURE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.txt')
        with error.open('x',encoding='utf8') as f:f.write(traceback.format_exc())
        raise
