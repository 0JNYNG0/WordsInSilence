"""Same preregistered gates and paired baseline arithmetic; no models."""
from pathlib import Path
from collections import Counter
import json
import hashlib

U=Path(__file__).resolve().parent
AB=U.parent/'word100_action_completion_supervision_20261102'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=U/'DECISION.json';assert not out.exists()
    c=read(U/'ALIGNMENT_FROZEN.json');ev=read(U/'EVALUATION.json')
    verified=read(U/'EVALUATION_VERIFIED.json')
    assert verified['status'].startswith('PASS')
    assert verified['evaluation_sha256']==sha(U/'EVALUATION.json')
    for p,h in c['pins'].items():assert sha(p)==h,p
    old=read(AB/'PROCESS_EVALUATION.json');oldauto=read(AB/'PROCESS_AUTO_EVALUATION.json')
    base={r['view_id']:r for r in old['rows']}
    bq={r['view_id']:r for r in oldauto['view_rows']}
    ba={r['query_id']:r for r in oldauto['automatic_rows']}
    rows=ev['rows'];paired=[]
    for r in rows:
        if r['origin']=='reviewed_view':
            b=base[r['case_id']];q=bq[r['case_id']]['qualification']
            assert b['truth']==r['truth'] and b['person']==r['person'] and b['physical_query']==r['physical_query']
            assert b['source_frames']==[a['source_frame'] for a in r['axis']]
            oldp=b['probability']
        else:
            b=ba[r['query_id']];q=b['qualification'];oldp=b['probability']
            assert b['truth']==r['truth'] and b['person']==r['person'] and b['physical_query']==r['physical_query']
            assert b['source_frame_ids']==[a['source_frame'] for a in r['axis']]
        paired.append(dict(case_id=r['case_id'],query_id=r['query_id'],role=r['role'],origin=r['origin'],
            physical_query=r['physical_query'],truth=r['truth'],
            old_probability=oldp,new_probability=r['probability'],
            old_confirmed=q['qualified_display']==1,new_confirmed=r['qualification']['qualified_display']==1))
    g=c['fixed_gates'];s=ev['summaries']['development'];tests=[]
    def low(name,value,required):tests.append(dict(name=name,actual=value,required_min=required,passed=value>=required))
    def high(name,value,required):tests.append(dict(name=name,actual=value,required_max=required,passed=value<=required))
    v=[r for r in rows if r['role']=='development' and r['origin']=='reviewed_view']
    a=[r for r in rows if r['role']=='development' and r['origin']=='runtime_proposal']
    low('raw_macro_query_balanced_accuracy',s['raw_macro_query_balanced_accuracy'],g['raw_macro_query_balanced_accuracy_min'])
    low('known_positive_raw_confident_recovered',s['raw']['class_counts']['1']['confident_correct'],g['known_positive_confident_required'])
    low('known_zero_raw_confident_recovered',s['raw']['class_counts']['0']['confident_correct'],g['known_negative_confident_required'])
    high('known0_raw_confident_false_positive',sum(r['truth']==0 and r['raw_display']==1 for r in v),g['known0_confident_false_positive_max'])
    high('native1_raw_confident_false_negative',sum(r['kind']=='native' and r['truth']==1 and r['raw_display']==0 for r in v),g['native1_confident_false_negative_max'])
    low('qualified_positive_recovered',s['qualification']['confirmed_positive_correct'],g['qualified_positive_required'])
    high('qualified_known0_false_confirmation',s['qualification']['confirmed_positive_wrong'],g['qualified_known0_false_confirmation_max'])
    high('qualified_unknown_false_confirmation',s['qualification']['unknown_confirmed_positive'],g['qualified_unknown_false_confirmation_max'])
    low('auto_native_recovered',s['automatic']['confirmed_positive_correct'],g['automatic_native_positive_required'])
    for q,x in s['automatic_per_query'].items():low('auto_per_query:'+q,x['confirmed_positive_correct'],g['automatic_per_query_required'])
    for q,x in s['native_per_query'].items():low('qualified_native_per_query:'+q,x['confirmed_positive_correct'],g['native_positive_per_query_required'])
    controls=[r for r in rows if r['role']=='fit' and r['origin']=='reviewed_view' and r['truth']==1 and r['kind'] in
        ['prefix','suffix','middle_splice','speed_0_8','speed_1_25','intact_context_trim','missing_rgb']]
    high('FIT_positive_control_confident_reversal',sum(r['raw_display']==0 for r in controls),0)
    high('old_frozen_files_changed',0,0)
    counts={}
    for origin in ('reviewed_view','runtime_proposal'):
        rr=[r for r in paired if r['role']=='development' and r['origin']==origin]
        counts[origin]=dict(total=len(rr),known_positive=sum(r['truth']==1 for r in rr),
            old_true_confirmed=sum(r['truth']==1 and r['old_confirmed'] for r in rr),
            new_true_confirmed=sum(r['truth']==1 and r['new_confirmed'] for r in rr),
            gained_true_confirmation=sum(r['truth']==1 and not r['old_confirmed'] and r['new_confirmed'] for r in rr),
            lost_true_confirmation=sum(r['truth']==1 and r['old_confirmed'] and not r['new_confirmed'] for r in rr),
            old_false_confirmation=sum(r['truth']==0 and r['old_confirmed'] for r in rr),
            new_false_confirmation=sum(r['truth']==0 and r['new_confirmed'] for r in rr),
            old_unknown_confirmation=sum(r['truth'] is None and r['old_confirmed'] for r in rr),
            new_unknown_confirmation=sum(r['truth'] is None and r['new_confirmed'] for r in rr))
    result=dict(status='ELIGIBLE_FOR_SEPARATE_RUNTIME_AND_LOCAL_TIME_VALIDATION' if all(t['passed'] for t in tests) else 'NOT_ADOPTED_FIXED_GATES_FAILED',
        gates=tests,passed=sum(t['passed'] for t in tests),failed=sum(not t['passed'] for t in tests),
        comparison=counts,paired=paired,new_training_runs=1,epochs=80,
        bundled_changes_not_isolated_causal_effect=True,
        baseline_rerun=False,old_C_T3F_web_preserved=True,
        actual_new_end_to_end_latency_measured=False,exact_local_span_accuracy_unverified=True,
        current_demo_updated=False,whole_goal_complete=False,
        limitations=['Repeated DEV IDs11-13, not independent final test',
            'Same-Codex physical annotations, not human/KSL expert correctness',
            'V2 adds point/motion diagnostics; only unchanged whole-view truth and matched original frame scope enter paired efficacy comparison',
            'Global/nominal model support is not whole-word correctness, required extent/repetition, or exact completion boundary'],
        pins={str(p):sha(p) for p in [U/'ALIGNMENT_FROZEN.json',U/'TRAIN_RESULT.json',U/'EVALUATION.json',U/'EVALUATION_VERIFIED.json',
            AB/'PROCESS_EVALUATION.json',AB/'PROCESS_AUTO_EVALUATION.json']})
    with out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(result['status'],json.dumps(counts,ensure_ascii=False))

if __name__=='__main__':main()
