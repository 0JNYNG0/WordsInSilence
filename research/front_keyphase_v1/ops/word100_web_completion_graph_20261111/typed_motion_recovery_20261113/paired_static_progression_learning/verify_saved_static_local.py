"""Independent saved-decision sums; no evaluation/model code executed."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,runpy

S=Path(__file__).resolve().parent
P=S.parent/'progression_presence_learning'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    dest=S/'INDEPENDENT_STATIC_LOCAL_REVIEW.json';assert not dest.exists()
    static=read(S/'static_evaluation/COMPARISON.json')
    local=read(S/'local_evaluation/COMPARISON.json')
    prior=read(P/'local_evaluation/COMPARISON.json')
    check_path=P/'local_evaluation/verify_saved_arithmetic.py'
    check=runpy.run_path(str(check_path))['check'] # stdlib-only arithmetic helper; main not invoked
    sm=dict(observed_eligible=lambda r:r['observation_eligible'],excluded_unobserved=lambda r:not r['observation_eligible'],
        false_progression_observed=lambda r:r['observation_eligible'] and r['predicted_progression'],
        rejected_static_observed=lambda r:r['observation_eligible'] and not r['predicted_progression'],
        predicted_progression_on_unobserved=lambda r:not r['observation_eligible'] and r['predicted_progression'])
    for table,keys in [('summary',['seed','split','family','duration_bucket','anchor']),('by_family',['seed','split','family']),
        ('by_duration',['seed','split','duration_bucket']),('by_anchor',['seed','split','anchor']),('overall',['seed','split'])]:
        saved=[dict(rows=r['generated'],**{k:v for k,v in r.items() if k!='generated'}) for r in static[table]]
        check(static['rows'],keys,saved,sm)
    cm=dict(positive_labels=lambda r:r['expected_progression'],negative_labels=lambda r:not r['expected_progression'],
        true_positive=lambda r:r['expected_progression'] and r['predicted_progression'],
        false_negative=lambda r:r['expected_progression'] and not r['predicted_progression'],
        false_positive=lambda r:not r['expected_progression'] and r['predicted_progression'],
        true_negative=lambda r:not r['expected_progression'] and not r['predicted_progression'])
    bm={k:(lambda r,k=k:r[k]) for k in ['any_progression_raw','any_known_positive_supported','only_unknown_progression',
        'old_chosen_known_full','known_old_full_extent_supported','known_old_full_extent_unsupported','old_hold_false_selection',
        'previous_null_hold_false_selection','hold_false_positive_any_raw','hold_false_positive_old_chosen']}
    bm['old_chosen_present']=lambda r:r['old_chosen_index'] is not None
    bm['old_chosen_supported']=lambda r:r['old_chosen_supported'] is True
    pm=dict(body_gate_admitted=lambda r:r['previous_body_gate_admitted'],old_null_suppressed=lambda r:r['old_null_suppressed'],
        new_progression_positive=lambda r:r['predicted_progression'],new_positive_after_saved_body_gate=lambda r:r['after_same_partial_body_gate'],
        recovered_from_null_suppression=lambda r:r['recovered_from_null_suppression'],lost_previous_above_null=lambda r:r['lost_previous_above_null'])
    tables=[('candidate_summary','candidate_rows',['seed','split','family','label_kind','duration_bucket'],cm),
        ('candidate_all_durations','candidate_rows',['seed','split','family','label_kind'],cm),
        ('candidate_all_families','candidate_rows',['seed','split','label_kind'],cm),
        ('matched_duration_scope','candidate_rows',['seed','split','family','duration_bucket_in_fit_matched','label_kind'],cm),
        ('bag_summary','bag_rows',['seed','split','origin','family'],bm),('bag_all_families','bag_rows',['seed','split','origin'],bm),
        ('partial_comparison','partial_rows',['seed','split','family','duration_bucket'],pm),('partial_all_families','partial_rows',['seed','split'],pm)]
    for table,rs,keys,metrics in tables:check(local[rs],keys,local[table],metrics)
    for unit,d in [('static_evaluation',static),('local_evaluation',local)]:
        frozen=read(S/unit/'RULES_FROZEN.json');assert sha(S/unit/'RULES_FROZEN.json')==d['rules_sha256']
        for p,h in {**frozen['pins'],**d['prediction_pins']}.items():assert sha(p)==h
    for r in static['rows']:
        assert r['observation_eligible']==(r['observation_mask']==1)
        assert r['expected_observed_progression']==(False if r['observation_eligible'] else None)
        assert r['predicted_progression']==(r['score']>=0)
    for r in local['candidate_rows']:
        assert r['predicted_progression']==(r['score']>=0)
        assert r['correct']==(r['expected_progression']==r['predicted_progression'])
    oldmap={(r['seed'],r['bag_index'],r['candidate_index']):r for r in prior['candidate_rows']}
    transitions=Counter();fail_cases=[]
    for r in local['candidate_rows']:
        old=oldmap[r['seed'],r['bag_index'],r['candidate_index']]
        assert all(r[k]==old[k] for k in ['span','label_kind','expected_progression','split','family'])
        if r['label_kind']=='partial':
            transitions[(r['split'],bool(old['predicted_progression']),bool(r['predicted_progression']))]+=1
            if old['predicted_progression'] and not r['predicted_progression']:
                fail_cases.append({k:r[k] for k in ['seed','bag_index','candidate_index','clip_id','split','family','frames','span','view','reverse']})
    oldbags={(r['seed'],r['bag_index']):r for r in prior['bag_rows']}
    for r in local['bag_rows']:
        old=oldbags[r['seed'],r['bag_index']]
        assert r['old_chosen_span']==old['old_chosen_span'] and r['old_chosen_index']==old['old_chosen_index']
    matched=[]
    for seed in [42,43,44]:
        rr=[r for r in static['matched_prior_rows'] if r['seed']==seed]
        obs=[r for r in rr if r['observation_eligible']];assert len(rr)==72
        assert all(r['in_new_training_augmentation_pool'] for r in rr)
        oldfp=sum(r['previous_false_progression'] for r in obs);newfp=sum(r['new_false_progression'] for r in obs)
        expected=next(r for r in static['matched_prior_summary'] if r['seed']==seed)
        assert expected['previous_false_progression']==oldfp and expected['new_false_progression']==newfp
        matched.append(dict(seed=seed,training_diagnostic_rows=72,old_fp=oldfp,new_fp=newfp))
    static_fp=[r for r in static['rows'] if r['observation_eligible'] and r['predicted_progression']]
    fp_by_source=Counter((r['seed'],r['split'],r['clip_id'],r['family'],r['anchor']) for r in static_fp)
    report=dict(at=datetime.now(timezone.utc).isoformat(),status='PASS_SAVED_ARITHMETIC_WITH_PARTIAL_REGRESSION',
        reviewer='/root/absence_contract_review_1113',static5_and_local8_summary_tables_exact=True,
        static_overall=static['overall'],matched_training_diagnostics=matched,
        partial_transitions=[dict(split=k[0],prior_positive=k[1],new_positive=k[2],candidates=v) for k,v in transitions.items()],
        lost_partial_candidates=fail_cases,
        static_false_progression_sources=[dict(seed=k[0],split=k[1],clip_id=k[2],family=k[3],anchor=k[4],count=v) for k,v in fp_by_source.items()],
        old_known_extent_supported=sum(r['known_old_full_extent_supported'] for r in local['bag_rows']),
        old_known_extent_total=sum(r['old_chosen_known_full'] for r in local['bag_rows']),
        source_hashes={str(p):sha(p) for p in [Path(__file__),check_path,S/'static_evaluation/COMPARISON.json',S/'local_evaluation/COMPARISON.json',P/'local_evaluation/COMPARISON.json']},
        model_inference=False,rescoring=False,generator_changes=False,full_path_results_not_reviewed_here=True,
        interpretation=[
            'Matched FIT static44/42/38 false positives become0 each, but these inputs entered the augmentation pool: training diagnostic only.',
            'New upstream-exposed probe static false positives are0/44/0 per6516 observed synthetic entries. No prior same-input probe baseline exists.',
            'Real hold FP remains0 and prior known full extents505/505 remain supported. Probe real partial positives fall4338 to4287/4356, a net51 loss. Do not claim strict partial preservation.',
            'Synthetic frame/anchor/view/reverse/seed counts are correlated, not independent people or sign correctness trials.',
            'The improvement supports separating static pose from input progression, but does not establish motion type, semantic completion or educational correctness.'
        ])
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],transitions=report['partial_transitions'],static_fp_sources=report['static_false_progression_sources']),ensure_ascii=False))

if __name__=='__main__':main()
