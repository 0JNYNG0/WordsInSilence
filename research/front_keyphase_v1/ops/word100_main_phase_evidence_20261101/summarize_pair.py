"""Precommitted acceptance gates. Reads saved results only; no path search."""
from collections import Counter,defaultdict
import numpy as np
from prepare_pair import U,read,write,sha
import natural_evaluator
EPS=1e-9
TIME_KEYS=['query_mean_max_error_frames','reference_mean_max_error_frames','peak_error_frames','query_wrong_occurrence_frames','reference_wrong_occurrence_frames','interior_mean_two_axes']
def main():
    man=read(U/'PAIR_MANIFEST.json');inp=read(U/'PAIR_INPUTS.json');cs=[];ns=[]
    assert len(man['controls'])==192 and len({r['id'] for r in man['controls']})==192
    assert {r['id'] for r in man['controls']}=={c['id'] for c in inp['controls']}
    assert len(man['natural'])==12 and len({r['id'] for r in man['natural']})==12
    assert {r['id'] for r in man['natural']}=={p['query']+'_'+p['reference'] for p in inp['natural']}
    for group,arr in [('controls',cs),('natural',ns)]:
        for rec in man[group]:
            assert sha(rec['file'])==rec['sha256'];arr.append(read(rec['file']))
    normal=[d for d in cs if d['normal_control']]
    assert len(normal)==128
    expected_controls={c['id']:c for c in inp['controls']}
    for d in cs:
        expected=expected_controls[d['id']]
        assert d['kind']==expected['kind'] and d['normal_control']==expected['normal_control']
    improved=[];regressed=[];nochange=[]
    for d in normal:
        a,b=d['unit']['truth'],d['bounded']['truth']
        bad=[k for k in TIME_KEYS if b[k]>a[k]+EPS]
        good=[k for k in TIME_KEYS if b[k]<a[k]-EPS]
        rec=dict(id=d['id'],kind=d['kind'],worse=bad,better=good,unit=a,bounded=b)
        if bad:regressed.append(rec)
        elif good:improved.append(rec)
        else:nochange.append(d['id'])
    shorts=[d for d in cs if 'short_shape_guard' in d];short_loss=[]
    assert len(shorts)==32
    for d in shorts:
        g=d['short_shape_guard'];lost=int(np.sum(np.array(g['query_frame_peaks_bounded'])+EPS<np.array(g['query_frame_peaks_unit'])))
        if lost or g['lost_supported_cells']:short_loss.append(dict(id=d['id'],lost_peak_frames=lost,lost_support=g['lost_supported_cells']))
    denominators=[]
    for d in cs+ns:
        for axis in ['query','reference']:
            a,b=d['unit']['evidence'][axis],d['bounded']['evidence'][axis]
            for k in ['total_slots','observed_slots','unobserved_slots']:
                if a[k]!=b[k]:denominators.append(dict(id=d.get('id',d.get('query')),axis=axis,key=k))
            if b['compared_slots']==0 and any(x['peak_difference'] is not None for x in b['shape'].values()):denominators.append(dict(id=d.get('id'),axis=axis,key='unknown called zero'))
    agg={mode:natural_evaluator.aggregate([d[mode]['natural_evaluation'] for d in ns]) for mode in ['unit','bounded']}
    natural_changes=[];natural_regressions=[];natural_improvements=[]
    for d in ns:
        a,b=d['unit']['natural_evaluation'],d['bounded']['natural_evaluation']
        for ap,bp in zip(a['parents'],b['parents']):
            assert ap['id']==bp['id']
            for af,bf in zip(ap['frames'],bp['frames']):
                assert af['reference_index']==bf['reference_index']
                keys=['any_different_physical_family','any_unknown_comparison']
                bad=[k for k in keys if bf['flags'][k] and not af['flags'][k]]
                good=[k for k in keys if af['flags'][k] and not bf['flags'][k]]
                if bad or good:
                    rec=dict(parent=ap['id'],reference_frame=af['reference_index'],query=d['query'],reference=d['reference'],unit_query=af['matched_query_indices'],bounded_query=bf['matched_query_indices'],worse=bad,better=good)
                    natural_changes.append(rec)
                    if bad:natural_regressions.append(rec)
                    elif good:natural_improvements.append(rec)
    gates=dict(normal_speed_hold_repeat_no_regression=not regressed,short_shape_support_and_peak_preserved=not short_loss,
               full_observed_missing_denominators_preserved=not denominators,natural_no_new_family_or_unknown_flags=not natural_regressions,
               known_time_useful_improvement=bool(improved),natural_draft_useful_improvement=bool(natural_improvements))
    bykind={}
    for kind in sorted({d['kind'] for d in cs}):
        arr=[d for d in cs if d['kind']==kind]
        bykind[kind]=dict(n=len(arr),normal=arr[0]['normal_control'],changed_paths=sum(d['unit']['path']!=d['bounded']['path'] for d in arr),
                         mean_source_frame_error={m:float(np.mean([d[m]['truth']['mean_two_axes'] for d in arr])) for m in ['unit','bounded']})
    s=dict(status='PAIRED_RESULTS_SUMMARIZED',decision='RESEARCH_CANDIDATE_PASSES_LIMITED_GATES' if all(gates.values()) else 'NOT_ADOPTED',
           controls=len(cs),normal_controls=len(normal),natural_pairs=len(ns),gates=gates,
           normal_improved=improved,normal_regressed=regressed,normal_unchanged=nochange,by_kind=bykind,
           short_cases=len(shorts),short_actually_changed_cases=sum(d['short_shape_guard']['actual_observed_changed_cells']>0 for d in shorts),
           short_actual_changed_cells=sum(d['short_shape_guard']['actual_observed_changed_cells'] for d in shorts),short_losses=short_loss,denominator_failures=denominators,
           natural_aggregate=agg,natural_changed_paths=sum(d['unit']['path']!=d['bounded']['path'] for d in ns),natural_frame_changes=natural_changes,
           natural_regressions=natural_regressions,natural_improvements=natural_improvements,
           runtime_seconds={m:sum(d[m]['seconds'] for d in cs+ns) for m in ['unit','bounded']},
           runtime_not_full_video_or_end_user_latency=True,no_new_training_or_detector_inference=True,
           natural_kind_flags_are_partial_physical_drafts_not_linguistic_accuracy=True,
           no_parameter_tuning_or_rerun_allowed=True)
    write(U/'PAIR_SUMMARY.json',s)
    print(json_summary(s))
def json_summary(s):
    import json
    return json.dumps({k:s[k] for k in ['decision','controls','normal_controls','natural_pairs','gates','by_kind','natural_changed_paths','runtime_seconds']},ensure_ascii=False,indent=2)
if __name__=='__main__':main()
