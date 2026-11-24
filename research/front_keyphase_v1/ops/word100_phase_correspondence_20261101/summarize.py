"""Aggregate saved paths only; never run candidate or rescore classifiers."""
from pathlib import Path
from collections import Counter
import numpy as np
from prepare import read,write,sha,load
U=Path(__file__).resolve().parent
NORMAL={'identity','speed_075','speed_125','hold15','repeat2_same','repeat3_same'}
def main():
    m=read(U/'RESULT_MANIFEST.json');inp=read(U/'INPUTS.json');controls=[];natural=[]
    for target,rows in [(controls,m['controls']),(natural,m['natural'])]:
        for a in rows:assert sha(a['file'])==a['sha256'];target.append(read(a['file']))
    by=[];regress=[];improve=[];shape_regress=[];scope_fail=[]
    for c in controls:
        b,a=c['baseline_truth'],c['candidate_truth']
        if c['kind'] in NORMAL:
            bad=any(a[k]>b[k]+1e-9 for k in ['query_mean_max_error_frames','reference_mean_max_error_frames','peak_error_frames','occurrence_mismatch_cells','query_wrong_occurrence_frames','reference_wrong_occurrence_frames'])
            good=not bad and a['mean_two_axes']<b['mean_two_axes']-1e-9 and a['interior_mean_two_axes']<b['interior_mean_two_axes']-1e-9
            if bad:regress.append(c['id'])
            if good:improve.append(c['id'])
        s=c['changed_shape']
        if s and s['actual_observed_changed_cells'] and (s['lost_compared_edited_cells']>0 or s['nonzero_after']<s['nonzero_before'] or any(a+1e-6<b for a,b in zip(s['after_max'],s['before_max']))):shape_regress.append(c['id'])
        for ax in ['query','reference']:
            b,a=c['baseline_evidence'][ax],c['candidate_evidence'][ax]
            if any(a[k]!=b[k] for k in ['total_slots','observed_slots','unobserved_slots']):scope_fail.append(c['id'])
    for kind in sorted({c['kind'] for c in controls}):
        cs=[c for c in controls if c['kind']==kind]
        by.append({'kind':kind,'cases':len(cs),'changed_paths':sum(c['candidate']['path']!=c['baseline_path'] for c in cs),'candidate_states':dict(Counter(c['candidate'].get('status','unspecified') for c in cs)),'baseline_mean_two_axis_error':float(np.mean([c['baseline_truth']['mean_two_axes'] for c in cs if c['baseline_truth']])),'candidate_mean_two_axis_error':float(np.mean([c['candidate_truth']['mean_two_axes'] for c in cs if c['candidate_truth']])),'worse_normal_ids':[c['id'] for c in cs if c['id'] in regress],'better_normal_ids':[c['id'] for c in cs if c['id'] in improve]})
    nsum=[]
    for d in natural:
        b,a=d['baseline_evidence'],d['candidate_evidence']
        for ax in ['query','reference']:
            if any(a[ax][k]!=b[ax][k] for k in ['total_slots','observed_slots','unobserved_slots']):scope_fail.append(d['panel']+'_'+d['query']+'_'+d['reference'])
        nsum.append({'query':d['query'],'reference':d['reference'],'panel':d['panel'],'changed_path':d['candidate']['path']!=d['baseline_path'],'status':d['candidate'].get('status'),'height_conflicts_before':len(b['height_conflicts']),'height_conflicts_after':len(a['height_conflicts']),'height_observed_path_cells_before':b['height_supported_path_cells'],'height_observed_path_cells_after':a['height_supported_path_cells'],'no_semantic_accuracy_claim':True})
    # W06/W07 evaluator-only source review: no unique correct correspondence was supplied.
    residual=[]
    for d in natural:
        if (d['query'],d['reference'])!=('P066','P063'):continue
        q,r=load(d['input']),load(d['reference_input'])
        for name,path in [('global_saved',read(d['global_saved_source'])['path']),('same_input_unconstrained',d['baseline_path']),('candidate',d['candidate']['path'])]:
            for frame in [68,71,72,73,74,75,76,77,78,79]:
                refs=[int(j) for i,j in path if i==frame]
                signs=[]
                for j in refs:
                    valid=q['mask'][frame,[1,29]].all() and r['mask'][j,[1,29]].all()
                    signs.append({'reference_frame':j,'both_wrist_heights_observed':bool(valid),'opposite_height_order':bool((q['x'][frame,1]-q['x'][frame,29])*(r['x'][j,1]-r['x'][j,29])<0) if valid else None})
                residual.append({'panel':d['panel'],'condition':name,'query_frame':frame,'correspondences':signs,'unique_true_reference_frame':None})
    gates={'whole_input_axes_preserved':not scope_fail,'known_normal_time_no_regression':not regress,'short_shape_peaks_preserved':not shape_regress,'known_time_useful_improvement':bool(improve),'requires_independent_natural_semantic_validation':True}
    useful=gates['whole_input_axes_preserved'] and gates['known_normal_time_no_regression'] and gates['short_shape_peaks_preserved'] and gates['known_time_useful_improvement']
    summary={'status':'COMPLETE_SAVED_RESULTS_SUMMARIZED','counts':{'controls':len(controls),'normal_time_controls':sum(c['kind'] in NORMAL for c in controls),'natural_panels':len(natural),'unique_natural_pairs':12,'parent_intervals_kept':39},'control_summary':by,'gates':gates,'normal_regressions':regress,'known_time_improvements':improve,'short_shape_regressions':shape_regress,'scope_failures':scope_fail,'natural_summary':nsum,'source_review_followup':residual,'decision':'RESEARCH_PATH_CANDIDATE_ONLY_PENDING_NATURAL_TRUTH' if useful else 'NOT_ADOPTED','timing':{'candidate_seconds_median':float(np.median([c['seconds'] for c in controls+natural])),'candidate_seconds_max':float(max(c['seconds'] for c in controls+natural)),'scope':'cached69 input processing only; excludes detection/video I/O/model/end-to-end wait'},'new_training':0,'new_detector_inference':0,'C_T3F_operational_DTW_unchanged':True,'limitations':['strict sign events can be created by jitter or missed between samples','all-maximal matching consensus is not proof of linguistic occurrence','new copies are feature-space stress tests, not natural repeated signing','speed125 discards samples; source mapping is known but short information may be lost','no user/mentor recordings or new words were used','range/low score/withheld count is not correctness']}
    write(U/'SUMMARY.json',summary)
    print({'decision':summary['decision'],'gates':gates,'normal_regressions':regress,'known_time_improvements':improve,'short_shape_regressions':shape_regress})
if __name__=='__main__':main()
