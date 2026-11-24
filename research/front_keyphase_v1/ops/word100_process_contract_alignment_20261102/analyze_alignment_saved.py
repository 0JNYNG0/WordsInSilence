"""Fixed-threshold descriptive analysis of immutable saved scores; no model import."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import statistics

U = Path(__file__).resolve().parent
A = U.parent / 'word100_action_completion_supervision_20261102'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def barriers(r):
    q = r['qualification']
    c = q['interval_candidates']
    key = 'connected_process_probability' if r['model'] == 'AC' else 'motion_probability'
    b = []
    if r['probability'] < .8:
        b.append('global_below_0.8')
    if not c:
        b.append('no_ordered_point_candidate')
    elif max(x[key] for x in c) < .8:
        b.append('all_local_below_0.8')
    return '+'.join(b) or 'confirmed'

def metrics(rows):
    pos = [r for r in rows if r['truth'] == 1]
    neg = [r for r in rows if r['truth'] == 0]
    unk = [r for r in rows if r['truth'] is None]
    known = pos + neg
    return dict(rows=len(rows), source_clips=len({r['clip_id'] for r in rows}), source_queries=len({r['query_id'] for r in rows}),
                positives=len(pos), negatives=len(neg), unknown=len(unk),
                raw_correct=sum((r['probability'] >= .5) == bool(r['truth']) for r in known), known=len(known),
                high_positive=sum(r['probability'] >= .8 for r in pos),
                confirmed_positive=sum(r['qualification']['qualified_display'] == 1 for r in pos),
                false_confirmation=sum(r['qualification']['qualified_display'] == 1 for r in neg),
                unknown_confirmation=sum(r['qualification']['qualified_display'] == 1 for r in unk),
                positive_barriers=dict(Counter(barriers(r) for r in pos)))

def point_metrics(rows):
    out = {}
    for side in ['setup', 'terminal']:
        entries = [p for r in rows for p in r.get('sparse_point_scores', []) if p[side + '_truth'] is not None]
        pos = [p for p in entries if p[side + '_truth'] == 1]
        neg = [p for p in entries if p[side + '_truth'] == 0]
        out[side] = dict(positive=len(pos),negative=len(neg),
            positive_ge_05=sum(p[side+'_probability'] >= .5 for p in pos),
            positive_ge_08=sum(p[side+'_probability'] >= .8 for p in pos),
            negative_ge_05=sum(p[side+'_probability'] >= .5 for p in neg),
            negative_ge_08=sum(p[side+'_probability'] >= .8 for p in neg))
    return out

def compact(r):
    q = r['qualification']
    key = 'connected_process_probability' if r['model'] == 'AC' else 'motion_probability'
    frames = [x['source_frame'] for x in r['axis']] if 'axis' in r else r.get('source_frames', r.get('source_frame_ids'))
    cs = []
    for c in q['interval_candidates']:
        ii = c['output_indices']
        ff = [frames[j] for j in ii]
        support = c.get('rgb_context_support', {})
        cs.append(dict(output_indices=ii,source_frames=ff,
             source_frame_range=[ff[0],ff[-1]],source_seams=[dict(output_index=ii[k],before=ff[k-1],after=ff[k]) for k in range(1,len(ii)) if ff[k] != ff[k-1]+1],
             setup_probability=c['setup_probability'],terminal_probability=c['terminal_probability'],
             local_probability=c[key],local_meaning=key,
             outside_nominal_source_frames=support.get('outside_nominal_observed_source_frames',[])))
    p = r['point_probabilities']
    return dict(case_id=r['case_id'],query_id=r['query_id'],clip_id=r['clip_id'],person=r['person'],model=r['model'],origin=r['origin'],kind=r.get('kind'),
        physical_query=r['physical_query'],truth=r['truth'],raw_probability=r['probability'],qualified=q['qualified_display'],status=q['status'],barriers=barriers(r),
        candidate_count=len(cs),max_setup=max(x[0] for x in p),max_terminal=max(x[1] for x in p),
        high_setup_source_frames=[frames[i] for i,x in enumerate(p) if x[0]>=.8],
        high_terminal_source_frames=[frames[i] for i,x in enumerate(p) if x[1]>=.8],
        candidate_cap_reached=q['candidate_cap_reached'],candidates=cs,
        sparse_points=r.get('sparse_point_scores',[]),annotated_netmotion=r.get('interval_scores',[]),
        oracle_qualified=r.get('oracle_known_seam_diagnostic',{}).get('qualified_display'))

def main():
    inputs = [U/'EVALUATION.json',U/'EVALUATION_VERIFIED.json',A/'PROCESS_EVALUATION.json',A/'PROCESS_AUTO_EVALUATION.json']
    acd,verified,abr,aba = map(read,inputs)
    assert verified['status'].startswith('PASS') and verified['evaluation_sha256']==sha(inputs[0])
    ac = [dict(r,model='AC') for r in acd['rows']]
    raw = {r['view_id']:r for r in abr['rows']}
    ab = []
    for q in aba['view_rows']:
        r = dict(raw[q['view_id']], **{k:v for k,v in q.items() if k not in raw[q['view_id']]})
        ab.append(dict(r,model='AB',case_id=r['view_id'],origin='reviewed_view'))
    for r in aba['automatic_rows']:
        ab.append(dict(r,model='AB',case_id=r['query_id']+'__runtime_proposal',origin='runtime_proposal',kind='automatic_native'))
    byold = {r['case_id']:r for r in ab}
    assert len(ac)==len(ab)==357 and set(byold)=={r['case_id'] for r in ac}
    for r in ac:
        o=byold[r['case_id']]
        assert r['truth']==o['truth'] and r['role']==o['role'] and r['query_id']==o['query_id']
        assert r['image_available']==o['image_available']
        oldframes=o.get('source_frames',o.get('source_frame_ids'))
        assert [x['source_frame'] for x in r['axis']]==oldframes
    summary = {}
    for model,rows in [('AB',ab),('AC',ac)]:
        summary[model]={}
        for role in ['fit','development']:
            summary[model][role]={}
            for origin in ['reviewed_view','runtime_proposal']:
                selected=[r for r in rows if r['role']==role and r['origin']==origin]
                summary[model][role][origin]=dict(all=metrics(selected),
                    by_query={q:metrics([r for r in selected if r['physical_query']==q]) for q in sorted({r['physical_query'] for r in selected})},
                    native=metrics([r for r in selected if r.get('kind')=='native']),
                    sparse_point_diagnostics=point_metrics(selected))
    detailed=[]
    false=[]
    transitions=[]
    for r in ac:
        o=byold[r['case_id']]
        if r['truth']==1:
            transitions.append(dict(case_id=r['case_id'],role=r['role'],origin=r['origin'],kind=r.get('kind'),physical_query=r['physical_query'],
                    before=barriers(o),after=barriers(r),before_raw=o['probability'],after_raw=r['probability']))
        if r['role']=='development' and (r['physical_query'] in ('right_hand_toward_left_wrist_along_arm','right_hand_rises_relative_left') and r['truth']==1):
            detailed.append(dict(before=compact(o),after=compact(r)))
        if r['role']=='development' and r['truth']==0 and r['qualification']['qualified_display']==1:
            false.append(dict(before=compact(o),after=compact(r)))
    nominal=[p for r in ac for p in r.get('nominal_process_scores',[])]
    nominal_counts={k:{'total':sum(p['binding_kind']==k for p in nominal),
       'known':sum(p['binding_kind']==k and p['truth'] is not None for p in nominal),
       'raw_correct':sum(p['binding_kind']==k and p['truth'] is not None and (p['probability']>=.5)==bool(p['truth']) for p in nominal),
       'high_positive':sum(p['binding_kind']==k and p['truth']==1 and p['probability']>=.8 for p in nominal),
       'high_false_positive':sum(p['binding_kind']==k and p['truth']==0 and p['probability']>=.8 for p in nominal),
       'unknown_high_positive':sum(p['binding_kind']==k and p['truth'] is None and p['probability']>=.8 for p in nominal)} for k in sorted({p['binding_kind'] for p in nominal})}
    byid={r['case_id']:r for r in ac}
    native_auto=[]
    for r in ac:
        if r['origin']!='runtime_proposal':continue
        n=byid[r['query_id']+'__native']
        native_auto.append(dict(query_id=r['query_id'],role=r['role'],physical_query=r['physical_query'],truth=r['truth'],
            native_global=n['probability'],auto_global=r['probability'],native_barriers=barriers(n),auto_barriers=barriers(r),
            native_max_local=max([c['connected_process_probability'] for c in n['qualification']['interval_candidates']],default=None),
            auto_max_local=max([c['connected_process_probability'] for c in r['qualification']['interval_candidates']],default=None)))
    out=dict(status='SAVED_FIXED_THRESHOLD_DIAGNOSTIC_NOT_NEW_EVALUATION',input_pins={str(p):sha(p) for p in inputs},script_sha256=sha(Path(__file__)),
       thresholds={'raw':.5,'high':.8,'low':.2},all357_identity_truth_axis_checks=True,
       scope_note='AB local score is net-motion; AC local score is linked-process. Components are not same-target causal ablations. Added v2 point/motion diagnostic labels make full sparse populations unmatched.',
       summaries=summary,positive_transitions=transitions,development_arm_and_rise=detailed,development_false_confirmations=false,
       nominal_fit_diagnostics=nominal_counts,native_auto_pairs=native_auto,
       model_calls=0,training_steps=0,threshold_sweep=False,old_result_mutation=False)
    p=U/'ALIGNMENT_CAUSE_COUNTS.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'saved':str(p),'sha256':sha(p),'rows':len(ac),'summary':{m:{role:{origin:summary[m][role][origin]['all'] for origin in ['reviewed_view','runtime_proposal']} for role in ['fit','development']} for m in ['AB','AC']},'nominal':nominal_counts},ensure_ascii=False))

if __name__=='__main__':main()
