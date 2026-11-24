"""Independent saved arithmetic path; no model inference or new labels."""
from collections import Counter
import numpy as np
from prepare_holdout import H,N,R,Q,O,read,sha,exclusive

def main():
    f=read(H/'FROZEN.json'); rows=read(H/'COMPARISON.json')['rows']; episodes=read(H/'PHYSICAL_EPISODES.json')['rows']
    inputs={r['clip_id']+'_'+r['kind']:r for r in read(H/'INPUT_MANIFEST.json')['rows']}
    targets=read(N/'LOCKED_PHYSICAL_TARGETS.json')['rows']; assert len(rows)==192 and len(episodes)==432
    assert len({(r['condition'],r['seed'],r['case']) for r in rows})==192
    summary={}; direct=[]
    for cond in f['candidates']:
        summary[cond]={k:dict(Counter(m['status'] for r in rows if r['condition']==cond and r['kind']==k for m in r['motions'])) for k in f['conditions']}
        for seed in f['seeds']:
            ee=[e for e in episodes if e['condition']==cond and e['seed']==seed]; assert len(ee)==72
            p=np.load(H/'predictions'/f'{cond}_{seed}.npz')
            np.testing.assert_allclose(p['p'],np.asarray([e['probabilities'] for e in ee]),rtol=0,atol=0)
            for k in f['conditions']:
                chosen=[e for e in ee if e['variant']==k]; assert len(chosen)==18
                scores=[]
                for j in range(5):
                    recalls=[]
                    for value in (0,1):
                        valid=[e for e in chosen if e['mask'][j] and e['y'][j]==value]; assert valid
                        recalls.append(sum((e['probabilities'][j]>=.5)==bool(value) for e in valid)/len(valid))
                    scores.append(sum(recalls)/2)
                direct.append(dict(condition=cond,seed=seed,variant=k,per_target_balanced_accuracy=scores,mean_balanced_accuracy=float(np.mean(scores))))
    for e in episodes:
        t=targets[e['target_index']]; r=inputs[e['case']]; mapping=np.asarray(r['source_frame_indices'])
        assert e['y']==t['y'] and e['mask']==t['mask'] and r['clip_id']==t['clip_id']
        assert e['frames']==np.flatnonzero((mapping>=t['frames'][0])&(mapping<=t['frames'][1])).tolist()
    for r in rows:
        out=read(H/'feedback'/f"{r['condition']}_{r['seed']}_{r['case']}.json"); base=read(inputs[r['case']]['result'])
        assert out['path']==base['path'] and out['motion_observations']==r['motions'] and len(out['stages'])==len(base['stages'])
        assert all(s['features']==b['features'] and s['relations']==b['relations'] for s,b in zip(out['stages'],base['stages']))
    assert summary==read(H/'COMPARISON_SUMMARY.json')
    primary=[r for r in rows if r['condition']=='fixed_rgb_residual']
    counts={k:sum(m['status']=='DIRECTED_MOTION_OBSERVED' for r in primary if r['kind']==k for m in r['motions']) for k in f['conditions']}
    assert sum(len(r['motions']) for r in primary)==120
    gate=read(H/'EVALUATION_GATE.json'); assert counts['original']==gate['original_confirmed'] and sum(counts.values())==gate['all_normal_confirmed']
    prior={(r['seed'],r['case'],m['target']):m['status'] for r in rows if r['condition']=='previous_separated_relative' for m in r['motions']}
    new_opposite=[dict(case=r['case'],seed=r['seed'],target=m['target']) for r in primary for m in r['motions'] if m['status']=='OPPOSITE_DIRECTION_CANDIDATE' and prior[(r['seed'],r['case'],m['target'])]!='OPPOSITE_DIRECTION_CANDIDATE']
    checks=dict(original_normal_motion_confirmed_at_least27_of30=counts['original']>=27,
        all_normal_variants_confirmed_at_least108_of120=sum(counts.values())>=108,no_new_opposite_direction_candidates_on_normal=not new_opposite,old_features_and_DTW_preserved=True)
    assert checks==gate['checks'] and new_opposite==gate['new_opposite']
    assert (gate['status']=='PASS_TO_INTEGRATED_DIAGNOSTICS')==all(checks.values())
    for p,h in f['pins'].items(): assert sha(p)==h
    for p,h in read(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json')['source_pins'].items(): assert sha(p)==h
    base=np.asarray([r['seconds'] for r in inputs.values()]);head=np.asarray([r['seconds'] for r in primary])
    exclusive(H/'SAVED_VERIFICATION.json',dict(status='PASS_SAVED_ARITHMETIC_AND_PRESERVATION',counts=counts,gate_status=gate['status'],
        physical_episode_scores=direct,base_pipeline_seconds=dict(min=float(base.min()),median=float(np.median(base)),max=float(base.max())),
        additional_motion_head_seconds_median=float(np.median(head)),new_inference=False,whole_goal_complete=False,
        verifier='Same Codex separate saved-result program; not independent human/KSL truth',training_on_these_inputs=False))
    print(gate['status'],counts,flush=True)

if __name__=='__main__':main()
