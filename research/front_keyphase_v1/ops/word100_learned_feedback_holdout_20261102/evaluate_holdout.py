"""One fixed comparison, all seeds, no fitting or selection on these results."""
import sys,time,traceback
from collections import Counter
from prepare_holdout import H,N,R,Q,read,sha,exclusive,write,clock
sys.path.insert(0,str(R))
from rgb_runtime import RGBMotionObserver
from rgb_model import rgb_input,predict
from evaluate_path import PathObserver
from motion_model import features,normalize
import numpy as np,torch

def main():
    torch.set_num_threads(4)
    assert read(H/'INPUT_VERIFICATION.json')['status'].startswith('PASS')
    assert not (H/'COMPARISON.json').exists()
    frozen=read(H/'FROZEN.json'); inputs=read(H/'INPUT_MANIFEST.json')['rows']; targets=read(N/'LOCKED_PHYSICAL_TARGETS.json')['rows']
    cache=[]
    for r in inputs:
        a=dict(np.load(r['input'])); cache.append((r,a,read(r['result'])))
    rows=[]; episodes=[]
    for condition in frozen['candidates']:
        for seed in frozen['seeds']:
            obs=RGBMotionObserver('rgb_residual',seed) if condition=='fixed_rgb_residual' else PathObserver('separated_relative',seed)
            seq=[]; labels=[]
            for r,a,base in cache:
                key=r['clip_id']+'_'+r['kind']; dest=H/'feedback'/f'{condition}_{seed}_{key}.json'
                # Explicitly resume only a completed saved result; never silently rerun it.
                if dest.exists(): out=read(dest)
                else:
                    out=obs.attach(base,a,a['retained_probability'],a['relation_probability']); exclusive(dest,out)
                assert out['path']==base['path']
                assert all(s['features']==b['features'] and s['relations']==b['relations'] for s,b in zip(out['stages'],base['stages']))
                rows.append(dict(condition=condition,seed=seed,case=key,kind=r['kind'],person=r['person'],word=r['word_id'],
                    motions=out['motion_observations'],seconds=out['motion_head_seconds'],base_seconds=r['seconds'],old_outputs_preserved=True))
                z,m=features(a,a['retained_probability'],a['relation_probability']); x=normalize(z,m,obs.stats)
                if condition=='fixed_rgb_residual': x=np.concatenate([x,rgb_input(a,obs.rgb_stats)],axis=1)
                mapping=np.asarray(r['source_frame_indices'])
                for tid,t in enumerate(targets):
                    if t['clip_id']!=r['clip_id']: continue
                    ix=np.flatnonzero((mapping>=t['frames'][0])&(mapping<=t['frames'][1]))
                    seq.append(x[ix]); labels.append(dict(target_index=tid,case=key,person=r['person'],word=r['word_id'],variant=r['kind'],
                        physical_kind=t['kind'],frames=ix.tolist(),y=t['y'],mask=t['mask']))
            pred=H/'predictions'/f'{condition}_{seed}.npz'; pred.parent.mkdir(exist_ok=True)
            if pred.exists():
                p=np.load(pred)['p']; assert len(p)==len(labels)
            else:
                p=predict(obs.model,seq); np.savez_compressed(pred,p=p,y=np.asarray([t['y'] for t in labels]),mask=np.asarray([t['mask'] for t in labels]))
            for t,pp in zip(labels,p): episodes.append(dict(condition=condition,seed=seed,**t,probabilities=pp.tolist()))
            write(H/'EVALUATION_PROGRESS.json',dict(condition=condition,seed=seed,comparisons=len(rows),at=clock()))
            print('evaluated locked',condition,seed,flush=True)
    assert len(rows)==192 and len(episodes)==432
    for p,h in frozen['pins'].items(): assert sha(p)==h
    exclusive(H/'COMPARISON.json',dict(rows=rows,no_training=True,no_selection=True))
    exclusive(H/'PHYSICAL_EPISODES.json',dict(rows=episodes,scope='Coarse visual physical directions/stability, not exact boundaries or complete sign correctness'))
    summary={c:{k:dict(Counter(m['status'] for r in rows if r['condition']==c and r['kind']==k for m in r['motions'])) for k in frozen['conditions']} for c in frozen['candidates']}
    exclusive(H/'COMPARISON_SUMMARY.json',summary)
    primary=[r for r in rows if r['condition']=='fixed_rgb_residual']
    normal=sum(m['status']=='DIRECTED_MOTION_OBSERVED' for r in primary for m in r['motions'])
    original=sum(m['status']=='DIRECTED_MOTION_OBSERVED' for r in primary if r['kind']=='original' for m in r['motions'])
    previous={(r['seed'],r['case'],m['target']):m['status'] for r in rows if r['condition']=='previous_separated_relative' for m in r['motions']}
    new_opposite=[dict(case=r['case'],seed=r['seed'],target=m['target']) for r in primary for m in r['motions']
        if m['status']=='OPPOSITE_DIRECTION_CANDIDATE' and previous[(r['seed'],r['case'],m['target'])]!='OPPOSITE_DIRECTION_CANDIDATE']
    checks=dict(original_normal_motion_confirmed_at_least27_of30=original>=27,all_normal_variants_confirmed_at_least108_of120=normal>=108,
        no_new_opposite_direction_candidates_on_normal=not new_opposite,old_features_and_DTW_preserved=True)
    exclusive(H/'EVALUATION_GATE.json',dict(status='PASS_TO_INTEGRATED_DIAGNOSTICS' if all(checks.values()) else 'LOCKED_GENERALIZATION_NOT_PASSED',
        checks=checks,original_confirmed=original,all_normal_confirmed=normal,new_opposite=new_opposite,no_selection=True,whole_goal_complete=False))
    print('Locked gate',original,normal,checks,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (H/'EVALUATION_FIRST_FAILURE.txt').exists():(H/'EVALUATION_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
