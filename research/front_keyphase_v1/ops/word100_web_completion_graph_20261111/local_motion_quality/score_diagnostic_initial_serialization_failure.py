"""Frozen arithmetic only. Source-index truth is never passed to a model."""
from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
W=Path(__file__).resolve().parent;U=W.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def truth_rows(rows):
    sources={r['clip_id']:r for r in read(U/'reviews/sequence_target/MOTION_EXTENTS.json')['rows']}
    targets=[];groups={}
    for r in rows:
        if r['clip_id'] not in sources or r['kind'] not in ('full_normal','dynamic_complete_blocks') or r['moving']!='R' or r['axis'] not in (0,1):continue
        target=sources[r['clip_id']];m=target['motions']['B' if r['axis']==0 else 'C'];a,b=r['span']
        key='normal_query_indices' if r['kind']=='full_normal' else 'permuted_query_indices_in_source_order'
        on=set(m['onset_uncertainty'][key]);off=set(m['offset_uncertainty'][key]);witness=set(m['definite_moving_witness'][key]);envelope=set(m['possible_motion_envelope'][key])
        actual=set(range(a,b+1));splices=[] if r['kind']=='full_normal' else [s['query_right'] for s in target['splice_joins'] if s['synthetic_discontinuity']]
        crosses=any(a<j<=b for j in splices)
        label=-1;reason='unreviewed_or_mixed'
        if b<=a:reason='single_observation'
        elif r['observed_frames']<r['frames']:reason='incomplete_observation'
        elif crosses:reason='synthetic_join_not_a_new_semantic_error_label'
        elif a in on and b in off:label=2;reason='reviewed_extent'
        elif actual.issubset(envelope) and bool(witness&actual):label=1;reason='reviewed_motion_fragment'
        else:
            for hold in m['post_motion_hold_ranges']:
                if actual.issubset(set(hold[key])):label=0;reason='reviewed_hold';break
        v=dict(id=r['id'],clip_id=r['clip_id'],kind=r['kind'],axis=r['axis'],origin=r['origin'],span=r['span'],target=label,reason=reason,witness_covered=witness.issubset(actual) and not crosses,extent_match=a in on and b in off and not crosses,synthetic_join=crosses)
        targets.append(v)
        if r['origin']=='boundary':groups.setdefault((r['clip_id'],r['kind'],r['axis']),[]).append(v)
    return targets,groups

def freeze():
    assert not (W/'EVALUATION_FROZEN.json').exists()
    # No new probability/checkpoint values are read here, only hashes/manifests.
    rows=read(W/'EVAL_INPUTS_COMPLETE.json')['rows'];targets,groups=truth_rows(rows)
    write(W/'DIAGNOSTIC_TARGETS.json',dict(rows=targets,scope='Existing source-reviewed B/C extent and hold bands of three development videos, mapped to normal/rearranged query indices for arithmetic only. Not human-independent or unseen evaluation.'))
    paths=[W/'evaluate.py',Path(__file__),W/'DIAGNOSTIC_TARGETS.json',W/'EVAL_INPUTS_COMPLETE.json',W/'TRAINING_COMPLETE.json',W/'quality_model.py',U/'reviews/sequence_target/MOTION_EXTENTS.json',U/'reviews/local_quality_input_verification.json',U/'reviews/local_quality_evaluation_preflight.json',U/'reviews/local_quality_training_verification.json']
    paths += [Path(r['path']) for r in read(W/'TRAINING_COMPLETE.json')['runs']]
    write(W/'EVALUATION_FROZEN.json',dict(at=stamp(),pins={str(p):sha(p) for p in paths},conditions=['A_cached_seed42','B42','B43','B44','C42','C43','C44'],mode='shadow_component_diagnostic_only_no_selector_adoption',metrics=['125 continuation-excluded probe candidates confusion/macro recall (already exposed to older model)','direct-reviewed extent/fragment/hold candidate confusion with unknown denominator','12 source motion groups: any witness-covering boundary candidate assigned class1/2, and any reviewed full extent assigned class2','diagnostic-only max-p2 boundary ranking per axis; no runtime selection, no confidence tuning','all124 phase candidate class changes; short windows and all unknowns retained'],ranking='Unchanged pool, max p2 then p1 then earliest boundary; source truth only scores the output.',unscored='Left-moving accuracy, semantic clause identity, educational correctness, observed-vs-edited continuity, actual user feedback and end-to-end latency',decision='Cannot pass G2 from this component result. No thresholds, checkpoint choice or runtime changes based on diagnostic scores.',root_finished_design_after_agent_limit=True,agent_review_basis='Prior agent review rejected a quality-only selector; generated data and evaluation-input independent audits passed.'))
    print('Frozen diagnostic labels',len(targets),'boundary motion groups',len(groups),collections.Counter(t['target'] for t in targets))

def score():
    assert not (W/'DIAGNOSTIC_SUMMARY.json').exists()
    frozen=read(W/'EVALUATION_FROZEN.json')
    for p,h in frozen['pins'].items():assert sha(p)==h,p
    rows=read(W/'EVAL_INPUTS_COMPLETE.json')['rows'];targets,groups=truth_rows(rows)
    assert targets==read(W/'DIAGNOSTIC_TARGETS.json')['rows']
    predictions={'A_cached_seed42':np.asarray([r['old_quality'] for r in rows])}
    timings={}
    for r in read(W/'EVALUATION_COMPLETE.json')['runs']:
        assert sha(r['path'])==r['sha256'];d=read(r['path']);name=r['condition']+str(r['seed']);predictions[name]=np.asarray(d['probabilities']);timings[name]=d['seconds']
    output={}
    for name,probs in predictions.items():
        assert probs.shape==(len(rows),4) and np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-5)
        p=probs.argmax(1);conf=np.zeros((4,4),int);unknown=0
        for t in targets:
            if t['target']<0:unknown+=1
            else:conf[t['target'],p[t['id']]]+=1
        gr=[]
        for key,candidates in groups.items():
            winner=max(candidates,key=lambda c:(probs[c['id'],2],probs[c['id'],1],-c['span'][0],-c['span'][1]))
            gr.append(dict(clip_id=key[0],kind=key[1],axis=key[2],witness_retained=any(c['witness_covered'] and p[c['id']] in (1,2) for c in candidates),extent_retained=any(c['extent_match'] and p[c['id']]==2 for c in candidates),rank_winner=dict(id=winner['id'],span=winner['span'],witness_covered=winner['witness_covered'],extent_match=winner['extent_match'],synthetic_join=winner['synthetic_join'],probabilities=probs[winner['id']].tolist())))
        phase=[r for r in rows if r['origin']=='phase'];dist=collections.Counter(f"{r['kind']}/{int(p[r['id']])}" for r in phase)
        output[name]=dict(direct_confusion=conf.tolist(),direct_correct=int(np.trace(conf)),direct_total=int(conf.sum()),direct_unknown=unknown,groups=gr,aggregate={kind:dict(total=sum(g['kind']==kind for g in gr),witness_retained=sum(g['kind']==kind and g['witness_retained'] for g in gr),extent_retained=sum(g['kind']==kind and g['extent_retained'] for g in gr),rank_witness=sum(g['kind']==kind and g['rank_winner']['witness_covered'] for g in gr),rank_extent=sum(g['kind']==kind and g['rank_winner']['extent_match'] for g in gr)) for kind in ['full_normal','dynamic_complete_blocks']},phase_distributions=dict(dist),phase_changed_vs_A=sum(p[r['id']]!=predictions['A_cached_seed42'][r['id']].argmax() for r in phase),short_phase_count=sum(r['frames']<=3 for r in phase),seconds=timings.get(name))
    probe=[]
    for r in read(W/'TRAINING_COMPLETE.json')['runs']:
        d=read(Path(r['path']).parent/'HEAD_PROBE.json');conf=np.zeros((4,4),int)
        for s in d['rows']:
            if s['extent_mask']:conf[s['extent_target'],s['prediction']]+=1
        probe.append(dict(condition=r['condition'],seed=r['seed'],correct=int(np.trace(conf)),total=int(conf.sum()),confusion=conf.tolist(),macro_recall=float(np.mean(np.diag(conf)/conf.sum(1)))))
    write(W/'DIAGNOSTIC_SUMMARY.json',dict(at=stamp(),conditions=output,head_probe=probe,scope=frozen['mode'],decision='G2 remains REWORK; no new selector or runtime change. Candidate-quality and source coverage diagnostics do not prove semantic feedback correctness.',private_access=False))
    print(json.dumps({k:{'accuracy':[v['direct_correct'],v['direct_total']],'aggregate':v['aggregate']} for k,v in output.items()},indent=2))

if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='score':score()
    else:raise ValueError('choose freeze/score')
