"""Run only the NEW A model on the predeclared 324 rotated inputs.

U and R rotated results already exist and are read from disk; their models are
never instantiated or inferred. A unrotated outputs also come only from its
saved canonical evaluation. This is not a fresh independent generalization set.
"""
from pathlib import Path
import sys, json, hashlib, datetime, time, traceback
from collections import defaultdict
import numpy as np
from hand_core import HandObserver, padded, torch
from articulation_core import articulation24

A = Path(__file__).resolve().parent
R = A.parent/'hand_rotation_invariant_learning'
ORIGINAL = A.parent/'hand_transition_learning'
sys.path.append(str(R))
from rotation_core import rotate_hand63
from runtime_contract import prepare_shape_observation
D = A/'rotation_controls'
ANGLES = [-90,-60,-30,30,60,90]


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True); assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def metrics(rows):
    eligible = [r for r in rows if r['eligible']]
    emitted = [r for r in eligible if r['confidence'] >= .65]
    return dict(total=len(rows),eligible=len(eligible),observation_unknown=len(rows)-len(eligible),
        argmax_correct=sum(r['prediction']==r['target'] for r in eligible),emitted=len(emitted),
        confident_wrong=sum(r['prediction']!=r['target'] for r in emitted),low_confidence=len(eligible)-len(emitted),
        changed_from_saved_unrotated=sum(r['prediction']!=r['baseline_prediction'] for r in eligible))


def groups(rows, fields, fn=metrics):
    g=defaultdict(list)
    for r in rows:g[tuple(r[f] for f in fields)].append(r)
    return [dict(**dict(zip(fields,k)),counts=fn(v)) for k,v in sorted(g.items())]


def main():
    assert not D.exists(), 'Preserve first outcome, no automatic rerun'
    plan=read(A/'ROTATION_CONTROL_PLAN.json');assert plan['angles_degrees']==ANGLES
    for p,h in plan['pins'].items():assert sha(p)==h,p
    for cfg in [A/'FROZEN.json',A/'TRAINING_CONTRACT.json']:
        for p,h in read(cfg)['pins'].items():assert sha(p)==h,p
    assert (A/'saved_evaluation/COMPLETE.json').exists()
    assert all((A/f'saved_evaluation/PAIRED_COMPARISON_{tag}.json').exists() for tag in ['R','U'])
    cases=read(A/'saved_evaluation/CANONICAL_SOURCE_ROWS.json')
    oldcases=read(ORIGINAL/'saved_evaluation/CANONICAL_SOURCE_ROWS.json')
    key=lambda c:(c['person'],c['seed'],c['clip_id'],c['kind'])
    oldmap={key(c):c for c in oldcases};assert len(oldmap)==len(cases)==54
    for c in cases:
        o=oldmap[key(c)];assert c['source_frames']==o['source_frames'] and c['target']==o['target'] and c['reverse']==o['reverse']
    ai=read(A/'EVALUATION_ITEMS.json');oi=read(ORIGINAL/'EVALUATION_ITEMS.json');assert ai==oi
    meta=read(ORIGINAL/'INPUTS_COMPLETE.json')
    needed={c['clip_id'] for c in cases}
    sources={(r['clip_id'],r['side']):r for r in meta['sequences'] if r['clip_id'] in needed and r['side']=='R'}
    for r in sources.values():assert sha(r['path'])==r['sha256']
    done=read(A/'TRAINING_COMPLETE.json');assert len(done['runs'])==18
    runs={(r['heldperson'],r['seed']):r for r in done['runs']}
    for r in runs.values():
        assert sha(r['model'])==r['model_sha256'] and r['heldperson'] not in r['split']['trainpeople']
    saved_parents=read(R/'rotation_controls/CASE_RESULTS.json')
    parentkey=lambda c:(c['person'],c['seed'],c['clip_id'],c['kind'],c['angle'])
    parentmaps={name:{parentkey(r):r for r in saved_parents if r['condition']==condition}
                for name,condition in [('U','original'),('R','canonical_rotation')]}
    assert all(len(v)==324 for v in parentmaps.values())
    frozen=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),angles_degrees=ANGLES,new_forward_cases=324,
        only_model_executed='hand_articulation_learning A; no U/R model load or inference',
        selection='All54source-label-selected canonical cases times6nonzeroangles, unchanged labels and all denominators.',
        input='Original U handlocal63 -> fixedwindowrotation -> R canonical basis once -> articulation24 appended. All availability and unknown gates retained.',
        labels='Projected rotation preserves handshape transition target, not a new normal filmed sign/semantic label.',
        decision='Same argmax/maxprob>=.65 and observation eligibility; unavailable counts retained as unknown.',
        limitation='Development-exposed six people/one transition expression. No3D occlusion proof or educational error benchmark.',
        model_pins={r['model']:r['model_sha256'] for r in runs.values()},
        pins={str(p):sha(p) for p in [Path(__file__),A/'hand_core.py',A/'articulation_core.py',A/'ROTATION_CONTROL_PLAN.json',
            A/'TRAINING_COMPLETE.json',A/'EVALUATION_ITEMS.json',A/'saved_evaluation/CANONICAL_SOURCE_ROWS.json',
            R/'rotation_core.py',R/'runtime_contract.py',R/'rotation_controls/CASE_RESULTS.json']+[Path(r['path']) for r in sources.values()]})
    write(D/'FROZEN.json',frozen)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cache={k:np.load(r['path'],allow_pickle=False) for k,r in sources.items()}
    out=[];audit=[];timing=[]
    for person in range(2,8):
        for seed in [42,43,44]:
            selected=[c for c in cases if c['person']==person and c['seed']==seed];assert len(selected)==3
            controls=[]
            for c in selected:
                for angle in ANGLES:
                    row=dict(person=person,seed=seed,clip_id=c['clip_id'],kind=c['kind'],angle=angle,target=c['target'],
                        source_frames=c['source_frames'],reverse=c['reverse'],baseline_prediction=c.get('prediction'),baseline_confidence=c.get('confidence'))
                    if not c['available']:
                        controls.append((row,None,False));continue
                    w=ai['windows'][c['window_id']];a,b=w['span'];raw=cache[c['clip_id'],'R']['x'][a:b+1].copy()
                    if w['reverse']:raw=raw[::-1].copy()
                    canonical,observed,info=prepare_shape_observation(rotate_hand63(raw,angle))
                    base,baseobserved,_=prepare_shape_observation(raw)
                    extra=articulation24(canonical);baseextra=articulation24(base)
                    assert np.array_equal(observed,baseobserved) and np.allclose(canonical,base,atol=2e-6,rtol=1e-6)
                    assert np.array_equal(extra[:,12:],baseextra[:,12:]) and np.allclose(extra,baseextra,atol=2e-5,rtol=1e-5)
                    eligible=bool(observed.all());x=np.c_[canonical,extra].astype(np.float32)
                    audit.append(dict(**row,frames=len(raw),eligible=eligible,basis_valid_frames=int(info['basis_valid'].sum()),
                        observed_frames=int(observed.sum()),feature_unavailable=int((extra[:,12:]==0).sum()),
                        canonical_max_error=float(np.max(np.abs(canonical-base))),articulation_max_error=float(np.max(np.abs(extra-baseextra)))))
                    controls.append((row,x,eligible))
            run=runs[person,seed];state=torch.load(run['model'],map_location='cpu',weights_only=True)
            model=HandObserver(state['mean'],state['std'],state['articulation_mean'],state['articulation_std'])
            model.load_state_dict(state);model.eval();model.requires_grad_(False)
            ids=[i for i,(_,_,ok) in enumerate(controls) if ok];prediction={};elapsed=0.
            if ids:
                x,lens=padded([controls[i][1] for i in ids]);tic=time.perf_counter()
                with torch.inference_mode():logits=model(x,lens)['transition'].numpy()
                elapsed=time.perf_counter()-tic;assert np.isfinite(logits).all()
                prediction=dict(zip(ids,logits))
            timing.append(dict(person=person,seed=seed,new_forward_cases=len(ids),compute_batch_seconds=elapsed))
            for i,(row,x,eligible) in enumerate(controls):
                row.update(eligible=eligible,condition='articulation_residual',model_sha256=run['model_sha256'])
                if eligible:
                    logits=prediction[i].astype(float);p=np.exp(logits-logits.max());p/=p.sum();j=int(p.argmax());conf=float(p[j])
                    row.update(prediction=j,confidence=conf,emitted=conf>=.65,probabilities=p.tolist(),logits=logits.tolist(),state='observed' if conf>=.65 else 'uncertain')
                else:row.update(prediction=None,confidence=None,emitted=False,probabilities=None,logits=None,state='observation_unknown')
                out.append(row)
            del model,state
    assert len(out)==324
    pair=[]
    for row in out:
        for name,lookup in parentmaps.items():
            old=lookup[parentkey(row)];assert old['target']==row['target'] and old['eligible']==row['eligible']
            pair.append(dict(parent=name,person=row['person'],seed=row['seed'],kind=row['kind'],angle=row['angle'],eligible=row['eligible'],
                old_correct=old['prediction']==row['target'] if row['eligible'] else False,new_correct=row['prediction']==row['target'] if row['eligible'] else False))
    def pm(rows):
        v=[r for r in rows if r['eligible']]
        return dict(total=len(rows),eligible=len(v),old_correct=sum(r['old_correct'] for r in v),new_correct=sum(r['new_correct'] for r in v),
            repaired=sum(not r['old_correct'] and r['new_correct'] for r in v),regressed=sum(r['old_correct'] and not r['new_correct'] for r in v))
    write(D/'INPUT_AUDIT.json',audit);write(D/'CASE_RESULTS.json',out)
    result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='A_ONLY_FIXED_ROTATION_CONTROLS_COMPLETE',
        total=metrics(out),by_seed=groups(out,['seed']),by_angle=groups(out,['angle']),by_person_kind=groups(out,['person','kind']),
        saved_parent_comparison=groups(pair,['parent'],pm),saved_parent_by_seed=groups(pair,['parent','seed'],pm),
        timing=timing,old_R_U_model_forwards=0,new_A_forward_cases=324,unrotated_inputs_reinferred=0,
        independent_source_people=6,transition_expressions=1,artifacts={str(p):sha(p) for p in [D/'FROZEN.json',D/'INPUT_AUDIT.json',D/'CASE_RESULTS.json']})
    write(D/'RESULTS.json',result);write(D/'COMPLETE.json',dict(results=str(D/'RESULTS.json'),sha256=sha(D/'RESULTS.json'),training_executed=False))
    print(json.dumps(result['total']),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        (A/f'ROTATION_CONTROL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
