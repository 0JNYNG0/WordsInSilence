"""Fixed six-angle NEW-input diagnostic. Root launches after both saved audits.

Original unrotated inputs are NEVER inferred again: both baselines come from
saved canonical case records. No model optimization, label changes or selection.
"""
from pathlib import Path
from collections import defaultdict
import datetime,hashlib,importlib.util,json,time,traceback,sys
import numpy as np
import torch
from rotation_core import rotate_hand63,canonicalize_hand63
from runtime_contract import prepare_shape_observation

sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent
OLD=U.parent/'hand_transition_learning'
D=U/'rotation_controls'
ANGLES=[-90,-60,-30,30,60,90]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def metrics(rows):
    observed=[r for r in rows if r['eligible']]
    emitted=[r for r in observed if r['confidence']>=.65]
    return dict(total=len(rows),observation_unknown=len(rows)-len(observed),eligible=len(observed),
        argmax_correct=sum(r['prediction']==r['target']for r in observed),
        emitted=len(emitted),confident_wrong=sum(r['prediction']!=r['target']for r in emitted),
        low_confidence=len(observed)-len(emitted),
        changed_from_saved_unrotated=sum(r['prediction']!=r['baseline_prediction']for r in observed),
        original_correct_became_wrong=sum(r['baseline_prediction']==r['target']and r['prediction']!=r['target']for r in observed),
        original_wrong_became_correct=sum(r['baseline_prediction']!=r['target']and r['prediction']==r['target']for r in observed))

def groups(rows,fields):
    g=defaultdict(list)
    for r in rows:g[tuple(r[f]for f in fields)].append(r)
    return [dict(**dict(zip(fields,k)),counts=metrics(v))for k,v in sorted(g.items())]

def main():
    assert not D.exists(),'Preserve first diagnostic outcome; never silently rerun it'
    plan=read(U/'ROTATION_CONTROL_PLAN.json');assert plan['angles_degrees']==ANGLES
    for p,h in plan['pins'].items():assert sha(p)==h,p
    for unit in [OLD,U]:
        for p,h in read(unit/'FROZEN.json')['pins'].items():assert sha(p)==h,p
        for p,h in read(unit/'TRAINING_CONTRACT.json')['pins'].items():assert sha(p)==h,p
    assert (U/'saved_evaluation/COMPLETE.json').exists() and (U/'saved_evaluation/PAIRED_COMPARISON.json').exists()
    assert (OLD/'saved_evaluation/COMPLETE.json').exists()
    olddone=read(OLD/'TRAINING_COMPLETE.json');newdone=read(U/'TRAINING_COMPLETE.json')
    olditems=read(OLD/'EVALUATION_ITEMS.json');newitems=read(U/'EVALUATION_ITEMS.json');assert olditems==newitems
    oldcases=read(OLD/'saved_evaluation/CANONICAL_SOURCE_ROWS.json')
    newcases=read(U/'saved_evaluation/CANONICAL_SOURCE_ROWS.json')
    key=lambda r:(r['person'],r['seed'],r['clip_id'],r['kind'])
    newer={key(r):r for r in newcases};assert len(newer)==len(oldcases)==54
    assert {r['person']for r in oldcases}==set(range(2,8))
    assert {r['seed']for r in oldcases}=={42,43,44}
    for c in oldcases:
        n=newer[key(c)];assert c['target']==n['target'] and c.get('source_frames')==n.get('source_frames')
    runs={}
    for condition,done in [('original',olddone),('canonical_rotation',newdone)]:
        assert len(done['runs'])==18
        for r in done['runs']:
            assert sha(r['model'])==r['model_sha256']
            assert r['heldperson']not in r['split']['trainpeople']
            runs[condition,r['heldperson'],r['seed']]=r
    metadata=read(OLD/'INPUTS_COMPLETE.json')
    rows={(r['clip_id'],r['side']):r for r in metadata['sequences']}
    needed={c['clip_id']for c in oldcases};source_paths=[]
    for (cid,side),r in rows.items():
        if cid in needed and side=='R':
            assert sha(r['path'])==r['sha256'];source_paths.append(Path(r['path']))
    frozen=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='FIXED_NEW_INPUT_RULES_BEFORE_ANY_MODEL_FORWARD',angles_degrees=ANGLES,
        per_model_cases=324,total_new_forward_cases=648,
        selection='All54alreadylabelselected source/seed canonical cases times6fixed nonzeroangles. No seed/person/angle selection by outcomes.',
        labels='Same observed hand-state transition/stable targets. Rotating a hand coordinate system does not create a filmed normal KSL video or semantic-correctness label.',
        original_input='Original prepared hand-local63, canonical case span/reverse, then one constant rotationangle acrosswindow.',
        new_input='The SAME rotated original hand-local63 passed through frozen canonicalize_hand63 once, before the corresponding newmodel. No secondleftreflection or scalechange.',
        abstention='All54×6 remain in each denominator. Missing/degeneratebasis or requiredpoint unavailable: unknown, no modelpositive. Modelconfidence cannot override observability.',
        classification='Fixed argmax andmaxsoftmax>=.65; no confidence calibration or threshold selection.',
        baselines='Each model own unrotated outputs read from saved canonical records only; neither original nor canonical completed evaluation rerun.',
        latency='Compute-only batch seconds, not end-to-end webcam/feedback latency.',
        limitations=['Data already informed development, especially REAL05. Not a fresh final test.','Constant imageplane rotation controls do not test3D foreshortening, occlusion or natural filmedwrongactions.','No new training, download, private data, final8 or web change.'],
        cases=oldcases,
        pins={str(p):sha(p)for p in [Path(__file__),U/'rotation_core.py',U/'runtime_contract.py',U/'ROTATION_CONTROL_PLAN.json',
            OLD/'hand_core.py',OLD/'TRAINING_COMPLETE.json',U/'TRAINING_COMPLETE.json',OLD/'EVALUATION_ITEMS.json',U/'EVALUATION_ITEMS.json',
            OLD/'saved_evaluation/CANONICAL_SOURCE_ROWS.json',U/'saved_evaluation/CANONICAL_SOURCE_ROWS.json']+source_paths},
        model_pins={r['model']:r['model_sha256']for r in runs.values()})
    write(D/'FROZEN.json',frozen)
    spec=importlib.util.spec_from_file_location('_rotation_controls_original_architecture',OLD/'hand_core.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cache={k:np.load(r['path'],allow_pickle=False)for k,r in rows.items()if k[0]in needed and k[1]=='R'}
    outputs=[];timing=[];audit=[]
    for person in range(2,8):
        for seed in [42,43,44]:
            cases=[c for c in oldcases if c['person']==person and c['seed']==seed]
            assert len(cases)==3
            controls=[]
            for c in cases:
                for angle in ANGLES:
                    base=dict(person=person,seed=seed,clip_id=c['clip_id'],kind=c['kind'],angle=angle,target=c['target'],
                              source_frames=c.get('source_frames'),reverse=c.get('reverse'),source_case_available=c['available'])
                    if not c['available']:
                        controls.append((base,None,None,False));continue
                    w=olditems['windows'][c['window_id']];z=cache[c['clip_id'],'R'];a,b=w['span']
                    raw=z['x'][a:b+1].copy()
                    if w['reverse']:raw=raw[::-1].copy()
                    rot=rotate_hand63(raw,angle);canonical,observed,ca=prepare_shape_observation(rot)
                    original_canonical,original_valid,_=canonicalize_hand63(raw)
                    assert np.array_equal(rot[:,42:],raw[:,42:])
                    assert np.array_equal(ca['basis_valid'],original_valid)
                    assert np.allclose(canonical,original_canonical,atol=2e-6,rtol=1e-6)
                    # Originalhandstate and transition targets remain source targets;
                    # no prediction is allowed to relabel this geometric control.
                    eligible=bool(observed.all())
                    audit.append(dict(**base,frames=len(raw),eligible=eligible,
                                      basis_valid_frames=int(ca['basis_valid'].sum()),observed_frames=int(observed.sum()),
                                      canonical_transform_max_error=float(np.max(np.abs(canonical-original_canonical)))))
                    controls.append((base,rot,canonical,eligible))
            for condition,column in [('original',1),('canonical_rotation',2)]:
                run=runs[condition,person,seed]
                state=torch.load(run['model'],map_location='cpu',weights_only=True)
                model=core.HandObserver(state['mean'],state['std']);model.load_state_dict(state);model.eval();model.requires_grad_(False)
                eligible_ids=[i for i,r in enumerate(controls)if r[3]]
                predictions={};elapsed=0.
                if eligible_ids:
                    x,lens=core.padded([controls[i][column]for i in eligible_ids]);tic=time.perf_counter()
                    with torch.inference_mode():logits=model(x,lens)['transition'].numpy()
                    elapsed=time.perf_counter()-tic
                    assert np.isfinite(logits).all()
                    for i,v in zip(eligible_ids,logits):predictions[i]=v
                timing.append(dict(condition=condition,person=person,seed=seed,new_forward_cases=len(eligible_ids),batch_seconds=elapsed))
                for i,(c,raw,canonical,eligible)in enumerate(controls):
                    original=next(k for k in cases if k['clip_id']==c['clip_id']and k['kind']==c['kind'])
                    baseline=original if condition=='original'else newer[key(original)]
                    row=dict(c,condition=condition,eligible=eligible,baseline_prediction=baseline.get('prediction'),
                             baseline_confidence=baseline.get('confidence'),baseline_pass=baseline['passed'],
                             model_sha256=run['model_sha256'])
                    if not eligible:
                        row.update(state='observation_unknown',prediction=None,confidence=None,emitted=False,probabilities=None)
                    else:
                        v=predictions[i].astype(float);p=np.exp(v-v.max());p/=p.sum();j=int(p.argmax());conf=float(p[j])
                        row.update(state='observed'if conf>=.65 else'uncertain',prediction=j,confidence=conf,
                                   emitted=conf>=.65,probabilities=p.tolist())
                    outputs.append(row)
                del model,state
    assert len(outputs)==648
    assert all(sum(r['condition']==c for r in outputs)==324 for c in ['original','canonical_rotation'])
    write(D/'INPUT_AUDIT.json',audit);write(D/'CASE_RESULTS.json',outputs)
    result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='FIXED_NEW_ROTATION_INPUT_DIAGNOSTIC_COMPLETE',
        by_model=groups(outputs,['condition']),by_model_angle=groups(outputs,['condition','angle']),
        by_model_seed=groups(outputs,['condition','seed']),by_model_kind=groups(outputs,['condition','kind']),
        by_model_person_kind=groups(outputs,['condition','person','kind']),
        by_model_seed_angle=groups(outputs,['condition','seed','angle']),
        compute_only_timing=timing,original_inputs_reinferred=0,new_inputs_per_model=324,
        original_source_people=6,source_transition_expression_count=1,
        limits=frozen['limitations'],threshold_or_label_changed=False,
        artifacts={str(p):sha(p)for p in[D/'FROZEN.json',D/'INPUT_AUDIT.json',D/'CASE_RESULTS.json']})
    write(D/'RESULTS.json',result)
    write(D/'COMPLETE.json',dict(at=result['at'],results=str(D/'RESULTS.json'),sha256=sha(D/'RESULTS.json'),
                               original_inputs_reinferred=0,training_executed=False))
    print(json.dumps(result['by_model'],ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (U/f'ROTATION_CONTROL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
