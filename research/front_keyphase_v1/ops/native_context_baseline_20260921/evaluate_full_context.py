"""Fixed-candidate native development diagnostic. Old annotation predictions are reused."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib
import json
import sys
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;LAB=HERE.parents[1]
sys.path[:0]=[str(LAB),str(LAB/'src'),str(LAB.parent/'sign_model_pilot_v1/src')]
import tools_prep_union93_candidate as union
from pilot.data import collate
from wis_lab.service.neural import PROJECT_18

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    plan=read(HERE/'plan.json')
    assert not (HERE/'result.json').exists() and not (HERE/'features_and_scores.npz').exists()
    for path,digest in plan['source_pins'].items():assert sha(path)==digest,path
    assert sha(plan['checkpoint']['path'])==plan['checkpoint']['sha256']
    payload=torch.load(plan['checkpoint']['path'],map_location='cpu',weights_only=True)
    model,classes=union.rebuild_from_payload(payload)
    assert len(classes)==93
    model.eval().requires_grad_(False);torch.set_num_threads(2)
    targets=set(PROJECT_18.values());target_ids=[i for i,c in enumerate(classes) if c in targets]
    weight=payload['state_dict']['head.weight'].numpy().astype(np.float64)
    bias=payload['state_dict']['head.bias'].numpy().astype(np.float64)
    captured=[]
    hook=model.head.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().numpy().copy()[0]))
    rows=[];features=[];scores_saved=[];ids=[];max_affine=0.
    try:
        with torch.inference_mode():
            for spec in plan['inputs']:
                row={k:spec[k] for k in ('clip_id','signer_id','reference_label','annotation_reference_id')}
                row['status']='ERROR'
                try:
                    assert spec['signer_id'] in (11,12,13) and spec['split_role']=='validation'
                    assert sha(spec['npz'])==spec['npz_sha256']
                    with np.load(spec['npz'],allow_pickle=False) as saved:
                        x=saved['x'].copy()
                    assert x.shape==(spec['frames'],280) and np.isfinite(x).all()
                    bx,pad,_=collate([{'x':x,'frames':len(x),'label':0}])
                    direct=model(torch.from_numpy(bx),torch.from_numpy(pad))[0].numpy()
                    feature=captured[-1].astype(np.float64)
                    scores=weight@feature+bias
                    max_affine=max(max_affine,float(np.max(np.abs(scores-direct))))
                    assert max_affine<1e-4
                    features.append(feature);scores_saved.append(scores);ids.append(spec['clip_id'])
                    yi=classes.index(spec['reference_label']);scopes={'full93':list(range(93))}
                    if spec['reference_label'] in targets:scopes['project18']=target_ids
                    assert ('project18' in scopes)==(spec['annotation_prediction_project18'] is not None)
                    comparisons={}
                    for scope,indices in scopes.items():
                        rank=sorted(indices,key=lambda i:-float(scores[i]))
                        old=spec['annotation_prediction_'+scope]
                        old_ok=old==spec['reference_label'];new_ok=rank[0]==yi
                        other=next(i for i in rank if i!=yi)
                        comparisons[scope]={'annotation_prediction':old,'annotation_correct':old_ok,
                            'whole_prediction':classes[rank[0]],'whole_correct':new_ok,'whole_true_rank':rank.index(yi)+1,
                            'whole_true_margin':float(scores[yi]-scores[other]),
                            'change':'right_to_wrong' if old_ok and not new_ok else 'wrong_to_right' if not old_ok and new_ok else 'correct_both' if old_ok else 'wrong_both'}
                    row.update(status='EVALUATED',input_frames=len(x),annotation_frames=spec['annotation_frames'],
                        input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),comparisons=comparisons)
                except Exception as exc:
                    row.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
                rows.append(row)
    finally:
        hook.remove()
    metrics={}
    for scope in ('full93','project18'):
        applicable=[r for r in rows if r['status']=='EVALUATED' and scope in r['comparisons']]
        metrics[scope]={'n':len(applicable),'ground_truth_classes':len({r['reference_label'] for r in applicable}),
            'annotation_correct':sum(r['comparisons'][scope]['annotation_correct'] for r in applicable),
            'whole_correct':sum(r['comparisons'][scope]['whole_correct'] for r in applicable),
            'changes':dict(Counter(r['comparisons'][scope]['change'] for r in applicable)),
            'per_label':{label:{'n':sum(r['reference_label']==label for r in applicable),
                'annotation_correct':sum(r['comparisons'][scope]['annotation_correct'] for r in applicable if r['reference_label']==label),
                'whole_correct':sum(r['comparisons'][scope]['whole_correct'] for r in applicable if r['reference_label']==label)} for label in sorted({r['reference_label'] for r in applicable})}}
    for path,digest in plan['source_pins'].items():assert sha(path)==digest,path
    assert sha(plan['checkpoint']['path'])==plan['checkpoint']['sha256']
    with (HERE/'features_and_scores.npz').open('xb') as f:
        np.savez_compressed(f,features=np.stack(features),scores=np.stack(scores_saved),clip_ids=np.array(ids),classes=np.array(classes))
    result={'status':'COMPLETE_PENDING_INDEPENDENT_RECOUNT','at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'plan_sha256':sha(HERE/'plan.json'),'script_sha256':sha(__file__),'arrays_sha256':sha(HERE/'features_and_scores.npz'),
        'checkpoint':plan['checkpoint'],'rows':rows,'metrics':metrics,'new_encoder_forwards':len(captured),
        'reused_annotation_predictions':74,'errors':sum(r['status']=='ERROR' for r in rows),'max_affine_error':max_affine,
        'training':0,'user_arrays_read':0,'reserved_arrays_read':0,'train_arrays_read':0,'new_extraction':0,
        'model_or_runtime_changes':0,'limits':['Public development/native detector domain; not an unseen-person webcam test',
            'Scope of project18 subset is30 records with6 actual labels, not all18-word coverage',
            'Context, duration and normalization vary together; exact source-video alignment and complete annotation coverage unverified',
            'Old annotation predictions have no saved logits; no annotation margins were inferred or recomputed']}
    with (HERE/'result.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'new_forwards':len(captured),'errors':result['errors'],'metrics':metrics},ensure_ascii=False),flush=True)
    assert len(captured)==74 and result['errors']==0


if __name__=='__main__':main()
