"""Public-only fixed evaluation, reusing shared encoder and existing public features."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import torch

OPS=Path(__file__).resolve().parent;WEB=OPS.parent.parent
sys.path[:0]=[str(WEB),str(WEB/'src'),str(WEB.parent/'sign_model_pilot_v1/src')]
import tools_prep_union93_candidate as u
import tools_visibility_experiment as v
from pilot.data import collate
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (OPS/'PUBLIC_EVALUATION.json').exists()
audit=read(OPS/'REAL_TRAINING_AUDIT.json');assert audit['status']=='PASS'
models={}
for name,spec in audit['models'].items():
    assert sha(spec['path'])==spec['sha256'];models[name]=torch.load(spec['path'],map_location='cpu',weights_only=True)
parent_path=Path(read(OPS/'lambda_0p1_plan.json')['checkpoint']);parent=torch.load(parent_path,map_location='cpu',weights_only=True)
classes=parent['classes'];assert all(ck['classes']==classes for ck in models.values())
model,_=u.rebuild_from_payload(parent);model.eval();model.requires_grad_(False);torch.set_num_threads(2)
provenance=read(WEB/'runs/visibility_candidate_v1/provenance.json')
reference_path=Path(provenance['source_experiment']['directory'])/'inputs.json'
inputs=read(reference_path);items=[];eligible={};pins={str(reference_path):sha(reference_path)}
for row in inputs['nikl']:
    p=WEB/row['npz'];assert sha(p)==row['npz_sha256'];pins[str(p)]=row['npz_sha256']
    native,present,timestamps=u._native_from_nikl(str(p),row['npz_sha256'])
    items.append(dict(clip_id=row['id'],label=row['label'],nat=native,present=present,ts=timestamps))
    if row.get('reviewed_non_sign_hand') in ('left','right'):eligible[row['id']]={'side':row['reviewed_non_sign_hand']}
assert len(items)==18 and len(eligible)==5
clips=v.reference_clips(items,eligible);assert len(clips)==23
captured=[];features=[];rows=[]
hook=model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy().copy()[0]))
max_affine=0.
try:
    with torch.inference_mode():
        for clip in clips:
            bx,pad,_=collate([dict(x=clip['x'],frames=len(clip['x']),label=0)])
            actual=model(torch.from_numpy(bx),torch.from_numpy(pad))[0].numpy();z=captured[-1].astype(np.float64)
            exact=parent['state_dict']['head.weight'].numpy().astype(np.float64)@z+parent['state_dict']['head.bias'].numpy().astype(np.float64)
            max_affine=max(max_affine,float(np.max(np.abs(exact-actual))));assert max_affine<1e-4
            features.append(z);rows.append(dict(id=clip['clip_id'],true=clip['label'],condition=clip['condition'],domain='public_reference',models={}))
finally:hook.remove()
# This cache contains only9 public training examples; no participant arrays are opened.
probe=WEB/'ops/public_cached_domain_probe_20260921'
fit=read(probe/'TRAIN_FIT_RESULT.json');fitplan=read(probe/'TRAIN_FIT_PLAN.json')
assert sha(probe/'train_features_and_scores.npz')==fit['feature_scores_sha256']
assert fitplan['models']['parent']['sha256']==sha(parent_path)
pins[str(probe/'train_features_and_scores.npz')]=sha(probe/'train_features_and_scores.npz')
with np.load(probe/'train_features_and_scores.npz',allow_pickle=False) as z:
    assert list(z['classes'])==classes and list(z['ids'])==[r['id'] for r in fit['rows']]
    for i,item in enumerate(fit['rows']):
        features.append(z['features'][i].copy());rows.append(dict(id=item['id'],true=item['reference_label'],condition='additional_train',domain='native_source_training',signer_id=item['signer_id'],models={}))
assert len(captured)==23 and len(rows)==32
indices={'project18':[classes.index(label) for label in sorted({i['label'] for i in items})],'full93':list(range(93))}
score_rows=[]
for row,z in zip(rows,features):
    outputs=[]
    for name,ck in models.items():
        logit=ck['state_dict']['head.weight'].numpy().astype(np.float64)@z+ck['state_dict']['head.bias'].numpy().astype(np.float64);outputs.append(logit)
        row['models'][name]={}
        for scope,ids in indices.items():
            order=sorted(ids,key=lambda i:-logit[i]);yi=classes.index(row['true']);rival=next(i for i in order if i!=yi)
            row['models'][name][scope]={'pred':classes[order[0]],'correct':order[0]==yi,'rank':order.index(yi)+1,'margin':float(logit[yi]-logit[rival])}
    score_rows.append(outputs)
summaries={name:{scope:{condition:{'n':sum(r['condition']==condition for r in rows),'correct':sum(r['condition']==condition and r['models'][name][scope]['correct'] for r in rows),'errors':[{'id':r['id'],'true':r['true'],'pred':r['models'][name][scope]['pred']} for r in rows if r['condition']==condition and not r['models'][name][scope]['correct']]} for condition in ['intact','reviewed_masked','additional_train']} for scope in indices} for name in models}
for p,digest in pins.items():assert sha(p)==digest
np.savez_compressed(OPS/'public_features_scores.npz',features=np.asarray(features),scores=np.asarray(score_rows),classes=np.array(classes),models=np.array(list(models)),ids=np.array([r['id'] for r in rows]))
result={'status':'COMPLETED_PENDING_RECOUNT','policy_sha256':sha(OPS/'comparison_policy.json'),'models':audit['models'],'rows':rows,'summary':summaries,'public_inputs':pins,'encoder_forwards':23,'public_train_features_reused':9,'linear_head_outputs':128,'optimizer_steps':0,'user_records_or_scores_read':0,'reserved_arrays_read':0,'max_parent_affine_error':max_affine,'features_scores_sha256':sha(OPS/'public_features_scores.npz')}
with (OPS/'PUBLIC_EVALUATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':result['status'],'summary':summaries,'encoder_forwards':23},ensure_ascii=False))
