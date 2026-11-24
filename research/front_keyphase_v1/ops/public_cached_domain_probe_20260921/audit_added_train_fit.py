"""Fixed complete9 added-training-input diagnostic; no updates or new extraction."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import torch

OPS=Path(__file__).resolve().parent;WEB=OPS.parent.parent
sys.path[:0]=[str(WEB),str(WEB/'src'),str(WEB.parent/'sign_model_pilot_v1/src')]
import tools_prep_union93_candidate as u
from pilot.data import collate
from wis_lab.service import neural

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(OPS/'TRAIN_FIT_PLAN.json');assert not (OPS/'TRAIN_FIT_RESULT.json').exists()
models={}
for name,item in plan['models'].items():
    assert sha(item['path'])==item['sha256'];models[name]=torch.load(item['path'],map_location='cpu',weights_only=True)
parent=models['parent'];classes=parent['classes'];model,_=u.rebuild_from_payload(parent)
model.eval();model.requires_grad_(False);torch.set_num_threads(2)
with np.load(OPS/'features_and_scores.npz',allow_pickle=False) as z:
    saved_feature=z['features'][list(z['conditions']).index('native_annotation36')].copy()
features=[];head_scores=[];rows=[];captured=[]
hook=model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy().copy()[0]))
indices={'project18':[classes.index(x) for x in neural.PROJECT_18.values()],'full93':list(range(93))}
try:
    with torch.inference_mode():
        for item in plan['inputs']:
            assert sha(item['npz'])==item['npz_sha256'];assert item['split_role']=='train_candidate'
            if item['signer_id']==2:z=saved_feature
            else:
                with np.load(item['npz'],allow_pickle=False) as source:x=source['x'].copy()
                bx,pad,_=collate([{'x':x,'frames':len(x),'label':0}])
                model(torch.from_numpy(bx),torch.from_numpy(pad));z=captured[-1].astype(np.float64)
            features.append(z);outputs=[];reports={}
            for name,ck in models.items():
                logits=ck['state_dict']['head.weight'].numpy().astype(np.float64)@z+ck['state_dict']['head.bias'].numpy().astype(np.float64)
                outputs.append(logits);reports[name]={}
                for scope,ids in indices.items():
                    order=sorted(ids,key=lambda i:-logits[i]);yi=classes.index(plan['reference_label']);rival=next(i for i in order if i!=yi)
                    reports[name][scope]={'top1':classes[order[0]],'source_label_agreement':order[0]==yi,'source_label_rank':order.index(yi)+1,'margin':float(logits[yi]-logits[rival])}
            head_scores.append(outputs);rows.append({'id':item['clip_id'],'signer_id':item['signer_id'],'reference_label':plan['reference_label'],'source_sha256':item['npz_sha256'],'models':reports})
finally:hook.remove()
assert len(captured)==8 and len(rows)==9
summary={scope:{name:{'agree':sum(r['models'][name][scope]['source_label_agreement'] for r in rows),'total':9,'disagree_signers':[r['signer_id'] for r in rows if not r['models'][name][scope]['source_label_agreement']]} for name in models} for scope in indices}
np.savez_compressed(OPS/'train_features_and_scores.npz',features=np.stack(features),scores=np.asarray(head_scores),classes=np.array(classes),models=np.array(list(models)),ids=np.array([r['id'] for r in rows]))
report={'status':'COMPLETED_PENDING_RECOUNT','plan_sha256':sha(OPS/'TRAIN_FIT_PLAN.json'),'rows':rows,'summary':summary,'additional_encoder_calls':8,'reused_encoder_features':1,'optimizer_steps':0,'user_records_read':0,'reserved_data_read':0,'feature_scores_sha256':sha(OPS/'train_features_and_scores.npz'),'is_generalization':False}
with (OPS/'TRAIN_FIT_RESULT.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False))
