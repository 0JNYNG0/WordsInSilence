"""Read-only diagnostic using existing public caches and unchanged inference code."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

OPS=Path(__file__).resolve().parent;WEB=OPS.parent.parent
sys.path[:0]=[str(WEB),str(WEB/'src'),str(WEB.parent/'sign_model_pilot_v1/src')]
import tools_prep_union93_candidate as u
from pilot.data import collate
from wis_lab.core.observation import Observation
from wis_lab.service import neural

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(OPS/'plan.json')
assert not (OPS/'result.json').exists() and not (OPS/'features_and_scores.npz').exists()
for p,digest in plan['source_protected'].items():assert sha(p)==digest,p
payloads={}
for role,spec in plan['models'].items():
    assert sha(spec['path'])==spec['sha256']
    payloads[role]=torch.load(spec['path'],map_location='cpu',weights_only=True)
parent=payloads['parent'];classes=parent['classes']
assert len(classes)==93 and all(p['classes']==classes for p in payloads.values())
# Existing frozen-tensor proofs apply to these exact model hashes; no retraining.
model,rebuilt_classes=u.rebuild_from_payload(parent);assert rebuilt_classes==classes
model.eval();model.requires_grad_(False);torch.set_num_threads(2)
obs=Observation.load(plan['observation']);assert obs.n_frames==118
clips={};supports={}
with np.load(plan['native']['npz'],allow_pickle=False) as z:
    clips['native_annotation36']=z['x'].copy()
    supports['native_annotation36']={'frames':len(z['x']),'nominal_frame_ids':[int(z['frame_index'][0]),int(z['frame_index'][-1])],'timebase':'nominal30fps; video frame origin remains unverified'}
with np.load(plan['cached_web_input'],allow_pickle=False) as z:
    clips['web_entire118']=z['x'].copy()
    supports['web_entire118']={'frames':len(z['x']),'frame_ids':[int(z['frame_id'][0]),int(z['frame_id'][-1])],'pts_us':[int(z['timestamp_us'][0]),int(z['timestamp_us'][-1])],'input':'previously verified live-adapter cache, not recomputed'}
for condition in plan['conditions'][2:]:
    lo,hi=[x+condition['offset'] for x in plan['nominal_start_end']]
    assert 0<=lo<=hi<obs.n_frames
    window=slice(lo,hi+1)
    fields={key:getattr(obs,key)[window].copy() for key in ['xy_px','timestamp_us','frame_id','observed','usable','conf','hand_score']}
    cut=Observation(**fields,sidecar=dict(obs.sidecar))
    clip,why,stats=neural._channels(cut);assert clip is not None,why
    clips[condition['id']]=clip['x']
    supports[condition['id']]={'frames':len(clip['x']),'observation_row_indices':[lo,hi],'frame_ids':[int(cut.frame_id[0]),int(cut.frame_id[-1])],'pts_us':[int(cut.timestamp_us[0]),int(cut.timestamp_us[-1])],'adapter_support':stats,'normalization':'unchanged adapter recomputes on declared input window'}
captured=[]
hook=model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy().copy()[0]))
scope_indices={'project18':[classes.index(x) for x in neural.PROJECT_18.values()],'full93':list(range(93))}
assert len(scope_indices['project18'])==18
rows=[];features=[];all_scores=[];model_order=list(payloads);max_affine=0.
try:
    with torch.inference_mode():
        for ident,x in clips.items():
            assert x.ndim==2 and x.shape[1]==280 and np.isfinite(x).all()
            bx,pad,_=collate([{'x':x,'frames':len(x),'label':0}])
            parent_logits=model(torch.from_numpy(bx),torch.from_numpy(pad))[0].numpy()
            z=captured[-1].astype(np.float64);features.append(z)
            scores={role:p['state_dict']['head.weight'].numpy().astype(np.float64)@z+p['state_dict']['head.bias'].numpy().astype(np.float64) for role,p in payloads.items()}
            max_affine=max(max_affine,float(np.max(np.abs(scores['parent']-parent_logits))));assert max_affine<1e-4
            all_scores.append(np.stack([scores[role] for role in model_order]))
            record={'condition':ident,'input_shape':list(x.shape),'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'support':supports[ident],'models':{}}
            yi=classes.index(plan['reference_label'])
            for role,logits in scores.items():
                record['models'][role]={}
                for scope,ids in scope_indices.items():
                    rank=sorted(ids,key=lambda i:-float(logits[i]));other=next(i for i in rank if i!=yi)
                    record['models'][role][scope]={'top1':classes[rank[0]],'source_label_agreement':rank[0]==yi,'source_label_rank':rank.index(yi)+1,'source_margin':float(logits[yi]-logits[other]),'top3':[{'label':classes[i],'score':float(logits[i])} for i in rank[:3]]}
            rows.append(record)
finally:
    hook.remove()
assert len(captured)==len(rows)==plan['encoder_forwards_planned']==4
assert sum(len(r['models']) for r in rows)==plan['linear_head_outputs_planned']==12
for p,digest in plan['source_protected'].items():assert sha(p)==digest,p
np.savez_compressed(OPS/'features_and_scores.npz',features=np.stack(features),scores=np.stack(all_scores),classes=np.array(classes),models=np.array(model_order),conditions=np.array(list(clips)))
result={'status':'COMPLETED_PENDING_INDEPENDENT_RECOUNT','plan_sha256':sha(OPS/'plan.json'),'script_sha256':sha(__file__),'models':plan['models'],'rows':rows,'encoder_forwards':4,'linear_head_outputs':12,'optimizer_steps':0,'extractor_calls':0,'user_records_opened':0,'reserved_data_read':0,'max_parent_affine_error':max_affine,'source_pins_unchanged':True,'features_sha256':sha(OPS/'features_and_scores.npz'),'limits':[plan['role'],plan['semantic_limit'],'Different source conventions and unknown frame origin prevent isolating detector causality. Window results do not change production behavior.']}
with (OPS/'result.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':result['status'],'rows':[{k:v for k,v in r.items() if k in ['condition','input_shape','models']} for r in rows],'encoder_forwards':4},ensure_ascii=False))
