"""One fixed local RGB development pair. Train first; evaluation is a separate command."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F

U=Path(__file__).resolve().parent; B=U.parents[1]; OUT=B/'runs/rgb_tcn_development_pair_v1'
sys.path[:0]=[str(B.parent/'sign_model_pilot_v1/src'), str(B/'ops/rgb_pose_bridge_20261010'), str(B/'ops/rgb_features_and_browser_20261010')]
from pilot.model_masked_v1 import MaskedSignTCN
from pilot.data import build_channels
from rgb_bridge import PoseRGBBridge, FrozenRGBEncoder
from feature_cache import load as load_features, model_inputs


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, obj):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def init():
    c=read(U/'FROZEN_COMPARISON.json')
    assert all(sha(p)==h for p,h in c['pins'].items())
    torch.set_num_threads(2);torch.manual_seed(c['train']['seed']);torch.use_deterministic_algorithms(True)
    ck=torch.load(B/'runs/partial_encoder_pair_v1/checkpoint_control_final.pt',map_location='cpu',weights_only=True)
    parent=MaskedSignTCN(width=ck['width'],dilations=tuple(ck['dilations']),kernel_size=ck['kernel_size'],dropout=ck['dropout'],n_classes=len(ck['classes']))
    parent.load_state_dict(ck['state_dict'],strict=True);parent.eval();parent.requires_grad_(False)
    return c,parent,ck['classes']


def pack(items, rgb):
    n=len(items);t=max(len(i['x']) for i in items)
    x=torch.zeros(n,t,280);pad=torch.zeros(n,t);feat=torch.zeros(n,t,2,576);av=torch.zeros(n,t,2,dtype=torch.bool);est=torch.zeros_like(av)
    for j,it in enumerate(items):
        l=len(it['x']);x[j,:l]=torch.from_numpy(it['x']);pad[j,:l]=1
        if rgb and 'features' in it:
            feat[j,:l]=torch.from_numpy(it['features']);av[j,:l]=torch.from_numpy(it['available'])
    return (x,pad,feat,av,est),torch.tensor([i['y'] for i in items],dtype=torch.long)


def training_data(classes):
    cache=load(B/'ops/motion_input_contract_20260922/training_motion_inputs.npz')
    meta=read(B/'ops/context_training_pair_20260921/feature_metadata.json')
    assert cache['classes'].tolist()==classes
    replay=[]
    for group in ('base','pair_context'):
        for i,r in enumerate(meta[group]):
            a,b=cache[group+'_offsets'][i:i+2];y=int(cache[group+'_y'][i]);assert classes[y]==r['label'] and r['role']=='train'
            replay.append({'id':group+':'+r['id'],'x':cache[group+'_x'][a:b].copy(),'y':y})
    paired=[];pins={}
    for r in read(B/'ops/nikl18_rgb_pairing_20261010/TRAINING_INPUT_MANIFEST.json'):
        assert r['role']=='training_only_parent_exposed_original' and not r['id'].startswith('KSL')
        f=load_features(r['feature_path'],r['feature_sha256']);v=model_inputs(f,'B');pins[r['feature_path']]=r['feature_sha256']
        paired.append({'id':r['id'],'x':v['pose_x'].numpy(),'features':v['features'].numpy(),'available':v['available'].numpy(),'y':classes.index(r['label'])})
    assert len(replay)==2346 and len(paired)==131 and len({r['y'] for r in paired})==82
    assert not any('KSL' in r['id'] for r in replay)
    return replay,paired,pins


def balanced_schedule(pool,rng):
    labels=sorted({r['y'] for r in pool});groups={y:np.array([i for i,r in enumerate(pool) if r['y']==y]) for y in labels}
    return np.array([[rng.choice(groups[int(rng.choice(labels))]) for _ in range(16)] for _ in range(660)],dtype=np.int64)


def changed(model,before):
    return [k for k,v in model.state_dict().items() if not torch.equal(v,before[k])]


def train():
    c,parent,classes=init();OUT.mkdir(exist_ok=False)
    replay,paired,pins=training_data(classes)
    write(OUT/'SOURCE_PINS.json',dict(c['pins'],**pins))
    rng=np.random.RandomState(c['train']['seed']);ri=balanced_schedule(replay,rng);pi=balanced_schedule(paired,rng)
    with (OUT/'SCHEDULE.npz').open('xb') as f:np.savez_compressed(f,replay=ri,paired=pi)
    teacher=[]
    with torch.inference_mode():
        for first in range(0,len(replay),32):
            args,_=pack(replay[first:first+32],False);teacher.append(parent(args[0],args[1]).double())
    teacher=torch.cat(teacher)
    with (OUT/'TEACHER_TRAIN_ONLY.npz').open('xb') as f:np.savez_compressed(f,logits=teacher.numpy(),ids=np.array([r['id'] for r in replay]))
    probes=paired[:2]+[r for r in paired if r['id'] in ('NIKL_GONGBU_01','NIKL_MEOKDA_01')]
    bridge=PoseRGBBridge(parent);args,y=pack(probes,True)
    with torch.no_grad():
        assert torch.equal(bridge(*args),parent(args[0],args[1]))
    bridge.base.head.requires_grad_(True)
    F.cross_entropy(bridge(*args),y).backward()
    grads=[k for k,p in bridge.named_parameters() if p.grad is not None and bool(p.grad.abs().sum()>0)]
    assert 'rgb_projection.weight' in grads and all(k.startswith('base.head.') or k=='rgb_projection.weight' for k in grads)
    write(OUT/'PREFLIGHT.json',{'status':'PASS','replay':len(replay),'paired':len(paired),'paired_classes':82,
          'zero_projection_equals_parent_exactly':True,'gradient_reaches_only_allowed_params':grads,
          'no_KSL_or_user_training':True,'same_schedule_for_both_arms':True,'teacher_checkpoint':c['parent'],
          'old_ancestor_teacher_cache_ignored':True,'config_sha256':sha(U/'FROZEN_COMPARISON.json'),'implementation_sha256':sha(__file__)})
    records=[];start=time.perf_counter()
    for arm in ('A','B'):
        m=PoseRGBBridge(parent).eval();m.base.head.requires_grad_(True);m.rgb_projection.requires_grad_(arm=='B')
        before={k:v.detach().clone() for k,v in m.state_dict().items()};named=[(k,p) for k,p in m.named_parameters() if p.requires_grad]
        opt=torch.optim.AdamW([p for k,p in named],lr=.001,weight_decay=.001,betas=(.9,.999),eps=1e-8)
        losses=[]
        for step in range(660):
            items=[replay[i] for i in ri[step]]+[paired[i] for i in pi[step]];args,y=pack(items,arm=='B');scores=m(*args)
            ce=F.cross_entropy(scores,y)
            anchor=sum((p.double()-before[k].double()).square().sum() for k,p in named)*.005
            delta=scores[:16].double()-teacher[ri[step]];delta=delta-delta.mean(1,keepdim=True);retention=delta.square().mean()*.5
            loss=ce+anchor+retention;assert bool(torch.isfinite(loss))
            opt.zero_grad(set_to_none=True);loss.backward();assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for k,p in named)
            opt.step();losses.append([float(ce.detach()),float(anchor.detach()),float(retention.detach()),float(loss.detach())])
            if (step+1)%100==0 or step==659:
                progress={'arm':arm,'step':step+1,'steps':660,'elapsed_seconds':time.perf_counter()-start}
                temp=OUT/'progress.tmp';temp.write_text(json.dumps(progress));temp.replace(OUT/'progress.json');print(progress,flush=True)
        changes=changed(m,before);assert all(k.startswith('base.head.') or (arm=='B' and k=='rgb_projection.weight') for k in changes)
        if arm=='A':assert not m.rgb_projection.weight.any()
        else:assert 'rgb_projection.weight' in changes
        dest=OUT/('checkpoint_'+arm+'.pt')
        torch.save({'state_dict':m.state_dict(),'classes':classes,'arm':arm,'parent_sha256':c['parent'],'config_sha256':sha(U/'FROZEN_COMPARISON.json'),'steps':660},dest)
        with (OUT/('losses_'+arm+'.npz')).open('xb') as f:np.savez_compressed(f,losses=np.array(losses))
        records.append({'arm':arm,'path':str(dest),'sha256':sha(dest),'changed_tensors':changes,'trainable_tensors':[k for k,p in named],'trainable_parameters':sum(p.numel() for k,p in named)})
    assert all(sha(p)==h for p,h in read(OUT/'SOURCE_PINS.json').items())
    write(OUT/'FROZEN_MODELS.json',{'status':'BOTH_TRAINED_FROZEN_BEFORE_EVALUATION','models':records,'schedule_sha256':sha(OUT/'SCHEDULE.npz'),'steps_each':660,'seconds':time.perf_counter()-start})


def load_models(parent):
    rec=read(OUT/'FROZEN_MODELS.json');models={}
    for r in rec['models']:
        assert sha(r['path'])==r['sha256'];m=PoseRGBBridge(parent);m.load_state_dict(torch.load(r['path'],map_location='cpu',weights_only=True)['state_dict'],strict=True);m.eval();m.requires_grad_(False);models[r['arm']]=m
    return models


def evaluate():
    c,parent,classes=init();assert not (OUT/'EVALUATION_SCORES.npz').exists();models=load_models(parent)
    old=read(B/'runs/stgcn_localshape_v1/EVALUATION_ROWS.json');assert old['classes']==classes
    rows=[];items=[];saved=[]
    for r in old['rows']:
        if r['group']!='annotation_dev':continue
        assert r['signer_id'] in (11,12,13) and sha(r['path'])==r['sha256']
        z=load(r['path']);x=build_channels(z['xy_native'],z['coordinate_present'],z['timestamp_nominal_us'])['x']
        items.append({'x':x,'y':classes.index(r['label'])});rows.append({k:r[k] for k in ('id','label','group','signer_id')});saved.append(r['baseline_scores'])
    assert len(items)==699
    for r in read(B/'runs/ksl77_rgb_paired_v1/MANIFEST.json'):
        z=load_features(r['feature_path'],r['feature_sha256']);v=model_inputs(z,'B');target=next(o for o in old['rows'] if o['group']=='ksl36' and o['id']==r['sample_id'])
        assert target['label']==r['label_metadata_only']
        items.append({'x':v['pose_x'].numpy(),'features':v['features'].numpy(),'available':v['available'].numpy(),'y':classes.index(r['label_metadata_only'])})
        rows.append({'id':r['sample_id'],'label':r['label_metadata_only'],'group':'ksl_previous2' if r['split']=='previous2' else 'ksl_primary34','signer_id':None})
        saved.append(target['baseline_scores'])
    write(OUT/'EVALUATION_ROWS.json',{'classes':classes,'rows':rows})
    scores={'baseline':np.array(saved),'y':np.array([i['y'] for i in items])}
    with torch.inference_mode():
        for arm,m in models.items():
            blocks=[]
            for first in range(0,len(items),16):
                args,_=pack(items[first:first+16],arm=='B');blocks.append(m(*args).numpy())
            scores[arm]=np.concatenate(blocks)
    with (OUT/'EVALUATION_SCORES.npz').open('xb') as f:np.savez_compressed(f,**scores)
    write(OUT/'EVALUATION_COMPLETE.json',{'rows':len(rows),'config_sha256':sha(U/'FROZEN_COMPARISON.json'),'scores_sha256':sha(OUT/'EVALUATION_SCORES.npz'),'models_sha256':sha(OUT/'FROZEN_MODELS.json'),'baseline_scores_reused':True,'no_settings_changed_after_scores':True})
    print('evaluation saved',len(rows),flush=True)


if __name__=='__main__':
    try:
        if sys.argv[1]=='train':train()
        elif sys.argv[1]=='evaluate':evaluate()
        else:raise ValueError('Expected train or evaluate')
    except Exception:
        if not (U/'FIRST_FAILURE.json').exists():write(U/'FIRST_FAILURE.json',{'stage':sys.argv[1:],'traceback':traceback.format_exc()})
        raise
