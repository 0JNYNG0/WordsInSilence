"""Hand-adapted encoder comparison. Preserve earlier heads; open final test once."""
from pathlib import Path
import argparse,json,time,os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import encode_public as enc
import train_observer as tr
U=Path(__file__).resolve().parent

def load_ft(name,seed):
    p=U/'models'/f'ft_{name}_{seed}.pt';z=torch.load(p,map_location='cpu',weights_only=True)
    m,forward,_=enc.load_encoder(name);m.load_state_dict(z['model'],strict=True);m.eval().requires_grad_(False)
    return m,forward,z,p

def prepare(c):
    complete=tr.read(U/'FINETUNE_COMPLETE.json')
    assert all(tr.sha(Path(p))==h for p,h in complete['checkpoints'].items())
    selection=U/'ADAPTED_ENCODER_SELECTION.json'
    if not selection.exists():
        candidates=[]
        for name in c['encoder_candidates']:
            group=[tr.read(U/'models'/f'ft_{name}_{seed}.json') for seed in c['seeds']]
            candidates.append({'name':name,'validation_macro_f1':float(np.mean([r['validation']['macro_f1_supported'] for r in group])),
                              'validation_accuracy':float(np.mean([r['validation']['accuracy'] for r in group])),
                              'p95_ms':float(np.mean([r['cpu_full_model_ms']['p95'] for r in group]))})
        best=max(r['validation_macro_f1'] for r in candidates)
        chosen=min([r for r in candidates if r['validation_macro_f1']>=best-.01],key=lambda r:r['p95_ms'])['name']
        tr.save(selection,{'selected':chosen,'candidates':candidates,'seed_for_common_features':c['seeds'][0],
          'selection':'Mean validation macroF1; within1pp lower latency; fixed first seed for common features, not best run',
          'test_seen':False,'scope':'Bounded last-stage adaptation recipe, not global architectural optimum'})
    selected=tr.read(selection);name=selected['selected'];seed=selected['seed_for_common_features']
    fp=U/'data/adapted_rgb_features.npz';meta=U/'data/ADAPTED_RGB_META.json'
    if fp.exists():
        assert tr.read(meta)['features_sha256']==tr.sha(fp)
        z=np.load(fp,allow_pickle=False);assert list(z['ids'])==[r['id'] for r in c['rows']]
        return torch.tensor(z['features'])
    m,forward,cp,p=load_ft(name,seed);out=[]
    with torch.inference_mode():
        for start in range(0,len(c['rows']),32):
            x=torch.stack([enc.crop_tensor(r['image']) for r in c['rows'][start:start+32]])
            f=F.normalize(forward(enc.normalize(x,name)),dim=-1);assert torch.isfinite(f).all();out.append(f.numpy())
    features=np.concatenate(out);np.savez_compressed(fp,features=features,ids=np.array([r['id'] for r in c['rows']]))
    tr.save(meta,{'features_sha256':tr.sha(fp),'checkpoint_sha256':tr.sha(p),'selected':name,'seed':seed,
                 'contract_sha256':tr.sha(U/'EXPERIMENT.json'),'test_scores_seen':False,'labels_used_for_extraction':False})
    return torch.tensor(features)

def train(c):
    rgb=prepare(c);joint,_=tr.joints(c);rr=tr.roles(c);y=torch.tensor([r['target'] for r in c['rows']]);spec=c['temporal']
    plan=U/'ADAPTED_FUSION_PROTOCOL.json'
    if not plan.exists():tr.save(plan,{'experiment_sha256':tr.sha(U/'EXPERIMENT.json'),'selection_sha256':tr.sha(U/'ADAPTED_ENCODER_SELECTION.json'),
      'feature_sha256':tr.sha(U/'data/adapted_rgb_features.npz'),'joint_sha256':tr.sha(U/'data/PUBLIC_JOINTS.npz'),
      'code_sha256':tr.sha(Path(__file__)),'new_conditions':['R_T','RJ_F','RJ_T'],'unchanged_conditions_reused':['J_T','MASK_T'],
      'seeds':c['seeds'],'training':spec,'test_seen':False,
      'purpose':'Same four inputs after source-hand RGB adaptation; preserve all prior frozen-feature comparisons'})
    for cond in ['R_T','RJ_F','RJ_T']:
        x=tr.contexts(c,rgb,joint,cond)
        for seed in c['seeds']:
            stem=f'adapted_{cond}_{seed}';result=U/'models'/(stem+'.json');cp=U/'models'/(stem+'.pt')
            if result.exists():assert tr.read(result)['checkpoint_sha256']==tr.sha(cp);continue
            torch.manual_seed(seed);m=tr.Observer(x.shape[-1],45,cond!='RJ_F');t=time.perf_counter()
            trace=tr.train(m,x,y,rr['fit'],seed=seed,**{k:spec[k] for k in ['epochs','batch','lr','weight_decay']})
            val=tr.logits(m,x,rr['validation'])
            tr.checkpoint(cp,m,c,{'kind':'adapted_observer','condition':cond,'seed':seed,'dim':x.shape[-1]})
            vp=U/'models'/(stem+'_validation.npz');assert not vp.exists();np.savez_compressed(vp,indices=rr['validation'].numpy(),logits=val)
            obj={'condition':cond,'seed':seed,'checkpoint_sha256':tr.sha(cp),'protocol_sha256':tr.sha(plan),'trace':trace,
                 'validation':tr.metrics(y[rr['validation']].numpy(),val,c['classes']),'elapsed_seconds':time.perf_counter()-t,'test_seen':False}
            tr.save(result,obj);print(stem,obj['validation']['correct'],'/',284,flush=True)
    tr.save(U/'ADAPTED_TRAINING_FROZEN.json',{'protocol_sha256':tr.sha(plan),'checkpoints':{str(p):tr.sha(p) for p in (U/'models').glob('adapted_*.pt')},'test_seen':False})

def final(c):
    assert (U/'ADAPTED_TRAINING_FROZEN.json').exists()
    ft_pins=tr.read(U/'FINETUNE_COMPLETE.json')['checkpoints']
    adapted_pins=tr.read(U/'ADAPTED_TRAINING_FROZEN.json')['checkpoints']
    assert tr.read(U/'FINETUNE_COMPLETE.json')['protocol_sha256']==tr.sha(U/'FINETUNE_PROTOCOL.json')
    assert tr.read(U/'ADAPTED_TRAINING_FROZEN.json')['protocol_sha256']==tr.sha(U/'ADAPTED_FUSION_PROTOCOL.json')
    assert set(ft_pins)=={str(U/'models'/f'ft_{name}_{seed}.pt') for name in c['encoder_candidates'] for seed in c['seeds']}
    assert set(adapted_pins)=={str(U/'models'/f'adapted_{condition}_{seed}.pt') for condition in ['R_T','RJ_F','RJ_T'] for seed in c['seeds']}
    assert all(tr.sha(Path(p))==h for p,h in {**ft_pins,**adapted_pins}.items())
    pins={str(p):tr.sha(p) for p in [U/'EXPERIMENT.json',U/'FINETUNE_PROTOCOL.json',U/'ADAPTED_FUSION_PROTOCOL.json',U/'ADAPTED_ENCODER_SELECTION.json',U/'data/adapted_rgb_features.npz',U/'data/PUBLIC_JOINTS.npz',U/'adapted_comparison.py']}
    pins.update(ft_pins);pins.update(adapted_pins)
    pinfile=U/'ADAPTED_TEST_PINS.json'
    if pinfile.exists():assert tr.read(pinfile)['pins']==pins
    else:tr.save(pinfile,{'pins':pins,'test_started':False})
    rr=tr.roles(c);ii=rr['test'].numpy();y=np.array([r['target'] for r in c['rows']]);joint,seen=tr.joints(c)
    rgb=torch.tensor(np.load(U/'data/adapted_rgb_features.npz',allow_pickle=False)['features']);results=[]
    for p in [Path(p) for p in sorted(ft_pins)]+[Path(p) for p in sorted(adapted_pins)]:
        stem=p.stem;out=U/'models'/(stem+'_test.json');numeric=U/'models'/(stem+'_test.npz')
        if out.exists():
            old=tr.read(out)
            assert old['checkpoint_sha256']==tr.sha(p) and old['input_pins_sha256']==tr.sha(pinfile)
            assert old['numeric_sha256']==tr.sha(numeric)
            results.append(old);continue
        if numeric.exists():
            z=np.load(numeric,allow_pickle=False);assert str(z['checkpoint_sha256'])==tr.sha(p) and str(z['input_pins_sha256'])==tr.sha(pinfile)
            assert np.array_equal(z['indices'],ii);pred=z['logits']
        else:
            cp=torch.load(p,map_location='cpu',weights_only=True)
            if stem.startswith('ft_'):
                name=cp['model_name'];m,forward,_,_=load_ft(name,cp['seed']);head=nn.Linear(cp['head']['weight'].shape[1],45);head.load_state_dict(cp['head']);head.eval();outputs=[]
                with torch.inference_mode():
                    for start in range(0,len(ii),32):
                        im=torch.stack([enc.crop_tensor(c['rows'][i]['image']) for i in ii[start:start+32]])
                        outputs.append(head(F.normalize(forward(enc.normalize(im,name)),dim=-1)).numpy())
                pred=np.concatenate(outputs)
            else:
                meta=cp['meta'];x=tr.contexts(c,rgb,joint,meta['condition']);m=tr.Observer(meta['dim'],45,meta['condition']!='RJ_F');m.load_state_dict(cp['state_dict']);m.eval();pred=tr.logits(m,x,rr['test'])
            assert pred.shape==(382,45) and np.isfinite(pred).all()
            np.savez_compressed(numeric,indices=ii,logits=pred,checkpoint_sha256=np.array(tr.sha(p)),input_pins_sha256=np.array(tr.sha(pinfile)))
        assert pred.shape==(382,45) and np.isfinite(pred).all()
        detail={'stem':stem,'checkpoint_sha256':tr.sha(p),'input_pins_sha256':tr.sha(pinfile),'numeric_sha256':tr.sha(numeric),
                'test':tr.metrics(y[ii],pred,c['classes']),'subgroups':{}}
        masks={'detector_missing':~seen[ii],'detector_present':seen[ii]}
        for signer in c['summary']['test']['signers']:masks[signer]=np.array([c['rows'][i]['signer']==signer for i in ii])
        masks['near_label_change']=np.array([any(j>=0 and c['rows'][j]['target']!=c['rows'][i]['target'] for j in c['rows'][i]['context_indices']) for i in ii])
        for key,mask in masks.items():
            if mask.any():detail['subgroups'][key]=tr.metrics(y[ii][mask],pred[mask],c['classes'])
        tr.save(out,detail);results.append(detail);print(stem,'test',detail['test']['correct'],'/',382,flush=True)
    tr.save(U/'ADAPTED_COMPARISON.json',{'results':results,'test_tuning':False,'scope':'External manual45-handshape held-signers only'})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['train','final']);args=a.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    lock=U/'ADAPTED_RUNNING.json';assert not lock.exists();lock.write_text(json.dumps({'pid':os.getpid(),'stage':args.stage,'time':time.time()}))
    try:globals()[args.stage](tr.read(U/'EXPERIMENT.json'))
    except BaseException as e:
        (U/'logs'/('ADAPTED_ERROR_'+str(time.time_ns())+'.json')).write_text(json.dumps({'type':type(e).__name__,'message':str(e)}));raise
    finally:lock.unlink()
