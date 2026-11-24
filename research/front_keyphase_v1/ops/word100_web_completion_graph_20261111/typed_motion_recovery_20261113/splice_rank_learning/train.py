"""Copied local_full rank continuation. CLI freeze is read-only toward old assets.

No training occurs on import or freeze. `run-reviewed` is reserved for root's
explicit post-review execution. No encoder, guide, source order or join input.
"""
import collections, sys, time, traceback
import numpy as np
from prepare import U, W, L, read, write, sha, check_pins
sys.path.insert(0, str(W))
from local_rank_core import LocalRankHead
import torch

def freeze():
    assert not (U/'TRAINING_FROZEN.json').exists()
    cfg=read(U/'FROZEN.json'); data=read(U/'INPUTS_COMPLETE.json')
    check_pins(cfg['pins'])
    assert read(U/'VERIFICATION.json')['status']=='SAVED_INPUT_CONTRACT_PASS_NO_TRAINING'
    recipe=cfg['proposed_training']
    pins={str(p):sha(p)for p in [U/'FROZEN.json',U/'INPUTS_COMPLETE.json',U/'VERIFICATION.json',
          U/'train.py',W/'local_rank_core.py',W/'train_local_rank.py']}
    for r in data['replay']+data['bags']:
        if not r.get('skipped',False):pins[r['path']]=r['sha256']
    for r in cfg['prior_models']:pins[r['model']]=r['model_sha256']
    write(U/'TRAINING_FROZEN.json',dict(status='IMPLEMENTED_PENDING_ROOT_REVIEW_NOT_EXECUTED',
        at=time.time(),recipe=recipe,architecture='Unchanged1160->96 GELU->5',
        input_fields=['local_full','y'],input_excluded=['word','guide_order','source_index','join','provenance','timestamp_position'],
        optimizer_steps=1200,source_family_pairing='Exact clip_id/view/reverse/family metadata for pairing ONLY; never model input.',
        per_step_loss='Mean over5 families; each family averages its2 sampled original source bags. Families2/3 use .5 old+.5 paired splice where full positives exist. Families4/5/7 old only.',
        initialization='Matching-seed original local_full state_dict copied exactly, including mean/std buffers; no normalization refit.',
        output_routing='Only family2/3 can use new scores. Every other runtime score/card/qualifier comes from saved old output.',
        no_model_selection=True,final_checkpoint_only=True,pins=pins))
    print('Training contract frozen; no training')

def load_training_data(meta):
    originals=[]; new={}
    def item(r):
        z=np.load(r['path']); x=z['local_full'].copy(); y=z['y'].copy()
        assert x.ndim==2 and x.shape[1]==1160 and np.isfinite(x).all()
        assert y.dtype==np.bool_ and y.any() and (~y).any()
        if 'rank_mask'in z.files:assert z['rank_mask'].all()
        return dict(row=r,x=x,y=y)
    key=lambda r:(r['clip_id'],r['view'],r['reverse'],r['family'])
    for r in meta['replay']:
        assert r['split']=='train' and r['person']in range(2,8)
        originals.append(item(r))
    for r in meta['bags']:
        assert r['family']in [2,3] and r['split']=='train' and r['person']in range(2,8)
        if r['skipped']:
            assert r['positive']==0 or r['negative']==0
            continue
        k=key(r);assert k not in new;new[k]=item(r)
    assert len(originals)==178
    assert set(new)<={key(d['row'])for d in originals}
    return originals,new,key

def sampled_batch(originals,new,key,groups,rng):
    xs=[]; pieces=[]; start=0
    for family in [2,3,4,5,7]:
        for index in rng.choice(groups[family],2,replace=False):
            original=originals[int(index)]; paired=new.get(key(original['row'])) if family in [2,3] else None
            entries=[(original,.5),(paired,.5)] if paired is not None else [(original,1.)]
            # Divide by2 sources and5 families. Every source retains equal total weight.
            for data,weight in entries:
                positive=np.flatnonzero(data['y']); negative=np.flatnonzero(~data['y'])
                negative=rng.choice(negative,min(56,len(negative)),replace=False)
                ids=np.r_[positive,negative];xs.append(data['x'][ids])
                pieces.append((start,start+len(ids),len(positive),[2,3,4,5,7].index(family),weight/10))
                start+=len(ids)
    assert np.isclose(sum(p[-1]for p in pieces),1)
    return np.concatenate(xs),pieces

def train():
    assert not (U/'TRAINING_COMPLETE.json').exists()
    assert not (U/'runs').exists(), 'Never overwrite or resume a partly executed experiment'
    cfg=read(U/'FROZEN.json'); frozen=read(U/'TRAINING_FROZEN.json'); meta=read(U/'INPUTS_COMPLETE.json')
    check_pins(cfg['pins']);check_pins(frozen['pins'])
    assert frozen['optimizer_steps']==1200
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    originals,new,key=load_training_data(meta)
    groups={f:[i for i,d in enumerate(originals)if d['row']['family']==f]for f in [2,3,4,5,7]}
    assert {f:len(v)for f,v in groups.items()}=={2:36,3:36,4:34,5:36,7:36}
    priors={r['seed']:r for r in cfg['prior_models']}; runs=[]
    recipe=frozen['recipe'];(U/'runs').mkdir()
    for seed in recipe['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        prior=priors[seed];state=torch.load(prior['model'],map_location='cpu',weights_only=True)
        model=LocalRankHead(state['mean'].numpy(),state['std'].numpy());model.load_state_dict(state)
        assert all(torch.equal(v,state[k])for k,v in model.state_dict().items())
        assert list(model.mean.shape)==[1160] and list(model.std.shape)==[1160]
        opt=torch.optim.AdamW(model.parameters(),lr=recipe['lr'],weight_decay=recipe['weight_decay'])
        model.train();log=[];tick=time.perf_counter()
        for step in range(recipe['steps']):
            x,pieces=sampled_batch(originals,new,key,groups,rng)
            logits=model(torch.tensor(x))
            loss=sum(weight*(torch.logsumexp(logits[a:b,j],0)-torch.logsumexp(logits[a:a+np_,j],0))
                     for a,b,np_,j,weight in pieces)
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
            if step%300==0 or step==recipe['steps']-1:
                record=dict(step=step+1,loss=float(loss.detach()));log.append(record)
                print('splice rank',seed,step+1,record['loss'],flush=True)
        model.eval()
        assert torch.equal(model.mean,state['mean']) and torch.equal(model.std,state['std'])
        dest=U/'runs'/str(seed);dest.mkdir();torch.save(model.state_dict(),dest/'model.pt')
        # Only the new head is evaluated on FIT data here; old outputs stay saved.
        predictions={}; references=[]
        with torch.inference_mode():
            for origin,items in [('original',originals),('splice',list(new.values()))]:
                for i,d in enumerate(items):
                    name=f'{origin}_{i}';predictions[name]=model(torch.tensor(d['x'])).numpy()
                    references.append(dict(key=name,path=d['row']['path'],sha256=d['row']['sha256'],family=d['row']['family']))
        np.savez_compressed(dest/'fit_predictions.npz',**predictions)
        write(dest/'FIT_PREDICTION_INDEX.json',references)
        runs.append(dict(seed=seed,condition='splice_rank',model=str(dest/'model.pt'),model_sha256=sha(dest/'model.pt'),
                         initial_model=prior['model'],initial_model_sha256=prior['model_sha256'],normalization_unchanged=True,
                         fit_predictions=str(dest/'fit_predictions.npz'),fit_predictions_sha256=sha(dest/'fit_predictions.npz'),
                         seconds=time.perf_counter()-tick,log=log,steps=recipe['steps']))
        write(U/'TRAINING_PROGRESS.json',dict(completed=runs))
    check_pins(cfg['pins']);check_pins(frozen['pins'])
    write(U/'TRAINING_COMPLETE.json',dict(status='TRAINED_NOT_SELECTED_OR_DEPLOYED',at=time.time(),rows=runs,
          source_contract_sha256=sha(U/'FROZEN.json'),training_contract_sha256=sha(U/'TRAINING_FROZEN.json'),
          encoder_unchanged=True,probe_dev_training_inputs=False,old_predictions_reused_only=True,web_changed=False))

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1]in ['freeze','run-reviewed']
        freeze() if sys.argv[1]=='freeze' else train()
    except BaseException:
        (U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
