"""One fixed FIT-only continuation; no encoder, probe, DEV, or web inference."""
from pathlib import Path
import json, sys, time, traceback
import numpy as np
import torch
from model import BodyProgressHead, FAMILIES

U = Path(__file__).resolve().parent
sys.path.insert(0, str(U.parent))
from common import read, write, sha, stamp

def stop():
    for p in [U, U.parent, U.parent.parent]:
        for n in ['STOP', 'STOP.txt', 'STOP_REQUESTED', 'STOP_REQUESTED.json']:
            assert not (p/n).exists()

def pins_match(pins):
    for p, h in pins.items():
        assert sha(p) == h, p

def protected_match(pins):
    assert isinstance(pins, dict) and len(pins) == 477
    assert pins == read(U.parent.parent/'PROTECTIONS.json')
    pins_match(pins)

def freeze():
    stop()
    assert not (U/'TRAINING_FROZEN.json').exists()
    d = read(U/'INPUTS_COMPLETE.json')
    protected = d['protected_pins']
    protected_match(protected)
    assert d['family_order'] == FAMILIES
    rows = d['sequences']
    assert len(rows) == 144 and len({r['clip_id'] for r in rows}) == 24
    assert {r['person'] for r in rows} == {2, 3, 4, 5, 6, 7}
    assert all(r['split'] == 'train' for r in rows)
    assert {r['seed'] for r in d['parent_models']} == {42, 43, 44}
    pins = dict(d['pins'])
    for p in [Path(__file__), U/'model.py', U/'INPUTS_COMPLETE.json',
              U.parent/'observed_action_anchor_design/progress_order.py']:
        pins[str(p)] = sha(p)
    for r in rows:
        pins[r['path']] = r['sha256']
    for r in d['parent_models']:
        pins[r['model']] = r['model_sha256']
        pins[r['predictions']] = r['predictions_sha256']
    pins_match(pins)
    write(U/'TRAINING_FROZEN.json', dict(at=stamp(), status='FIXED_FIT_BODY_PROGRESS_SUPERVISION',
        seeds=[42,43,44], steps=1200, lr=.001, weight_decay=.001, gradient_clip=2,
        architecture='150 body context ->96 GELU ->5 family body-progress logits',
        initialization='Matching immutable boundary_extended hidden layer and output rows [2,5,8,11,14]; original mean/std preserved.',
        loss='Full batch masked BCE; each family/class 1/10, equal source, then equal eligible views/reversals, then equal known points.',
        labels='Existing directly reviewed movement witnesses positive. Eligible reviewed body holds outside uncertain boundary bands negative. Unknown points remain untrained.',
        teacher_scope='Body movement vs held body pose only; held pose is not an incorrect sign or absent action. No completion labels.',
        rate='Original sequence and original timestamps only; no new temporal augmentation in this controlled continuation.',
        inference='New model FIT final predictions once. Old initial logits reused from immutable parent prediction arrays.',
        runtime_readout='Fixed >=0 logits on qualified original body observations; retain all support islands, require >=2 distinct points. Original candidate extents retained separately; no expected order, no overlap tolerance.',
        no_encoder_training=True, no_word_model_change=True, no_private=True, no_final8=True,
        no_probe_or_DEV=True, web_changed=False, pins=pins, protected_pins=protected))

def metrics(logits, old, data):
    rows=[]
    for j, family in enumerate(FAMILIES):
        for label in [0,1]:
            take=(data['mask'][:,j]>0)&(data['y'][:,j]==label)
            rows.append(dict(family=family, label=label, count=int(take.sum()),
                old_correct=int(((old[take,j]>=0)==bool(label)).sum()),
                new_correct=int(((logits[take,j]>=0)==bool(label)).sum())))
    return rows

def run():
    stop()
    assert not (U/'TRAINING_STARTED.json').exists() and not (U/'TRAINING_COMPLETE.json').exists()
    cfg=read(U/'TRAINING_FROZEN.json'); pins_match(cfg['pins'])
    meta=read(U/'INPUTS_COMPLETE.json'); sequences=meta['sequences']
    assert cfg['protected_pins'] == meta['protected_pins']
    protected_match(cfg['protected_pins'])
    data=[]
    for r in sequences:
        assert sha(r['path']) == r['sha256'] and r['split']=='train'
        with np.load(r['path'], allow_pickle=False) as z:
            a={k:z[k].copy() for k in ['x','y','mask','observed','weights','label_provenance']}
        n=len(a['x'])
        assert a['x'].shape==(n,150) and all(a[k].shape==(n,5) for k in a if k!='x')
        assert np.isfinite(a['x']).all() and np.isfinite(a['weights']).all()
        assert (a['weights']>=0).all() and np.all(a['weights'][a['mask']==0]==0)
        assert np.all(a['mask']<=a['observed'])
        data.append(a)
    flat={k:np.concatenate([a[k] for a in data]) for k in data[0]}
    assert np.isclose(flat['weights'].sum(),1)
    for j in range(5):
        for label in [0,1]:
            take=(flat['mask'][:,j]>0)&(flat['y'][:,j]==label)
            assert np.isclose(flat['weights'][take,j].sum(),.1)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    x=torch.from_numpy(flat['x']); y=torch.as_tensor(flat['y'],dtype=torch.float32)
    weight=torch.as_tensor(flat['weights'],dtype=torch.float32)
    write(U/'TRAINING_STARTED.json',dict(at=stamp(),frozen_sha256=sha(U/'TRAINING_FROZEN.json')))
    runs=[]
    for seed in cfg['seeds']:
        stop(); torch.manual_seed(seed)
        parent=next(r for r in meta['parent_models'] if r['seed']==seed)
        st=torch.load(parent['model'],map_location='cpu',weights_only=True)
        net=BodyProgressHead.from_boundary(st)
        dst=U/'runs'/str(seed); dst.mkdir(parents=True,exist_ok=False)
        opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        logs=[]; tick=time.perf_counter(); net.train()
        for step in range(cfg['steps']):
            stop()
            pred=net(x)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(pred,y,reduction='none')*weight).sum()
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']); opt.step()
            if step%300==0 or step==cfg['steps']-1:
                logs.append(dict(step=step+1,loss=float(loss.detach())))
                print('body progress',seed,step+1,float(loss.detach()),flush=True)
        seconds=time.perf_counter()-tick
        assert torch.equal(net.mean,st['mean']) and torch.equal(net.std,st['std'])
        net.eval(); torch.save(net.state_dict(),dst/'model.pt')
        preds={}; initial=[]; offsets=[]; cursor=0
        with np.load(parent['predictions'],allow_pickle=False) as old:
            with torch.inference_mode():
                for i,(r,a) in enumerate(zip(sequences,data)):
                    preds[f'logits_{i}']=net(torch.from_numpy(a['x'])).numpy()
                    initial.append(old[f"logits_{r['original_sequence_index']}"][:,:,2].copy())
                    offsets.append(dict(input_index=i,original_sequence_index=r['original_sequence_index'],clip_id=r['clip_id'],view=r['view'],reverse=r['reverse'],start=cursor,length=len(a['x'])))
                    cursor+=len(a['x'])
        np.savez_compressed(dst/'FIT_PREDICTIONS.npz',**preds)
        write(dst/'FIT_PREDICTION_INDEX.json',dict(rows=offsets,source_inputs_sha256=sha(U/'INPUTS_COMPLETE.json')))
        new=np.concatenate(list(preds.values())); old=np.concatenate(initial)
        assert new.shape==old.shape==flat['y'].shape
        row=dict(seed=seed,condition='body_progress_vs_reviewed_hold',model=str(dst/'model.pt'),model_sha256=sha(dst/'model.pt'),
                 predictions=str(dst/'FIT_PREDICTIONS.npz'),predictions_sha256=sha(dst/'FIT_PREDICTIONS.npz'),
                 prediction_index=str(dst/'FIT_PREDICTION_INDEX.json'),prediction_index_sha256=sha(dst/'FIT_PREDICTION_INDEX.json'),
                 initial_model=parent['model'],initial_model_sha256=parent['model_sha256'],normalization_unchanged=True,
                 steps=cfg['steps'],seconds=seconds,log=logs,FIT=metrics(new,old,flat))
        runs.append(row);write(U/'TRAINING_PROGRESS.json',dict(at=stamp(),completed=len(runs),rows=runs))
    pins_match(cfg['pins']);protected_match(cfg['protected_pins']);stop()
    write(U/'TRAINING_COMPLETE.json',dict(at=stamp(),status='BODY_PROGRESS_TRAINED_NOT_ADOPTED',rows=runs,
        source_clips=24,people=6,sequences=144,training_frozen_sha256=sha(U/'TRAINING_FROZEN.json'),
        whole_action_or_educational_completion=None,no_old_model_inference=True,no_encoder_training=True,
        no_private=True,no_probe_or_DEV=True,no_final8=True,web_changed=False))

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1] in ['freeze','run-reviewed']
        freeze() if sys.argv[1]=='freeze' else run()
    except BaseException:
        U.mkdir(parents=True,exist_ok=True)
        (U/f'FIRST_TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8')
        raise
