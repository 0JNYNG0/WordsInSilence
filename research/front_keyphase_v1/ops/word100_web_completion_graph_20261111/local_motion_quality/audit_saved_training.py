"""Independent saved-artifact arithmetic; never runs a model or optimizer."""
from pathlib import Path
import sys,json,hashlib,math,collections,datetime
import numpy as np
import torch
sys.dont_write_bytecode=True
from quality_model import LocalMotionQuality

W=Path(__file__).resolve().parent;U=W.parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def forbidden(*a,**kw):raise AssertionError('Saved-artifact audit forbids model forward')

def main():
    torch.nn.Module._call_impl=forbidden
    cfg=read(W/'FROZEN.json');ad=read(W/'TRAIN_ADMISSION.json');manifest=read(W/'INPUTS_COMPLETE.json');done=read(W/'TRAINING_COMPLETE.json')
    assert ad['approved'] is True
    assert all(sha(W/n)==h for n,h in ad['pins'].items())
    assert sha(ad['review'])==ad['review_sha256']
    pin_checks=[dict(path=p,expected=h,actual=sha(p)) for p,h in cfg['pins'].items()]
    assert all(r['actual']==r['expected'] for r in pin_checks)
    sources={r['clip_id']:r for r in cfg['sources']}
    ready={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['clip_id'] in sources}
    assert len(sources)==len(ready)==30
    assert all(r['role']=='fit' and r['person'] in (1,3,5,7) for r in ready.values())
    rows=manifest['rows'];train=[r for r in rows if r['split']=='train'];probe=[r for r in rows if r['split']=='head_probe']
    assert len(rows)==853 and len(train)==728 and len(probe)==125
    assert all(r['clip_id'] in sources and r['split']==cfg['split'][r['clip_id']] and r['source_sha256']==sources[r['clip_id']]['source_sha256'] for r in rows)
    assert {r['source_sha256'] for r in train}.isdisjoint({r['source_sha256'] for r in probe})
    assert len({r['source_sha256'] for r in train})==26 and len({r['source_sha256'] for r in probe})==4
    assert all(r['kind']=='source_original' and r['extent_mask']==1 and r['aux_mask']==0 for r in rows)
    for r in rows:
        assert sha(r['path'])==r['sha256']
        with np.load(r['path'],allow_pickle=False) as z:
            assert z['summary'].shape==(304,) and z['sequence'].shape==(r['frames'],11)
            assert np.isfinite(z['summary']).all() and np.isfinite(z['sequence']).all()
            assert np.diff(z['timestamps_us']).min()>0
            for key in ('extent_target','extent_mask','aux_target','aux_mask'):assert z[key].item()==r[key]
    class_counts=np.bincount([r['extent_target'] for r in train],minlength=4)
    assert class_counts.tolist()==[126,160,375,67]
    expected_steps=cfg['epochs']*math.ceil(len(train)/cfg['batch'])
    assert cfg['epochs']==40 and cfg['lr']==.001 and cfg['batch']==64 and cfg['weight_decay']==.01 and expected_steps==480
    train_source=(W/'train.py').read_text(encoding='utf-8')
    assert 'np.random.default_rng(seed*1000+epoch).permutation(len(train))' in train_source
    assert '1/np.sqrt(counts)' in train_source and "lr=cfg['lr'],weight_decay=cfg['weight_decay']" in train_source
    expected_pairs={(c,s) for c in ('B','C') for s in (42,43,44)}
    assert {(r['condition'],r['seed']) for r in done['runs']}==expected_pairs and len(done['runs'])==6
    probe_by_path={r['path']:r for r in probe};reports=[]
    for run in done['runs']:
        c,s=run['condition'],run['seed'];folder=W/'runs'/c/str(s)
        assert run==read(folder/'TRAINING.json')
        assert run['train_items']==728 and run['probe_items']==125 and run['class_counts']==class_counts.tolist()
        assert run['steps']==expected_steps and run['epochs']==40 and len(run['history'])==40
        assert np.isfinite(run['history']).all() and run['aux_enabled'] is False
        assert sha(run['path'])==run['sha256']
        initial=LocalMotionQuality(c,s,O).state_dict()  # Constructors only, forward is blocked.
        final=torch.load(run['path'],map_location='cpu',weights_only=True)
        assert initial.keys()==final.keys()
        deltas=[]
        for key in initial:
            d=(final[key]-initial[key]).double()
            deltas.append(dict(parameter=key,changed=not torch.equal(final[key],initial[key]),changed_elements=int(torch.count_nonzero(d)),elements=d.numel(),max_abs=float(d.abs().max()),l2=float(torch.linalg.vector_norm(d))))
        unchanged={r['parameter'] for r in deltas if not r['changed']}
        changed={r['parameter'] for r in deltas if r['changed']}
        assert {'relation_mean','relation_std','edit_aux.weight','edit_aux.bias'} <= unchanged
        assert {'summary.weight','summary.bias','extent.weight','extent.bias'} <= changed
        allowed=('summary.','extent.') if c=='B' else ('summary.','extent.','conv.','gru.','observation_projection.')
        assert all(k.startswith(allowed) for k in changed)
        if c=='C':
            assert any(k.startswith('conv.') for k in changed) and any(k.startswith('gru.') for k in changed)
            assert 'observation_projection.weight' in changed and bool((final['summary.weight'][:,304:]!=0).any())
        hp=read(folder/'HEAD_PROBE.json');pr=hp['rows']
        assert len(pr)==125 and {r['input_path'] for r in pr}==set(probe_by_path)
        assert len({r['input_path'] for r in pr})==125
        confusion=np.zeros((4,4),int)
        by_clip=collections.defaultdict(lambda:[0,0])
        for r in pr:
            source=probe_by_path[r['input_path']];p=np.array(r['probability'])
            assert r['clip_id']==source['clip_id'] and r['extent_target']==source['extent_target'] and r['extent_mask']==source['extent_mask']
            assert p.shape==(4,) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and abs(p.sum()-1)<1e-6
            assert r['prediction']==int(p.argmax()) and r['auxiliary_used'] is False
            y=source['extent_target'];pred=r['prediction'];confusion[y,pred]+=1
            by_clip[r['clip_id']][0]+=int(y==pred);by_clip[r['clip_id']][1]+=1
        denom=confusion.sum(1);recall=np.divide(confusion.diagonal(),denom,out=np.full(4,np.nan),where=denom>0)
        reports.append(dict(condition=c,seed=s,steps=run['steps'],seconds=run['seconds'],checkpoint_sha256=run['sha256'],
            parameter_deltas=deltas,aux_parameters_unchanged=True,normalization_buffers_unchanged=True,
            head_probe=dict(correct=int(confusion.trace()),total=int(confusion.sum()),accuracy=float(confusion.trace()/confusion.sum()),
                confusion_true_rows_pred_columns=confusion.tolist(),per_class_recall=recall.tolist(),macro_recall=float(np.nanmean(recall)),by_clip=dict(by_clip),
                interpretation='Continuation-excluded diagnostic only; REAL07 was already exposed to source encoder/localizer/quality training. Not fresh generalization.')))
    protections=read(U/'PROTECTIONS.json')
    keep={p:h for p,h in protections.items() if p.endswith(('training_C_20261013_best.pt','T3F_20261013_best.pt')) or 'shared100_42' in p or ('word100_action_feedback_demo_20261108' in p and Path(p).name in ('model.pt','consumer.py','app.js','CARDS.json','PREPARED.json'))}
    protected_checks=[dict(path=p,expected=h,actual=sha(p)) for p,h in keep.items()]
    assert len(protected_checks)>=8 and all(r['expected']==r['actual'] for r in protected_checks)
    report=dict(status='PASS_SAVED_TRAINING_INTEGRITY_ONLY_NOT_RUNTIME_ACCEPTANCE',at=datetime.datetime.now().isoformat(),
        reviewer='Separate Codex saved tensor/hash/arithmetic audit; no human label validation',
        counts=dict(source_clips=30,train_clips=26,probe_clips=4,train_samples=728,probe_samples=125,train_extent_counts=class_counts.tolist(),conditions=2,seeds=3,runs=6,epochs_each=40,steps_each=480),
        equal_comparison=dict(shared_manifest=True,same_order_per_matched_seed_epoch=True,same_optimizer_and_lr=True,same_train_only_class_weights=True,final_epoch_only=True),
        source_and_code_pin_checks=pin_checks,protected_checks=protected_checks,runs=reports,
        no_private_or_dev_training=True,auxiliary_training_or_use=False,model_forward_calls=0,optimizer_steps=0,
        limitations=['All admitted measured-motion sides are R; no demonstrated L transfer.',
            'REAL07 was learned by previous source models; this continuation exclusion is not a new participant evaluation.',
            'All accepted candidates have observed relations; this does not validate missing-joint recovery.',
            'Extent class3 now means multiple reviewed episodes, not every movement-plus-hold mixture.',
            'High head-probe scores do not establish selection/order correction, educational feedback correctness or web readiness.',
            'Same minibatch schedule follows frozen deterministic training code; individual minibatch hashes were not logged.'],
        evidence={n:sha(W/n) for n in ('FROZEN.json','TRAIN_ADMISSION.json','INPUTS_COMPLETE.json','TRAINING_COMPLETE.json','quality_model.py','train.py')})
    out=U/'reviews/local_quality_training_verification.json'
    assert not out.exists(),'do not overwrite completed audit'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],head_probe=[dict(condition=r['condition'],seed=r['seed'],correct=r['head_probe']['correct'],total=125,macro_recall=r['head_probe']['macro_recall']) for r in reports],protected=len(protected_checks),output=str(out))))

if __name__=='__main__':main()
