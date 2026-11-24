"""Independently verify saved nine fine-tuning runs; no model forward/training.

Reads only this unit's research artifacts and named public pretrained weights.
Does not read user data, operating C/T3F weights, or final test scores.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import argparse
import hashlib
import json
import math
import sys
import numpy as np
import torch

U=Path(__file__).resolve().parents[1]
V=U/'verification'
torch.set_num_threads(2)

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def th(t):
    a=t.detach().cpu().contiguous()
    h=hashlib.sha256(str(a.dtype).encode()+str(tuple(a.shape)).encode())
    h.update(a.numpy().tobytes());return h.hexdigest()
def scalar_close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12)
def same(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return scalar_close(a,b)
    return a==b
def metrics(truth,logits,classes):
    nclass=len(classes); pred=np.argmax(logits,axis=1)
    cm=np.bincount(truth*nclass+pred,minlength=nclass*nclass).reshape(nclass,nclass)
    support=cm.sum(1);pn=cm.sum(0);tp=np.diag(cm)
    pre=np.divide(tp,pn,out=np.zeros(nclass,dtype=float),where=pn>0)
    rec=np.divide(tp,support,out=np.zeros(nclass,dtype=float),where=support>0)
    f1=np.divide(2*tp,pn+support,out=np.zeros(nclass,dtype=float),where=(pn+support)>0)
    return {'n':len(truth),'correct':int(tp.sum()),'accuracy':float(tp.sum()/len(truth)),
            'macro_f1_supported':float(f1[support>0].mean()),'supported_classes':int((support>0).sum()),
            'per_class':[{'class':c,'support':int(support[i]),'predicted':int(pn[i]),'tp':int(tp[i]),
                          'precision':float(pre[i]),'recall':float(rec[i]),'f1':float(f1[i])} for i,c in enumerate(classes)],
            'confusion':cm.tolist()}

def main():
    cfg=read(U/'EXPERIMENT.json');p=read(U/'FINETUNE_PROTOCOL.json');done=read(U/'FINETUNE_COMPLETE.json')
    rows=cfg['rows'];classes=cfg['classes']; ph=sha(U/'FINETUNE_PROTOCOL.json')
    checks=[]
    def check(name,condition,detail=None):
        checks.append({'name':name,'passed':bool(condition),'detail':detail});assert condition,name
    check('completed_without_finetune_lock',not (U/'FINETUNE_RUNNING.json').exists()
          and done['status']=='NINE_FIXED_EPOCH20_RUNS_COMPLETE_TEST_UNSEEN')
    check('protocol_experiment_and_code_hashes',p['experiment_sha256']==sha(U/'EXPERIMENT.json')
          and p['code_sha256']==sha(U/'fine_tune_public.py')
          and p['encode_code_sha256']==sha(U/'encode_public.py')
          and p['metrics_code_sha256']==sha(U/'train_observer.py') and done['protocol_sha256']==ph)
    fit=[i for i,r in enumerate(rows) if r['role']=='fit'];val=[i for i,r in enumerate(rows) if r['role']=='validation']
    test=[i for i,r in enumerate(rows) if r['role']=='test'];active=fit+val
    check('only_fit_and_validation_in_prefix_cache',len(fit)==2693 and len(val)==284 and len(test)==382
          and p['active_indices']==active and not set(active)&set(test)
          and p['active_id_sha256']==hashlib.sha256('\n'.join(rows[i]['id'] for i in active).encode()).hexdigest())
    check('fixed_epoch20_seed_budget',p['epochs']==p['selection_epoch']==20 and p['batch']==32
          and not p['best_epoch_selection'] and p['seeds']==[27101,27102,27103]
          and p['backbone_lr']==.0002 and p['head_lr']==.003 and p['weight_decay']==.01)
    check('original24_research_heads_unchanged',len(p['original_heads'])==24
          and all(sha(Path(k))==v for k,v in p['original_heads'].items()))
    expected={f'ft_{name}_{seed}' for name in p['models'] for seed in p['seeds']}
    check('nine_expected_checkpoints',len(done['checkpoints'])==9 and {Path(k).stem for k in done['checkpoints']}==expected
          and all(sha(Path(k))==v for k,v in done['checkpoints'].items()))
    order_hashes={}
    for seed in p['seeds']:
        gen=torch.Generator().manual_seed(seed);values=[]
        for epoch in range(20):
            order=torch.randperm(len(fit),generator=gen)
            assert torch.equal(torch.sort(order).values,torch.arange(len(fit)))
            values.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
        order_hashes[seed]=values
    summaries=[];frozen_counts={};changed_counts={};pins={}
    for name in p['models']:
        spec=p['architectures'][name];official=Path(spec['official_weight_path'])
        check('official_weight_file_'+name,sha(official)==spec['official_weight_sha256'])
        # The official TinyNeXt wrapper contains the ordinary argparse.Namespace
        # metadata object. Only this declared benign global is allowlisted.
        unsafe=set(torch.serialization.get_unsafe_globals_in_checkpoint(official))
        check('bounded_official_pickle_globals_'+name,unsafe <= {'argparse.Namespace'},sorted(unsafe))
        with torch.serialization.safe_globals([argparse.Namespace]):
            original=torch.load(official,map_location='cpu',weights_only=True,mmap=True)
        if 'model' in original:original=original['model']
        check('initial_tensor_hashes_against_official_'+name,set(original)==set(spec['initial_state_hashes'])
              and all(th(v)==spec['initial_state_hashes'][k] for k,v in original.items()))
        allowed=set(spec['trainable_names']);frozen=set(spec['frozen_state_names'])
        check('parameter_partition_'+name,not allowed&frozen and allowed|frozen==set(original)
              and sum(original[k].numel() for k in allowed)==spec['trainable_parameters'])
        # All running means, variances, and batch counters must remain frozen.
        check('all_buffers_frozen_'+name,all(k in frozen for k in original
              if k.endswith(('running_mean','running_var','num_batches_tracked'))))
        cpfx=U/'data'/('ft_prefix_'+name+'.pt');meta=read(cpfx.with_suffix('.json'))
        check('prefix_file_hash_'+name,sha(cpfx)==meta['sha256'] and meta['protocol_sha256']==ph and meta['test_inputs']==0)
        z=torch.load(cpfx,map_location='cpu',weights_only=True,mmap=True)
        check('prefix_indices_shape_finite_'+name,z['indices']==active and z['protocol_sha256']==ph
              and list(z['prefix'].shape)==[2977]+spec['prefix_shape'] and str(z['prefix'].dtype)=='torch.float32'
              and all(bool(torch.isfinite(z['prefix'][i:i+64]).all()) for i in range(0,len(active),64)))
        pins[str(cpfx.relative_to(U))]=meta['sha256'];del z
        per=[]
        for seed in p['seeds']:
            stem=f'ft_{name}_{seed}';cp=U/'models'/(stem+'.pt');rp=cp.with_suffix('.json')
            r=read(rp);out=torch.load(cp,map_location='cpu',weights_only=True,mmap=True)
            check('run_metadata_'+stem,r['name']==out['model_name']==name and r['seed']==out['seed']==seed
                  and r['epoch']==out['epoch']==20 and r['protocol_sha256']==out['protocol_sha256']==ph
                  and r['test_used'] is False and out['test_used'] is False and r['test_predictions']==0
                  and sha(cp)==r['checkpoint_sha256'])
            final={k:th(v) for k,v in out['model'].items()}
            actualchanged=[k for k in spec['trainable_names'] if final[k]!=spec['initial_state_hashes'][k]]
            check('frozen_and_trained_weights_'+stem,set(out['model'])==set(original)
                  and all(final[k]==spec['initial_state_hashes'][k] for k in frozen)
                  and bool(actualchanged) and actualchanged==r['frozen_check']['changed_trainable']
                  and final==r['frozen_check']['final'] and spec['initial_state_hashes']==r['frozen_check']['initial']
                  and r['frozen_check']['all_frozen_equal'] is True and r['frozen_check']['frozen_state_count']==len(frozen)
                  and all(bool(torch.isfinite(v).all()) for v in out['model'].values()))
            check('new_head_shapes_hashes_'+stem,list(out['head']['weight'].shape)==[45,spec['feature_dim']]
                  and list(out['head']['bias'].shape)==[45]
                  and {k:th(v) for k,v in out['head'].items()}==r['head_final_hashes']
                  and all(th(v)!=r['head_initial_hashes'][k] for k,v in out['head'].items())
                  and all(bool(torch.isfinite(v).all()) for v in out['head'].values()))
            tr=r['trace']
            check('exposure_trace_and_seed_order_'+stem,len(tr)==20 and r['fit_exposures']==2693*20
                  and all(t['epoch']==i+1 and t['seen_count']==t['unique_image_count']==2693
                          and t['fit_order_indices_sha256']==order_hashes[seed][i]
                          and t['test_predictions']==t['validation_predictions']==0 and math.isfinite(t['fit_loss'])
                          for i,t in enumerate(tr)))
            vp=U/'models'/(stem+'_validation.npz');v=np.load(vp,allow_pickle=False)
            check('stored_validation_indices_and_hash_'+stem,sha(vp)==r['validation_sha256']
                  and np.array_equal(v['indices'],val) and list(v['ids'])==[rows[i]['id'] for i in val]
                  and v['logits'].shape==(284,45) and np.isfinite(v['logits']).all())
            calc=metrics(np.asarray([rows[i]['target'] for i in val]),v['logits'],classes)
            check('validation_arithmetic_'+stem,same(calc,r['validation']))
            latency=r['cpu_full_model_ms'];samples=np.asarray(latency['samples'])
            check('latency_saved_arithmetic_'+stem,len(samples)==40 and np.isfinite(samples).all() and (samples>0).all()
                  and scalar_close(float(np.percentile(samples,50)),latency['p50'])
                  and scalar_close(float(np.percentile(samples,95)),latency['p95'])
                  and latency['threads']==4 and latency['batch']==1)
            per.append({'name':name,'seed':seed,'n':calc['n'],'correct':calc['correct'],'accuracy':calc['accuracy'],
                        'macro_f1_supported':calc['macro_f1_supported'],'cpu_p95_ms':latency['p95'],
                        'checkpoint_sha256':sha(cp),'validation_sha256':sha(vp)})
            frozen_counts[stem]=len(frozen);changed_counts[stem]=len(actualchanged)
            del out,v
        summaries+=per
        agg=next(x for x in done['models'] if x['name']==name)
        check('three_seed_aggregate_'+name,agg['seeds']==p['seeds']
              and scalar_close(agg['validation_accuracy_mean'],np.mean([x['accuracy'] for x in per]))
              and scalar_close(agg['validation_macro_f1_mean'],np.mean([x['macro_f1_supported'] for x in per]))
              and same(agg['validation_accuracy_range'],[min(x['accuracy'] for x in per),max(x['accuracy'] for x in per)])
              and same(agg['validation_macro_f1_range'],[min(x['macro_f1_supported'] for x in per),max(x['macro_f1_supported'] for x in per)])
              and scalar_close(agg['cpu_p95_mean_ms'],np.mean([x['cpu_p95_ms'] for x in per])))
    groups=done['models'];best=max(x['validation_macro_f1_mean'] for x in groups)
    eligible=[x for x in groups if x['validation_macro_f1_mean']>=best-.01]
    winner=min(eligible,key=lambda x:x['cpu_p95_mean_ms'])['name']
    review={'status':'PASS','at':datetime.now(timezone.utc).isoformat(),'checks':checks,'count':len(checks),
            'independent_verification':'No producer imports, new inference, optimizer steps, or model creation. NumPy score arithmetic, saved tensor/official-weight comparison, and seeded permutation reconstruction.',
            'runs':summaries,'protocol_sha256':ph,'completion_sha256':sha(U/'FINETUNE_COMPLETE.json'),
            'code_sha256':sha(Path(__file__)),'prefix_cache_hashes':pins,'frozen_tensor_counts':frozen_counts,
            'changed_allowed_tensor_counts':changed_counts,'total_fit_exposures':2693*20*9,
            'optimizer_steps_from_declared_batch_contract_per_run':math.ceil(2693/32)*20,
            'validation_selected_architecture_recomputed':winner,
            'test_score_files_read':0,'private_or_operating_model_files_read':0,
            'old_24_research_heads_unchanged':True,
            'limitations':['Stored epoch order hashes reconstruct all 2693 unique fit indices per epoch; these are execution-produced order records, not a separate per-batch source-ID sensor log.',
                           'Frozen-prefix equivalence is a recorded three-fit-image preflight, not re-inference by this reviewer.',
                           'Three architectures adapt different last-stage parameter counts; this is a fixed practical adaptation comparison, not equal-trainable-capacity proof.',
                           'Validation accuracy is on 284 externally labelled hand crops, not KSL recognition or feedback accuracy.',
                           'Final test predictions remain separate and are not read in this audit.'],
            'checker_compatibility_repair':'Initial direct official TinyNeXt weights-only inspection rejected argparse.Namespace metadata. Original failed inspection preserved separately; verifier now verifies official SHA and permits only that known global under weights_only=True.'}
    out=V/'FINETUNE_INDEPENDENT_VERIFY.json';assert not out.exists(),'Preserve existing audit'
    out.write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','checks':len(checks),'runs':9,'test_scores_read':0,'winner_from_validation':winner}))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');main()
