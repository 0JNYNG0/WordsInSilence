"""Separate arithmetic verification of saved logits and training-only BN moments."""
import json
import numpy as np
import torch
from probe import U,OLD,RUNS,T,read,write,sha,pin_check
from pilot.model_masked_v1 import MaskedBatchNorm1d

def main():
    assert not (U/'INDEPENDENT_CHECK.json').exists()
    f=pin_check();manifest={r['clip_id']:r for r in read(OLD/'MANIFEST.json') if r['role']=='development_only'}
    factorial=read(U/'FACTORIAL.json');population=read(U/'POPULATION_RESULT.json')
    checks={};expected={}
    for arm,conditions in factorial['conditions'].items():
        for cond,result in conditions.items():expected[f'{arm}_{cond}.json']=result
    for arm,result in population['results'].items():expected[f'population_{arm}.json']=result
    for filename,result in expected.items():
        rows=read(U/filename);assert len(rows)==300 and {r['clip_id'] for r in rows}==set(manifest)
        assert all(r['label_index']==manifest[r['clip_id']]['label_index'] and r['signer_id']==manifest[r['clip_id']]['signer_id'] for r in rows)
        logits=np.asarray([r['logits'] for r in rows],dtype=np.float64);y=np.array([r['label_index'] for r in rows]);rank=np.argsort(-logits,axis=1)
        correct=rank[:,0]==y;assert all(bool(c)==r['correct'] and int(a)==r['prediction'] for c,a,r in zip(correct,rank[:,0],rows))
        z=logits-logits.max(1,keepdims=True);ce=float(np.mean(np.log(np.exp(z).sum(1))-z[np.arange(300),y]))
        assert int(correct.sum())==result['correct'] and abs(ce-result['ce'])<2e-6
        checks[filename]=dict(correct=int(correct.sum()),accuracy_percent=float(correct.mean()*100),ce_float64=ce,
            by_signer={str(s):int(sum(c for c,r in zip(correct,rows) if r['signer_id']==s)) for s in [11,12,13]},
            top5_correct=int(np.any(rank[:,:5]==y[:,None],axis=1).sum()),sha256=sha(U/filename))
    train,dev,invalid=T.datasets(f);assert not invalid and {c['signer'] for c in train}==set(range(1,10))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    stats_checks={}
    for name,path in RUNS.items():
        for which in ['best','last']:
            key=f'{name}_{which}';stats=population['results'][key]['statistics']
            ck=torch.load(path/f'{which}.pt',weights_only=True,map_location='cpu');state=ck['state_dict' if which=='best' else 'model_state']
            m=T.model(f);m.load_state_dict(state);m.eval();acc={};handles=[]
            def capture(name):
                def hook(mod,args):
                    x,mask=args;a=x.detach().numpy().astype(np.float64);mask=mask.detach().numpy().astype(np.float64)
                    n=int(mask.sum());s=(a*mask).sum((0,2));q=(a*a*mask).sum((0,2))
                    if name not in acc:acc[name]=[n,s,q]
                    else:acc[name][0]+=n;acc[name][1]+=s;acc[name][2]+=q
                return hook
            for n,layer in m.named_modules():
                if isinstance(layer,MaskedBatchNorm1d):
                    with torch.no_grad():
                        layer.running_mean.copy_(torch.tensor(stats[n]['mean']))
                        layer.running_var.copy_(torch.tensor(stats[n]['unbiased_variance']))
                    handles.append(layer.register_forward_pre_hook(capture(n)))
            with torch.inference_mode():
                for at in range(0,len(train),17):m(*T.batch(train[at:at+17])[:2])
            for h in handles:h.remove()
            errors=[]
            for n,(count,s,q) in acc.items():
                mean=s/count;var=(q-s*s/count)/(count-1);target=stats[n]
                assert count==target['valid_frame_count']
                assert np.allclose(mean,target['mean'],atol=1e-4,rtol=1e-5)
                assert np.allclose(var,target['unbiased_variance'],atol=1e-4,rtol=1e-5)
                errors.append(dict(layer=n,mean_max_abs=float(np.max(np.abs(mean-target['mean']))),variance_max_abs=float(np.max(np.abs(var-target['unbiased_variance'])))))
            stats_checks[key]=dict(status='PASS',batch_size=17,aggregator='float64 sum and sum-of-squares, all8 layers simultaneously with recalculated upstream state',layers=errors)
    guard=read(U.parent/'word100_robustness_20261013/ORIGINAL_GUARD.json');pins=read(U/'PLAN.json')['source_sha256'];allpins={**guard,**pins}
    assert all(sha(k)==v for k,v in allpins.items())
    write(U/'INDEPENDENT_CHECK.json',dict(status='PASS',saved_prediction_files=checks,population_moments=stats_checks,
        original_files_preserved=len(allpins),original_web_model_sha256=sha(RUNS['current']/'best.pt'),
        gradient_training_runs=0,new_checkpoint_files=list(str(p) for p in U.glob('*.pt')),
        implementer='Codex',verification='Separate arithmetic code and alternate batching; not a separate person/agent/external expert review'))
    print(json.dumps({k:{q:r[q] for q in ['correct','accuracy_percent','by_signer','top5_correct']} for k,r in checks.items()},ensure_ascii=False))

if __name__=='__main__':main()
