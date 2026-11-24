"""Fresh100-class masked TCN: frozen whole vs core-context development comparison."""
from collections import Counter
import copy
import hashlib
import json
import math
import os
import random
import time

from common import U,OUT,RUN,read,write,sha,pin_check
import numpy as np
import torch
from pilot.model_masked_v1 import MaskedSignTCN

def batch(clips,epoch=0,arm='whole'):
    xx=[]
    for c in clips:
        context=arm=='whole_context' and ((epoch+c['parity'])%2==1)
        xx.append(c['context'] if context else c['whole'])
    m=max(len(x) for x in xx); a=np.zeros((len(xx),m,280),np.float32); mask=np.zeros((len(xx),m),np.float32)
    for i,x in enumerate(xx):a[i,:len(x)]=x;mask[i,:len(x)]=1
    return torch.from_numpy(a),torch.from_numpy(mask),torch.tensor([c['y'] for c in clips],dtype=torch.long)

def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)

def model(f):
    cfg=dict(f['model']);cfg.pop('type');return MaskedSignTCN(**cfg)

def datasets(f):
    rows=read(U/'MANIFEST.json');train=[];dev=[];invalid=[]
    checked=read(U/'INPUT_CHECK.json');assert checked['status']=='PASS'
    for r in rows:
        p=OUT/'records'/f"{r['clip_id']}.json";record=read(p)
        assert sha(p)==checked['record_hashes'][r['clip_id']]
        if record['channel_file'] is None:
            invalid.append(dict(clip_id=r['clip_id'],role=r['role'],label_index=r['label_index'],signer_id=r['signer_id'],reason=record['feature_check']['reason']));continue
        assert sha(record['channel_file'])==record['channel_sha256']
        with np.load(record['channel_file'],allow_pickle=False) as z:
            c=dict(id=r['clip_id'],y=r['label_index'],signer=r['signer_id'],whole=z['x_whole'],
                   parity=int(hashlib.sha256(r['clip_id'].encode()).hexdigest(),16)%2,
                   context=z['x_context'] if r['role']=='train_candidate' else None)
        (train if r['role']=='train_candidate' else dev).append(c)
    assert len(train)+sum(r['role']=='train_candidate' for r in invalid)==899
    assert len(dev)+sum(r['role']=='development_only' for r in invalid)==300
    assert set(c['signer'] for c in train)<=set(range(1,10)) and set(c['signer'] for c in dev)<={11,12,13}
    assert set(c['y'] for c in train)==set(range(100))
    return train,dev,invalid

def predict(m,clips,bs):
    m.eval();pred=[];total_loss=0.
    with torch.no_grad():
        for start in range(0,len(clips),bs):
            cc=clips[start:start+bs];x,p,y=batch(cc);logits=m(x,p)
            assert logits.shape==(len(cc),100) and torch.isfinite(logits).all()
            total_loss+=torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
            for c,l in zip(cc,logits.numpy()):
                pred.append(dict(clip_id=c['id'],label_index=c['y'],signer_id=c['signer'],prediction=int(l.argmax()),logits=l.tolist(),status='OK'))
    return total_loss/len(clips),pred

def checkpoint(m,f,epoch,step,ce,arm):
    return dict(state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},classes=f['labels'],word_codes=f['classes'],
        config={k:v for k,v in f['model'].items() if k not in ['type','n_classes','in_channels']},
        in_channels=280,feature_schema=dict(builder='pilot.data.build_channels'),model_type=m.model_type,model_version=m.model_version,
        epoch=epoch,step=step,development_ce=ce,arm=arm,initialization='fresh random AIHub-only',
        frozen_experiment_sha256=sha(U/'FROZEN_EXPERIMENT.json'),input_check_sha256=sha(U/'INPUT_CHECK.json'))

def summarize(pred,invalid,f):
    allrows=list(pred)
    for r in invalid:
        if r['role']=='development_only':allrows.append(dict(clip_id=r['clip_id'],label_index=r['label_index'],signer_id=r['signer_id'],prediction=None,status=r['reason'],logits=None))
    assert len(allrows)==300 and len({r['clip_id'] for r in allrows})==300
    for r in allrows:r['correct']=r['prediction']==r['label_index']
    correct=sum(r['correct'] for r in allrows)
    signer={str(s):dict(correct=sum(r['correct'] for r in allrows if r['signer_id']==s),total=100) for s in [11,12,13]}
    per=[dict(code=c,label=f['labels'][i],correct=sum(r['correct'] for r in allrows if r['label_index']==i),total=3) for i,c in enumerate(f['classes'])]
    summary=dict(correct=correct,total=300,accuracy_percent=100*correct/300,macro_recall_percent=float(np.mean([p['correct']/3 for p in per])*100),
                 valid_only_correct=correct,valid_only_total=len(pred),valid_only_accuracy_percent=100*correct/len(pred),input_failed=300-len(pred),by_signer=signer,by_class=per)
    return summary,sorted(allrows,key=lambda r:r['clip_id'])

def main():
    f=pin_check();train,dev,invalid=datasets(f);t=f['train']
    assert read(U/'TRAIN_CODE_PIN.json')['sha256']==sha(__file__)
    torch.set_num_threads(t['cpu_threads']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    RUN.mkdir(parents=True,exist_ok=True)
    results={}
    for arm in t['arms']:
        d=RUN/arm;d.mkdir(exist_ok=True)
        if (d/'RESULT.json').exists():
            results[arm]=read(d/'RESULT.json');assert sha(d/'best.pt')==results[arm]['checkpoint_sha256'];continue
        assert not (d/'last.pt').exists(), 'Interrupted training requires explicit checkpoint recovery review; never silently restart'
        seed(t['seed']);m=model(f);initial={k:v.clone() for k,v in m.state_dict().items()}
        init_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
        opt=torch.optim.AdamW(m.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
        start=time.perf_counter();best=float('inf');best_ep=0;step=0;history=[];stopped=False
        for epoch in range(1,t['epochs_max']+1):
            m.train(); order=np.random.default_rng(t['seed']+epoch).permutation(len(train));loss_sum=0.
            for at in range(0,len(order),t['batch_size']):
                cc=[train[i] for i in order[at:at+t['batch_size']]];x,p,y=batch(cc,epoch,arm)
                opt.zero_grad(set_to_none=True);logit=m(x,p);loss=torch.nn.functional.cross_entropy(logit,y)
                assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(m.parameters(),t['gradient_clip_norm']);assert torch.isfinite(norm)
                opt.step();step+=1;loss_sum+=loss.item()*len(cc)
            ce,pred=predict(m,dev,t['batch_size']);correct=sum(x['prediction']==x['label_index'] for x in pred)
            if ce < best-1e-6:
                best=ce;best_ep=epoch;torch.save(checkpoint(m,f,epoch,step,ce,arm),d/'best.pt')
            row=dict(epoch=epoch,step=step,train_ce=loss_sum/len(train),development_ce=ce,development_correct=correct,
                best_epoch=best_ep,seconds=time.perf_counter()-start)
            history.append(row);write(d/'history.json',history)
            torch.save(dict(model_state=m.state_dict(),optimizer_state=opt.state_dict(),torch_rng=torch.get_rng_state(),
                epoch=epoch,step=step,best_epoch=best_ep,best_ce=best,history=history,arm=arm,
                frozen_experiment_sha256=sha(U/'FROZEN_EXPERIMENT.json')),d/'last.pt')
            write(RUN/'progress.json',dict(status='TRAINING',pid=os.getpid(),arm=arm,**row));print(json.dumps(dict(arm=arm,**row)),flush=True)
            if epoch-best_ep>=t['early_stopping_patience']:stopped=True;break
            if time.perf_counter()-start>t['wall_safety_seconds_per_arm']:
                raise RuntimeError('Training exceeded safety wall budget; saved last.pt; run incomplete')
        best_ck=torch.load(d/'best.pt',map_location='cpu',weights_only=True);m.load_state_dict(best_ck['state_dict'])
        ce,pred=predict(m,dev,t['batch_size']);s,allrows=summarize(pred,invalid,f)
        assert abs(ce-best)<1e-6
        write(d/'PREDICTIONS.json',allrows)
        result=dict(status='COMPLETE_DEVELOPMENT_ONLY',arm=arm,initial_state_sha256=init_hash,checkpoint_sha256=sha(d/'best.pt'),
            selected_epoch=best_ep,epochs_executed=epoch,early_stopped=stopped,steps=step,selected_development_ce=ce,
            seconds=time.perf_counter()-start,parameter_count=m.parameter_count(),metrics=s,
            note='Same300 development rows selected the epoch and report performance; not a blind final test. No user data, no old93-class score comparison.')
        write(d/'RESULT.json',result);results[arm]=result
    a=read(RUN/'whole/PREDICTIONS.json');b=read(RUN/'whole_context/PREDICTIONS.json')
    assert [r['clip_id'] for r in a]==[r['clip_id'] for r in b]
    assert results['whole']['initial_state_sha256']==results['whole_context']['initial_state_sha256']
    gains=[x['clip_id'] for x,y in zip(a,b) if not x['correct'] and y['correct']]
    losses=[x['clip_id'] for x,y in zip(a,b) if x['correct'] and not y['correct']]
    comparison=dict(status='COMPLETE_PENDING_INDEPENDENT_METRICS',arms=results,
        percentage_point_difference=results['whole_context']['metrics']['accuracy_percent']-results['whole']['metrics']['accuracy_percent'],
        improved=gains,regressed=losses,net_correct=len(gains)-len(losses),
        primary_success=len(gains)>len(losses),final_blind_test=False,production_model_replaced=False,
        frozen_experiment_sha256=sha(U/'FROZEN_EXPERIMENT.json'))
    write(RUN/'COMPARISON.json',comparison);write(RUN/'progress.json',dict(status='TRAINING_AND_DEVELOPMENT_COMPLETE',run=str(RUN)))
    pin_check()

if __name__=='__main__': main()
