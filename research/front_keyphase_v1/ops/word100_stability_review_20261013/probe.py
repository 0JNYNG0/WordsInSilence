"""Preregistered no-training factorial diagnostic of weights and BN buffers.

No deployable model is created. Counterfactual buffer swaps are diagnostic interventions,
not legitimate trained checkpoints or new-user performance evidence.
"""
from pathlib import Path
import sys
import argparse
import hashlib
import json
import math
import numpy as np
U=Path(__file__).resolve().parent
OLD=U.parent/'word100_training_20261013'
sys.path.insert(0,str(OLD))
from common import BASE,ROOT,read,write,sha,pin_check
import train as T
import torch

RUNS={'current':BASE/'runs/word100_tcn_v1/whole_context','speed':BASE/'runs/word100_temporal_robustness_v1'}
def bn(k):return k.endswith(('running_mean','running_var','num_batches_tracked'))
def digest(state):return hashlib.sha256(b''.join(k.encode()+v.numpy().tobytes() for k,v in state.items())).hexdigest()

def freeze():
    assert not (U/'PLAN.json').exists()
    pins=[Path(__file__),OLD/'train.py',OLD/'FROZEN_EXPERIMENT.json',OLD/'MANIFEST.json',OLD/'INPUT_CHECK.json',ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py',BASE/'src/wis_lab/service/word100.py',BASE/'src/wis_lab/service/neural.py']
    for run in RUNS.values():pins.extend([run/p for p in ['best.pt','last.pt','RESULT.json','PREDICTIONS.json','history.json']])
    write(U/'PLAN.json',dict(status='FROZEN_BEFORE_PROBE',authorization='User requests repeated analyses/reviews and a fact-based expected-value plan; diagnostic only, no new training/deployment',
        question='How much of the stored best-to-last inference change depends on learned weights versus running normalization state?',
        conditions=['best weights/best buffers: reuse saved300 predictions','last weights/last buffers: reproduce final history score','best weights/last buffers: swap running stats only','last weights/best buffers: swap running stats only'],
        replicates='current whole_context and already rejected speed model; not independent random-seed replications',
        invariance='First16 development clips, same fixed model, batch1 vs16 and64 appended zero-masked frames: all predicted labels unchanged and max logits difference<1e-4',
        evidence_rule='If only buffers change while all learned parameters are identical and predictions change, inference sensitivity to buffers is demonstrated. It does not prove buffers are the sole cause, small final batch is the cause, or stale cross-epoch buffers are deployable.',
        no_score_based_selection=True,new_training=0,optimizer_steps=0,new_checkpoints=0,user_data=False,heldout=False,
        source_sha256={str(p):sha(p) for p in pins}))

def run():
    assert not (U/'FACTORIAL.json').exists(),'Reuse completed diagnostic'
    p=read(U/'PLAN.json');assert all(sha(k)==h for k,h in p['source_sha256'].items())
    f=pin_check();tr,dv,invalid=T.datasets(f);assert not invalid
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    results={};invariance=None;histories={};tails={};buffer_change={}
    for name,path in RUNS.items():
        bestck=torch.load(path/'best.pt',map_location='cpu',weights_only=True)
        lastck=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        states={'best':bestck['state_dict'],'last':lastck['model_state']}
        hashes={k:digest(s) for k,s in states.items()}
        paramkeys=[k for k in states['best'] if not bn(k)];bufkeys=[k for k in states['best'] if bn(k)]
        assert len([k for k in bufkeys if k.endswith('running_mean')])==8
        buffer_change[name]={k:dict(max_abs=float((states['best'][k]-states['last'][k]).abs().max()),median_abs=float((states['best'][k]-states['last'][k]).abs().float().median())) for k in bufkeys if not k.endswith('num_batches_tracked')}
        rr={};saved=read(path/'PREDICTIONS.json');result=read(path/'RESULT.json')
        rr['Wbest_Bbest']=dict(correct=result['metrics']['correct'],ce=result['selected_development_ce'],source='saved predictions')
        write(U/f'{name}_Wbest_Bbest.json',saved)
        for w,b in [('last','last'),('best','last'),('last','best')]:
            state={k:states[b if bn(k) else w][k].clone() for k in states[w]}
            assert all(torch.equal(state[k],states[w][k]) for k in paramkeys)
            assert all(torch.equal(state[k],states[b][k]) for k in bufkeys)
            m=T.model(f);m.load_state_dict(state);before=digest(m.state_dict())
            ce,pred=T.predict(m,dv,32);metric,allrows=T.summarize(pred,[],f)
            assert digest(m.state_dict())==before
            key=f'W{w}_B{b}'
            rr[key]=dict(correct=metric['correct'],ce=ce,by_signer={s:q['correct'] for s,q in metric['by_signer'].items()},learned_state=w,buffers=b)
            write(U/f'{name}_{key}.json',allrows)
            if w==b=='last':
                final=read(path/'history.json')[-1]
                assert metric['correct']==final['development_correct'] and abs(ce-final['development_ce'])<2e-6
        assert all(digest(states[k])==v for k,v in hashes.items())
        if name=='current':
            m=T.model(f);m.load_state_dict(states['best']);m.eval();cc=dv[:16];x,pad,y=T.batch(cc)
            with torch.inference_mode():
                joined=m(x,pad).numpy();padded=m(torch.nn.functional.pad(x,(0,0,0,64)),torch.nn.functional.pad(pad,(0,64))).numpy()
                single=np.stack([m(*T.batch([clip])[:2])[0].numpy() for clip in cc])
            savedmap={r['clip_id']:r for r in saved};target=np.asarray([savedmap[c['id']]['logits'] for c in cc])
            errors={'batch1_vs16':float(np.max(np.abs(single-joined))),'padding64':float(np.max(np.abs(padded-joined))),'saved_replay':float(np.max(np.abs(target-joined)))}
            assert max(errors.values())<1e-4
            assert np.array_equal(single.argmax(1),joined.argmax(1)) and np.array_equal(padded.argmax(1),joined.argmax(1))
            invariance=dict(status='PASS',clips=16,max_abs_logit_difference=errors)
        h=read(path/'history.json');late=[r for r in h if r['epoch']>=max(1,len(h)-19)]
        histories[name]=dict(selected_epoch=result['selected_epoch'],last_epoch=len(h),selected_train_ce=h[result['selected_epoch']-1]['train_ce'],last_train_ce=h[-1]['train_ce'],
          last20_correct=[r['development_correct'] for r in late],last20_range=[min(r['development_correct'] for r in late),max(r['development_correct'] for r in late)],
          last20_mean=float(np.mean([r['development_correct'] for r in late])),adjacent_changes_10_or_more=sum(abs(a['development_correct']-b['development_correct'])>=10 for a,b in zip(late,late[1:])))
        tails[name]={}
        for epoch in [result['selected_epoch'],len(h)]:
            order=np.random.default_rng(f['train']['seed']+epoch).permutation(len(tr));tail=[tr[i] for i in order[-3:]]
            tails[name][str(epoch)]=[dict(clip=c['id'],signer=c['signer'],label=f['labels'][c['y']],context=(epoch+c['parity'])%2==1) for c in tail]
        results[name]=rr
    write(U/'FACTORIAL.json',dict(status='DIAGNOSTIC_COMPLETE_PENDING_RECOUNT',conditions=results,invariance=invariance,history=histories,final_small_batch=tails,
        bn_running_state_delta=buffer_change,layout=dict(train_clips=899,batch_size=32,full_batches=28,last_batch_clips=3,bn_momentum=.1,last7_updates_weight=1-.9**7),
        warning='Mixed weights/buffers are counterfactuals, not selected model candidates. No new-user efficacy claim.'))
    print(json.dumps(dict(conditions=results,invariance=invariance,history=histories),ensure_ascii=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);args=a.parse_args();globals()[args.action]()
