"""Recount saved300-row logits/labels/trace without rescoring or selecting models."""
from collections import Counter
import math
import numpy as np
from common import U,RUN,read,write,sha,pin_check

def main():
    assert not (U/'METRICS_CHECK.json').exists(), 'Reuse completed check'
    f=pin_check();rows=read(U/'MANIFEST.json');dev={r['clip_id']:r for r in rows if r['role']=='development_only'}
    assert len(dev)==300
    result=read(RUN/'COMPARISON.json');observed={};counts={}
    for arm in ['whole','whole_context']:
        rr=read(RUN/arm/'PREDICTIONS.json');summary=read(RUN/arm/'RESULT.json');history=read(RUN/arm/'history.json')
        assert len(rr)==len({x['clip_id'] for x in rr})==300 and {x['clip_id'] for x in rr}==set(dev)
        ce=[];n=0;correct=0;byclass=Counter();bysigner=Counter();pairs={}
        for r in rr:
            src=dev[r['clip_id']];assert src['label_index']==r['label_index'] and src['signer_id']==r['signer_id']
            pred=None
            if r['status']=='OK':
                z=np.array(r['logits'],np.float64);assert z.shape==(100,) and np.isfinite(z).all()
                pred=int(np.argmax(z)); mx=z.max();ce.append(float(mx+np.log(np.exp(z-mx).sum())-z[r['label_index']]))
                n+=1
            assert pred==r['prediction']
            yes=pred==r['label_index'];assert yes==r['correct'];correct+=yes
            byclass[r['label_index']]+=yes;bysigner[r['signer_id']]+=yes;pairs[r['clip_id']]=bool(yes)
        s=summary['metrics'];assert correct==s['correct'] and n==s['valid_only_total']
        assert abs(s['accuracy_percent']-100*correct/300)<1e-9
        assert abs(s['macro_recall_percent']-np.mean([byclass[i]/3 for i in range(100)])*100)<1e-9
        assert all(s['by_signer'][str(i)]['correct']==bysigner[i] for i in [11,12,13])
        assert all(p['correct']==byclass[i] and p['code']==f['classes'][i] for i,p in enumerate(s['by_class']))
        assert abs(float(np.mean(ce))-summary['selected_development_ce'])<2e-6
        assert len(history)==summary['epochs_executed']<=80
        assert [x['epoch'] for x in history]==list(range(1,len(history)+1))
        best=math.inf;epoch=0
        for h in history:
            assert h['step']==h['epoch']*math.ceil(read(U/'INPUT_CHECK.json')['counts']['train_candidate']/32)
            if h['development_ce']<best-1e-6:best=h['development_ce'];epoch=h['epoch']
            assert h['best_epoch']==epoch
        assert epoch==summary['selected_epoch'] and abs(best-summary['selected_development_ce'])<1e-6
        assert len(history)==80 or (len(history)-epoch==12 and summary['early_stopped'])
        assert sha(RUN/arm/'best.pt')==summary['checkpoint_sha256']
        assert summary==result['arms'][arm]
        observed[arm]=pairs;counts[arm]=correct
    a,b=observed['whole'],observed['whole_context']
    gains=sorted(k for k in a if not a[k] and b[k]);losses=sorted(k for k in a if a[k] and not b[k])
    assert gains==sorted(result['improved']) and losses==sorted(result['regressed'])
    assert len(gains)-len(losses)==result['net_correct']==counts['whole_context']-counts['whole']
    assert abs(result['percentage_point_difference']-(counts['whole_context']-counts['whole'])/3)<1e-9
    check=dict(status='PASS',verifier='Codex separate NumPy saved-logit recount, metadata labels, epoch-selection trace and artifact hashes',
        correct=counts,total=300,improved=len(gains),regressed=len(losses),
        final_blind_test=False,new_inference=0,new_training=0,
        artifacts={str(p):sha(p) for arm in ['whole','whole_context'] for p in [RUN/arm/'PREDICTIONS.json',RUN/arm/'RESULT.json',RUN/arm/'best.pt']})
    write(U/'METRICS_CHECK.json',check);print(check)

if __name__=='__main__':main()
