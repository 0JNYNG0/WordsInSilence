"""Independent scalar arithmetic and original-logit provenance check; no model execution."""
from common import *
import math
def main():
    f=read(U/'CLASSIFIER_FROZEN.json');idx=read(U/'CLASSIFIER_INPUTS.json');check(f['pins']);check(idx['pins'])
    result=read(U/'CLASSIFIER_RESULT.json');verified=0;prob_error=0.;original_reuse=0
    scores={name:read(U/(name+'_SCORES.json')) for name in list(f['models'])+['E3']}
    for name,data in scores.items():
        if name!='E3':
            old={r['clip_id']:r for r in read(f['prior'][name])['predictions']['masked']}
            for kind in f['kinds']:
                for r,rr in zip(data['predictions'][kind],idx['rows']):
                    h=rr['arrays'][kind]
                    if h in idx['original_array_to_clip']:
                        assert r['logits']==old[idx['original_array_to_clip'][h]]['logits'];original_reuse+=1
            assert data['unique_forwards']==sum(p['kind']=='new_forward' for p in data['provenance'].values())
            assert data['model_state_unchanged'] and data['training']==0
        base={r['clip_id']:r for r in data['predictions']['native']}
        for kind,rows in data['predictions'].items():
            assert len(rows)==300;correct=top5=0;gain=[];loss=[]
            for i,r in enumerate(rows):
                assert r['clip_id']==f['rows'][i]['clip_id'];truth=f['rows'][i]['label_index']
                if r['logits'] is None:assert not r['correct'];continue
                a=r['logits'];rank=sorted(range(100),key=lambda j:(-a[j],j))
                assert r['prediction']==rank[0] and r['top5']==rank[:5]
                assert r['correct']==(rank[0]==truth) and r['top5_correct']==(truth in rank[:5])
                correct+=rank[0]==truth;top5+=truth in rank[:5];verified+=1
                if r['correct'] and not base[r['clip_id']]['correct']:gain.append(r['clip_id'])
                if not r['correct'] and base[r['clip_id']]['correct']:loss.append(r['clip_id'])
                if name=='E3':
                    pp=[]
                    for s in (20261013,20261014,20261015):
                        b=scores[f'T3F{s}']['predictions'][kind][i]['logits'];m=max(b);e=[math.exp(x-m) for x in b];total=sum(e);pp.append([x/total for x in e])
                    error=max(abs(math.exp(a[j])-sum(p[j] for p in pp)/3) for j in range(100));prob_error=max(prob_error,error);assert error<1e-12
            rr=result['summary'][name][kind];assert rr['correct']==correct and rr['top5']==top5 and rr['gained']==gain and rr['lost']==loss
    check(read(U/'FROZEN.json')['protected'])
    write(U/'CLASSIFIER_VERIFIED.json',dict(at=now(),status='PASS',scores_recounted=verified,original_logits_reuse_checked=original_reuse,
       ensemble_probability_max_error=prob_error,reinference=0,training=0,independent_people_ground_truth=False,
       method='Separate stored-logit ranking, gains/losses, original byte-identical input/logit reuse and scalar softmax arithmetic by same Codex'))
    print('SCORES VERIFIED',verified,flush=True)
if __name__=='__main__':main()
