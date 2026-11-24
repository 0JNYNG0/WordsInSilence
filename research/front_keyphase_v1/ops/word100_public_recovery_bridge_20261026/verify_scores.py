from common import *
def main():
    f=read(U/'CLASSIFIER_FROZEN.json');idx=read(U/'CLASSIFIER_INPUTS.json');check(f['pins']);check(idx['pins'])
    result=read(U/'CLASSIFIER_RESULT.json');verified=0;prob_error=0.
    all_scores={name:read(U/(name+'_SCORES.json')) for name in list(f['models'])+['E3']}
    for name,data in all_scores.items():
        for kind,rows in data['predictions'].items():
            correct=top5=0
            for i,r in enumerate(rows):
                assert r['clip_id']==f['rows'][i]['clip_id']
                if r['logits'] is None:assert not r['correct'];continue
                logits=r['logits'];rank=sorted(range(100),key=lambda j:(-logits[j],j));truth=f['rows'][i]['label_index']
                assert r['prediction']==rank[0] and r['top5']==rank[:5]
                assert r['correct']==(rank[0]==truth) and r['top5_correct']==(truth in rank[:5])
                correct+=rank[0]==truth;top5+=truth in rank[:5];verified+=1
                if name=='E3':
                    import math
                    probabilities=[]
                    for s in (20261013,20261014,20261015):
                        a=all_scores[f'T3F{s}']['predictions'][kind][i]['logits'];m=max(a);e=[math.exp(x-m) for x in a];total=sum(e);probabilities.append([x/total for x in e])
                    expected=[sum(p[j] for p in probabilities)/3 for j in range(100)]
                    err=max(abs(math.exp(logits[j])-expected[j]) for j in range(100));prob_error=max(prob_error,err);assert err<1e-12
            assert result['summary'][name][kind]['correct']==correct and result['summary'][name][kind]['top5']==top5
    check(read(U/'FROZEN.json')['protected'])
    write(U/'CLASSIFIER_VERIFIED.json',dict(at=now(),status='PASS',scores_recounted=verified,ensemble_probability_max_error=prob_error,
        reinference=0,training=0,independent_people_ground_truth=False,method='Separate stored-logit ranking and scalar softmax arithmetic by same Codex'))
    print('SCORES VERIFIED',verified,flush=True)
if __name__=='__main__':main()
