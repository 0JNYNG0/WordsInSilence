"""Contract checks without training, encoder inference, or model predictions."""
import ast, collections, inspect
import numpy as np
from prepare import U, read, write, load
_train=load('_splice_training_check',U/'train.py')
_evaluate=load('_splice_evaluation_check',U/'evaluate.py')
sampled_batch=_train.sampled_batch
select_card,original_orders,AX=_evaluate.select_card,_evaluate.original_orders,_evaluate.AX

def main():
    assert not (U/'IMPLEMENTATION_CHECK.json').exists()
    checks=[]
    for name in ['train.py','evaluate.py']:
        ast.parse((U/name).read_text(encoding='utf-8'));checks.append(name+' syntax')
    originals=[];new={};key=lambda r:(r['clip_id'],r['view'],r['reverse'],r['family'])
    for family in [2,3,4,5,7]:
        for i in range(4):
            row=dict(clip_id=f'toy{i}',view='full',reverse=False,family=family)
            d=dict(row=row,x=np.full((61,1160),family,np.float32),y=np.r_[np.ones(3,bool),np.zeros(58,bool)])
            originals.append(d)
            if family in [2,3]:new[key(row)]=dict(d,x=np.full((61,1160),family+10,np.float32))
    groups={f:[i for i,d in enumerate(originals)if d['row']['family']==f]for f in [2,3,4,5,7]}
    x,pieces=sampled_batch(originals,new,key,groups,np.random.default_rng(42))
    weights=collections.defaultdict(float)
    for a,b,np_,j,w in pieces:
        assert b-a==59 and np_==3
        origin='new'if x[a,0]>=10 else'old';weights[j,origin]+=w
    for j in [0,1]:assert np.isclose(weights[j,'old'],.1)and np.isclose(weights[j,'new'],.1)
    for j in [2,3,4]:assert np.isclose(weights[j,'old'],.2)and weights[j,'new']==0
    checks+=['equal old/new source-family loss','all5 families replay','all positives plus56 negatives']
    _,held=sampled_batch(originals,{},key,groups,np.random.default_rng(42))
    assert len(held)==10 and all(np.isclose(p[-1],.1)for p in held)
    checks.append('held new bag falls back to original only')
    c=dict(card='toy',family=2,truth=dict(witness=[2],wrong_witness=[5]))
    spans=np.array([[0,3],[1,4]],np.int32);q=np.zeros((6,10,10),np.float32);q[:,:,8]=1
    out=select_card(c,spans,np.array([-1,0]),np.array([100.,1.]),q)
    assert out['chosen']['span']==[1,4] and out['chosen']['hand_qualified']==[]
    assert out['truth']['all_witness'] and out['truth']['wrong_points']==0
    checks+=['old presence gate retained','wrong hand does not exclude attempt','body qualification unchanged']
    out=select_card(c,spans,np.array([-1,-1]),np.array([100.,1.]),q)
    assert out['chosen']is None;checks.append('no body admission stays absent')
    params=set(inspect.signature(select_card).parameters)
    assert params=={'original','spans','presence','logits','qr'}
    checks.append('no provenance/order argument in selector')
    cards=[dict(card='a',family=2,chosen=dict(span=[10,15])),dict(card='b',family=3,chosen=dict(span=[0,5]))]
    assert original_orders(cards)[0]['state']=='reverse';checks.append('order diagnosed without forced correspondence')
    assert not (U/'runs').exists() and not (U/'TRAINING_COMPLETE.json').exists()
    checks.append('no training/model output exists')
    write(U/'IMPLEMENTATION_CHECK.json',dict(status='PASS_NO_TRAINING_OR_MODEL_INFERENCE',checks=checks,count=len(checks),
        limitation='Toy loss/selector contracts and syntax, not performance or post-training verification. Parent already read train/evaluate implementation.'))
    print('Implementation contracts PASS',len(checks))

if __name__=='__main__':main()
