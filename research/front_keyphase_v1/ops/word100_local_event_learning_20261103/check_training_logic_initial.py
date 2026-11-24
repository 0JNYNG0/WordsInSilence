"""Synthetic loss/readout checks; no checkpoint, media, or real data inference."""
import math
import numpy as np
import torch
from train_local import U,write,sha,class_weights,group_loss
from evaluate_local import score_bag

def main():
    tests=[]
    def ok(name,value):
        assert value,name
        tests.append(name)
    groups=[]
    for t in range(5):
        for i in range(9):
            bags=[dict(indices=[0,2],label=1,train_eligible=True,evidence_complete=True)]
            if i<7:bags.append(dict(indices=[3,5],label=0,train_eligible=True,evidence_complete=True))
            bags.append(dict(indices=[6,7],label=None,train_eligible=False,evidence_complete=True))
            groups.append(dict(target_index=t,bags=bags))
    weights,counts=class_weights(groups,list(range(5)))
    for t in range(5):
        for y in [0,1]:ok(f'Equal target/class global contribution {t}:{y}',abs(weights[f'{t}:{y}']*counts[(t,y)]/45-.1)<1e-12)
    z=torch.zeros((1,8,5),requires_grad=True)
    out=dict(logits=z,time_valid=torch.ones((1,8),dtype=torch.bool),motion_anchor=torch.ones((1,8),dtype=torch.bool))
    loss=torch.stack([group_loss(out,g,weights) for g in groups]).mean()
    ok('Global zero-logit loss is log2',abs(float(loss.detach())-math.log(2))<1e-6)
    loss.backward();ok('Unknown-only rows have zero supervised gradient',not z.grad[0,6:].any())
    copied=dict(groups[0],bags=groups[0]['bags']+[dict(groups[0]['bags'][0])])
    ok('Duplicate dependent same-class bag does not add weight',torch.equal(group_loss(out,copied,weights),group_loss(out,groups[0],weights)))
    p=dict(ids=np.arange(8),ts=np.arange(8)*33333,support=dict(row_consumed_support=[[i] for i in range(8)]))
    b=dict(bag_id='x',kind='reviewed_local',label=0,frames=[0,7],indices=[0,7],evidence_complete=False)
    prob=np.full((8,5),.99);anchor=np.ones(8,dtype=bool);anchor[3]=False
    v=score_bag(prob,anchor,p,b,0,True)
    ok('Partial observation cannot become high-confirmed',v['status']=='INSUFFICIENT_OBSERVATION' and not v['high_confirmation'])
    ok('Missing row retained in denominator',v['prediction_coverage']==7/8 and v['unavailable_native_frames']==[3])
    v=score_bag(prob,anchor,p,b,0,False)
    ok('Unsupported score hidden',v['status']=='UNSUPPORTED_QUERY' and v['probability'] is None)
    v=score_bag(prob,np.zeros(8,dtype=bool),p,b,0,True)
    ok('Entire absence of evidence not normal',v['status']=='INSUFFICIENT_OBSERVATION' and v['probability'] is None)
    write(U/'TRAINING_LOGIC_CHECKS.json',dict(status='PASS',checks=tests,count=len(tests),
        code_pins={str(U/n):sha(U/n) for n in ['local_events.py','train_local.py','evaluate_local.py','check_training_logic.py']},
        real_video_model_forwards=0,checkpoint_loads=0,optimizer_steps=0))
    print(len(tests),'synthetic loss/readout checks PASS')
if __name__=='__main__':main()
