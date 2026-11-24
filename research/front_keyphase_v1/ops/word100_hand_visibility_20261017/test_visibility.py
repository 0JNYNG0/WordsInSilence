"""Synthetic safety checks; no public observations or model loading."""
import json, hashlib
from pathlib import Path
import numpy as np
from visibility import HandVisibility
U=Path(__file__).resolve().parent
results=[]
def scene(left=True,right=True):
    xy=np.full((75,2),np.nan);ok=np.zeros(75,bool);conf=np.ones(33)
    for i,p in {11:(200,100),12:(400,100),13:(200,230),14:(400,230),15:(310,250),16:(315,250)}.items():xy[i]=p;ok[i]=True
    for side,yes in enumerate((left,right)):
        if yes:xy[33+21*side:54+21*side]=[310+5*side,250];ok[33+21*side:54+21*side]=True
    return xy,ok,conf
def step(v,a,t=100000,**kw):return v.step(*a,t,600,500,**kw)
def check(name,cond):
    assert cond,name
    results.append(name)
a=scene();v=HandVisibility();check('observed unchanged',[x['state'] for x in step(v,a,0)]==['observed','observed'])
b=scene(False,True);before=[x.copy() for x in b];r=step(v,b);check('overlap requires own track',r[0]['state']=='overlap_likely')
check('immutable arrays',all(np.array_equal(x,y,equal_nan=True) for x,y in zip(b,before)))
check('no track inside fallback',step(HandVisibility(),b)[0]['state']=='inside_missing')
check('expired track no overlap',step(v,b,600000)[0]['state']=='inside_missing')
v=HandVisibility();step(v,a,0);check('withheld never promoted',step(v,b,unassigned_raw=True)[0]['state']=='assignment_uncertain')
check('partial remains partial',step(v,b,200000,assigned=(True,True))[0]['state']=='partial_observation')
c=scene(False,True);c[0][15]=[200,510];c[1][15]=False
check('outside wrist is diagnostic estimate',step(HandVisibility(),c)[0]['state']=='boundary_outside_likely')
c[0][15]=[200,480];check('boundary partially cut distinguished',step(HandVisibility(),c)[0]['state']=='boundary_outside_likely')
c[0][13]=[200,490];check('short invalid arm unknown',step(HandVisibility(),c)[0]['state']=='unknown')
c=scene(False,True);c[2][15]=0.1;check('weak pose unknown',step(HandVisibility(),c)[0]['state']=='unknown')
c=scene(False,True);c[0][15]=np.nan;check('nan pose unknown',step(HandVisibility(),c)[0]['state']=='unknown')
c=scene(False,True);c[0][12]=c[0][11];check('zero scale unknown',step(HandVisibility(),c)[0]['state']=='unknown')
v=HandVisibility();step(v,a,500000);check('time reversal clears track',step(v,b,400000)[0]['state']=='inside_missing')
v=HandVisibility();step(v,a,0);check('dimension change clears track',v.step(*b,100000,800,500)[0]['state']=='inside_missing')
v=HandVisibility();step(v,a,0);v.reset();check('new clip clears track',step(v,b)[0]['state']=='inside_missing')
v=HandVisibility();step(v,a,0);d=scene(False,False);check('no other hand no overlap',step(v,d)[0]['state']=='inside_missing')
v=HandVisibility();step(v,a,0);d=scene(False,True);d[0][15]=[200,350];check('not close no overlap',step(v,d)[0]['state']=='inside_missing')
v=HandVisibility();step(v,a,0);d=scene(False,True);d[0][15]=[200,400];d[0][54:75]=[205,400];check('no own track continuity no overlap',step(v,d)[0]['state']=='inside_missing')
# Resolution scaling preserves dimensionless evidence categories.
v1=HandVisibility();v2=HandVisibility()
for t,z in [(0,a),(100000,b)]:
    q=list(z);q[0]=z[0]*2
    x=step(v1,z,t);y=v2.step(*q,t,1200,1000)
    check('resolution scaling '+str(t),[i['state'] for i in x]==[i['state'] for i in y])
# Equal input prefixes yield identical outputs regardless of later frame.
v1=HandVisibility();v2=HandVisibility();prefix=[]
for t,z in [(0,a),(100000,b),(200000,b)]:
    x=step(v1,z,t);y=step(v2,z,t);prefix.append(x==y)
step(v1,a,300000);step(v2,d,300000);check('causal prefix stability',all(prefix))
check('no recovery requested',all(x['coordinate_repair'] is False for x in r))
with (U/'SYNTHETIC_TESTS.json').open('x',encoding='utf-8') as f:json.dump({'status':'PASS','checks':results,'code_sha256':hashlib.sha256((U/'visibility.py').read_bytes()).hexdigest()},f,indent=2)
print('PASS',len(results))
