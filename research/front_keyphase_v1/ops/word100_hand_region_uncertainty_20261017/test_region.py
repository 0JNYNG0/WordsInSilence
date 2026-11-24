import json,hashlib
from pathlib import Path
import numpy as np
from region import HandRegion
U=Path(__file__).resolve().parent
checks=[]
def scene(left=(450.,550.),right=(470.,550.)):
    p=np.zeros((75,2),np.float32);m=np.zeros(75,bool);c=np.zeros(33,np.float32)
    p[[11,12,13,14,15,16]]=[[400,300],[600,300],[400,450],[600,450],left,right]
    m[[11,12,13,14,15,16]]=True;c[[11,12,13,14,15,16]]=1
    offsets=np.array([(x,y) for y in np.linspace(0,80,7) for x in (-30,0,30)],np.float32)
    offsets[0]=0
    for s,w in enumerate((left,right)):p[33+s*21:54+s*21]=offsets+np.array(w);m[33+s*21:54+s*21]=True
    return p,m,c
def run(model,arr,t=0,size=(1000,1000),missing=False,withheld=False):
    p,m,c=[x.copy() for x in arr]
    if missing:m[33:54]=False;p[33:54]=0
    before=[x.copy() for x in (p,m,c)]
    result=model.step(p,m,c,t,*size,assigned=(not missing,True),unassigned_raw=withheld)
    assert all(np.array_equal(x,y,equal_nan=True) for x,y in zip(before,(p,m,c)))
    return result
def ok(name,condition):assert condition,name;checks.append(name)
a=scene();h=HandRegion();r=run(h,a)
ok('observed preserved without changing input',all(x['state']=='observed' for x in r))
r=run(h,a,33333,missing=True)
ok('nearby whole regions support overlap',r[0]['state']=='overlap_likely')
ok('no coordinate recovery',all(x['coordinate_repair'] is False for x in r))
u=r[0]['evidence']['uncertainty_pixels'];r=run(h,a,200000,missing=True)
ok('uncertainty grows with missing duration',r[0]['evidence']['uncertainty_pixels']>u)
r=run(h,a,450000,missing=True)
ok('wide uncertainty abstains',r[0]['state']=='region_uncertain')
r=run(h,a,600001,missing=True)
ok('old hand region expires',r[0]['region_status']=='unavailable')
h=HandRegion();run(h,a);r=run(h,a,33333,missing=True,withheld=True)
ok('ambiguous identity not promoted',r[0]['state']=='assignment_uncertain')
h=HandRegion();run(h,a);r=run(h,a,0,missing=True)
ok('nonmonotonic timestamps reset region',r[0]['region_status']=='unavailable')
h=HandRegion();run(h,a);r=run(h,a,33333,(900,1000),missing=True)
ok('new dimensions reset region',r[0]['region_status']=='unavailable')
h=HandRegion();near=scene(left=(450,860),right=(620,550))
# Valid arm geometry for lowered wrist: elbow tracks lower too.
near[0][13]=[425,650]
near[0][[11,12],1]=500
run(h,near);r=run(h,near,33333,(1000,1000),missing=True)
ok('well inside lower hand extent stays inside',r[0]['state']=='inside_missing')
edge=scene(left=(450,925),right=(620,550));edge[0][13]=[425,700]
edge[0][[11,12],1]=500
h=HandRegion();run(h,near);r=run(h,edge,33333,missing=True)
ok('wrist inside but fingers may cross edge abstains',r[0]['state']=='boundary_uncertain')
far=scene(left=(450,970),right=(620,550));far[0][13]=[425,730]
far[0][[11,12],1]=500
h=HandRegion();run(h,near);r=run(h,far,33333,missing=True)
ok('region persists beyond edge across envelope',r[0]['state']=='boundary_outside_likely')
h=HandRegion();clip=scene(left=(450,970));clip[0][13]=[425,730]
clip[0][[11,12],1]=500
run(h,clip);r=run(h,clip,33333,missing=True)
ok('partly out-of-frame coordinates not saved as full hand history',r[0]['region_status']=='unavailable')
h=HandRegion();run(h,a);altered=scene();altered[0][34:54,0]+=15
run(h,altered,33333);r=run(h,altered,66666,missing=True)
ok('relative shape change expands uncertainty',r[0]['evidence']['residual_speed_pixels_per_second']>0)
h=HandRegion();run(h,a);p,m,c=a
p2=p*2
r=run(h,a,33333,missing=True)
j=HandRegion();run(j,(p2,m,c),0,(2000,2000));s=run(j,(p2,m,c),33333,(2000,2000),missing=True)
ok('coordinate scale invariance',r[0]['state']==s[0]['state'] and abs(r[0]['evidence']['uncertainty_shoulders']-s[0]['evidence']['uncertainty_shoulders'])<1e-6)
j=HandRegion();run(j,a);prefix=run(j,a,33333,missing=True)
run(j,scene(left=(500,650)),66666)
k=HandRegion();run(k,a);prefix2=run(k,a,33333,missing=True)
ok('prefix independent of future',prefix==prefix2)
out={'status':'PASS','checks':checks,'code_sha256':hashlib.sha256((U/'region.py').read_bytes()).hexdigest()}
with (U/'SYNTHETIC_TESTS.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
print(json.dumps({'status':'PASS','checks':len(checks)}))
