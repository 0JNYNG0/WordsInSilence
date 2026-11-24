import json,copy
from pathlib import Path
import numpy as np
from spatial import extend_evidence
U=Path(__file__).resolve().parent
xy=np.zeros((75,2),np.float32);mask=np.ones(75,bool);conf=np.ones(33,np.float32)
xy[11]=[30,30];xy[12]=[70,30]
xy[[15,17,19,21]]=[[45,45],[55,55],[45,55],[55,45]]
xy[54:75]=np.resize(np.array([[48,48],[58,48],[58,58],[48,58]],np.float32),(21,2))
base=[{'side':s,'position':{'state':'wrist_interior' if s==0 else 'observed_points'},
 'shape':{'state':'unobserved' if s==0 else 'usable_observed_points'},'overlap':{'occlusion_cause_confirmed':False},
 'coordinate_repair':False,'eligible_for_automatic_exclusion':False} for s in (0,1)]
cases=[]
def run(name,b=base,p=xy,m=mask,c=conf,w=100,h=100):
 originals=copy.deepcopy(b);arrays=[a.copy() for a in (p,m,c)]
 out=extend_evidence(b,p,m,c,w,h)
 assert b==originals and all(np.array_equal(a,z,equal_nan=True) for a,z in zip((p,m,c),arrays))
 assert all(all(r[k]==b[s][k] for k in b[s]) for s,r in enumerate(out))
 assert all(not r['current_overlap']['occlusion_cause_confirmed'] and not r['current_extent']['whole_hand_visibility_proven'] for r in out)
 cases.append(name);return out[0]
r=run('positive polygon intersection');assert r['current_overlap']['status']=='current_2d_intersection_suspected'
p=xy.copy();p[[17,19],1]=110
r=run('wrist interior but distal points outside',p=p);assert r['current_extent']['crossed_edges']==['bottom'] and r['current_overlap']['status']=='boundary_competes_with_overlap'
p=xy.copy();p[54:75]+=30
assert run('disjoint polygons',p=p)['current_overlap']['status']=='no_current_2d_intersection'
p=xy.copy();p[[15,17,19,21]]=40
assert run('degenerate region',p=p)['current_extent']['status']=='degenerate_pose_region'
for val in (0.,float('nan')):
 c=conf.copy();c[17]=val;assert run('low/nonfinite pose confidence',c=c)['current_extent']['status']=='coarse_pose_unavailable'
p=xy.copy();p[17,0]=float('nan');assert run('nonfinite coordinate',p=p)['current_extent']['status']=='coarse_pose_unavailable'
for scale in (.25,2.,10.):
 r=run('scale equivariance',p=xy*scale,w=100*scale,h=100*scale);assert r['current_overlap']['status']=='current_2d_intersection_suspected'
p=xy.copy();p[:,0]=100-p[:,0];assert run('horizontal mirror',p=p)['current_overlap']['status']=='current_2d_intersection_suspected'
for state in ('observed_points','partial_points','assignment_unresolved','unknown'):
 b=copy.deepcopy(base);b[0]['position']['state']=state;assert run('protect '+state,b=b)['current_extent']['status']=='not_assessed'
b=copy.deepcopy(base);b[1]['shape']['state']='unobserved';assert run('other missing',b=b)['current_overlap']['status']=='other_hand_not_observed'
with (U/'CONTRACT_TESTS.json').open('x',encoding='utf-8') as f:json.dump({'status':'PASS','cases':cases,'count':len(cases)},f,indent=2)
print('PASS',len(cases))
