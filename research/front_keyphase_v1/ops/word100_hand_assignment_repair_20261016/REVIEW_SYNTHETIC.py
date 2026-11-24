"""Independent reviewer scenarios, no public result selection or model calls."""
from pathlib import Path
import hashlib,json,sys,importlib.util
import numpy as np
U=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('review_repair_assignment',U/'assignment.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def palm(x,y):return np.array([[x+(j%5)*4,y+(j//5)*5] for j in range(21)],float)
def body():
    p=np.zeros((33,2));v=np.zeros(33,bool);p[11]=[300,100];p[12]=[500,100];v[[11,12]]=True
    return p,v
passed=[]

# Incorrect high-confidence label does not veto a uniquely nearby anatomical wrist.
p,v=body();h=np.array([palm(600,300)]);p[16]=h[0,0];v[16]=True
a=m.HandAssigner();r,d=a.assign(h,[0],[.99],p,v,0,1000,800)
assert r=={0:1} and d['reasons']==['bounded_pose_over_label'];passed.append('bounded_pose_over_wrong_high_label')

# Same-side duplicate alternatives retain the original pose-supported hypothesis,
# not whichever happens to lie nearest the previous palm trajectory.
p,v=body();h=np.array([palm(600,300),palm(610,300)]);p[16]=h[0,0];v[16]=True
a=m.HandAssigner();a.assign(h[:1],[1],[.99],p,v,0,1000,800)
p[16]=h[1,0];r,d=a.assign(h,[1,1],[.99,.81],p,v,33333,1000,800)
assert r=={0:None,1:1} and d['same_side_competition']['winner_raw_index']==1
passed.append('nearest_current_pose_not_temporal_residual')
b=m.HandAssigner();p0,v0=body();p0[16]=h[0,0];v0[16]=True;b.assign(h[:1],[1],[.99],p0,v0,0,1000,800)
r2,d2=b.assign(h[::-1],[1,1],[.81,.99],p,v,33333,1000,800)
assert r2=={0:1,1:None};passed.append('same_side_selection_order_invariant')

# Two very close real hands with distinct anatomical histories are not merged.
p,v=body();h=np.array([palm(600,300),palm(602,300)]);p[[15,16]]=h[:,0];v[[15,16]]=True
a=m.HandAssigner();r,d=a.assign(h,[0,1],[.99,.99],p,v,0,1000,800)
assert r=={0:0,1:1} and d['close_palms'];passed.append('close_palms_distinct_pose_preserved')
v[15]=False;r,d=a.assign(h,[0,1],[.99,.99],p,v,33333,1000,800)
assert r=={0:0,1:1} and d['same_side_competition'] is None;passed.append('close_palms_distinct_tracks_preserved')

# Known information limit: without distinct pose/history evidence, two true hands
# can compete for the only available pose wrist. Record, do not label duplicate GT.
p,v=body();p[16]=h[1,0];v[16]=True;a=m.HandAssigner();r,d=a.assign(h,[0,1],[.99,.99],p,v,0,1000,800)
assert r=={0:None,1:1};limit={'scenario':'true_close_pair_single_pose_no_history','output':r,'reason':d['reasons'],'meaning':'Not a guarantee that suppressed hypothesis is a false duplicate.'}

# Wrong-side initialization correction still works after two consecutive confirmations.
p,v=body();h=np.array([palm(600,300)]);a=m.HandAssigner();a.assign(h,[1],[.99],p,v,0,1000,800)
p[15]=h[0,0];v[15]=True;r,_=a.assign(h,[0],[.99],p,v,33333,1000,800);assert r=={0:None}
r,d=a.assign(h,[0],[.99],p,v,66666,1000,800);assert r=={0:0};passed.append('wrong_init_confirmation_recovery_preserved')

result={'status':'PASS_WITH_RECORDED_INFORMATION_LIMIT','tests_passed':passed,'limitation':limit,'code_sha256':hashlib.sha256((U/'assignment.py').read_bytes()).hexdigest(),'scope_sha256':hashlib.sha256((U/'SCOPE.json').read_bytes()).hexdigest(),'detector_calls':0,'classifier_calls':0,'training':0}
with (U/'REVIEW_SYNTHETIC_RESULT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
