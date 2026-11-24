from pathlib import Path
import json,numpy as np
from candidate_v2 import inspect_pair
U=Path(__file__).resolve().parent
a=np.zeros((21,2),float)
for f in range(5):
 for j in range(4):a[1+4*f+j]=[100+f*20,100-j*25]
a[0]=[140,140];v=np.ones(21,bool)
def flag(x,y,va=v,vb=v,s=300):return inspect_pair(x,y,va,vb,s)['hold_new']
tests={}
b=a.copy();b[0]+=[100,70]
tests['wrong_wrist_cannot_bypass_same_fingers']=flag(a,b)
tests['source_order_symmetry']=flag(a,b)==flag(b,a)
for name,mat in [('mirror',np.diag([-1.,1.])),('rotation',np.array([[0.,-1.],[1.,0.]]))]:tests[name]=flag(a@mat,b@mat)
tests['translation_scale']=flag(a*1.7+[80,90],b*1.7+[80,90],s=510)
tests['distinct_hand']=not flag(a,a+[100,0])
b=a.copy();b[[4,8,12,16,20]]=a[4]
tests['ambiguous_identical_tip_positions_not_unique']=not flag(b,b)
vv=v.copy();vv[20]=False
tests['missing_tip_no_forced_identity']=not flag(a,a,v,vv)
b=a+[100,0];b[1:5]=a[1:5]
tests['one_contact_not_duplicate']=not flag(a,b)
assert all(tests.values()),tests
dest=U/'INVARIANTS_V2.json';assert not dest.exists()
dest.write_text(json.dumps({'status':'PASS','tests':tests,'scope':'Synthetic invariants, not detection accuracy'},indent=2),encoding='utf-8')
print(len(tests),'V2 invariants PASS')
