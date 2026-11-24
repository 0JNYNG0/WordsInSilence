from pathlib import Path
import json,numpy as np
from candidate import inspect_pair
U=Path(__file__).resolve().parent
a=np.zeros((21,2),float)
for f in range(5):
 for j in range(4):a[1+4*f+j]=[100+f*20,100-j*25]
a[0]=[140,140];v=np.ones(21,bool);scale=300.
tests={}
tests['exact_duplicate']=inspect_pair(a,a,v,v,scale)['hold_new']
b=a.copy();b[0]+=[100,70]
tests['wrist_displacement_cannot_bypass']=inspect_pair(a,b,v,v,scale)['hold_new']
b=a+[250,0];tests['separate_hand_retained']=not inspect_pair(a,b,v,v,scale)['hold_new']
b=a+[100,0];b[1:5]=a[1:5]
tests['one_touching_finger_not_duplicate']=not inspect_pair(a,b,v,v,scale)['hold_new']
b=a.copy();b[1:5]+=[90,60];tests['four_agreeing_fingers_hold']=inspect_pair(a,b,v,v,scale)['hold_new']
vv=v.copy();vv[[4,8]]=False;tests['missing_fingers_not_imputed']=inspect_pair(a,a,v,vv,scale)['status']=='insufficient_finger_evidence'
tests['zero_scale_unknown']=inspect_pair(a,a,v,v,0)['status']=='insufficient_finger_evidence'
for name,aa,bb,sc in [('translate',a+[80,90],a+[80,90],300),('scale',a*1.7,a*1.7,510),('mirror',a*np.array([-1,1]),a*np.array([-1,1]),300)]:
 tests[name]=inspect_pair(aa,bb,v,v,sc)['hold_new']
tests['source_order_symmetry']=inspect_pair(a,b,v,v,300)['hold_new']==inspect_pair(b,a,v,v,300)['hold_new']
assert all(tests.values()),tests
(U/'INVARIANTS.json').write_text(json.dumps({'status':'PASS','tests':tests,'scope':'Synthetic policy invariants, not real accuracy'},indent=2),encoding='utf-8');print(len(tests),'invariantsPASS')

