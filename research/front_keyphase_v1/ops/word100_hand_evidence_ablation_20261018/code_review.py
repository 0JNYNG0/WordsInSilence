"""Structural comparison with the previously verified trainer; no model execution."""
from common import *
import ast

def functions(p):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}

a,b=functions(SRC/'train.py'),functions(U/'train.py')
for name in ('initialize','statehash','savept','event','scores','run_one'):
 assert a[name]==b[name],name
old=(SRC/'train.py').read_text(encoding='utf-8')
patched=old.replace(" if arm=='C0':x[:,:,280:]=0", " off,size=GROUPS[arm.removeprefix('minus_')]\n for side in (0,1):x[:,:,280+side*15+off:280+side*15+off+size]=0")
node=next(n for n in ast.parse(patched).body if isinstance(n,ast.FunctionDef) and n.name=='batch')
assert ast.dump(node,include_attributes=False)==b['batch']
write(U/'IMPLEMENTATION_REVIEW.json',{'status':'PASS','at':now(),'identical_functions':['initialize','statehash','savept','event','scores','run_one'],'batch_change_only':'Zero specified added group on both hands, in allocated batch copy, both training and evaluation.','loadrows_change':'Read only previous immutable inputs_v2; no new features.','main_change':'New unit context/READY/locks and four-arm sequence.','training_updates_during_review':0,'reviewer':'Codex local structural check; not independent agent'},new=True)
print('PASS: initialization, optimizer/selection/checkpoint and scoring routines unchanged.')
