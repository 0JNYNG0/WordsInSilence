"""Synthetic contract test only: paired IDs, unavailable inputs and independent recount."""
from common import *
import importlib.util, tempfile

def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 agg=module('fresh_aggregation_test',U/'aggregate_fresh.py');ver=module('fresh_verification_test',U/'verify_fresh.py')
 testroot=U/'synthetic_reporting_contract';testroot.mkdir(exist_ok=False);f=testroot/'fresh199';f.mkdir()
 rows=[dict(clip_id=f'SYNTHETIC_{i:03}',label_index=0,signer_id=14 if i<99 else 15) for i in range(199)]
 write(testroot/'FROZEN.json',{'pins':{},'protected':{}});write(testroot/'FRESH_SELECTION_CORRECTION.json',{'synthetic':True})
 write(f/'EVAL_CONTEXT.json',{'rows':rows,'pins':{}});write(f/'INPUTS_VERIFIED.json',{'status':'PASS','synthetic_only':True})
 # Sample0: baseline is wrong, other two members rescue. Sample1: unavailable in every model stays incorrect in n199.
 for arm,seeds in [('C',(20261013,)),('T3F',SEEDS)]:
  for seed in seeds:
   pred=[]
   for i,r in enumerate(rows):
    z=[0.0]*100;z[0]=3.0
    if i==0 and seed==20261013:z[0]=0;z[1]=3
    if i==1:z=None
    rank=sorted(range(100),key=lambda j:(-z[j],j)) if z else []
    pred.append({**r,'logits':z,'correct':bool(rank and rank[0]==0),'top5_correct':0 in rank[:5]})
   write(f/f'{arm}_{seed}_SCORES.json',dict(context_sha256=sha(f/'EVAL_CONTEXT.json'),model_state_before='synthetic',model_state_after='synthetic',training=0,checkpoint_reselected=False,checkpoint={},predictions={v:pred for v in VIEWS}))
 agg.U=testroot;agg.F=f;agg.main()
 # Verifier needs actual code hashes, never uses these fixture copies to execute code.
 for name in ['aggregate_fresh.py','verify_fresh.py']:(testroot/name).write_text((U/name).read_text('utf8'),encoding='utf8')
 ver.U=testroot;ver.F=f;ver.main()
 out=read(f/'COMPARISON.json')
 for v in VIEWS:
  assert out['scores'][v]['T3F13']['correct']==197 and out['scores'][v]['E3']['correct']==198
  assert out['scores'][v]['E3']['n']==199
  assert out['scores'][v]['E3_vs_T3F13']['gains']==['SYNTHETIC_000']
 # A mismatched ID must reject aggregation, not silently pair by order.
 q=read(f/'T3F_20261014_SCORES.json');q['predictions']['sensor'][0]['clip_id']='BAD_ID';write(f/'T3F_20261014_SCORES.json',q,new=False)
 caught=False
 try:agg.main()
 except AssertionError:caught=True
 assert caught
 write(U/'REPORTING_SELFTEST.json',dict(status='PASS',synthetic_not_real_scores=True,checks=['unavailable_kept_in_denominator','equal_prob_mean_rescues_wrong_baseline','paired_ID_mismatch_rejected','separate_scalar_recount_agrees'],code_pins={str(U/n):sha(U/n) for n in ['aggregate_fresh.py','verify_fresh.py','test_fresh_reporting.py']},training=0,new_inference=0))
 print('REPORTING SYNTHETIC CONTRACT PASS')
if __name__=='__main__':main()
