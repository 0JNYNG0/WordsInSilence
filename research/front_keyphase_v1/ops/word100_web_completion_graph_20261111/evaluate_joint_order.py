from graph import *
from prepare_linker import load
import sys,numpy as np,torch
from collections import Counter
O=U.parent;G=O/'word100_guide_requirement_learning_20261111';R=O/'word100_candidate_role_learning_20261111'
def main():
 assert not (U/'JOINT_ORDER_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_joint_order_req',G/'requirement_core.py');sys.modules['requirement_core']=rc;om=load('_joint_order_model',G/'order_binding.py');cards=rc.CARDS
 src={(s['clip_id'],s['kind']):s for s in read(R/'EVALUATION_INPUTS.json')['rows']};selected=read(U/'UNORDERED_SELECTION_RESULTS.json');cache={k:np.load(r['path'])['fh'] for k,r in src.items()}
 write(U/'JOINT_ORDER_FROZEN.json',dict(basis='Combine exact source-selection bugfix with previously trained antisymmetric order head; no new training',inputs='independently selected actual observations, then chronological order and learned endpoint compatibility',decision='supported or reversed only when temporal order and neural head agree at original .35/.65; otherwise uncertain',seeds=[42,43,44],normal_reference='source supplied normal/speed9distinctclause pairs, not provided12annotation pairs',new_model_inference_only='frozen order head on new correspondence windows; prior completed windows not rerun',no_new_thresholds=True))
 out=[]
 for seed in [42,43,44]:
  net=om.OrderedBinding();net.load_state_dict(torch.load(G/'order_runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval()
  for row in selected:
   h=cache[row['clip_id'],row['kind']];word=row['clip_id'].split('_')[2];pairs=[]
   for p in row['pairs']:
    st=p['state'];sa,sb=p['spans'];v=None
    if sa is not None and sb is not None:
     a,b=sorted([int(round(sum(sa)/2)),int(round(sum(sb)/2))]);ga=next(c['groups'][0][0] for c in cards[word] if c['id']==p['cards'][0]);gb=next(c['groups'][0][0] for c in cards[word] if c['id']==p['cards'][1])
     with torch.inference_mode():v=float(net(torch.tensor(rc.observation(h,a,b)[None]),torch.tensor(rc.requirement('order',guides=[ga,gb])[None])).sigmoid()[0])
     st='supported' if st=='supported' and v>=.65 else 'reversed' if st=='reversed' and v<=.35 else 'ambiguous'
    pairs.append(dict(**{k:v for k,v in p.items() if k!='state'},state=st,order_probability=v))
   out.append(dict(seed=seed,clip_id=row['clip_id'],kind=row['kind'],pairs=pairs))
 write(U/'JOINT_ORDER_RESULTS.json',out);summary={str(s):{k:dict(Counter(p['state'] for r in out if r['seed']==s and r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in out})} for s in [42,43,44]};write(U/'JOINT_ORDER_COMPLETE.json',dict(summary=summary,web_changed=False));print(summary)
if __name__=='__main__':main()
