from graph import *
from order_evidence import *
from prepare_linker import load
from collections import Counter
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111'
def main():
 assert not (U/'ORDER_EVIDENCE_COMPLETE.json').exists()
 rc=load('_order_mass_req',G/'requirement_core.py');src=read(R/'EVALUATION_INPUTS.json')['rows'];saved=read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows'];configs=rc.FAMILIES
 write(U/'ORDER_EVIDENCE_FROZEN.json',dict(reason='All-occurrence Boolean ordering lets tiny secondary support invalidate the main order; aggregate actual-time evidence without erasing any occurrence',threshold=.65,unchanged_role_checkpoint=True,frames_minimum=None,expected_order_assignment=False,word_exceptions=False,provenance='Deterministic arithmetic of saved probabilities; no model refit or repeated model inference',not_calibrated_error_probability=True,requires_distinct_repeat_check=True,normal_speed_gate='all nine distinct clause pairs supported in each seed',difference_gate='false support below five plus at least one reversed on whole-core diagnostic in every seed; ambiguity reported, no natural learner claim',code_sha=hashlib.sha256((U/'order_evidence.py').read_bytes()).hexdigest()))
 arrays={s['key']:{k:v for k,v in np.load(s['path']).items()} for s in src};meta={(s['clip_id'],s['kind']):s for s in src};out=[]
 for r in saved:
  cid,kind=r['clip_id'],r['kind'];s=meta[cid,kind];c=arrays[s['key']];b=read(r['result']);p=np.asarray(b['role_probabilities']);av=rc.prior.shared.available(c['x']);v=np.zeros_like(p,bool)
  v[:,1:4]=av[:,1,None];v[:,4:8]=(av[:,1]&av[:,2])[:,None];specs=[f for f in configs[cid.split('_')[2]] if f['required'] and not f.get('parent')];pairs=[]
  for a,bb in zip(specs,specs[1:]):
   fa,fb=a['families'][0],bb['families'][0]
   if fa==fb:continue
   pairs.append(dict(cards=[a['id'],bb['id']],families=[fa,fb],**temporal_order_evidence(p,v,c['ts'],fa,fb)))
  out.append(dict(seed=r['seed'],clip_id=cid,kind=kind,pairs=pairs))
 write(U/'ORDER_EVIDENCE_RESULTS.json',out);summary={str(s):{k:dict(Counter(p['state'] for r in out if r['seed']==s and r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in out})} for s in [42,43,44]};write(U/'ORDER_EVIDENCE_COMPLETE.json',dict(summary=summary,private_used=False,web_changed=False));print(summary)
if __name__=='__main__':main()
