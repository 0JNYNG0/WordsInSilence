"""Use body-context model only as an explicit contradiction guard, not replacement."""
from graph import *
from binding import *
from prepare_linker import load
from collections import Counter
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111'
def context_conflict(p,valid,span,wanted):
 if span is None:return dict(conflict=False,reason='no_selection')
 a,b=span;v=np.asarray(valid)[a:b+1];q=np.asarray(p)[a:b+1]
 if v.mean()<.8:return dict(conflict=False,reason='insufficient_context_observation')
 avg=q[v].mean(0);other=int(np.argmax(avg[1:8]))+1
 conflict=bool(other!=wanted and avg[other]>=.65 and avg[wanted]<=.35)
 return dict(conflict=conflict,expected=int(wanted),observed=other,expected_support=float(avg[wanted]),other_support=float(avg[other]),reason='explicit_other_body_context' if conflict else 'no_strong_contradiction')
def main():
 assert not (U/'CONTEXT_GUARD_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_guard_requirement',G/'requirement_core.py');cfg=read(U/'LINKER_FROZEN.json');sources={(s['clip_id'],s['kind']):s for s in read(R/'EVALUATION_INPUTS.json')['rows']};selection=read(U/'UNORDERED_SELECTION_RESULTS.json');joint={(r['seed'],r['clip_id'],r['kind']):r for r in read(U/'JOINT_ORDER_RESULTS.json')};truth=read(U/'reviews/sequence_truth_audit.json')
 write(U/'CONTEXT_GUARD_FROZEN.json',dict(reason='Body-only replacement regressed; test only explicit other-context contradiction while preserving existing positive/shape/direction evidence. No abstention inferred from weak body confidence.',thresholds=dict(valid=.8,other=.65,wanted=.35),scope='common family descriptor; no word/frame exception',baseline='fixed source decoder + frozen order-head agreement',known_truth='Two reviewed selected-span source-role conflicts. Other three historical supports not automatically false sign decisions; preserve interpretation correction.',not_whole_sign_classifier=True,new_training=False,freeze_after_prior_exploration=True,not_fresh_evaluation=True))
 results=[];cache={}
 for seed in [42,43,44]:
  net=BodyContextLinker(np.array(cfg['mean'],np.float32),np.array(cfg['std'],np.float32));net.load_state_dict(torch.load(U/'linker_runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval()
  for s in selection:
   cid,kind=s['clip_id'],s['kind'];meta=sources[cid,kind]
   if (cid,kind) not in cache:cache[cid,kind]={k:v for k,v in np.load(meta['path']).items()}
   c=cache[cid,kind];av=rc.prior.shared.available(c['x']);families=rc.FAMILIES[cid.split('_')[2]]
   with torch.inference_mode():p=net(torch.tensor(body_context(c['z'],c['ts']))).softmax(-1).numpy()
   checks={}
   for card in s['cards']:
    spec=next(f for f in families if f['id']==card['id']);f=spec['families'][0];valid=av[:,1] if f<=3 else av[:,1]&av[:,2];checks[card['id']]=context_conflict(p,valid,card['span'],f)
   old=joint[seed,cid,kind]['pairs'];pairs=[]
   for pair in old:
    conflicted=any(checks[k]['conflict'] for k in pair['cards']);pairs.append(dict(**{k:v for k,v in pair.items() if k!='state'},before=pair['state'],state='association_conflict' if conflicted else pair['state']))
   results.append(dict(seed=seed,clip_id=cid,kind=kind,checks=checks,pairs=pairs))
 write(U/'CONTEXT_GUARD_RESULTS.json',results);summary={str(s):{k:dict(Counter(p['state'] for r in results if r['seed']==s and r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in results})} for s in [42,43,44]};write(U/'CONTEXT_GUARD_COMPLETE.json',dict(summary=summary,private_used=False,web_changed=False));print(summary)
if __name__=='__main__':main()
