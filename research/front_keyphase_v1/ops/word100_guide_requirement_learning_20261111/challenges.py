"""Additional negative-order controls and corrected direction evidence semantics.
First saved evaluation is retained; no checkpoint, label, or threshold tuning.
"""
from requirement_core import *
import time,traceback
def main():
 assert (U/'EVALUATION_COMPLETE.json').exists() and not (U/'CHALLENGE_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cfg=read(U/'FROZEN.json');rows={r['clip_id']:r for r in public_rows() if r['role']=='development'}
 sources=read(R/'EVALUATION_INPUTS.json')['rows'];normal={r['clip_id']:r for r in sources if r['kind']=='full_normal'}
 scores=read(U/'SCORES.json');full=read(U/'FULLPATH_DIAGNOSTICS.json');oldnet=frozen_observer()
 plan=dict(order='New reversed-requirement query on the same reviewed chronological endpoints. Negative compatibility, not filmed learner-error truth.',direction_correction='Low expected-direction probability can mean hold, not opposite motion. Require explicit opposite evidence; preserve first outputs. Additional two direction queries do not retune models.',short_spans='No blanket semantic target for shortened range. Holds, no movement, and lost observations are separate.',fixed=True)
 write(U/'CHALLENGE_FROZEN.json',plan)
 write(U/'SEMANTIC_CORRECTION.json',dict(first_output=str(U/'FULLPATH_DIAGNOSTICS.json'),cause='Binary non-match was displayed as direction difference even when a short supplied window contains a hold. This is a consumer interpretation error, not proof of learner error.',correction='Use explicit opposite-direction support; hold/ambiguous interval cannot issue wrong-direction correction.',training_repeated=False,checkpoints_changed=False,thresholds_changed=False,raw_first_scores_preserved=True))
 outputs=[];corrected=[];gates=[];cache={}
 def get(cid,kind):
  key=cid+'_'+kind
  if key not in cache:
   rec=next(r for r in sources if r['key']==key);cache[key]={k:v for k,v in np.load(rec['path']).items()}
  return cache[key]
 for seed in cfg['seeds']:
  n=RequirementAdapter(np.array(cfg['mean'],np.float32),np.array(cfg['std'],np.float32));n.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));n.eval()
  def predict(ob,req,pr):
   with torch.inference_mode():return float(n(torch.tensor(ob[None]),torch.tensor(req[None]),torch.tensor([logit(pr)])).sigmoid()[0])
  for s in scores:
   if s['seed']!=seed or s['task']!='order':continue
   c=get(s['clip_id'],'full_normal');h=c['fh'];a,b=s['a'],s['b'];ga,gb=s['guides'][::-1]
   with torch.inference_mode():base=float(oldnet.ordering(torch.tensor(h[None]),torch.tensor([[0,a,b]]),torch.tensor([[ga,gb]])).sigmoid()[0])
   prob=predict(observation(h,a,b),requirement('order',guides=[ga,gb]),base)
   outputs.append(dict(seed=seed,clip_id=s['clip_id'],pair=s['pair'],target=0,baseline_probability=base,probability=prob,before=decide(base),after=decide(prob),test='reversed_guide_requirement'))
  for r in full:
   if r['seed']!=seed:continue
   c=get(r['clip_id'],r['kind']);h=c['fh'];x=c['x'];ts=c['ts'];out=[];word=r['clip_id'].split('_')[2];st=time.perf_counter()
   for card in r['cards']:
    spec=next(s for s in FAMILIES[word] if s['id']==card['card']);req=[]
    for e in card['requirements']:
     e=dict(e)
     if e['task']=='direction' and 'probability' in e:
      aa,bb=e['span'];axis=spec['axis'];side=spec['hands'][0]
      # Axis 3 (distance) is symmetric; other fixed-direction guide subjects are first hand.
      g,ok,_=prior.geometry(x,ts,aa,bb,axis,side,'L' if side=='R' else 'R');want=int(spec['sign']+1)
      with torch.inference_mode():bp=oldnet.movement(torch.tensor(h[None]),torch.tensor(g[None]),torch.tensor([[0,aa,bb]])).softmax(-1)[0].numpy()
      # Reuse the saved desired-class result; compute only new hold/opposite matcher queries.
      pr=[e['probability'] if k==want else predict(observation(h,aa,bb,g),requirement('direction',axis=axis,direction=k),float(bp[k])) for k in range(3)]
      e.update(first_state=e['state'],state=direction_verdict(pr,want,ok),class_probabilities=pr,baseline_classes=bp.tolist(),evidence_semantics='Explicit opposite direction only; hold is not error')
     req.append(e)
    states=[q['state'] for q in req]
    dirs={q['state'] for q in req if q['task']=='direction'}
    if {'supported','opposite_direction'}.issubset(dirs):status='association_unknown'
    elif any(q['state']=='difference' for q in req if q['task']=='shape') or 'opposite_direction' in dirs:status='difference'
    elif states and all(v=='supported' for v in states):status='supported_features_only'
    else:status='insufficient_evidence'
    out.append(dict(card,first_after=card['after'],after=status,requirements=req))
   corrected.append(dict(r,cards=out,additional_query_seconds=time.perf_counter()-st))
  # Classification semantics tested independently of the trained model.
  for desired in [0,2]:
   for v,expected in [([.05,.9,.05],'movement_not_established'),([.34,.33,.33],'uncertain')]:
    gates.append(dict(seed=seed,desired=desired,probabilities=v,expected=expected,actual=direction_verdict(v,desired)))
   p=[.05,.05,.05];p[2-desired]=.9;gates.append(dict(seed=seed,desired=desired,probabilities=p,expected='opposite_direction',actual=direction_verdict(p,desired)))
   gates.append(dict(seed=seed,desired=desired,probabilities=[.9,.05,.05],expected='observation_unknown',actual=direction_verdict([.9,.05,.05],desired,False)))
 write(U/'ORDER_CHALLENGES.json',outputs);write(U/'CORRECTED_FULLPATH_DIAGNOSTICS.json',corrected);write(U/'SEMANTIC_GATE_TESTS.json',gates)
 write(U/'CHALLENGE_COMPLETE.json',dict(order_negative_queries=len(outputs),consumer_semantic_tests=len(gates),all_semantic_tests_pass=all(r['expected']==r['actual'] for r in gates),first_model_comparison_still_not_adopted=True,only_saved_positive_queries_reused=True,new_order_requirement_queries=True))
 print(read(U/'CHALLENGE_COMPLETE.json'))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'CHALLENGE_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
