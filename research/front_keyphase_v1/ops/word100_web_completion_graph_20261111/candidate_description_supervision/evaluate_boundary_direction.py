from boundary_direction_core import *
from collections import Counter
import time,traceback
E=M/'evaluation'
def main():
 E.mkdir(exist_ok=True);assert not(E/'RESULTS.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_bd_eval_req',O/'word100_guide_requirement_learning_20261111/requirement_core.py');obs=rc.frozen_observer();models={}
 for run in read(M/'TRAINING_COMPLETE.json')['rows']:
  assert sha(run['model'])==run['model_sha256'];m=DirectionHead(obs);m.load_state_dict(torch.load(run['model'],map_location='cpu',weights_only=True));m.eval();models[run['seed']]=m
 shadow=read(L/'evaluation/shadow_components/RESULTS.json');comp=read(L/'evaluation/COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in comp['predictions']}
 write(E/'FROZEN.json',dict(at=stamp(),selection=dict(conditions=['baseline','local_old','local_tuned']),candidate_spans='Exactly saved local_full; no reselection',scope='Only direction axes1..4 copied head, old shapes and axis0 preserved',pins={str(p):sha(p)for p in[Path(__file__),W/'boundary_direction_core.py',M/'TRAINING_COMPLETE.json',M/'SCOPED_INPUTS.json',M/'TARGET_REVIEW.json',L/'evaluation/shadow_components/RESULTS.json',L/'evaluation/COMPARISON.json']},no_threshold_change=True,no_old_tcn_or_dtw_rerun=True,web_changed=False))
 cache={};out=[];records=[]
 for r in shadow['rows']:
  key=r['clip_id'],r['kind'];p=pm[r['seed'],*key]
  if key not in cache:
   assert sha(p['source'])==p['source_sha256'];z=np.load(p['source']);cache[key]={k:z[k]for k in ['x','fh','ts']}
  d=cache[key];conditions={'baseline':r['conditions']['baseline'],'local_old':r['conditions']['local_full']};cards=[]
  for old in r['conditions']['local_full']:
   card=copy.deepcopy(old)
   for event in card['evidence']:
    if event['task']!='direction'or'axis'not in event or event['axis']==0:continue
    a,b=event['span'];word=r['clip_id'].split('_')[2];spec=next(v for v in rc.FAMILIES[word]if v['id']==card['card']);side=spec['hands'][0];g,ok,info=rc.prior.geometry(d['x'],d['ts'],a,b,event['axis'],side,'L'if side=='R'else'R');ha,hb=d['fh'][[a,b]];ctx=np.r_[ha,hb,hb-ha]
    with torch.inference_mode():logits=models[r['seed']](torch.tensor(ctx[None]),torch.tensor(g[None]))[0];prob=logits.softmax(-1).numpy()
    event.update(probabilities=prob.tolist(),state=rc.direction_verdict(prob,event['desired'],ok),model_scope='typed_direction_axes1_to4')
    records.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=card['card'],span=[a,b],axis=event['axis'],desired=event['desired'],eligible=bool(ok),geometry=info,logits=logits.tolist(),probabilities=prob.tolist(),state=event['state'],source=p['source'],source_sha256=p['source_sha256']))
   states=[v['state']for v in card['evidence']]
   if card['span']is None:card['status']='association_unknown'
   elif 'difference'in states or'opposite_direction'in states:card['status']='difference_evidence'
   elif states and all(v=='supported'for v in states):card['status']='supported_features_only'
   else:card['status']='insufficient_evidence'
   cards.append(card)
  conditions['local_tuned']=cards;out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],conditions=conditions))
 summary=[]
 for seed in [42,43,44]:
  for mode in ['baseline','local_old','local_tuned']:
   for kind in sorted({r['kind']for r in out}):
    cc=[c for r in out if r['seed']==seed and r['kind']==kind for c in r['conditions'][mode]];items=[v for c in cc for v in c['evidence']];summary.append(dict(seed=seed,condition=mode,kind=kind,cards=len(cc),statuses=dict(Counter(c['status']for c in cc)),shape=dict(Counter(v['state']for v in items if v['task']=='shape')),direction=dict(Counter(v['state']for v in items if v['task']=='direction'))))
 write(E/'RESULTS.json',dict(at=stamp(),rows=out,direction_predictions=records,summary=summary,common100_changed=False,web_changed=False))
 for s in summary:
  if s['condition']=='local_tuned'and s['kind']in['full_normal','speed125','main_reversed','closed_hand_and_reversed']:print(s)
if __name__=='__main__':
 try:main()
 except BaseException:E.mkdir(exist_ok=True);(E/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
