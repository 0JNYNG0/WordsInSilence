from core import *
from prepare_boundary_repair import inputs
import itertools,time,traceback
R=W/'boundary_direction_repair';V=W/'direction_neutral_candidates';E=R/'evaluation'
def main():
 assert not (E/'COMPARISON.json').exists();E.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(1);role=load('_boundary_eval_role',O/'word100_candidate_role_learning_20261111/role_core.py');old=read(V/'COMPARISON.json');pmap={(r['seed'],r['clip_id'],r['kind']):r for r in old['predictions']};heads={};boundaries={}
 cfg=dict(at=stamp(),selection=dict(conditions=['baseline','previous_normal_body','normal_body','neutral_body']),scope='Union old and repaired boundary proposals, fixed candidate heads, no desired-sign gate',old_selection_reused=True,encoder_rerun=False,thresholds_unchanged=True,pins={str(p):sha(p) for p in [R/'FROZEN.json',R/'TRAINING_COMPLETE.json',V/'COMPARISON.json',Path(__file__)]});write(E/'FROZEN.json',cfg)
 for folder,name in [(W,'normal_body'),(V,'neutral_body')]:
  for r in read(folder/'TRAINING_COMPLETE.json')['rows']:
   assert sha(r['model'])==r['model_sha256'];st=torch.load(r['model'],weights_only=True,map_location='cpu');net=CandidateHead(st['mean'].numpy(),st['std'].numpy());net.load_state_dict(st);net.eval();heads[r['seed'],name]=net
 for r in read(R/'TRAINING_COMPLETE.json')['rows']:
  assert sha(r['model'])==r['model_sha256'];net=nn.Sequential(nn.Linear(145,64),nn.GELU(),nn.Linear(64,3));net.load_state_dict(torch.load(r['model'],weights_only=True,map_location='cpu'));net.eval();boundaries[r['seed']]=net
 out=[];predictions=[];timing=[]
 for rec in old['rows']:
  key=rec['seed'],rec['clip_id'],rec['kind'];p=pmap[key];assert sha(p['source'])==p['source_sha256'];d=np.load(p['source']);z=d['z'];q=z[:,292:392].reshape(-1,10,10);t=time.perf_counter()
  with torch.inference_mode():bp=boundaries[rec['seed']](torch.tensor(inputs(z))).sigmoid().numpy()
  proposed=role.jc.candidate_spans(bp,q,[]);candidate_lists=[];features_new=[];spec=[]
  cards0=rec['conditions']['baseline']['cards']
  for ci,(card,previous) in enumerate(zip(cards0,rec['candidates'])):
   cs=[dict(c) for c in previous];seen={tuple(c['span']) for c in cs};axis=AX[card['family']];k=axis*2
   for c in proposed:
    if c['axis']!=axis or c['moving']!='R' or (c['a'],c['b']) in seen:continue
    ix=np.arange(c['a'],c['b']+1);observed=ix[q[ix,k,8]>.5]
    if len(observed)<3 or len(observed)/len(ix)<.8:continue
    qual=observed[q[observed,k,9]>.5];cs.append(dict(span=[c['a'],c['b']],source='repaired_'+c['source'],body_observed=observed.tolist(),hand_qualified=qual.tolist(),semantic_completion=None));seen.add((c['a'],c['b']));features_new.append(features(z,c['a'],c['b']));spec.append((ci,len(cs)-1,F.index(card['family'])))
   candidate_lists.append(cs)
  for name in ['normal_body','neutral_body']:
   with torch.inference_mode():pr=heads[rec['seed'],name](torch.tensor(np.array(features_new))).sigmoid().numpy() if features_new else []
   for (ci,idx,j),pp in zip(spec,pr):candidate_lists[ci][idx][name]=pp[j].tolist()
  modes={'baseline':rec['conditions']['baseline'],'previous_normal_body':rec['conditions']['normal_body']}
  for name in ['normal_body','neutral_body']:
   cards=[]
   for original,cs in zip(cards0,candidate_lists):
    eligible=[c for c in cs if c[name][0]>=.5];best=max(eligible,key=lambda c:np.log(np.clip(c[name],1e-6,1-1e-6)).sum()) if eligible else None;card=dict(card=original['card'],family=original['family'],chosen=best,semantic_completion=None)
    if 'truth' in original:
     g=original['truth'];obs=set(best['body_observed']) if best else set();card['truth']=dict(witness=g['witness'],wrong_witness=g['wrong_witness'],all_witness=set(g['witness'])<=obs,wrong_points=len(set(g['wrong_witness'])&obs))
    cards.append(card)
   orders=[]
   for i,j in itertools.combinations(range(len(cards)),2):
    if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
    a,b=cards[i]['chosen'],cards[j]['chosen'];state='unobserved' if a is None or b is None else 'ambiguous_overlap' if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1]) else 'forward' if a['span'][1]<b['span'][0] else 'reverse';orders.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state))
   modes[name]=dict(cards=cards,order=orders)
  dst=E/'predictions'/str(rec['seed'])/f"{rec['clip_id']}_{rec['kind']}.npz";dst.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dst,boundary=bp);predictions.append(dict(p,path=str(dst),sha256=sha(dst)));out.append(dict(seed=rec['seed'],clip_id=rec['clip_id'],kind=rec['kind'],conditions=modes,candidates=candidate_lists,added_candidates=len(spec)));timing.append(dict(seed=rec['seed'],clip_id=rec['clip_id'],kind=rec['kind'],seconds=time.perf_counter()-t))
  if len(out)%60==0:print('boundary eval',len(out),flush=True)
 summary=[]
 for seed,mode,kind in itertools.product([42,43,44],cfg['selection']['conditions'],sorted({r['kind'] for r in out})):
  rr=[r for r in out if r['seed']==seed and r['kind']==kind];cards=[c for r in rr for c in r['conditions'][mode]['cards']];tt=[c['truth'] for c in cards if 'truth'in c];orders=[o['state'] for r in rr for o in r['conditions'][mode]['order']];summary.append(dict(seed=seed,condition=mode,kind=kind,cases=len(rr),cards=len(cards),missing=sum(c['chosen'] is None for c in cards),known_groups=len(tt),all_witness=sum(t['all_witness'] for t in tt),wrong_points=sum(t['wrong_points'] for t in tt),order={k:orders.count(k) for k in set(orders)},hand_qualified_points=sum(len(c['chosen']['hand_qualified']) if c['chosen'] else 0 for c in cards)))
 write(E/'COMPARISON.json',dict(at=stamp(),rows=out,summary=summary,predictions=predictions,timing=timing,upstream_inference_calls=0,web_changed=False))
 for s in summary:
  if s['kind'] in ['full_normal','dynamic_complete_blocks'] and s['condition']=='normal_body':print(s)
if __name__=='__main__':
 try:main()
 except BaseException:E.mkdir(exist_ok=True);(E/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
