"""G2 cause isolation: frozen semantic heads on changed spans, no runtime adoption."""
from core import *
from collections import Counter,defaultdict
import time,traceback
S=W/'shadow_components'
def main():
 assert not (S/'RESULTS.json').exists();S.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_candidate_shadow_req',O/'word100_guide_requirement_learning_20261111/requirement_core.py');net=rc.frozen_observer();comp=read(W/'COMPARISON.json');predmap={(r['seed'],r['clip_id'],r['kind']):r for r in comp['predictions']};caps={(r['word'],r['card'],r['feature'],r['expected']):r['qualified'] for r in read(O/'word100_guide_requirement_learning_20261111/GUIDE_CAPABILITY.json')['rows']}
 write(S/'FROZEN.json',dict(at=stamp(),scope='Shadow component regression on saved G2 candidate choices; G3 not promoted',conditions=read(W/'FROZEN.json')['selection']['conditions'],observer_checkpoint='word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt',new_training=False,thresholds='existing supported>=.65, contradiction<=.35; direction requires explicit opposite; shape visibility>=.8',scope_warning='Official normal source warning is an audit target, not automatically a verified false correction. Synthetic shape/direction edits are controlled stress, not filmed learner errors.',no_source_edit_metadata_in_selection=True,semantic_completion=None,pins={str(p):sha(p) for p in [W/'COMPARISON.json',Path(__file__),O/'word100_guide_requirement_learning_20261111/GUIDE_CAPABILITY.json']}))
 cache={};out=[];seconds=[]
 for r in comp['rows']:
  key=r['clip_id'],r['kind'];e=predmap[r['seed'],*key];word=r['clip_id'].split('_')[2];start=time.perf_counter()
  if key not in cache:
   assert sha(e['source'])==e['source_sha256'];z=np.load(e['source']);h=z['fh'];x=z['x'];ts=z['ts'];av=rc.prior.shared.available(x)
   with torch.inference_mode():shape=net.shared.base.attributes(torch.tensor(h)).sigmoid().numpy()
   cache[key]=dict(h=h,x=x,ts=ts,av=av,shape=shape,movement={})
  d=cache[key];conditions={}
  for mode,m in r['conditions'].items():
   cards=[]
   for c in m['cards']:
    spec=next(v for v in rc.FAMILIES[word] if v['id']==c['card']);guide=next(v for v in rc.CARDS[word] if v['id']==c['card']);chosen=c['chosen'];items=[]
    if chosen:
     a,b=chosen['span'];aa=a+(b-a)//2 if guide.get('shape_scope')=='late' else a
     for j,want in guide.get('features',{}).items():
      j=int(j);valid=d['av'][aa:b+1,j];ok=bool(valid.mean()>=.8);v=d['shape'][aa:b+1,j][valid];p=float(v.mean()) if len(v) else None;p=p if want or p is None else 1-p;qualified=caps.get((word,c['card'],j,want),False);state=rc.decide(p,ok,qualified) if p is not None else 'observation_unknown';items.append(dict(task='shape',feature=j,expected=want,span=[aa,b],probability=p,state=state,qualified=qualified,observed_count=int(valid.sum())))
     axis=spec['axis'];side=spec['hands'][0]
     if axis is not None:
      if spec['sign'] is None:items.append(dict(task='direction',state='sequence_required',reason='Alternation cannot be judged from first/last displacement alone'))
      else:
       mkey=a,b,axis,side
       if mkey not in d['movement']:
        g,ok,info=rc.prior.geometry(d['x'],d['ts'],a,b,axis,side,'L' if side=='R' else 'R')
        with torch.inference_mode():p=net.movement(torch.tensor(d['h'][None]),torch.tensor(g[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy()
        d['movement'][mkey]=dict(p=p.tolist(),eligible=bool(ok))
       v=d['movement'][mkey];want=int(spec['sign']+1);items.append(dict(task='direction',axis=axis,desired=want,span=[a,b],probabilities=v['p'],state=rc.direction_verdict(v['p'],want,v['eligible'])))
    states=[v['state'] for v in items];status='association_unknown' if chosen is None else 'difference_evidence' if 'difference'in states or 'opposite_direction'in states else 'supported_features_only' if states and all(v=='supported' for v in states) else 'insufficient_evidence'
    cards.append(dict(card=c['card'],span=chosen['span'] if chosen else None,status=status,evidence=items,completion=None))
   conditions[mode]=cards
  out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],conditions=conditions));seconds.append(time.perf_counter()-start)
  if len(out)%60==0:print('shadow',len(out),flush=True)
 summary=[]
 for seed in [42,43,44]:
  for mode in read(W/'FROZEN.json')['selection']['conditions']:
   for kind in sorted({r['kind'] for r in out}):
    cards=[c for r in out if r['seed']==seed and r['kind']==kind for c in r['conditions'][mode]];items=[v for c in cards for v in c['evidence']];summary.append(dict(seed=seed,condition=mode,kind=kind,cards=len(cards),statuses=dict(Counter(c['status'] for c in cards)),shape=dict(Counter(v['state'] for v in items if v['task']=='shape')),direction=dict(Counter(v['state'] for v in items if v['task']=='direction'))))
 assert all(v['state']=='observation_unknown' for r in out if r['kind']=='all_hand_missing' for cs in r['conditions'].values() for c in cs for v in c['evidence'] if v['task']=='shape')
 write(S/'RESULTS.json',dict(at=stamp(),rows=out,summary=summary,unique_sources=len(cache),observations_reused=True,web_changed=False,mean_seconds=float(np.mean(seconds)),timing_scope='cached latent -> semantic head/geometry; not pose or capture'))
 for s in summary:
  if s['seed']==42 and s['kind'] in ['full_normal','main_reversed','right_closed_hand_main','closed_hand_and_reversed']:print(s)
if __name__=='__main__':
 try:main()
 except BaseException:S.mkdir(exist_ok=True);(S/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
