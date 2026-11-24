"""Second arithmetic path: saved outputs only, no model inference/training."""
from core import *
from collections import Counter,defaultdict
import itertools,time
V=W/'parent_consistency_repair'

def verify_run(unit):
 t=read(unit/'TRAINING_COMPLETE.json');meta=read(unit/'INPUTS_COMPLETE.json');d=np.load(unit/'pairs.npz');probe=~d['train'];rows=[r for r in meta['rows'] if r['split']=='probe'];checks=[]
 for r in t['runs']:
  assert sha(r['model'])==r['sha256'];pd=np.load(unit/'runs'/str(r['seed'])/'PROBE.npz');assert np.array_equal(pd['y'],d['y'][probe]);assert np.array_equal(pd['mask'],d['mask'][probe]);p=pd['p'];assert np.isfinite(p).all() and np.all((p>=0)&(p<=1));assert np.allclose(p[:,1],p[:,0]*p[:,2])
  for view in ['full','upper']:
   for j,name in enumerate(['parent','atomic_given_parent']):
    cm=[[0,0,0],[0,0,0]]
    for i,row in enumerate(rows):
     if row['view']!=view or not pd['mask'][i,j]:continue
     truth=int(pd['y'][i,j]);pred=2 if not pd['eligible'][i] else int(p[i,0 if j==0 else 2]>=.5);cm[truth][pred]+=1
    assert cm==r['probe'][view][name]['confusion'];checks.append(dict(seed=r['seed'],view=view,head=name,correct=cm[0][0]+cm[1][1],total=sum(map(sum,cm))))
 return checks

def verify_comparison(unit):
 d=read(unit/'COMPARISON.json');cfg=read(unit/'C_FROZEN.json');pred={(r['seed'],r['clip_id'],r['kind']):r for r in d['predictions']};base=read(U/'episode_stage_contrast/BINDING_V2_SCORES.json')['rows'];base={(r['seed'],r['clip_id'],r['kind'],r['card']):r for r in base};count=0;raw=[];diagnostics=[]
 for r in d['rows']:
  pr=pred[r['seed'],r['clip_id'],r['kind']];assert sha(pr['path'])==pr['sha256'];pd=np.load(pr['path']);pairs={tuple(k.tolist()):float(v[0]) for k,v in zip(pd['pairs'],pd['p'])};assert all(a<b for a,b in pairs)
  for cond,result in r['conditions'].items():
   for c in result['cards']:
    orig=base[r['seed'],r['clip_id'],r['kind'],c['card']];selected=c['chosen'];assert c['semantic_completion'] is None
    if selected is None:assert not any(a['has_motion'] for a in orig['candidates']);continue
    assert any(a['span']==selected['span'] and a['source']==selected['source'] and a['score']==selected['score'] for a in orig['candidates']);assert selected['has_motion'];assert set(selected['hand_qualified'])<=set(selected['body_observed']);assert selected['semantic_completion'] is None
    ps=[np.log(np.clip(pairs[a,b],1e-6,1-1e-6)) for a,b in itertools.combinations(selected['representatives'],2)];assert np.isclose(selected['internal_log'],np.mean(ps) if ps else 0.)
    if cond=='baseline':assert selected['span']==orig['chosen']['span']
    if 'truth'in c:
     tt=c['truth'];covered=set(selected['body_observed']);assert tt['all_witness']==(set(tt['witness'])<=covered) and tt['wrong_points']==len(set(tt['wrong_witness'])&covered)
    if r['kind']=='all_hand_missing':assert selected['hand_qualified']==[]
    count+=1
   # Reconstruct objective for selected configuration, not trusting stored sum.
   total=sum(c['chosen']['score']+(c['chosen']['internal_log'] if cond in ['internal','combined'] else 0.) for c in result['cards'] if c['chosen'])
   relations={tuple(x['families']) for x in cfg['linkage']['relation_contract']}
   if cond in ['joint','combined']:
    for a,b in itertools.combinations(result['cards'],2):
     if tuple(sorted([a['family'],b['family']])) not in relations or not a['chosen'] or not b['chosen']:continue
     terms=[]
     for x,y in itertools.product(a['chosen']['representatives'],b['chosen']['representatives']):
      p=1. if x==y else pairs[tuple(sorted([x,y]))];terms.append(np.log(np.clip(1-p,1e-6,1-1e-6)))
     total+=np.mean(terms) if terms else 0.
   assert np.isclose(result['score'],total)
  b=r['conditions']['baseline'];c=r['conditions']['combined'];changed=[dict(card=a['card'],family=a['family'],before=a['chosen']['span'] if a['chosen'] else None,after=z['chosen']['span'] if z['chosen'] else None) for a,z in zip(b['cards'],c['cards']) if (a['chosen']['span'] if a['chosen'] else None)!=(z['chosen']['span'] if z['chosen'] else None)]
  if changed:raw.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],changes=changed))
  if r['kind']=='dynamic_complete_blocks' and any(x.get('truth',{}).get('wrong_points',0) or not x.get('truth',{'all_witness':True})['all_witness'] for x in c['cards']):diagnostics.append(dict(seed=r['seed'],clip_id=r['clip_id'],cards=c['cards'],order=c['order']))
 for s in d['summary']:
  rr=[r for r in d['rows'] if r['seed']==s['seed'] and r['kind']==s['kind']];cc=[c for r in rr for c in r['conditions'][s['condition']]['cards']];tt=[c['truth'] for c in cc if 'truth'in c];order=Counter(p['state'] for r in rr for p in r['conditions'][s['condition']]['order']);assert s['cases']==len(rr) and s['cards']==len(cc) and s['missing']==sum(c['chosen'] is None for c in cc) and s['all_witness']==sum(t['all_witness'] for t in tt) and s['wrong_points']==sum(t['wrong_points'] for t in tt) and s['order']==dict(order)
 return dict(checked_selected_candidates=count,changed_choices=raw,unresolved_reordered=diagnostics,summary=d['summary'])

def main():
 assert not (W/'VERIFICATION.json').exists();runs={str(u):verify_run(u) for u in [W,V]};comparisons={str(u):verify_comparison(u) for u in [W,V]};protected=read(U/'PROTECTIONS.json');bad=[]
 for p,h in protected.items():
  if sha(p)!=h:bad.append(p)
 assert not bad
 # Validate restored parent targets and weight mass independently.
 sd=np.load(V/'SUPERVISION.npz');d=np.load(V/'pairs.npz');train=d['train']&d['eligible'];assert np.all(sd['weights'][~train]==0);assert np.all(sd['weights'][sd['mask']==0]==0);assert np.allclose(sd['weights'].sum(0),1);assert np.array_equal(sd['mask'][:,0],d['mask'][:,0])
 versions=[]
 for unit in [W,V]:
  co=read(unit/'COMPARISON.json');rr=[]
  for seed in [42,43,44]:
   index={(s['condition'],s['kind']):s for s in co['summary'] if s['seed']==seed};normal=[index['combined',k] for k in ['full_normal','speed075','speed125']];b=index['baseline','dynamic_complete_blocks'];c=index['combined','dynamic_complete_blocks'];n0=index['baseline','full_normal'];n1=index['combined','full_normal'];rr.append(dict(seed=seed,normal_speed_candidates=sum(s['cards']-s['missing'] for s in normal),normal_speed_total=sum(s['cards'] for s in normal),normal_known_before=n0['all_witness'],normal_known_after=n1['all_witness'],normal_wrong_before=n0['wrong_points'],normal_wrong_after=n1['wrong_points'],reorder_known_before=b['all_witness'],reorder_known_after=c['all_witness'],reorder_wrong_before=b['wrong_points'],reorder_wrong_after=c['wrong_points'],reorder_order_before=b['order'],reorder_order_after=c['order']))
  times=[t['seconds'] for t in co['timing']];versions.append(dict(unit=str(unit),summary=rr,latency_cached_mean_seconds=float(np.mean(times)),latency_cached_p95_seconds=float(np.quantile(times,.95))))
 decision=dict(at=stamp(),status='PARTIAL_COMPONENT_GAIN_G2_REWORK',A='PASS exact source-feature/mask binding; shortcut audit caused balanced supervision/metrics repair.',B='Parent relation improved on exposed probe; atomic other-word transfer still inadequate. Upstream100 frozen.',C='Combined relation selection reduces wrong-action mixing in known reordered samples, but a repaired-model seed chooses preparation instead of actual movement. No adoption.',D='All82 saved exposed scenarios x3seeds x4selection modes compared for each of2 learned variants. Saved arithmetic verified; final integration gates NOT met.',versions=versions,protected_files=len(protected),web_changed=False,private_used=False,final8_read=False,semantic_completion_claimed=False,normal_cross_shape_direction='Old models preserved and missing-hand qualification checked. New chosen intervals change some normal/speed evidence; end-to-end shape/direction correctness under those intervals is NOT established.',causal_findings=['Dropping valid same-parent positives for repeat words caused parent split errors; restoring balanced supervision repaired this sampled relation error.','High pair accuracy does not guarantee correct candidate/description identity. Combined selector sometimes trades real-action candidate for preparation when a broad other-family candidate gets a high accumulated stage score.','Three representative points and mixed score units do not supply boundary correctness/completion labels; no claim that a weight change solves it.','Conditional repeat-cycle head has one mixed-outcome TRAIN word; other-word failure is a data coverage limitation, not solved by parent accuracy.'],next='Train candidate-level description compatibility with FIT source episode extents: valid interior, uncertain boundary, wrong-action mixed candidate, preparation/return, and partial vs whole observed occurrence. Preserve parent relation model; do not sweep DEV score coefficients. Joint selection must retain actual movement evidence before judging order. Exact linguistic completion remains separate.',reviewer='Root Codex implementation plus separately coded saved-array arithmetic; not independent human/KSL expert.')
 write(W/'D_REGRESSION_DETAIL.json',comparisons);write(W/'DECISION.json',decision);write(W/'VERIFICATION.json',dict(at=stamp(),status='PASS_ARITHMETIC_RESEARCH_PARTIAL_NOT_ADOPTED',run_checks=runs,comparisons_checked={k:v['checked_selected_candidates'] for k,v in comparisons.items()},protected_files=len(protected),bad_protected=bad,independent_human=False,pins={str(p):sha(p) for p in [Path(__file__),W/'DECISION.json',W/'D_REGRESSION_DETAIL.json',W/'COMPARISON.json',V/'COMPARISON.json',W/'TRAINING_COMPLETE.json',V/'TRAINING_COMPLETE.json']}))
 print('D verified: protected',len(protected),'versions',versions,flush=True)

if __name__=='__main__':main()
