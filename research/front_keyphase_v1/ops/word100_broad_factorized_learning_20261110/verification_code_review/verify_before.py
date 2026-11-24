"""Separate saved-output arithmetic and invariance checks. No retraining/rescoring."""
from core import *
from collections import Counter,defaultdict
def main():
 cfg=read(U/'FROZEN.json');ev=read(U/'EVALUATION_COMPLETE.json');full=read(U/'FULLPATH_COMPLETE.json')['rows'];checks=0
 for p,h in {**cfg['protected'],**cfg['source_pins'],**cfg['code_pins']}.items():assert sha(p)==h,p;checks+=1
 for p,h in read(U/'EVAL_CODE_PINS.json').items():assert sha(p)==h,p;checks+=1
 manifest=read(U/'READY.json')['rows'];fit={r['clip_id'] for r in manifest if r['role']=='fit'};dev={r['clip_id'] for r in manifest if r['role']=='development'}
 assert len(fit)==899 and len(dev)==300 and not fit&dev;checks+=1
 report={}
 for name,summ in ev['models'].items():
  ck=read(U/'runs'/name/'COMPLETE.json');assert set(ck['fit_seen'])==fit and ck['fit_words']==100 and not ck['development_training'];checks+=1
  assert all(h['all100_clips_seen']==899 for h in ck['history']);assert len(ck['history'])==12;checks+=2
  assert sha(U/'runs'/name/'model.pt')==ck['sha256'];checks+=1
  rr=read(U/'scores'/f'{name}.json');ww=read(U/'scores'/f'{name}_words.json');assert len(ww)==300 and {r['clip_id'] for r in ww}==dev;checks+=1
  for v,tasks in summ['by_variant'].items():
   for task,s in tasks.items():
    q=[r for r in rr if r['variant']==v and r['task']==task]
    recomputed={'n':len(q),'correct':sum(r['pred']==r['target'] for r in q),'eligible':sum(bool(r['eligible']) for r in q),'accepted':sum(bool(r['eligible'] and r['confident']) for r in q),'accepted_correct':sum(bool(r['eligible'] and r['confident'] and r['pred']==r['target']) for r in q)}
    assert s==recomputed,(name,v,task);checks+=6
  assert summ['by_variant']['normal']['original_shape']['n']==228;assert summ['by_variant']['normal']['movement']['n']==32;assert summ['by_variant']['normal']['order']['n']==12;assert summ['by_variant']['normal']['expanded_shape']['n']==133;checks+=4
  assert summ['by_variant']['missing_long']['original_shape']['accepted']==0;checks+=1
  fr=[r for r in full if r['model']==name];assert len(fr)==69;checks+=1
  group={}
  for r in fr:
   assert r['status']=='OK',r
   assert sha(r['result'])==r['sha256'];d=read(r['result']);c=r['candidate'];assert c['display']==[q['display_status'] for q in d['cards']];assert c['all_required_similar']==all(q['display_status']=='similar' for q in d['cards'] if q.get('required',True));checks+=4
   if 'tcn_identical' in r:assert r['tcn_identical'] and r['dtw_identical'] and r['frames_preserved'];checks+=3
   for card in d['cards']:
    if card.get('motion_evidence',{}):
     m=card['motion_evidence']
     if m.get('learned_eligible') is False:assert m['state']=='unknown';checks+=1
    if card.get('observation_quarantine_overlap'):assert card['display_status']=='uncertain';checks+=1
  for kind in sorted(set(r['kind'] for r in fr)):
   selected=[r for r in fr if r['kind']==kind];group[kind]={}
   for side in ['baseline','candidate']:
    group[kind][side]=dict(n=len(selected),all_similar=sum(r[side]['all_required_similar'] for r in selected),any_difference=sum('difference' in r[side]['display'] for r in selected),any_uncertain=sum('uncertain' in r[side]['display'] for r in selected),motion_supported=sum(bool(r[side]['motion_states']) and all(s=='supported' for s in r[side]['motion_states']) for r in selected),motion_contradicted=sum('contradicted' in r[side]['motion_states'] for r in selected))
  times=[r['seconds'] for r in fr if 'tcn_identical' in r];report[name]=dict(component_normal=summ['by_variant']['normal'],automatic=group,cached_fullpath_ms_median=float(np.median(times)*1000),cached_fullpath_ms_p95=float(np.percentile(times,95)*1000),aux_word100=summ['aux_word100'])
 # Independent equation-level contracts: changing fingers cannot affect wrist motion.
 torch.set_num_threads(4);norm=read(U/'NORMALIZATION.json');r=next(r for r in manifest if r['clip_id']=='NIA_SL_WORD1133_REAL01_F');z=np.load(r['path']);x=physical(z['x_full']);t=z['timestamp_us'];a,b=40,80;observations=[]
 for seed in cfg['seeds']:
  net=Network('factorized',norm);net.load_state_dict(torch.load(U/'runs'/f'factorized_{seed}'/'model.pt',weights_only=True,map_location='cpu'));net.eval()
  for axis in range(5):
   g,ok,_=geometry(x,t,a,b,axis);q=x.copy();q[:,18:186]=np.random.default_rng(seed).normal(size=q[:,18:186].shape);g2,ok2,_=geometry(q,t,a,b,axis);assert np.array_equal(g,g2) and ok==ok2;checks+=1
   q=x.copy();pts=q[:,:102].reshape(-1,51,2);pts[:]=pts*1.4+np.array([.31,-.22]);g3,ok3,_=geometry(q,t,a,b,axis);assert ok==ok3 and np.allclose(g,g3,atol=2e-5);checks+=1
   h=torch.randn(1,len(x),64);h2=torch.randn_like(h)
   with torch.inference_mode():p=net.movement(h,torch.tensor(g[None]),torch.tensor([[0,a,b]]));p2=net.movement(h2,torch.tensor(g2[None]),torch.tensor([[0,a,b]]))
   assert torch.equal(p,p2);checks+=1;observations.append(dict(seed=seed,axis=axis,finger_and_shared_latent_invariant=True,translation_scale_invariant=True))
 write(U/'SUMMARY.json',dict(models=report,invariance=observations,scope='Historical public DEV, not new independent or expert-validated correctness'))
 write(U/'VERIFICATION.json',dict(status='PASS',checks=checks,scope='Separate saved arithmetic plus contracts by same Codex; no independent human review',private_used=False,production_changed=False,all_fit_seen_each_epoch=True,errors_in_denominator=True));event('verified',checks=checks)
if __name__=='__main__':main()
