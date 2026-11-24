from requirement_core import *
from order_binding import OrderedBinding
import time,traceback
from collections import defaultdict
def main():
 assert (U/'ORDER_BINDING_COMPLETE.json').exists() and not (U/'ORDER_REPLAY_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cfg=read(U/'FROZEN.json'); rows={r['clip_id']:r for r in public_rows() if r['role']=='development'}
 source=read(R/'EVALUATION_INPUTS.json')['rows']; replay={(r['clip_id'],r['kind']):r for r in read(O/'word100_completion_fullpath_20261110/REPLAY_INPUTS.json')['rows']}
 rev={r['clip_id']:r for r in read(O/'word100_automatic_window_learning_20261110/FULLPATH_COMPLETE.json')['rows'] if r['model']=='learned_core_42' and r['kind']=='reversed_core_auto_order'}
 gold=[r for r in read(U/'SCORES.json') if r['seed']==42 and r['task']=='order'];before=read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows']
 write(U/'ORDER_REPLAY_FROZEN.json',dict(conditions=['speed075','speed125','reversed_core_auto_order','all_hand_missing','mask_final_event','omit_final_event'],guidance='Source-index correspondence used only to score supplied gold windows, never an inference input.',auto='Prior card spans; same learned binder, no expected answer used to choose intervals.',no_new_training=True,score_first_once=True))
 base=frozen_observer();out=[];automatic=[];cache={}
 def get(rec):
  if rec['key'] not in cache:cache[rec['key']]={k:v for k,v in np.load(rec['path']).items()}
  return cache[rec['key']]
 for seed in cfg['seeds']:
  n=OrderedBinding();n.load_state_dict(torch.load(U/'order_runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));n.eval()
  for rec in source:
   kind=rec['kind'];cid=rec['clip_id'];r=rows[cid];c=get(rec);h=c['fh'];valid=prior.shared.available(c['x']);ii=np.arange(len(r['frame_id']))
   if kind=='reversed_core_auto_order':
    lo,hi=rev[cid]['reversed_source_frame_span'];k=np.flatnonzero((r['frame_id']>=lo)&(r['frame_id']<=hi));ii[k]=ii[k[::-1]]
    normal=get(next(s for s in source if s['clip_id']==cid and s['kind']=='full_normal'))
    assert np.allclose(c['x'],normal['x'][ii],atol=1e-5),('Reversal provenance mismatch',cid)
   elif (cid,kind) in replay:ii=np.asarray(replay[cid,kind]['source_indices'])
   else:continue
   if kind not in ['speed075','speed125','reversed_core_auto_order','all_hand_missing','mask_final_event','omit_final_event']:continue
   assert len(ii)==len(h)
   for q in [q for q in gold if q['clip_id']==cid]:
    aa=int(np.argmin(abs(ii-q['a'])));bb=int(np.argmin(abs(ii-q['b']))); exact=abs(int(ii[aa])-q['a'])<=int(kind.startswith('speed')) and abs(int(ii[bb])-q['b'])<=int(kind.startswith('speed'))
    a,b=sorted([aa,bb]);eligible=bool(exact and aa!=bb and valid[[a,b]].any(1).all());target=int(aa<bb) if exact else None
    with torch.inference_mode():
     ob=torch.tensor(observation(h,a,b)[None]);req=torch.tensor(requirement('order',guides=q['guides'])[None]);p=float(n(ob,req).sigmoid()[0]);old=float(base.ordering(torch.tensor(h[None]),torch.tensor([[0,a,b]]),torch.tensor([q['guides']])).sigmoid()[0])
    out.append(dict(seed=seed,clip_id=cid,kind=kind,pair=q['pair'],target=target,eligible=eligible,source_indices=[int(ii[aa]),int(ii[bb])],query_indices=[a,b],baseline_probability=old,probability=p,before=decide(old,eligible),after=decide(p,eligible),known_absence=not exact))
   old=read(next(q['result'] for q in before if q['seed']==seed and q['clip_id']==cid and q['kind']==kind))
   cards=old['cards'];word=cid.split('_')[2]
   # Adjacent distinct guide clauses. Repetitions/parent-arrival are not invented separate requirements.
   pairs=[]
   for ca,cb in zip(CARDS[word],CARDS[word][1:]):
    sa=next(s for s in FAMILIES[word] if s['id']==ca['id']);sb=next(s for s in FAMILIES[word] if s['id']==cb['id'])
    if sa['families']==sb['families'] or not sa['required'] or not sb['required']:continue
    a=next(c for c in cards if c['id']==ca['id']);b=next(c for c in cards if c['id']==cb['id']);ga=ca['groups'][0][0];gb=cb['groups'][0][0]
    if not a.get('query_indices') or not b.get('query_indices'):
     pairs.append(dict(guides=[ga,gb],state='association_unknown'));continue
    aa=int(round(sum(a['query_indices'])/2));bb=int(round(sum(b['query_indices'])/2));lo,hi=sorted([aa,bb]);eligible=bool(lo!=hi and valid[[lo,hi]].any(1).all())
    with torch.inference_mode():p=float(n(torch.tensor(observation(h,lo,hi)[None]),torch.tensor(requirement('order',guides=[ga,gb])[None])).sigmoid()[0])
    pairs.append(dict(guides=[ga,gb],state=decide(p,eligible),probability=p,query_indices=[lo,hi],not_educational_error=True))
   automatic.append(dict(seed=seed,clip_id=cid,kind=kind,pairs=pairs))
 write(U/'ORDER_REPLAY_SCORES.json',out);write(U/'ORDER_AUTOMATIC_DIAGNOSTICS.json',automatic)
 sums={}
 for seed in cfg['seeds']:
  sums[str(seed)]={}
  for kind in sorted({r['kind'] for r in out}):
   a=[r for r in out if r['seed']==seed and r['kind']==kind];known=[r for r in a if r['eligible'] and r['target'] is not None]
   sums[str(seed)][kind]=dict(n=len(a),eligible=len(known),before_correct=sum((r['baseline_probability']>=.5)==bool(r['target']) for r in known),after_correct=sum((r['probability']>=.5)==bool(r['target']) for r in known),withheld=sum(r['after'] in ['uncertain','observation_unknown'] for r in a))
 write(U/'ORDER_REPLAY_COMPLETE.json',dict(summary=sums,scope='Supplied reviewed endpoints transformed by saved source mapping; separate automatic-window diagnostic. Synthetic reversal/deletion; no natural learner-error claim.'))
 print(sums)
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'ORDER_REPLAY_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
