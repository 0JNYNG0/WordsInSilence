from requirement_core import *
from collections import Counter
import os,time,traceback

def main():
 assert not (U/'FROZEN.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 write(U/'AUTHORIZATION.json',dict(user_request='Link important guide shape/direction/order to observed actions, verify labels, train and compare; no blanket short-motion errors.',private_data=False,models_preserved=True,web_change=False))
 write(U/'CURRENT_EXECUTION.json',dict(status='PREPARING_REQUIREMENT_LABELS',pid=os.getpid()))
 protections=read(V/'PROTECTIONS.json')
 assert all(sha(p)==s for p,s in protections.items())
 write(U/'PROTECTIONS.json',protections)
 rr=public_rows(); fit=[r for r in rr if r['role']=='fit'];dev=[r for r in rr if r['role']=='development']
 assert {r['person'] for r in fit}.isdisjoint({r['person'] for r in dev})
 assert {r['source_sha256'] for r in fit}.isdisjoint({r['source_sha256'] for r in dev})
 net=frozen_observer(); samples=[]; arrays=[]; pins={};links=[];unavailable=[]
 for r in fit:
  cid=r['clip_id'];pins[r['path']]=sha(r['path'])
  for view in ['full','upper']:
   try:h,cp=cached_h(r,view)
   except FileNotFoundError:
    unavailable.append(dict(clip_id=cid,view=view,reason='No saved observer features; not silently rerunning old model'));continue
   pins[str(cp)]=sha(cp);raw=prior.physical(r['x_'+view]);av=prior.shared.available(r['x_'+view]);th=torch.tensor(h[None])
   with torch.inference_mode():ap=net.shared.base.attributes(th)[0].sigmoid().numpy()
   def add(task,a,b,req,y,prob,eligible,g=None,**meta):
    s=dict(clip_id=cid,person=r['person'],word_id=r['word_id'],view=view,task=task,a=int(a),b=int(b),frames=[int(r['frame_id'][a]),int(r['frame_id'][b])],target=int(y),eligible=bool(eligible),**meta)
    s['sample']=len(samples);samples.append(s);arrays.append((observation(h,a,b,g),req,logit(prob),int(y),int(eligible)))
   for a,j in zip(*np.where(r['direct_ym']>0)):
    y=int(r['y'][a,j]);gl=guide_links(r,'shape',a,a,feature=int(j))
    if view=='full':links.append(dict(clip_id=cid,person=r['person'],task='shape',frame=int(r['frame_id'][a]),feature=int(j),observed=y,guide_clause_ids=gl,truth='Inherited direct RGB label; no DTW pseudo-label',coverage_known=bool(av[a,j])))
    for expected in [0,1]:add('shape',a,a,requirement('shape',feature=j,expected=expected),y==expected,ap[a,j] if expected else 1-ap[a,j],av[a,j],feature=int(j),expected=expected,guide_clause_ids=gl)
   for pi,p in enumerate(r['pairs']):
    a,b=p['a'],p['b'];gl=guide_links(r,'direction',a,b,axis=p['axis'])
    for rev in [False,True]:
     g,ok,info=prior.geometry(raw,r['timestamp_us'],a,b,p['axis'],p.get('moving','R'),p.get('anchor','L'),rev)
     with torch.inference_mode():pr=net.movement(th,torch.tensor(g[None]),torch.tensor([[0,b if rev else a,a if rev else b]])).softmax(-1)[0].numpy()
     actual=2-p['target'] if rev else p['target']
     for desired in range(3):
      add('direction',a,b,requirement('direction',axis=p['axis'],direction=desired),actual==desired,pr[desired],ok,g,axis=p['axis'],moving=p.get('moving','R'),expected=desired,observed=actual,reverse=rev,pair=pi,guide_clause_ids=gl)
    if view=='full':links.append(dict(clip_id=cid,person=r['person'],task='direction',frames=[p['frame_a'],p['frame_b']],axis=p['axis'],hand=p.get('moving','R'),observed=p['target'],guide_clause_ids=gl,truth='Reviewed body-relative direction; no semantic completion or exact contact label'))
   for pi,p in enumerate(r['orders']):
    a,b=p['a'],p['b'];valid=bool(av[[a,b]].any(1).all())
    for rev in [False,True]:
     ga,gb=(p['guide_b'],p['guide_a']) if rev else (p['guide_a'],p['guide_b'])
     with torch.inference_mode():pr=float(net.ordering(th,torch.tensor([[0,a,b]]),torch.tensor([[ga,gb]])).sigmoid()[0])
     add('order',a,b,requirement('order',guides=[ga,gb]),not rev,pr,valid,pair=pi,guides=[ga,gb],contrast='query_order_swapped' if rev else 'reviewed_order')
    if view=='full':links.append(dict(clip_id=cid,person=r['person'],task='order',frames=[p['frame_a'],p['frame_b']],guide_indices=[p['guide_a'],p['guide_b']],truth='Reviewed ordered events; reversed query is a contrast, not a wrong filmed sign'))
 data={k:np.asarray([a[i] for a in arrays],np.float32) for i,k in enumerate(['ob','req','base','y','mask'])}
 assert len(samples)>0 and all(np.isfinite(v).all() for v in data.values())
 mean=data['ob'][data['mask']>0].mean(0);std=np.maximum(data['ob'][data['mask']>0].std(0),.05)
 np.savez_compressed(U/'train.npz',**data)
 write(U/'TRAIN_MANIFEST.json',samples);write(U/'REQUIREMENT_LABELS.json',dict(rows=links,reviewer='Codex inherited sparse RGB annotations plus this-turn link audit; not independent human truth',unlinked_policy='Common component examples remain useful; no guessed clause assignment',unavailable=unavailable))
 write(U/'VISUAL_AUDIT.json',dict(fit_panels=['WORD0739_REAL01','WORD1133_REAL01','WORD1567_REAL01','WORD2523_REAL01'],observations=['Open face presentation -> closing downward -> raised closed hand is visibly distinct.','Palm sweeps along opposite forearm; later rest cannot prove extra required hold.','Alternation involves both hands; crossing itself is not error.','Egypt edge-on ending does not establish all-four-open; old draft requirement is not direct shape truth.'],policy='No new finger truth assigned from ambiguous images; no exact contact or amplitude tolerance invented'))
 pins.update({str(p):sha(p) for p in [U/'train.npz',U/'TRAIN_MANIFEST.json',U/'REQUIREMENT_LABELS.json',B/'ADDITIONAL_LABELS.json',prior.D/'CARDS.json',Q/'LABELS.json']})
 cfg=dict(status='FROZEN_BEFORE_TRAINING_AND_CANDIDATE_SCORES',seeds=[42,43,44],epochs=40,batch=128,lr=.001,weight_decay=.01,thresholds=[.35,.65],mean=mean.tolist(),std=std.tolist(),pins=pins,fit_ids=sorted({r['person'] for r in fit}),dev_ids=sorted({r['person'] for r in dev}),common100_backbone_frozen=True,baseline='coupled_42 saved predictions; same immutable observer features',not_whole_sign_correctness=True,missing_observation='mask, never negative',short_motion='No duration requirement or cropped-span semantic error label',final_epoch_only=True,adoption='Every seed: no normal or component regression; all unknown controls retained; improvement required on held-person evidence; no production change in this unit',historical_dev=True)
 write(U/'FROZEN.json',cfg)
 write(U/'PRETRAIN_CHECK.json',dict(pass_=True,fit_videos=len({s['clip_id'] for s in samples}),fit_words=len({s['word_id'] for s in samples}),direct_links=len(links),linked_shape_direction=sum(bool(a.get('guide_clause_ids')) for a in links if a['task']!='order'),samples=len(samples),eligible=int(data['mask'].sum()),tasks=dict(Counter(s['task'] for s in samples)),fit_dev_source_hash_disjoint=True,private_used=False,normal100_training_preserved=True,unknown_not_negative=True,review_scope='Schema, source hashes, inherited RGB-label consistency and four FIT panel spot checks; not expert validation'))
 write(U/'CURRENT_EXECUTION.json',dict(status='LABELS_VERIFIED_READY_TO_TRAIN',pid=None))
 print(json.dumps(read(U/'PRETRAIN_CHECK.json'),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
