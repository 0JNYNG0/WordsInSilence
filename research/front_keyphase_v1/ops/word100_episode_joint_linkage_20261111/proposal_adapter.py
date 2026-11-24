from joint_core import *
from collections import Counter
import os,traceback
V=U/'proposal_adapter';V.mkdir(exist_ok=True)
def assign(a,b,k,annotations):
 rows=[r for r in annotations if r['axis']*2+(r['moving']=='L')==k];es=[e for r in rows for e in r['events']];nonmotion=[s for r in rows for s in r['holds']+r['background']]
 if any(lo<=a<=b<=hi for lo,hi in nonmotion):return 0,None
 for j,e in enumerate(es):
  if e['onset'][0]<=a<=e['onset'][1] and e['offset'][0]<=b<=e['offset'][1]:return 2,j
  if e['onset'][0]<=a<=b<=e['offset'][1]:return 1,j
 contained=[j for j,e in enumerate(es) if a<=e['onset'][0] and b>=e['offset'][1]]
 if len(contained)>=2:return 3,None
 if contained and any(max(a,lo)<=min(b,hi) for lo,hi in nonmotion):return 3,None
 return -1,None
def fresh_heads(seed):
 torch.manual_seed(seed)
 return nn.Sequential(nn.Linear(304,64),nn.GELU(),nn.Linear(64,4)),nn.Sequential(nn.Linear(609,96),nn.GELU(),nn.Linear(96,3))
def main():
 assert not (V/'COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 # Adaptation is motivated by the FIT proposal audit. The ten earlier evaluation
 # videos are now disclosed development replays; they select no weights/epochs.
 write(V/'FROZEN.json',dict(seeds=[42,43,44],epochs=40,lr=.001,heldout_annotation_ids=[3,7],input='Only FIT public annotations and actual FIT proposals; no evaluation coordinates or outputs in training',frozen='100word encoder, role, boundary, shape, direction, DTW and all operational models',targets='Complete only when both endpoints fall in directly reviewed ranges; unknown proposals excluded. Multiple movements / reviewed nonmotion contamination distinguished.',development_replay='Previously inspected first-pass results are no longer a fresh test. New heldout public ID12 six-expression audit reserved before adapter scores.',adoption='No operational replacement; normal no extra warning and independent annotated expression checks required.'))
 net=JointNet();net.load_state_dict(torch.load(U/'runs/42/model.pt',weights_only=True,map_location='cpu'));net.eval();bn=bc.BoundaryNet();bn.load_state_dict(torch.load(B/'runs/42/model.pt',weights_only=True,map_location='cpu'));bn.eval();anns=read(U/'ALL_FIT_BOUNDARIES.json')['rows'];rr=[r for r in old.rows() if r['role']=='fit' and any(a['clip_id']==r['clip_id'] for a in anns)];records=[];audits=[]
 for r in rr:
  az=[a for a in anns if a['clip_id']==r['clip_id']];z=np.load(B/'cache'/f"{r['clip_id']}.npz");q=z['q_full'];g=z['g_full'];fh=z['frozen_h_full'];tq=torch.tensor(q[None]);tf=torch.tensor(fh[None]);tg=torch.tensor(g[None]);ll=torch.tensor([len(q)])
  with torch.inference_mode():
   rp,bp,_,h=net(tg,tq,tf,ll);bp=bp.sigmoid()[0].numpy();op,_=lc.be.infer(bn,prior.physical(r['x_full']),r['timestamp_us'],fh)
   legacy=[]
   for k in range(10):
    es,ps=bc.decode(op[:,k],q[:,k,-2:].max(-1)>.5);legacy.extend(dict(e,axis=k//2,moving='L' if k%2 else 'R') for e in es+ps)
   cand=candidate_spans(bp,q,legacy);spec=[];yy=[];ids=[];source=[]
   for e in cand:
    a,b,k=e['a'],e['b'],e['axis']*2+(e['moving']=='L');target,eid=assign(a,b,k,az)
    if target<0:continue
    spec.append((0,a,b,k));yy.append(target);ids.append((k,eid));source.append('actual_proposal')
   actual=len(spec)
   for a in az:
    k=a['axis']*2+(a['moving']=='L')
    for eid,e in enumerate(a['events']):
     for lo in sorted(set(e['onset'])):
      for hi in sorted(set(e['offset'])):
       spec.append((0,lo,hi,k));yy.append(2);ids.append((k,eid));source.append('reviewed_boundary_range')
       mid=(lo+hi)//2
       for aa,bb in [(lo,mid),(mid,hi)]:
        spec.append((0,aa,bb,k));yy.append(1);ids.append((k,eid));source.append('within_observed_event_fragment')
    for lo,hi in a['holds']+a['background']:spec.append((0,lo,hi,k));yy.append(0);ids.append((k,None));source.append('reviewed_nonmotion')
   valid=[bool((q[a:b+1,k,-2:].max(-1)>.5).mean()>=.8) for i,a,b,k in spec];keep=[i for i,v in enumerate(valid) if v];features=net.segment(h,tq,tf,spec).numpy();original=net.quality(torch.tensor(features)).argmax(-1).numpy();pf=[];py=[]
   for ii in keep:
    for jj in keep:
     i,a,b,k=spec[ii];j,c,d,l=spec[jj]
     if ii==jj or b>c or c-b>max(b-a+1,d-c+1) or ids[ii][1] is None or ids[jj][1] is None:continue
     label=0 if ids[ii]==ids[jj] else 1 if k==l else 2
     pf.append(np.r_[features[ii],features[jj],(c-b)/100.]);py.append(label)
   records.append(dict(clip=r['clip_id'],person=r['person'],features=features[keep],targets=np.asarray(yy)[keep],pair=np.asarray(pf,dtype=np.float32).reshape(-1,609),pair_targets=np.asarray(py),source=[source[i] for i in keep]));audits.append(dict(clip_id=r['clip_id'],person=r['person'],actual_proposals=actual,actual_correct=int(sum(original[i]==yy[i] for i in range(actual) if valid[i])),actual_observed=int(sum(valid[:actual])),known_quality=len(keep),pairs=len(py)))
 write(V/'FIT_PROPOSAL_AUDIT.json',dict(rows=audits,quality_classes=dict(Counter(int(y) for r in records for y in r['targets'])),pair_classes=dict(Counter(int(y) for r in records for y in r['pair_targets']))))
 for name,seed,holdout in [('annotation_probe',42,True)]+[(str(s),s,False) for s in [42,43,44]]:
  chosen=[r for r in records if not holdout or r['person'] not in [3,7]];X=torch.tensor(np.concatenate([r['features'] for r in chosen]));Y=torch.tensor(np.concatenate([r['targets'] for r in chosen]),dtype=torch.long);PX=torch.tensor(np.concatenate([r['pair'] for r in chosen]));PY=torch.tensor(np.concatenate([r['pair_targets'] for r in chosen]),dtype=torch.long);qh,ph=fresh_heads(seed);opt=torch.optim.AdamW(list(qh.parameters())+list(ph.parameters()),lr=.001,weight_decay=.01);qw=(1/torch.bincount(Y,minlength=4).float().clamp_min(1)).sqrt();pw=(1/torch.bincount(PY,minlength=3).float().clamp_min(1)).sqrt();history=[];start=time.perf_counter()
  for epoch in range(40):
   rng=np.random.default_rng(seed*1000+epoch);losses=[]
   for step in range(max(1,(len(X)+63)//64)):
    ix=rng.choice(len(X),min(64,len(X)),False);ip=rng.choice(len(PX),min(64,len(PX)),False);loss=F.cross_entropy(qh(X[ix]),Y[ix],weight=qw)+F.cross_entropy(ph(PX[ip]),PY[ip],weight=pw);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(qh.parameters())+list(ph.parameters()),2);opt.step();losses.append(float(loss.detach()))
   history.append(float(np.mean(losses)))
  dest=V/'runs'/name;dest.mkdir(parents=True,exist_ok=True);torch.save(dict(quality=qh.state_dict(),pair=ph.state_dict()),dest/'model.pt');summary=dict(seed=seed,history=history,quality_samples=len(X),pair_samples=len(PX),seconds=time.perf_counter()-start,sha256=sha(dest/'model.pt'))
  if holdout:
   vals=[]
   with torch.inference_mode():
    for r in records:
     if r['person'] not in [3,7]:continue
     vals.append(dict(clip_id=r['clip'],targets=r['targets'].tolist(),pred=qh(torch.tensor(r['features'])).argmax(-1).tolist(),pair_targets=r['pair_targets'].tolist(),pair_pred=ph(torch.tensor(r['pair'])).argmax(-1).tolist(),source=r['source']))
   write(V/'ANNOTATION_HOLDOUT.json',dict(rows=vals,encoder_previously_saw_fit_people=True))
  write(dest/'COMPLETE.json',summary);event('proposal_adapter_trained',run=name,quality=len(X),pairs=len(PX))
 write(V/'COMPLETE.json',dict(passed=True,seeds=[42,43,44],preserved_100word_encoder=True,encoder_sha256=sha(U/'runs/42/model.pt')))
if __name__=='__main__':
 try:main()
 except BaseException:(V/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
