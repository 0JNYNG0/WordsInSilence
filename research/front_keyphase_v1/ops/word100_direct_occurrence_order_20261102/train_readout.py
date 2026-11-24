import os,time,traceback,json
from readout_model import *

def preflight(rr,stats,flow):
 torch.manual_seed(42);t=ReadoutNet('tied_readout').eval()
 torch.manual_seed(42);d=ReadoutNet('direct_readout').eval()
 for k,v in t.state_dict().items():torch.testing.assert_close(v,d.state_dict()[k],rtol=0,atol=0)
 x,im,l,_,_=batch_j(rr[:2],stats);fl=flow_batch(rr[:2],flow)
 with torch.inference_mode():
  ta,tr,to=t(x,im,l,fl);da,dr,do=d(x,im,l,fl);ba,br,_=t.base(x,im,l,fl)
  torch.testing.assert_close(ta,da,rtol=0,atol=0);torch.testing.assert_close(ta,ba,rtol=0,atol=0)
  torch.testing.assert_close(tr,br,rtol=0,atol=0)
  torch.testing.assert_close(to-do,(tr[:,:,1]-tr[:,:,0])[:,:,None].expand_as(to),rtol=1e-6,atol=1e-6)
  for m in [t,d]:
   a,b,c=m(x[:1,:l[0]],im[:1,:l[0]],l[:1],fl[:1,:l[0]])
   aa,bb,cc=m(x,im,l,fl)
   for v,w in [(aa[0,:l[0]],a[0]),(bb[0,:l[0]],b[0]),(cc[0,:l[0]],c[0])]:torch.testing.assert_close(v,w,rtol=1e-5,atol=1e-5)
  torch.testing.assert_close(do,d(x,im,l,-fl)[2],rtol=0,atol=0)
  for condition in ['tied_readout','direct_readout']:
   av=torch.as_tensor(rr[0]['available'])
   p,_=order_probability(ta[0,:l[0]].softmax(-1),to[0,:l[0]].sigmoid(),av,0,1,condition)
   q,_=order_probability(ta[0,:l[0]].softmax(-1),to[0,:l[0]].sigmoid(),av,1,0,condition)
   torch.testing.assert_close(p+q,p.new_tensor(1.),rtol=1e-6,atol=1e-6)
   empty,mass=order_probability(ta[0].softmax(-1),to[0].sigmoid(),torch.zeros(len(to[0])),0,1,condition)
   assert float(empty)==.5 and float(mass)==0
 for m in [t,d]:
  ac,ro,oc=m(x,im,l,fl);ty=occurrence_targets(rr[:2],x.shape[1])
  loss=occurrence_loss(oc,ty,torch.ones(8));loss.backward()
  assert m.query.weight.grad.abs().sum()>0
  assert all(not p.requires_grad and p.grad is None for p in m.base.parameters())
  unk=[dict(r,order_queries=[dict(q,target=-1) for q in r['order_queries']]) for r in rr[:2]]
  missing,used=pair_loss(ac.softmax(-1),oc.sigmoid(),unk,m.condition)
  assert used==0 and float(missing.detach())==0
 z=torch.randn(1,4,8,requires_grad=True);ty=torch.full_like(z,-1);ty[0,1,0]=1
 occurrence_loss(z,ty,torch.ones(8)).backward();assert int((z.grad!=0).sum())==1
 size=sum(p.numel() for p in t.parameters() if p.requires_grad)
 assert size==sum(p.numel() for p in d.parameters() if p.requires_grad)==13824
 return dict(equal_initial_weights=True,equal_trainable_parameters=size,trainable_shapes={k:list(p.shape) for k,p in t.named_parameters() if p.requires_grad},base_actor_role_frozen=True,
  new_direct_head_independent_of_DTW_flow=True,padding_invariance=True,query_swap_complement=True,
  missing_images_no_pair_mass=True,unknown_frames_and_pairs_zero_loss=True,
  intended_contrast='tied adds fixed role prior and physical posterior product; direct removes both; not separate causal effects')

def main():
 torch.set_num_threads(2)
 assert not (Y/'FROZEN.json').exists() and not (OUT/'runs').exists()
 audit=read(Y/'INPUT_CONTRACT_REVIEW.json')
 assert audit.get('status')=='PASS_FIT_INPUT_REUSE_WITH_EXPLICIT_CONTROL_SCOPE' and audit['known_queries']==576
 review=read(Y/'CODE_REVIEW.json');assert review.get('training_allowed') is True
 protocol=read(Y/'EVALUATION_PROTOCOL.json');assert protocol['status']=='FROZEN_BEFORE_TRAINING_NO_PREDICTIONS'
 rr=attach_queries(get_rows('fit'));assert len(rr)==252
 stats=read(D/'NORMALIZATION.json');flow=read(FLOW_UNIT/'FLOW_INPUTS.json')
 pf=preflight(rr,stats,flow);exclusive(Y/'PREFLIGHT.json',dict(at=clock(),**pf))
 from pair_contract import behavior
 checks=behavior();exclusive(Y/'CONTRACT_PREFLIGHT.json',checks)
 pins=dict(read(OLD/'FROZEN.json')['pins'])
 for p,h in pins.items():assert sha(p)==h,p
 for r in rr:
  pins[r['cache']]=r['cache_sha256'];pins[r['input']]=r['input_sha256']
 for p in [INITIAL,OLD/'ORDER_SUPERVISION.json',D/'NORMALIZATION.json',FLOW_UNIT/'FLOW_INPUTS.json',
  Y/'readout_model.py',Y/'train_readout.py',Y/'pair_contract.py',Y/'evaluate_readout.py',Y/'prepare_comparison.py',Y/'EVALUATION_PROTOCOL.json',Y/'INPUT_CONTRACT_REVIEW.json',Y/'CODE_REVIEW.json',Y/'EVALUATION_DESIGN_REVIEW.json',Y/'PREFLIGHT.json',Y/'CONTRACT_PREFLIGHT.json']:
  pins[str(p)]=sha(p)
 for module in list(sys.modules.values()):
  file=getattr(module,'__file__',None)
  if file:
   p=Path(file).resolve()
   if p.suffix=='.py' and p.is_relative_to(O.resolve()):pins[str(p)]=sha(p)
 for p,h in pins.items():assert sha(p)==h,p
 plan=dict(at=clock(),conditions=['tied_readout','direct_readout'],primary='direct_readout',seed=42,epochs=80,
  batch_size=12,learning_rate=.0005,weight_decay=.01,order_loss_weight=.25,
  initialization=str(INITIAL),frozen_modules='all base encoder, physical actor, role head, learned prototype; existing RGBMotion external unchanged',
  trainable_parameters=pf['equal_trainable_parameters'],contrast=pf['intended_contrast'],checkpoint='final epoch only; no seed/epoch/DEV selection',
  fit_input_variants=252,known_queries=576,held_queries=54,original_order_relations=45,order_source_clips=27,
  fit_recorded_ids=list(range(1,10)),development_recorded_ids=[11,12,13],development_exposed=True,
  thresholds=dict(occurrence=.8,order=.8),
  primary_metric='Known unswapped pair-order classification and supported useful partial-pair count versus tied, source-grouped; not whole-sign correctness',
  gates=protocol['numeric_gates'],
  supervision='same masked sparse occurrence and known order;54held no binary loss; nonmain negatives only for relevance',
  inference_input='No recordID/edit kind/annotation index/ground truth; shared visual features plus fixed source query keys. DTW used only by frozen tied prior, not forced output.',
  not_complete_sign_or_repeat=True,pins=pins,whole_goal_complete=False)
 exclusive(Y/'FROZEN.json',plan)
 pos=np.zeros(8);neg=np.zeros(8)
 for r in rr:
  a=occurrence_targets([r],len(r['x']))[0].numpy();pos+=(a==1).sum(0);neg+=(a==0).sum(0)
 weight=torch.tensor(np.clip(neg/np.maximum(pos,1),.25,4),dtype=torch.float32);reports=[]
 for tag in plan['conditions']:
  torch.manual_seed(42);m=ReadoutNet(tag);rng=np.random.default_rng(42);before={k:v.clone() for k,v in m.state_dict().items()}
  dest=OUT/'runs'/tag;dest.mkdir(parents=True,exist_ok=False)
  opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0005,weight_decay=.01)
  history=[];start=time.perf_counter()
  for epoch in range(80):
   m.train();order=rng.permutation(len(rr));losses=[]
   for i in range(0,len(rr),12):
    br=[rr[k] for k in order[i:i+12]];x,im,l,_,_=batch_j(br,stats);fl=flow_batch(br,flow)
    ac,ro,oc=m(x,im,l,fl);ty=occurrence_targets(br,x.shape[1])
    cl=occurrence_loss(oc,ty,weight);ol,_=pair_loss(ac.softmax(-1),oc.sigmoid(),br,tag);loss=cl+.25*ol
    assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],2);opt.step()
    losses.append([float(loss.detach()),float(cl.detach()),float(ol.detach())])
   if epoch%10==0 or epoch==79:
    h=dict(epoch=epoch+1,losses=np.mean(losses,0).tolist(),seconds=time.perf_counter()-start)
    history.append(h);print(tag,h,flush=True);write(Y/'TRAIN_PROGRESS.json',dict(at=clock(),status='TRAINING',pid=os.getpid(),condition=tag,epoch=epoch+1))
  delta={k:float((v-before[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point}
  assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items() if k.startswith('base.'))
  assert all(d==0 for k,d in delta.items() if k.startswith('base.')) and delta['query.weight']>0
  torch.save(m.state_dict(),dest/'model.pt');r=dict(condition=tag,checkpoint_sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-start,history=history,changes=delta)
  exclusive(dest/'COMPLETE.json',r);reports.append(r)
 for p,h in pins.items():assert sha(p)==h,p
 exclusive(Y/'TRAINING_RESULT.json',dict(at=clock(),runs=reports,positive_counts=pos.tolist(),negative_counts=neg.tolist(),all_base_weights_exact=True,whole_goal_complete=False))
 write(Y/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None))

if __name__=='__main__':
 try:main()
 except BaseException:
  p=Y/'TRAIN_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
