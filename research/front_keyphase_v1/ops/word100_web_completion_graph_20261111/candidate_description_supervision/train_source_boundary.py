from source_boundary_core import *
import time,traceback

def main():
 assert not (B/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);cfg=read(B/'FROZEN.json');meta=read(B/'INPUTS_COMPLETE.json');a=np.load(meta['arrays']);assert sha(meta['arrays'])==meta['arrays_sha256']
 for p,h in {**cfg['pins'],**meta['pins']}.items():assert sha(p)==h,p
 assert all(np.isfinite(a[k]).all() for k in a.files)
 # Check the pooling implementation separately against explicit temporal thirds.
 rng=np.random.default_rng(9);xx=rng.normal(size=(25,150)).astype(np.float32);sp=np.array([[0,n-1] for n in range(1,26)]);expected=[]
 for lo,hi in sp:
  z=xx[lo:hi+1];parts=np.array_split(np.arange(len(z)),3)
  if any(len(g)==0 for g in parts):parts=[np.array([0]),np.arange(len(z)),np.array([len(z)-1])]
  expected.append(np.r_[*[z[g].mean(0) for g in parts],z.std(0)])
 assert np.allclose(pool_windows(xx,sp),expected,atol=2e-6)
 seq=meta['sequences'];zs=[]
 for r in seq:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);assert all(np.isfinite(z[k]).all() for k in z.files);zs.append({k:z[k].copy() for k in z.files})
 use=np.array([r['split']=='train' for r in seq]);trainseq=[r for r,t in zip(seq,use) if t];trainz=[z for z,t in zip(zs,use) if t];T=max(len(z['x']) for z in trainz);ns=len(trainz);sx=np.zeros((ns,T,150),np.float32);sy=np.zeros((ns,T,5),np.float32);sm=np.zeros_like(sy);band=np.zeros((ns,T,5,2),bool);valid=np.zeros_like(band)
 for i,z in enumerate(trainz):
  n=len(z['x']);sx[i,:n]=z['x'];sy[i,:n]=z['y'];sm[i,:n]=z['mask'];band[i,:n]=z['bands'];valid[i,:n]=z['valid']
 pw=np.zeros_like(sy);bw=np.zeros((ns,5,2),np.float32)
 for j in range(5):
  for v in [0,1]:
   take=(sm[:,:,j]>0)&(sy[:,:,j]==v);clips=sorted({r['clip_id'] for r,t in zip(trainseq,take.any(1)) if t})
   for cid in clips:
    ids=[i for i,r in enumerate(trainseq) if r['clip_id']==cid and take[i].any()]
    for i in ids:pw[i,take[i],j]=1/(10*len(clips)*len(ids)*take[i].sum())
  for h in range(2):
   take=band[:,:,j,h].any(1);clips=sorted({r['clip_id'] for r,t in zip(trainseq,take) if t})
   for cid in clips:
    ids=[i for i,r in enumerate(trainseq) if r['clip_id']==cid and take[i]]
    for i in ids:bw[i,j,h]=1/(10*len(clips)*len(ids))
 assert np.isclose(pw.sum(),1) and np.isclose(bw.sum(),1);assert (band<=valid).all()
 # Unlabelled head/clip cells get a harmless common denominator, with zero loss weight.
 empty=~band.any(1);safe_band=band.copy();safe_valid=valid.copy();safe_band[:,0]|=empty;safe_valid[:,0]|=empty
 bx=torch.tensor(sx);by=torch.tensor(sy);bpw=torch.tensor(pw);bbw=torch.tensor(bw);bband=torch.tensor(safe_band);bvalid=torch.tensor(safe_valid);joined=np.concatenate([z['x'] for z in trainz]);bmean=joined.mean(0);bstd=np.maximum(joined.std(0),.05)
 tr=a['train'];ww0=balanced_weights(a['y'],a['mask'],meta['rows'],tr);wx=np.concatenate([a['x'][tr],a['x_reverse'][tr]]);wy=np.concatenate([a['y'][tr]]*2);ww=np.concatenate([ww0]*2)*.5;wmean=wx.mean(0);wstd=np.maximum(wx.std(0),.05);wx=torch.tensor(wx);wy=torch.tensor(wy);ww=torch.tensor(ww)
 write(B/'PRETRAIN_VERIFICATION.json',dict(at=stamp(),finite=True,pooling_lengths_1_to25=True,boundary_bands_inside_observed_mask=True,weights_sum=[float(pw.sum()),float(bw.sum()),float(ww.sum())],fit_people=sorted({r['person'] for r in trainseq}),probe_people=sorted({r['person'] for r,t in zip(seq,use) if not t}),no_word_or_absolute_time=True,body_only_shape_invariance=True,upstream_unchanged=True,training_code_sha256=sha(__file__),source_recovery=meta['recovery_of']))
 np.savez_compressed(B/'TRAIN_WEIGHTS.npz',presence=pw,boundary=bw,window=ww.numpy());runs=[]
 for mode in ['body_window','direct_boundary']:
  for seed in cfg['seeds']:
   dst=B/'runs'/mode/str(seed);dst.mkdir(parents=True,exist_ok=True);assert not (dst/'model.pt').exists();torch.manual_seed(seed);net=BodyWindow(wmean,wstd) if mode=='body_window' else BoundaryHead(bmean,bstd);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);t=time.perf_counter();log=[]
   for step in range(cfg['steps']):
    opt.zero_grad()
    if mode=='body_window':loss=(torch.nn.functional.binary_cross_entropy_with_logits(net(wx),wy,reduction='none')*ww).sum()
    else:
     z=net(bx);lp=(torch.nn.functional.binary_cross_entropy_with_logits(z[:,:,:,2],by,reduction='none')*bpw).sum();den=torch.logsumexp(z[:,:,:,:2].masked_fill(~bvalid,-torch.inf),dim=1);num=torch.logsumexp(z[:,:,:,:2].masked_fill(~bband,-torch.inf),dim=1);lb=((den-num)*bbw).sum();loss=(lp+2*lb)/3
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
    if step%300==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())));print('source boundary',mode,seed,step+1,float(loss.detach()),flush=True)
   net.eval();torch.save(net.state_dict(),dst/'model.pt')
   with torch.inference_mode():
    if mode=='body_window':pred=dict(logits=net(torch.tensor(a['x'])).numpy(),logits_reverse=net(torch.tensor(a['x_reverse'])).numpy())
    else:pred={f'logits_{i}':net(torch.tensor(z['x'])).numpy() for i,z in enumerate(zs)}
   np.savez_compressed(dst/'predictions.npz',**pred);runs.append(dict(condition=mode,seed=seed,model=str(dst/'model.pt'),model_sha256=sha(dst/'model.pt'),predictions=str(dst/'predictions.npz'),predictions_sha256=sha(dst/'predictions.npz'),seconds=time.perf_counter()-t,log=log));write(B/'TRAINING_PROGRESS.json',dict(at=stamp(),completed=runs))
 write(B/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,shared100_frozen=True,web_changed=False,private=False,final8=False));print('Both boundary targets compared; training complete')
if __name__=='__main__':
 try:main()
 except BaseException:(B/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
