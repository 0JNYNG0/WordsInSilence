"""Public FIT-only source extension, then fixed fine-tuning of a shared head."""
from local_shape import *
ADDED={4:[84,90,99,108],5:[41,44,53,65],6:[48,51,57,66],7:[48,51,60,69],8:[51,60,72,84],9:[51,60,72,78]}

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 out=V/'expanded';out.mkdir(exist_ok=True);assert not(out/'TRAIN_FROZEN.json').exists()
 sys.path.insert(0,str(SOURCE));from finger_model import features,windows
 records={r['person']:r for r in read(U/'INPUTS.json')['rows']};d=np.load(SOURCE/'INPUTS.npz');old=read(SOURCE/'INPUTS.json')['rows'];new=[];xx=[];aa=[]
 for person,frames in ADDED.items():
  r=records[person];assert r['role']=='fit';assert sha(r['channels'])==r['channels_sha256'];z=np.load(r['channels']);f,a=features(z['tcn280'],'R')
  for frame in frames:
   ix=int(np.flatnonzero(z['frame_id']==frame)[0]);assert a[ix].all()
   new.append(dict(clip_id=r['clip_id'],word=r['word'],person=person,role='fit',frame=frame,side='R',targets=[-1,1,1,1,1],source_sha256=r['source_sha256'],source_review=str(U/'close_review'/f'{person:02}_0.jpg'),review='Codex RGB enlarged panel: visible four extended long fingers; thumb not labelled. Existing coordinates are inputs, not labels.'))
   xx.append(windows(f,[ix])[0]);aa.append(a[ix])
 write(out/'SOURCE_TRUTH.json',dict(rows=new,reviewer='Codex visual review; not independent sign-language expert',no_dev_to_fit=True))
 xraw=np.concatenate([d['x'],xx]);avail=np.concatenate([d['availability'],aa]);y=np.concatenate([d['y'],np.ones((len(new),4),np.int64)]);rows=old+new
 known=(y>=0).all(1)&avail.all(1);target=np.where((y==1).all(1),0,np.where((y==0).all(1),1,2));fit=np.array([r['role']=='fit'for r in rows])&known;dev=np.array([r['role']=='development'for r in rows])&known
 cfg=dict(seeds=[42,43,44],steps=300,lr=.0002,weight_decay=.001,confidence=.85,kind='temporal',parent={str(V/'temporal'/f'{s}.pt'):sha(V/'temporal'/f'{s}.pt')for s in [42,43,44]},labels=sha(out/'SOURCE_TRUTH.json'),normalization='Parent FIT-only normalization kept',selection='One predeclared continuation, no early stop/threshold search; compare full old DEV and previously-seen diagnostics. Reserve fresh-source check for subsequent admission.',no_private_data=True)
 write(out/'TRAIN_FROZEN.json',cfg)
 counts=collections.Counter((rows[i]['clip_id'],int(target[i]))for i in np.flatnonzero(fit));classes=collections.Counter(t for c,t in counts);weights=torch.tensor([1/(counts[rows[i]['clip_id'],int(target[i])]*classes[int(target[i])])for i in np.flatnonzero(fit)]);weights/=weights.sum();ty=torch.tensor(target[fit]);probs=[];runs=[]
 for seed in cfg['seeds']:
  ck=torch.load(V/'temporal'/f'{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);m=HandHead('temporal');m.load_state_dict(ck['state']);x=torch.tensor((xraw-ck['mean'])/ck['std']);opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.001);t=time.perf_counter()
  for step in range(300):
   loss=(F.cross_entropy(m(x[fit]),ty,reduction='none')*weights).sum();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
  m.eval()
  with torch.inference_mode():probs.append(m(x).softmax(-1).numpy())
  path=out/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=ck['mean'],std=ck['std'],kind='temporal'),path);runs.append(dict(seed=seed,seconds=time.perf_counter()-t,sha256=sha(path)));print(runs[-1],flush=True)
 ps=np.stack(probs);s=ps.argmax(-1);s[ps.max(-1)<.85]=-1;s=np.where((s==s[0]).all(0),s[0],-1);s[~avail.all(1)]=-1
 before=np.load(V/'temporal/predictions.npz')['states'];olddev=dev[:len(old)];ss=s[:len(old)];yy=target[:len(old)]
 result=dict(total=int(dev.sum()),correct=int((dev&(s==target)).sum()),wrong=int((dev&(s>=0)&(s!=target)).sum()),unknown=int((dev&(s<0)).sum()),old_correct_to_wrong=int((olddev&(before==yy)&(ss>=0)&(ss!=yy)).sum()),old_correct_to_unknown=int((olddev&(before==yy)&(ss<0)).sum()))
 np.savez_compressed(out/'predictions.npz',states=s,probabilities=ps,target=target,dev=dev,fit=fit);write(out/'TRAINED.json',dict(runs=runs,added_public_anchors=len(new),fit_anchors=int(fit.sum())));write(out/'DEVELOPMENT.json',result);print(result)
if __name__=='__main__':main()
