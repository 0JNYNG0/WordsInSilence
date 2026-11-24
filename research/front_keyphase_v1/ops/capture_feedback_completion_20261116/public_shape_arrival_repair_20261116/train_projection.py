"""One fixed continuation, RGB-reviewed public FIT only; no private inputs."""
from audit_public import *
from train_shape import HandHead,features,windows,S,F,collections,sha

def inputs():
 d=np.load(S/'INPUTS.npz');records=read(S/'INPUTS.json')['rows'];y=d['y'];target=np.where((y==1).all(1),0,np.where((y==0).all(1),1,2))
 xx=list(d['x']);av=list(d['availability']);yy=list(target);known=list((y>=0).all(1)&d['availability'].all(1))
 prior=read(V/'shape_continuation/SOURCE_TRUTH.json')['rows']+read(B/'local_shape/expanded/SOURCE_TRUTH.json')['rows']
 by={r['person']:r for r in read(B/'INPUTS.json')['rows']};seen=set()
 for r in prior:
  h=r.get('hand',r.get('side'));key=(r['clip_id'],r['frame'],h)
  if key in seen:continue
  seen.add(key);ch=np.load(by[r['person']]['channels']);i=int(np.flatnonzero(ch['frame_id']==r['frame'])[0]);f,a=features(ch['tcn280'],h)
  xx.append(windows(f,[i])[0]);av.append(a[i]);yy.append(0);known.append(a[i].all());records.append(r)
 for r in read(V/'shape_source_repair/SOURCE_TRUTH.json')['fit']:
  ch=np.load(r['channels']);i=int(np.flatnonzero(ch['frame_id']==r['frame'])[0]);f,a=features(ch['full'],r['hand'])
  xx.append(windows(f,[i])[0]);av.append(a[i]);yy.append(r['label']);known.append(a[i].all());records.append(r)
 added=[]
 for r in read(HERE/'public_views/INPUTS.json')['rows']:
  if r['person']not in[3,7]:continue
  raw=dict(np.load(r['raw']));ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  for h in['L','R']:
   f,a=features(ch['tcn280'],h)
   for i in[55,65,75]:
    row=dict(clip_id=r['id'],person=r['person'],role='fit',hand=h,frame=i,label=0,source_sha256=r['source_sha256'],video=r['video'],raw=r['raw'],review='Native public RGB panel: extended long-finger hand posture. Coarse label only; not individually visible 3D joint angles.',individual_finger_labels=None)
    added.append(row);records.append(row);xx.append(windows(f,[i])[0]);av.append(a[i]);yy.append(0);known.append(a[i].all())
 return np.array(xx),np.array(av),np.array(yy),np.array(known),records,added

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);out=HERE/'projection_head';out.mkdir(exist_ok=True);assert not(out/'FROZEN.json').exists()
 xraw,av,y,known,records,added=inputs();fit=np.array([r['role']=='fit'for r in records])&known
 write(out/'SOURCE_TRUTH.json',dict(rows=added,reviewer='Codex RGB review, not independent sign expert',image=str(HERE/'public_views/fit_source_review.jpg'),image_sha256=sha(HERE/'public_views/fit_source_review.jpg')))
 cfg=dict(seeds=[42,43,44],steps=300,lr=.0002,confidence=.85,fit_people=list(range(1,10)),extra_frames=len(added),
  parent={str(V/'shape_source_repair'/f'{s}.pt'):sha(V/'shape_source_repair'/f'{s}.pt')for s in[42,43,44]},
  integration='Only abstain when legacy calls folded AND this three-seed observer calls extended. Never change a folded prediction to normal. No word/side/person rule.',
  gate='Reduce source-confirmed false fold in non-fit public11 view; preserve all previously correct237 DEV anchor outputs and public normal/actual-fold controls.',
  selection='One fixed final checkpoint, no threshold or epoch search. Known development is not untouched test.',private_data=False)
 write(out/'FROZEN.json',cfg);counts=collections.Counter((records[i]['clip_id'],int(y[i]))for i in np.flatnonzero(fit));classes=collections.Counter(c for _,c in counts)
 w=torch.tensor([1/(counts[records[i]['clip_id'],int(y[i])]*classes[int(y[i])])for i in np.flatnonzero(fit)],dtype=torch.float32);w/=w.sum();ty=torch.tensor(y[fit]);runs=[]
 for seed in cfg['seeds']:
  ck=torch.load(V/'shape_source_repair'/f'{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);net=HandHead('temporal');net.load_state_dict(ck['state']);x=torch.tensor((xraw-ck['mean'])/ck['std'],dtype=torch.float32);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.001);t=time.perf_counter()
  for _ in range(cfg['steps']):
   loss=(F.cross_entropy(net(x[fit]),ty,reduction='none')*w).sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  net.eval();p=out/f'{seed}.pt';torch.save(dict(state=net.state_dict(),mean=ck['mean'],std=ck['std'],kind='temporal'),p);runs.append(dict(seed=seed,seconds=time.perf_counter()-t,sha256=sha(p)));print(runs[-1],flush=True)
 write(out/'TRAINED.json',dict(runs=runs,fit_rows=int(fit.sum()),fit_clips=len({records[i]['clip_id']for i in np.flatnonzero(fit)}),private_training=False,web_admitted=False))
if __name__=='__main__':main()
