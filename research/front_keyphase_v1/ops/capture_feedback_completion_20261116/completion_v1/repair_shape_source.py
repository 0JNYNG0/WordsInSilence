"""Append RGB-reviewed FIT curls and audit one incorrect legacy DEV label.

Neither DEV pixels/labels nor candidate predictions enter optimization.
"""
from train_shape import *
import copy
from shape_candidate import ContinuedShape

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 out=V/'shape_source_repair';out.mkdir(exist_ok=True);assert not(out/'FROZEN.json').exists()
 # Fixed after direct source panel inspection; omitted frames remain ambiguous.
 selected={
  ('WORD0678',1):{45:0,55:0,65:2,75:2},('WORD0678',4):{55:1,65:0,75:0},('WORD0678',7):{55:0,65:0,75:2},
  ('WORD0071',1):{55:0,65:0,75:2,85:2,95:2},('WORD0071',4):{55:1,65:0,75:0,85:0,95:2},('WORD0071',7):{75:0,85:0,95:0}}
 rows=read(V/'curl_review/INPUTS.json')['rows'];extra=[];xx=[];aa=[];yy=[]
 for r in rows:
  labels=selected[r['word'],r['person']]
  if r['frame']not in labels:continue
  z=np.load(r['channels']);x=z['full'];f,a=features(x,r['hand']);ix=int(np.flatnonzero(z['frame_id']==r['frame'])[0])
  extra.append(dict(**r,label=labels[r['frame']],review='RGB hand panel inspected before optimization; coarse visible long-finger posture, not per-joint label'))
  xx.append(windows(f,[ix])[0]);aa.append(a[ix]);yy.append(labels[r['frame']])
 write(out/'SOURCE_TRUTH.json',dict(fit=extra,excluded_ambiguous=len(rows)-len(extra),
  dev_correction=dict(clip_id='NIA_SL_WORD0071_REAL13_F',hand='L',frame=82,old_coarse=0,new_coarse=2,
      reason='Direct sequential RGB79/82/85: frame82 index/ thumb extended while other fingers flexed. Frame85 opens; old four-extended label is incorrect at82.',
      image=str(V/'regression_context.jpg'),image_sha256=sha(V/'regression_context.jpg'),training_use=False),
  unchanged_error=dict(clip_id='NIA_SL_WORD0678_REAL11_F',hand='R',frame=45,coarse=1,reason='Visible flexion confirmed; not a fist. 2D detected chains look straighter; keep this error case.',training_use=False)))
 # Rebuild first continuation's FIT inputs, preserving its original source pins.
 d=np.load(S/'INPUTS.npz');oldrows=read(S/'INPUTS.json')['rows'];target=np.where((d['y']==1).all(1),0,np.where((d['y']==0).all(1),1,2));known=(d['y']>=0).all(1)&d['availability'].all(1)
 allx=[d['x']];allav=[d['availability']];ally=[target];allknown=[known];records=list(oldrows)
 prior=read(V/'shape_continuation/SOURCE_TRUTH.json')['rows']
 oldextra=read(B/'local_shape/expanded/SOURCE_TRUTH.json')['rows'];by={r['person']:r for r in read(B/'INPUTS.json')['rows']};seen=set()
 px=[];pa=[]
 for r in prior+oldextra:
  hand=r.get('hand',r.get('side'));key=(r['clip_id'],r['frame'],hand)
  if key in seen:continue
  seen.add(key);ch=np.load(by[r['person']]['channels']);ix=int(np.flatnonzero(ch['frame_id']==r['frame'])[0]);f,a=features(ch['tcn280'],hand)
  px.append(windows(f,[ix])[0]);pa.append(a[ix]);records.append(dict(r,hand=hand))
 allx.extend([np.array(px),np.array(xx)]);allav.extend([np.array(pa),np.array(aa)]);ally.extend([np.zeros(len(px),int),np.array(yy)]);allknown.extend([np.array(pa).all(1),np.array(aa).all(1)]);records+=extra
 xraw=np.concatenate(allx);avail=np.concatenate(allav);target=np.concatenate(ally);known=np.concatenate(allknown)
 fit=np.array([r['role']=='fit'for r in records])&known
 write(out/'FROZEN.json',dict(parent={str(V/'shape_continuation'/f'{s}.pt'):sha(V/'shape_continuation'/f'{s}.pt')for s in[42,43,44]},
  labels=sha(out/'SOURCE_TRUTH.json'),steps=300,lr=.0002,seeds=[42,43,44],confidence=.85,
  no_dev_optimization=True,no_threshold_search=True,
  rule='Keep previously decided coarse results. For old unknown only, original expanded and continued model must agree without any known individual finger conflict.',
  acceptance='No extra definite wrong on source-corrected DEV; retain before-known outputs by contract. Compare original-label counts separately. Previously seen17 shape anchors not held-out.',web_admitted=False))
 counts=collections.Counter((records[i]['clip_id'],int(target[i]))for i in np.flatnonzero(fit));classes=collections.Counter(c for _,c in counts)
 w=torch.tensor([1/(counts[records[i]['clip_id'],int(target[i])]*classes[int(target[i])])for i in np.flatnonzero(fit)],dtype=torch.float32);w/=w.sum();ty=torch.tensor(target[fit]);runs=[]
 for seed in[42,43,44]:
  ck=torch.load(V/'shape_continuation'/f'{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);net=HandHead('temporal');net.load_state_dict(ck['state']);x=torch.tensor((xraw-ck['mean'])/ck['std'],dtype=torch.float32);opt=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.001);start=time.perf_counter()
  for _ in range(300):
   loss=(F.cross_entropy(net(x[fit]),ty,reduction='none')*w).sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  net.eval();path=out/f'{seed}.pt';torch.save(dict(state=net.state_dict(),mean=ck['mean'],std=ck['std'],kind='temporal'),path);runs.append(dict(seed=seed,seconds=time.perf_counter()-start,sha256=sha(path)))
 write(out/'TRAINED.json',dict(runs=runs,fit_only=True,added_source_frames=len(extra)));print(json.dumps(runs),flush=True)

if __name__=='__main__':main()
