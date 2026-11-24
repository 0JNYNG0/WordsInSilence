"""Fixed public FIT continuation of a shared L/R coarse handshape head."""
from pathlib import Path
import sys,json,time,hashlib,collections
import numpy as np,torch
from torch.nn import functional as F
V=Path(__file__).resolve().parent;O=V.parents[1];B=O/'basic_demo_feedback_20261115';S=O/'word100_finger_repair_20261115'
sys.path[:0]=[str(B),str(S)]
from local_shape import HandHead,LocalShape
from finger_model import features,windows
from shape_consensus import ShapeConsensus,combine
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 dest=V/'shape_continuation';dest.mkdir(exist_ok=True);assert not(dest/'FROZEN.json').exists()
 public=read(V/'REVIEW_INPUTS.json')['rows'];added=[];xs=[];av=[]
 # Reviewed original RGB panels p01..p09 before assigning coarse labels.
 # Coarse open is not an invented four-finger/individual-angle annotation.
 for r in public:
  z=np.load(r['channels']);ix=int(np.flatnonzero(z['frame_id']==r['frame'])[0])
  assert sha(r['video'])==r['source_sha256']
  for hand in ['L','R']:
   f,a=features(z['tcn280'],hand)
   added.append(dict(**r,hand=hand,coarse_label=0,individual_finger_labels=None,
      review='Codex RGB panels p01..09: visibly open hand; normal minor flexion retained. Not expert review.'))
   xs.append(windows(f,[ix])[0]);av.append(a[ix])
 write(dest/'SOURCE_TRUTH.json',dict(rows=added,public_fit_only=True))
 old=read(S/'INPUTS.json')['rows'];d=np.load(S/'INPUTS.npz');y=d['y'];target=np.where((y==1).all(1),0,np.where((y==0).all(1),1,2))
 known=(y>=0).all(1)&d['availability'].all(1)
 # Retain previously reviewed extra FIT sources too, without touching DEV labels.
 prior=read(B/'local_shape/expanded/SOURCE_TRUTH.json')['rows'];br={r['person']:r for r in read(B/'INPUTS.json')['rows']}
 keys={(r['clip_id'],r['frame'],r['hand']) for r in added}
 for r in prior:
  if (r['clip_id'],r['frame'],r['side']) in keys:continue
  ch=np.load(br[r['person']]['channels']);ix=int(np.flatnonzero(ch['frame_id']==r['frame'])[0]);f,a=features(ch['tcn280'],r['side'])
  added.append(dict(**r,hand=r['side'],coarse_label=0));xs.append(windows(f,[ix])[0]);av.append(a[ix])
 xraw=np.concatenate([d['x'],xs]);available=np.concatenate([d['availability'],av]);target=np.r_[target,np.zeros(len(added),int)]
 known=np.r_[known,np.array(av).all(1)];rows=old+added
 fit=np.array([r['role']=='fit' for r in rows])&known;dev=np.array([r['role']=='development' for r in rows])&known
 assert not ({r['person'] for i,r in enumerate(rows) if fit[i]} & {r['person'] for i,r in enumerate(rows) if dev[i]})
 write(dest/'FROZEN.json',dict(seeds=[42,43,44],steps=400,lr=.0002,weight_decay=.001,confidence=.85,
  parent={str(B/'local_shape/temporal'/f'{s}.pt'):sha(B/'local_shape/temporal'/f'{s}.pt') for s in [42,43,44]},
  truth_sha=sha(dest/'SOURCE_TRUTH.json'),selection='One fixed final checkpoint; no epoch/threshold search. Existing237 DEV coarse targets: no extra definite wrong and no lost prior correct.',
  input='Same four-chain geometry window; no word, person, target, whole-video time or body coordinates.',
  scope='Replace the older coarse observer in the existing consensus only if gates pass. Individual finger models and their conflict protection unchanged.',
  extra_fit_anchors=len(added),newly_reviewed=len(public)*2,dev_not_final=True,private_data=False))
 counts=collections.Counter((rows[i]['clip_id'],int(target[i])) for i in np.flatnonzero(fit));classes=collections.Counter(c for _,c in counts)
 weights=torch.tensor([1/(counts[rows[i]['clip_id'],int(target[i])]*classes[int(target[i])]) for i in np.flatnonzero(fit)],dtype=torch.float32);weights/=weights.sum()
 yy=torch.tensor(target[fit]);runs=[];allp=[]
 for seed in [42,43,44]:
  ck=torch.load(B/'local_shape/temporal'/f'{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed)
  net=HandHead('temporal');net.load_state_dict(ck['state']);xx=torch.tensor((xraw-ck['mean'])/ck['std'],dtype=torch.float32)
  opt=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.001);start=time.perf_counter()
  for _ in range(400):
   loss=(F.cross_entropy(net(xx[fit]),yy,reduction='none')*weights).sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  net.eval()
  with torch.inference_mode():allp.append(net(xx).softmax(-1).numpy())
  path=dest/f'{seed}.pt';torch.save(dict(state=net.state_dict(),mean=ck['mean'],std=ck['std'],kind='temporal'),path)
  runs.append(dict(seed=seed,seconds=time.perf_counter()-start,sha256=sha(path)));print('TRAIN',runs[-1],flush=True)
 ps=np.stack(allp);q=ps.argmax(-1);q[ps.max(-1)<.85]=-1;q=np.where((q==q[0]).all(0),q[0],-1);q[~available.all(1)]=-1
 # Reuse the already frozen legacy per-finger outputs for exactly the same rows.
 prev=np.load(B/'local_shape/CONSENSUS_DEVELOPMENT.npz');print('BASE_KEYS',prev.files,flush=True)
 np.savez_compressed(dest/'predictions.npz',probabilities=ps,states=q,targets=target,known=known,fit=fit,dev=dev)
 write(dest/'TRAINED.json',dict(runs=runs,fit_rows=int(fit.sum()),dev_rows=int(dev.sum()),weights_changed='new isolated coarse head only',web_admitted=False))

if __name__=='__main__':main()
