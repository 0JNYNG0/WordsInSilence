"""FIT-only supplemental coarse palm evidence; preserve old labels by rehearsal."""
from common import *
sys.path.insert(0,str(G/'apple_v1'))
from train_surface import HandHead,features,windows
import torch.nn.functional as F

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);dest=E/'palm';assert not(dest/'TRAINED.json').exists()
 legacy=read(G/'apple_v1/SURFACE_FROZEN.json')['rows']
 old_inputs=read(G/'apple_v1/INPUTS.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']
 mapping={r['source_sha256']:r for r in old_inputs if r['role']=='fit'}
 xx=[];yy=[];meta=[];cache={}
 for r in legacy:
  src=mapping[r['source']];assert src['role']=='fit'
  path=src['channels'];x=cache.setdefault(path,np.load(path)['tcn280']);f,a=features(x,r['hand']);i=r['frame'];assert a[i].all()
  xx.append(windows(f,[i])[0]);yy.append(r['label']);meta.append(dict(r,role='fit',provenance='previous source-reviewed coarse-form truth'))
 # Native RGB contact sheets were reviewed this turn: separated hands only,
 # avoiding the overlap. These are WHOLE palm-form labels, not hidden digits.
 reviewed={1:[45,47,53,57],3:[53,56,65,77],7:[45,48,57,65]}
 by={r['id']:r for r in sources()}
 for person,ids in reviewed.items():
  cid=f'NIA_SL_WORD1290_REAL{person:02d}_L';src=by[cid];ch=channels(dict(np.load(src['raw'])))
  for hand in['L','R']:
   f,a=features(ch['tcn280'],hand)
   for i in ids:
    if not a[i].all():continue
    xx.append(windows(f,[i])[0]);yy.append(0);meta.append(dict(clip=cid,person=person,frame=i,hand=hand,label=0,role='fit',source=src['raw'],review_image=str(D/'audit'/f'{cid}.jpg'),provenance='Native RGB visible open/curved palm; individual hidden extension not labeled'))
 x=np.array(xx,np.float32);y=torch.tensor(yy);counts=np.bincount(yy,minlength=3);cw=torch.tensor(len(y)/(3*counts),dtype=torch.float32)
 write(dest/'FROZEN.json',dict(rows=meta,seeds=[42,43,44],steps=500,lr=.0003,weight_decay=.005,confidence=.85,labels=['open_or_curved_palm','clenched_hand','distinct_finger_pattern'],use='Supplemental disagreement guard; no automatic override or hidden-finger completion',selection='Fixed protocol; all new labels FIT only; known DEV is development, not unseen confirmation.'))
 runs=[]
 for seed in[42,43,44]:
  ck=torch.load(G/'apple_v1/runs'/f'surface_{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);m=HandHead('temporal');m.load_state_dict(ck['state']);tx=torch.tensor((x-ck['mean'])/ck['std']);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.005)
  for _ in range(500):
   loss=F.cross_entropy(m(tx),y,weight=cw);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
  p=dest/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=ck['mean'],std=ck['std']),p);runs.append(dict(seed=seed,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 write(dest/'TRAINED.json',dict(runs=runs,total_rows=len(meta),added_rows=len(meta)-len(legacy),class_counts=counts.tolist(),private_data=False));print('PALM_TRAINED',len(meta),len(meta)-len(legacy),flush=True)

class SupplementalPalm:
 def __init__(self):
  self.models=[]
  for seed in[42,43,44]:
   ck=torch.load(E/'palm'/f'{seed}.pt',map_location='cpu',weights_only=False);m=HandHead('temporal');m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))
 def sequence(self,x,hand):
  f,a=features(x,hand);w=windows(f,list(range(len(x))));ss=[]
  with torch.inference_mode():
   for m,mu,sd in self.models:
    p=m(torch.tensor((w-mu)/sd)).softmax(-1).numpy();s=p.argmax(-1);s[p.max(-1)<.85]=-1;ss.append(s)
  ss=np.array(ss);return np.where((ss==ss[0]).all(0)&a.all(1),ss[0],-1)
if __name__=='__main__':main()
