from diagnose import *
from train_palm import HandHead,features,windows
from train_surface import Surface
import torch.nn.functional as F

class VisiblePalm(Surface):
 def __init__(self):
  self.models=[]
  for seed in [42,43,44]:
   ck=torch.load(L/'palm'/f'{seed}.pt',map_location='cpu',weights_only=False);m=HandHead('temporal');m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);out=L/'palm';out.mkdir(exist_ok=True);assert not(out/'TRAINED.json').exists()
 rows=current_sources()+read(K/'fit_02/INPUTS.json')['rows']+read(K/'fit_05/INPUTS.json')['rows'];by={r['id']:r for r in rows};forms=list(read(J/'palm/FROZEN.json')['rows']);oldn=len(forms)
 for person,frames in [(2,[49,58,64,66,68,70]),(5,[41,50])]:
  for camera in ['L','R']:
   cid=f'NIA_SL_WORD1290_REAL{person:02d}_{camera}'
   for hand in ['L','R']:
    for i in frames:
     forms.append(dict(clip=cid,frame=i,hand=hand,label=0,role='fit',source=by[cid]['raw'],review_image=str(K/('fit_review_second' if person==2 and i>=68 else 'fit_review')/(cid+'.jpg')),provenance='Native RGB and raw overlay reviewed: separated open/curved palm. No hidden individual finger labels.'))
 inputs=read(G/'apple_v1/INPUTS.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows'];mapping={r['source_sha256']:r for r in inputs if r['role']=='fit'};cache={};xx=[];yy=[]
 for r in forms:
  path=by[r['clip']]['raw'] if 'clip' in r else mapping[r['source']]['channels']
  if path not in cache:
   z=dict(np.load(path));cache[path]=channels(z)['tcn280'] if 'xy_px' in z else z['tcn280']
  x=cache[path];f,a=features(x,r['hand']);i=r['frame'];assert a[i].all();xx.append(windows(f,[i])[0]);yy.append(r['label'])
 write(out/'FROZEN.json',dict(rows=forms,old_rows=oldn,new_rows=len(forms)-oldn,steps=500,lr=.0003,seeds=[42,43,44],confidence=.85,base=str(J/'palm'),scope='Shared coarse palm only; hand-local input, no word/person/guide. Exact fingers frozen. FIT only, no DEV epoch selection.'))
 x=np.array(xx,np.float32);y=torch.tensor(yy);counts=np.bincount(yy,minlength=3);cw=torch.tensor(len(y)/(3*counts),dtype=torch.float32);runs=[]
 for seed in [42,43,44]:
  ck=torch.load(J/'palm'/f'{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);m=HandHead('temporal');m.load_state_dict(ck['state']);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.005);tx=torch.tensor((x-ck['mean'])/ck['std'])
  for _ in range(500):
   loss=F.cross_entropy(m(tx),y,weight=cw);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
  p=out/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=ck['mean'],std=ck['std']),p);runs.append(dict(seed=seed,sha256=hashlib.sha256(p.read_bytes()).hexdigest()));print('TRAINED',seed,flush=True)
 write(out/'TRAINED.json',dict(runs=runs,rows=len(forms),private_data=False,frozen100=True))
if __name__=='__main__':main()
