from view_common import *
from plane_model import PlaneHead,hand_plane_features
from train_palm import HandHead,features,windows
import torch.nn.functional as lossfn

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not(J/'TRAINED.json').exists()
 by={r['id']:r for r in sources()};cache={}
 def ch(cid):
  if cid not in cache:cache[cid]=channels(dict(np.load(by[cid]['raw'])))
  return cache[cid]
 xx=[];yy=[];labels=[]
 for r in read(V/'plane/SOURCE_TRUTH.json')['rows']:
  c=ch(r['clip_id']);i=int(np.flatnonzero(c['frame_id']==r['frame'])[0]);x,a=hand_plane_features(c,r['hand'])
  if a[i]:xx.append(x[i]);yy.append(r['plane']);labels.append(dict(r,role='fit',origin='original reviewed plane truth'))
 oldn=len(labels);frames={1:[45,47,53,57],3:[53,56,65,77],7:[45,48,57,65]};newforms=[]
 for person,ids in frames.items():
  for camera in['L','R']:
   cid=f'NIA_SL_WORD1290_REAL{person:02d}_{camera}';c=ch(cid)
   for hand,y in [('L',1),('R',0)]:
    x,a=hand_plane_features(c,hand)
    for i in ids:
     if not a[i]:continue
     label=dict(clip_id=cid,frame=i,hand=hand,plane=y,role='fit',origin='Native RGB paired raw/skeleton reviewed this turn; visible upright/flat, not arbitrary 3D surface',review_image=str(J/'audit'/(cid+'.jpg')))
     xx.append(x[i]);yy.append(y);labels.append(label)
     if camera=='R':newforms.append(dict(clip=cid,frame=i,hand=hand,label=0,role='fit',source=by[cid]['raw'],review_image=label['review_image'],provenance='Source visible open/curved hand; no exact hidden digit extension truth'))
 x=np.array(xx,np.float32);y=torch.tensor(yy);out=J/'plane';out.mkdir(exist_ok=True)
 write(out/'FROZEN.json',dict(rows=labels,original_rows=oldn,added_rows=len(labels)-oldn,steps=400,lr=.001,seeds=[42,43,44],confidence=.85,protocol='Existing42-input plane head, fine-tune from original weights with old truth rehearsal; no DEV epoch or threshold selection',physical_3d_truth=False));plane_runs=[]
 for seed in[42,43,44]:
  ck=torch.load(V/'plane'/f'{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);m=PlaneHead();m.load_state_dict(ck['state']);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);tx=torch.tensor((x-ck['mean'])/ck['std'])
  for _ in range(400):
   loss=lossfn.cross_entropy(m(tx),y);opt.zero_grad();loss.backward();opt.step()
  p=out/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=ck['mean'],std=ck['std']),p);plane_runs.append(dict(seed=seed,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 # Rehearse every prior coarse hand-form label; add right camera examples.
 old=read(E/'palm/FROZEN.json')['rows'];inputs=read(G/'apple_v1/INPUTS.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows'];mapping={r['source_sha256']:r for r in inputs if r['role']=='fit'};xx=[];yy=[];forms=old+newforms;local={}
 for r in forms:
  if 'clip'in r:c=ch(r['clip']);z=c['tcn280']
  else:
   src=mapping[r['source']];path=src['channels']
   if path not in local:local[path]=np.load(path)['tcn280']
   z=local[path]
  f,a=features(z,r['hand']);i=r['frame'];assert a[i].all();xx.append(windows(f,[i])[0]);yy.append(r['label'])
 x=np.array(xx,np.float32);y=torch.tensor(yy);counts=np.bincount(yy,minlength=3);cw=torch.tensor(len(y)/(3*counts),dtype=torch.float32);out=J/'palm';out.mkdir(exist_ok=True);runs=[]
 write(out/'FROZEN.json',dict(rows=forms,original_rows=len(old),added_rows=len(newforms),steps=500,lr=.0003,seeds=[42,43,44],confidence=.85,scope='Visible coarse palm form, no individual hidden finger state; fixed FIT-only protocol'))
 for seed in[42,43,44]:
  ck=torch.load(E/'palm'/f'{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);m=HandHead('temporal');m.load_state_dict(ck['state']);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.005);tx=torch.tensor((x-ck['mean'])/ck['std'])
  for _ in range(500):
   loss=lossfn.cross_entropy(m(tx),y,weight=cw);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
  p=out/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=ck['mean'],std=ck['std']),p);runs.append(dict(seed=seed,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 write(J/'TRAINED.json',dict(plane=dict(runs=plane_runs,fit_rows=len(labels),new_rows=len(labels)-oldn),palm=dict(runs=runs,fit_rows=len(forms),new_rows=len(newforms)),new_source_people=[1,3,7],new_source_views=['L','R'],private_data=False,frozen100=True));print('TRAINED',len(labels),len(forms),flush=True)
if __name__=='__main__':main()
