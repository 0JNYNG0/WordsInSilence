from view_common import *
from palm_geometry import palm_geometry,PalmRelation
from core import Features,PhaseNet

class ViewRelation(PalmRelation):
 def __init__(self):
  self.models=[]
  for seed in[42,43,44]:
   ck=torch.load(J/'relation'/f'{seed}.pt',map_location='cpu',weights_only=False);m=PhaseNet(len(ck['mean']),'frame_mlp');m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);out=J/'relation';out.mkdir(exist_ok=True);assert not(out/'TRAINED.json').exists();rows=list(read(E/'palm_relation/FROZEN.json')['rows'])
 extra={1:{1:[47,53,57],2:[50,62]},3:{1:[56,65],2:[61,62,69]},7:{1:[48,57,61],2:[53,54,63]}}
 for person,labels in extra.items():
  cid=f'NIA_SL_WORD1290_REAL{person:02d}_R'
  for state,frames in labels.items():
   rows += [dict(clip=cid,frame=i,state=state,split='fit',review=str(J/'audit_movement'/(cid+'.jpg')),note='Native RGB projected separation/near, not physical3Dcontact')for i in frames]
 by={r['id']:r for r in sources()};net=Features();cache={};xx=[];yy=[];included=[]
 for r in rows:
  cid=r['clip']
  if cid not in cache:
   ch=channels(dict(np.load(by[cid]['raw'])));f,_,_=net(ch);x,good=palm_geometry(f);cache[cid]=(ch,x,good)
  ch,x,good=cache[cid];i=int(np.flatnonzero(ch['frame_id']==r['frame'])[0])
  if good[i]:xx.append(x[i]);yy.append(r['state']);included.append(r)
 x=np.array(xx,np.float32);y=torch.tensor(yy);counts=np.bincount(yy,minlength=3);cw=torch.tensor(len(y)/(3*counts),dtype=torch.float32)
 write(out/'FROZEN.json',dict(rows=included,steps=500,lr=.0003,seeds=[42,43,44],confidence=.65,added_source_rows=sum('_R'in r['clip']for r in included),review='FIT1/3/7 right camera native key positions; ambiguous 01_R65 and03_R68 omitted',protocol='Fine-tune prior51-feature body/palm head with original146 anchors rehearsed; no hidden state or DEV labels'))
 runs=[]
 for seed in[42,43,44]:
  ck=torch.load(E/'palm_relation'/f'{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);m=PhaseNet(x.shape[1],'frame_mlp');m.load_state_dict(ck['state']);tx=torch.tensor((x-ck['mean'])/ck['std']);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01)
  for _ in range(500):
   loss=torch.nn.functional.cross_entropy(m(tx[None])[0],y,weight=cw);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
  p=out/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=ck['mean'],std=ck['std']),p);runs.append(dict(seed=seed,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 write(out/'TRAINED.json',dict(runs=runs,fit_rows=len(x),class_counts=counts.tolist(),private_data=False));print('RELATION_TRAINED',len(x),flush=True)
if __name__=='__main__':main()
