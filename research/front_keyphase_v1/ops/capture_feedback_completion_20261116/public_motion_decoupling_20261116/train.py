from common import *
from core import Features
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);dest=E/'relation';assert not(dest/'TRAINED.json').exists()
 write(E/'CONTRACT.json',dict(goal='Separate body-relative endpoint posture from finger shape. Improve missed genuine repetition without inventing completion.',graph=['A public source truth/observation audit','B fixed small geometry-head training','C complete/partial/missing and shape-change verification','D repair actual causes and revalidate only affected checks','E admitted integration, timing, resume record'],loop='Continue cause-based correction across stages without stopping after a rejected candidate; avoid repeated tests without new evidence.',private_data=False,frozen100=True,gate='No new normal warning or premature completion; keep existing clear handshape errors and speed. Distinguish source data problem from model error.'))
 truths=read(B/'source_corrected/SOURCE_TRUTH.json')['rows']+read(D/'relation/SOURCE_TRUTH.json')['rows'];rows={r['id']:r for r in sources()};base={r['clip_id']:r for r in read(B/'INPUTS.json')['rows']};net=Features();cache={};xx=[];yy=[];meta=[]
 for r in truths:
  if r['split']!='fit':continue
  cid=r['clip_id']
  if cid not in cache:
   src=rows.get(cid,base.get(cid));raw=dict(np.load(src.get('raw',src.get('raw_path'))));ch=channels(raw);f,_,_=net(ch);x,good=geometry(f);cache[cid]=(ch,x,good)
  ch,x,good=cache[cid];i=int(np.flatnonzero(ch['frame_id']==r['frame'])[0])
  if good[i]:xx.append(x[i]);yy.append(r['state']);meta.append(dict(clip=cid,frame=r['frame'],state=r['state'],split='fit'))
 x=np.array(xx,np.float32);y=np.array(yy,np.int64);mu=x.mean(0);sd=np.maximum(x.std(0),.02);weights=torch.tensor(len(y)/(3*np.maximum(np.bincount(y,minlength=3),1)),dtype=torch.float32)
 write(dest/'FROZEN.json',dict(rows=meta,labels=['outside_observed_interaction','separated_hands','projected_near_hands'],features='33 body-relative geometry/availability values; no fingers, word, person, whole-video GRU embedding or elapsed phase',seeds=[42,43,44],steps=800,lr=.001,weight_decay=.01,confidence=.65,selection='Fixed run; no DEV epoch/threshold selection. Small additional head, not full-backbone fine-tuning.'))
 runs=[]
 for seed in[42,43,44]:
  torch.manual_seed(seed);m=PhaseNet(x.shape[1],'frame_mlp');opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);tx=torch.tensor((x-mu)/sd);ty=torch.tensor(y);started=time.perf_counter()
  for _ in range(800):
   loss=torch.nn.functional.cross_entropy(m(tx[None])[0],ty,weight=weights);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
  m.eval();p=dest/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=mu,std=sd),p);runs.append(dict(seed=seed,seconds=time.perf_counter()-started,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 write(dest/'TRAINED.json',dict(runs=runs,fit_rows=len(x),fit_clips=len(cache),feature_dim=x.shape[1],class_counts=np.bincount(y).tolist(),private_data=False));print('TRAINED',len(x),x.shape[1],len(cache),flush=True)
if __name__=='__main__':main()
