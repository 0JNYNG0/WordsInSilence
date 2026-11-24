from core import *
import shutil
def main():
 assert not (U/'runs').exists()
 E=O/'word100_event_completion_20261109';cfg=read(U/'FROZEN.json');extra=read(U/'ADDITIONAL_LABELS.json');before=U/'before_order_augmentation';before.mkdir(exist_ok=True)
 for f in ['FROZEN.json','READY.json','ADDITIONAL_LABELS.json']:shutil.copy2(U/f,before/f)
 manifest={r['clip_id']:r for r in read(B/'READY.json')['rows']};orders={};mapped=[0,1,2,3,4,5,6,10,11]
 for r in read(E/'REVIEW_LABELS.json')['rows']:
  row=manifest[r['clip_id']];assert row['role']=='fit' and row['source_sha256']==r['source_sha256'];assert sha(r['sheet'])==r['sheet_sha256']
  z=np.load(row['path']);runs=[]
  for q in r['anchors']:
   if q['label']>0 and q['label']%2==0:
    if runs and runs[-1][0]==q['label']:runs[-1][1].append(q['frame'])
    else:runs.append([q['label'],[q['frame']]])
  out=[]
  for a,b in zip(runs,runs[1:]):
   fa=a[1][len(a[1])//2];fb=b[1][len(b[1])//2];ia=np.flatnonzero(z['frame_id']==fa);ib=np.flatnonzero(z['frame_id']==fb)
   if not len(ia) or not len(ib):continue
   out.append(dict(a=int(ia[0]),b=int(ib[0]),frame_a=fa,frame_b=fb,guide_a=mapped[a[0]//2-1],guide_b=mapped[b[0]//2-1],origin=str(E/'REVIEW_LABELS.json'),meaning='Ordered observed event pair, not a new obligatory step or exact contact truth'))
  orders[r['clip_id']]=out
 extra['orders']=orders;extra['order_source']=dict(path=str(E/'REVIEW_LABELS.json'),sha256=sha(E/'REVIEW_LABELS.json'),reviewer='Inherited Codex RGB event review; not new expert labels',mapping=mapped);write(U/'ADDITIONAL_LABELS.json',extra)
 rr=rows();fit=[r for r in rr if r['role']=='fit'];ready=read(U/'READY.json');ready.update(new_order_pairs=sum(map(len,orders.values())),order_pairs=sum(len(r['orders']) for r in fit));write(U/'READY.json',ready)
 cfg['revision']='Before any training/evaluation: link previously RGB-reviewed event order across all9 FIT IDs, not only original1/4/7. Initial preparation retained.'
 cfg['authorization']='Latest user: maintain100 common training; reinforce shared shape/direction/order; initially validate feedback on4; keep C/T3F, no private training/tuning.'
 cfg['protected']=read(E/'FROZEN.json')['protected']
 cfg['source_pins'][str(E/'REVIEW_LABELS.json')]=sha(E/'REVIEW_LABELS.json')
 for p in cfg['source_pins']:cfg['source_pins'][p]=sha(p)
 for f in ['core.py','prepare.py','train.py','pretrain.py']:cfg['code_pins'][str(U/f)]=sha(U/f)
 write(U/'FROZEN.json',cfg)
 checks=0;norm=read(U/'NORMALIZATION.json');torch.set_num_threads(4);net=Network('factorized',norm).eval();sample=next(r for r in fit if r['pairs']);x=physical(sample['x_full']);q=x.copy();q[:,18:102]+=3.7;q[:,102:186]*=-1 # Finger positions/local shapes altered, body wrists preserved.
 for axis in range(5):
  for side in ['R','L']:
   g,ok,_=geometry(x,sample['timestamp_us'],40,80,axis,side,'L' if side=='R' else 'R');g2,ok2,_=geometry(q,sample['timestamp_us'],40,80,axis,side,'L' if side=='R' else 'R');assert ok==ok2 and np.array_equal(g,g2);checks+=1
   h=torch.randn(1,len(x),64);h2=torch.randn_like(h);ii=torch.tensor([[0,40,80]])
   with torch.inference_mode():assert torch.equal(net.movement(h,torch.from_numpy(g[None]),ii),net.movement(h2,torch.from_numpy(g[None]),ii));checks+=1
 # A missing body wrist never becomes a usable direction measurement.
 q=x.copy();q[:,186+5]=0;q[:,186+6]=0
 for axis in range(5):assert not geometry(q,sample['timestamp_us'],40,80,axis)[1];checks+=1
 assert {r['person'] for r in fit}==set(range(1,10));assert len({r['word_id'] for r in fit})==100;checks+=2
 for r in fit:
  for p in r['pairs']+r['orders']:assert p['a']<p['b'] and p['b']<len(r['x_full']);checks+=1
 for p,h in cfg['protected'].items():assert sha(p)==h;checks+=1
 write(U/'PRETRAIN_CHECK.json',dict(status='PASS',checks=checks,frozen_sha256=sha(U/'FROZEN.json'),shape_motion_invariance=True,review='Same Codex code/data contract check',no_new_scores=True));event('pretrain_checked',checks=checks,new_order_pairs=ready['new_order_pairs'],total_order_pairs=ready['order_pairs'])
if __name__=='__main__':main()
