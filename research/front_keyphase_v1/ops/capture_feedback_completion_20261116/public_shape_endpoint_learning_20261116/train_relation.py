"""Fixed public-FIT continuation of the relation observer, never user tuning."""
from audit import *
from core import Features,PhaseNet,states,approaches
from basic_runtime import RelationObserver
from torch.nn import functional as F
import collections,time

class RelationCandidate(RelationObserver):
 def __init__(self):
  self.models=[]
  for seed in [42,43,44]:
   ck=torch.load(D/'relation'/f'{seed}.pt',map_location='cpu',weights_only=False);m=PhaseNet(len(ck['mean']),'frame_mlp');m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);dest=D/'relation';dest.mkdir(exist_ok=True);assert not(dest/'FROZEN.json').exists()
 # Native RGB panels reviewed before labels. No automatic view-label transfer.
 labels={'NIA_SL_WORD1290_REAL01_L':{1:[45,47,56,57],2:[50,61,62]},
         'NIA_SL_WORD1290_REAL03_L':{0:[45],1:[53,56,57,65,77,80],2:[61,62,68,69]},
         'NIA_SL_WORD1290_REAL07_L':{1:[45,48,49,51,57,60,65,69,72,78],2:[53,54,61,63]}}
 src={r['id']:r for r in read(H/'PUBLIC_INPUTS.json')['rows']+read(H/'public_views/INPUTS.json')['rows']}
 new=[]
 for cid,groups in labels.items():
  r=src[cid];assert r['person'] in [1,3,7]
  for label,ids in groups.items():
   for i in ids:new.append(dict(clip_id=cid,person=r['person'],split='fit',frame=i,state=label,source='Native RGB contact sheets; visible separated/near projected hands, not physical contact',raw=r['raw'],video=r['video'],individual_finger_labels=None))
 write(dest/'SOURCE_TRUTH.json',dict(rows=new,reviewer='Codex native RGB review, not sign expert',images=[str(D/'audit'/f'{x}.jpg')for x in labels]))
 truth=read(B/'source_corrected/SOURCE_TRUTH.json')['rows'];inputs={r['clip_id']:r for r in read(B/'INPUTS.json')['rows']};anchors=[r for r in truth if r['split']=='fit']+new
 ff=Features();cache={};xx=[];yy=[];meta=[]
 for row in anchors:
  cid=row['clip_id']
  if cid not in cache:
   r=src.get(cid,inputs.get(cid));path=r.get('raw',r.get('raw_path'));z=dict(np.load(path));ch=adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']));f,quality,_=ff(ch);cache[cid]=(ch,f,quality)
  ch,f,quality=cache[cid];i=int(np.flatnonzero(ch['frame_id']==row['frame'])[0])
  if quality[i]:xx.append(f[i]);yy.append(row['state']);meta.append(dict(clip=cid,frame=row['frame'],label=row['state'],context='full'))
 # Actual observed prefix examples protect against completion being inferred
 # merely from a sequence ending. Each label is from an observed source anchor.
 for cid in cache:
  ch,_,_=cache[cid];n=len(ch['frame_id']);rr=[r for r in anchors if r['clip_id']==cid]
  for cls in [0,1,2]:
   group=[r for r in rr if r['state']==cls]
   if not group:continue
   for row in {r['frame']:r for r in [group[0],group[-1]]}.values():
    i=int(np.flatnonzero(ch['frame_id']==row['frame'])[0]);prefix={k:(v[:i+1]if isinstance(v,np.ndarray)and v.ndim and len(v)==n else v)for k,v in ch.items()};f,q,_=ff(prefix)
    if q[-1]:xx.append(f[-1]);yy.append(cls);meta.append(dict(clip=cid,frame=row['frame'],label=cls,context='observed_prefix'))
 xraw=np.array(xx,np.float32);target=np.array(yy,np.int64);write(dest/'INPUTS.json',dict(rows=meta,private_data=False))
 cfg=dict(steps=300,lr=.0001,seeds=[42,43,44],confidence=.65,parent='unchanged source_corrected frame_mlp',selection='One fixed endpoint, no epoch/threshold search; known DEV labels plus no premature-completion controls must pass.',new_anchors=len(new),train_rows=len(target),word_person_inputs=False,frozen100_backbone=True)
 write(dest/'FROZEN.json',cfg);weights=torch.tensor(len(target)/(3*np.maximum(np.bincount(target,minlength=3),1)),dtype=torch.float32);runs=[]
 for seed in cfg['seeds']:
  ck=torch.load(B/'source_corrected/runs'/f'frame_mlp_{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);net=PhaseNet(len(ck['mean']),'frame_mlp');net.load_state_dict(ck['state']);x=torch.tensor((xraw-ck['mean'])/ck['std']);y=torch.tensor(target);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.0001);started=time.perf_counter()
  for _ in range(cfg['steps']):
   loss=F.cross_entropy(net(x[None])[0],y,weight=weights);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
  net.eval();p=dest/f'{seed}.pt';torch.save(dict(state=net.state_dict(),mean=ck['mean'],std=ck['std']),p);runs.append(dict(seed=seed,seconds=time.perf_counter()-started,sha256=hashlib.sha256(p.read_bytes()).hexdigest()));print(runs[-1],flush=True)
 write(dest/'TRAINED.json',dict(runs=runs,fit_rows=len(target),new_source_anchors=len(new),backbone_changed=False,private_training=False))
if __name__=='__main__':main()
