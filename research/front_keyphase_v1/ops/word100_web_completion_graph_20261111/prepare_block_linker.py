"""FIT-only rework: reorder whole observed role blocks, never reverse their frames."""
from graph import *
from prepare_linker import load
import numpy as np,torch,time,traceback,sys
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111'
def permutation(y,w,mode):
 anchors=np.flatnonzero((y>0)&(y<8)&(w==1))
 if not len(anchors):return None
 groups=[]
 for i in anchors:
  if not groups or y[i]!=groups[-1][0]:groups.append([int(y[i]),[int(i)]])
  else:groups[-1][1].append(int(i))
 # Repeated same action states are retained. Don't relabel them as different actions.
 if len(groups)<2 or len(groups)>5 or len({a for a,_ in groups})!=len(groups):return None
 edges=[int(anchors[0])]+[(a[-1]+b[0])//2+1 for (_,a),(_,b) in zip(groups,groups[1:])]+[int(anchors[-1])+1]
 blocks=[np.arange(a,b) for a,b in zip(edges,edges[1:])];order=list(range(len(blocks)))[::-1] if mode=='reverse_blocks' else list(range(1,len(blocks)))+[0]
 parts=[np.arange(edges[0])]+[blocks[i] for i in order]+[np.arange(edges[-1],len(y))];ii=np.concatenate(parts);seams=np.cumsum([len(a) for a in parts])[:-1]
 return ii,seams,dict(groups=[a for a,_ in groups],source_edges=edges,block_order=order,source_indices=ii.tolist(),within_block_direction_preserved=True)

def main():
 assert not (U/'BLOCK_INPUT_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);sys.dont_write_bytecode=True
 rc=load('_block_requirement',G/'requirement_core.py');role=load('_block_role',R/'role_core.py');rows={r['clip_id']:r for r in rc.public_rows() if r['role']=='fit'};labels={r['clip_id']:r for r in read(R/'ROLE_LABELS.json')['rows']};sources=[r for r in read(R/'TRAIN_INPUTS_COMPLETE.json')['rows'] if not r['reverse'] and r['clip_id'] in labels]
 write(U/'BLOCK_FROZEN.json',dict(reason='Body-only candidate loses normal contextual evidence. Preserve temporal model; train ordered block-permutation contrasts from FIT direct labels.',seeds=[42,43,44],epochs=25,lr=.0003,weight_decay=.01,batch_sequences=1,teacher_kl_weight=.2,normal_teacher_scope='normal FIT only, teacher matches corresponding seed',confidence=.65,selection='final epoch, no dev-based seed or threshold selection',pseudo_direct_labels=False,seam_guard_us=133333,weak_interpolation_used=False,whole_reverse_not_used=True,public_fit_only=True))
 base=rc.frozen_observer();frozen=role.frozen_model();records=[];t=time.perf_counter()
 for s in sources:
  cid=s['clip_id'];r=rows[cid];z=np.load(s['path']);ts=r['timestamp_us'];normal_y=z['y'].copy();normal_w=(z['w']==1).astype(np.float32);items=[('normal',z['z'],normal_y,normal_w,None)]
  modes=['reverse_blocks','rotate_blocks']
  for mode in modes:
   p=permutation(normal_y,normal_w,mode)
   if p is None:continue
   ii,seams,meta=p
   if mode=='rotate_blocks' and len(meta['block_order'])==2:continue # same permutation, no duplicate examples
   x=rc.prior.physical(r['x_'+s['view']])[ii].copy();yy=normal_y[ii].copy();ww=normal_w[ii].copy()
   for seam in seams:
    anchor=min(int(seam),len(ts)-1);ww[np.abs(ts-ts[anchor])<=133333]=0
   fh,_=role.lc.be.ev.frozen_forward(base,x,r['word']);ob=role.features(frozen,x,ts,fh)
   items.append((mode,ob['z'],yy,ww,meta))
  for mode,zz,yy,ww,meta in items:
   cp=U/'block_inputs'/f"{cid}_{s['view']}_{mode}.npz";cp.parent.mkdir(exist_ok=True);np.savez_compressed(cp,z=zz,y=yy,w=ww)
   records.append(dict(clip_id=cid,person=r['person'],word_id=r['word_id'],source_sha256=r['source_sha256'],view=s['view'],mode=mode,path=str(cp),sha256=rc.sha(cp),direct_frames=int(ww.sum()),permutation=meta))
  print(cid,s['view'],len(items),flush=True)
 write(U/'BLOCK_INPUT_COMPLETE.json',dict(rows=records,seconds=time.perf_counter()-t,private_used=False,source_videos=len({r['clip_id'] for r in records}),observed_frames=sum(r['direct_frames'] for r in records),new_input_inference_only=True))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'BLOCK_PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
