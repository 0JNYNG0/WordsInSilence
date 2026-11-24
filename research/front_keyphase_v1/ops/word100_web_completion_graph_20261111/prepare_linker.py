from graph import *
from binding import *
import importlib.util,sys
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111'
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 assert not (U/'LINKER_FROZEN.json').exists()
 sys.dont_write_bytecode=True
 rc=load('_graph_requirement',G/'requirement_core.py');rows={r['clip_id']:r for r in rc.public_rows() if r['role']=='fit'}
 labels={r['clip_id']:r for r in read(R/'ROLE_LABELS.json')['rows']};sources=read(R/'TRAIN_INPUTS_COMPLETE.json')['rows'];xx=[];yy=[];ww=[];manifest=[];pins={};stat={}
 for rec in sources:
  if rec['reverse'] or rec['clip_id'] not in labels:continue
  p=Path(rec['path']);assert rc.sha(p)==rec['sha256'];z=np.load(p);r=rows[rec['clip_id']]
  features=body_context(z['z'],r['timestamp_us']);mask=(z['w']==1)&(z['y']>=0)&(z['y']<10)
  # Direct/reviewed anchors/intervals only. Weak interpolation, reverse pseudo-targets excluded.
  xx.append(features[mask]);yy.append(z['y'][mask]);ww.extend([r['person']]*int(mask.sum()));stat[rec['clip_id']]=int(mask.sum())
  manifest.append(dict(clip_id=rec['clip_id'],person=r['person'],word_id=r['word_id'],source_sha256=r['source_sha256'],view=rec['view'],observed_indices=np.flatnonzero(mask).tolist(),frames=int(mask.sum()),path=str(p),sha256=rec['sha256']))
  pins[str(p)]=rec['sha256']
 x=np.concatenate(xx);y=np.concatenate(yy).astype(np.int64);person=np.array(ww)
 assert np.isfinite(x).all() and len(x)==len(y)
 np.savez_compressed(U/'linker_fit.npz',x=x,y=y,person=person)
 counts=np.bincount(y,minlength=11);weights=np.zeros(11,np.float32);weights[counts>0]=np.sqrt(len(y)/(max(1,sum(counts>0))*counts[counts>0]));weights=np.minimum(weights,3)
 mean=x.mean(0);std=np.maximum(x.std(0),.05)
 write(U/'LINKER_MANIFEST.json',manifest)
 write(U/'LINKER_FROZEN.json',dict(seeds=[42,43,44],epochs=60,lr=.001,batch=128,weight_decay=.01,mean=mean.tolist(),std=std.tolist(),weights=weights.tolist(),class_counts=counts.tolist(),normal_views=['full','upper'],weak_labels_used=False,reverse_labels_used=False,word_id_input=False,handshape_input=False,desired_direction_input=False,expected_order_input=False,threshold=.65,final_epoch=True,pins=pins,train_hash=rc.sha(U/'linker_fit.npz'),repeated_development=True,scope='Shared body-context correspondence component, not semantic correctness classifier',predeclared_comparison=['old card query_indices','saved learned role occurrences','new body-only learned occurrences']))
 write(U/'LABEL_CHECK.json',dict(pass_=True,videos=len(stat),unique_frames=sum(stat.values()),view_samples=len(y),performer_ids=sorted(set(ww)),class_counts=counts.tolist(),handshape_difference_cannot_filter_correspondence=True,provenance='Existing direct/reviewed body context annotations, not new independent human gold',unobserved_not_negative=True,shared100_frozen=True))
 print(read(U/'LABEL_CHECK.json'))
if __name__=='__main__':main()
