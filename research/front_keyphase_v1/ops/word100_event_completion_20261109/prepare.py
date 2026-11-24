from core import *
def main():
 assert not (E/'INPUTS.json').exists();labels={r['clip_id']:r for r in read(E/'REVIEW_LABELS.json')['rows']};rows=[]
 for r in read(N/'INPUTS.json')['rows']:
  if r['role']!='fit' or r['kind'] not in ['full_normal','actual_crop','upper_normal','speed075','speed125','gap2']:continue
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);a=labels[r['clip_id']]['anchors'];y,w=targets(z['frame_id'],a)
  # Hidden joints never become labels that the system observed an action.
  visible=(z['raw'][:,237:279].sum(1)>=5);w[~visible]=0;y[~visible]=-1
  def add(kind,ix):
   p=E/'inputs'/f"{r['clip_id']}_{kind}.npz";p.parent.mkdir(exist_ok=True);raw=z['raw'][ix];fid=z['frame_id'][ix];ts=z['timestamp_us'][ix].copy()
   if np.any(np.diff(ts)<=0):raise AssertionError('source time order')
   # A cut removes elapsed source time. It does not add evidence to complete the sign.
   if kind.startswith('partial'):ts=np.arange(len(ix),dtype=np.int64)*int(np.median(np.diff(z['timestamp_us'])))
   np.savez_compressed(p,raw=raw,timestamp_us=ts,frame_id=fid,target=y[ix],weight=w[ix]);rows.append(dict(clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],kind=kind,path=str(p),sha256=sha(p),source_sha256=r['source_sha256'],original_input=r['path']))
  add(r['kind'],np.arange(len(y)))
  if r['kind']!='full_normal':continue
  # FIT-only unfinished-action controls, before the first reviewed core of each event.
  for c in sorted(set(q['label'] for q in a if q['label']>0 and q['label']%2==0)):
   first=min(q['frame'] for q in a if q['label']==c);prior=[q for q in a if q['frame']<first and q['label']==c-1]
   if not prior:continue
   stop=max(q['frame'] for q in prior);ix=np.flatnonzero(z['frame_id']<=stop)
   rest=[q['frame'] for q in a if q['frame']>first and q['label']==0]
   if rest:ix=np.r_[ix,np.flatnonzero((z['frame_id']>=min(rest))&(y==0))]
   if len(ix)>5:add(f'partial_before_core{c}',ix)
 write(E/'INPUTS.json',dict(rows=rows,original_fit_clips=len(labels),no_dev=True))
 cfg=dict(status='FROZEN_BEFORE_TRAINING_AND_EVAL',model='Frozen geometry Conv+BiGRU representation + normalized geometry; 130->64->19 progression output',epochs=100,lr=.001,weight_decay=.01,batch_clips=6,seeds=[42,43,44],primary=42,folds=[[1,2,3],[4,5,6],[7,8,9]],threshold=.65,minimum_evidence_ms=60,labels_sha256=sha(E/'REVIEW_LABELS.json'),inputs_sha256=sha(E/'INPUTS.json'),protected=read(N/'FROZEN.json')['protected'],gates=['normal full/crop/speed/crossing do not increase warnings or withheld cases versus deployed baseline','normal useful similarity retained','unfinished/masked controls cannot gain full-completion claims','wrong shape/reverse/combined alert and motion coverage retained','no unseen coordinates are called correct'],scope='237 fixed historical regression conditions; new FIT subject-heldout core and unfinished controls; no new independent DEV; no threshold/candidate tuning from evaluation')
 cfg['protected'].update({str(N/'runs'/name/'model.pt'):sha(N/'runs'/name/'model.pt') for name in ['final_42','final_43','final_44','heldout_1_2_3','heldout_4_5_6','heldout_7_8_9']})
 write(E/'FROZEN.json',cfg);print('prepared',len(rows))
if __name__=='__main__':main()
