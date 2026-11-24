from core import *
from collections import Counter
def main():
 assert not (U/'FROZEN.json').exists()
 sources=read(U/'REVIEW_NEW_INPUTS.json');by={}
 for q in sources:by.setdefault(q['clip_id'],[]).append(q)
 motion_new={};shape_new={};audit=[];bm={r['clip_id']:r for r in shared.load_rows()}
 for clip,rr in by.items():
  r=bm[clip];assert r['role']=='fit' and r['person']<=9;word=r['word_id'];person=r['person'];frames=[q['frame_id'] for q in rr];pairs=[];shape=[]
  def add(a,b,axis,target,moving='R'):
   ia=int(np.flatnonzero(r['frame_id']==frames[a])[0]);ib=int(np.flatnonzero(r['frame_id']==frames[b])[0]);pairs.append(dict(a=ia,b=ib,frame_a=frames[a],frame_b=frames[b],axis=axis,target=target,moving=moving,anchor='L' if moving=='R' else 'R',origin='RGB_review_20261110',meaning='Observed component only; not whole sign correctness'))
  if word=='WORD0003':add(0,2,4,2);add(2,4,4,0)
  if word=='WORD0500':add(0,2,0,0);add(2,4,0,1);add(0,2,0,2,'L')
  if word=='WORD0678':add(0,2,0,0);add(2,4,0,1)
  if word=='WORD1139':
   if person in (2,3):add(0,2,3,0)
   add(2,4,3,1)
  if word=='WORD1290':add(0,4,0,1,'L') # Supporting hand; right hand cycles, NOT labelled hold.
  if word=='WORD2551' and person in (1,2):add(1,4,1,1)
  for k in (1,3):
   feat={}
   if word in ('WORD0003','WORD0500','WORD0678','WORD1139'):feat={'1':0,'3':0} # Partial long-finger extension; index-only not inferred.
   if word=='WORD1290':feat={'1':1,'2':1,'3':0,'4':0}
   if word=='WORD2551' and k==3 and person in (1,3):feat={'1':1,'3':0}
   if feat:shape.append(dict(frame=frames[k],features=feat,source=rr[k]['source']))
  motion_new[clip]=pairs;shape_new[clip]=shape
  audit.append(dict(clip_id=clip,reviewer='Codex visual review, not independent human/expert',frames=frames,new_motion=len(pairs),shape_cells=sum(len(q['features']) for q in shape),sources=[dict(path=q['source'],sha256=sha(q['source'])) for q in rr]))
 write(U/'ADDITIONAL_LABELS.json',dict(motion=motion_new,shape=shape_new,review=audit,limitations=['No exact contact, hidden thumb, 3D rotation or linguistic tolerance labels.','Holding refers to the specified component, not absence of all hand/finger motion.','Direction labels describe coarse interval progression; no claim of frame-complete motion truth.']))
 rr=rows();fit=[r for r in rr if r['role']=='fit'];dev=[r for r in rr if r['role']=='development'];assert len(fit)==899 and len(dev)==300
 assert len(set(r['word_id'] for r in fit))==100 and {r['person'] for r in fit}.isdisjoint({r['person'] for r in dev})
 gg=[];eligible=0;labelled=[r for r in fit if r['has_labels']]
 for r in labelled:
  b=component_batch([r]);gg.extend(b['g'].numpy());eligible+=len(b['my'])
 mat=np.array(gg);norm=dict(mean=mat.mean(0).tolist(),std=np.maximum(mat.std(0),.05).tolist(),fit_only=True);write(U/'NORMALIZATION.json',norm)
 write(U/'READY.json',dict(fit_clips=899,development_clips=300,words=100,fit_ids=sorted(set(r['person'] for r in fit)),development_ids=sorted(set(r['person'] for r in dev)),fine_label_clips=len(labelled),fine_label_words=len(set(r['word_id'] for r in labelled)),new_motion_intervals=sum(len(v) for v in motion_new.values()),new_shape_cells=sum(len(q['features']) for v in shape_new.values() for q in v),total_fit_motion=sum(len(r['pairs']) for r in labelled),eligible_motion_including_reverse=eligible,order_pairs=sum(len(r['orders']) for r in labelled),rows=[{k:r[k] for k in ('clip_id','person','word_id','role','path','sha256')} for r in rr]))
 pins=[B/'READY.json',M/'READY.json',U/'ADDITIONAL_LABELS.json',U/'NORMALIZATION.json',U/'READY.json',shared.BASE,D/'consumer.py',D/'runs/42/model.pt']+[B/'runs'/f'shared100_{s}'/'model.pt' for s in (42,43,44)]
 cfg=dict(status='FROZEN_BEFORE_NEW_TRAINING_AND_SCORES',conditions=['coupled','factorized'],seeds=[42,43,44],primary_seed=42,epochs=12,steps_per_epoch=38,aux_batch=24,supervised_batch=8,lr_base=.00005,lr_head=.001,weight_decay=.01,weights=dict(shape=1.,weak_stage=.15,prototype=.05,word=.2,movement=.5,order=.2),confidence=.65,init='Continue corresponding stored shared100 seed; no new model download',all_fit_required=899,selection='Fixed final epoch all seeds. No development threshold/seed/architecture selection.',runtime='Preserve current encoder/phase/difference/completion/TCN/DTW branch; only isolated shape evidence and direction observations receive new outputs. Order readout remains diagnostic.',evaluation=dict(component=['normal','upper','slow','fast','missing_2','missing_long','reversed_main'],fullpath='60 stored raw public conditions plus9 shape/direction factorial edited channels',broad='all300 historical DEV word auxiliary; not TCN accuracy or fresh independent test'),source_pins={str(p):sha(p) for p in pins},code_pins={str(U/k):sha(U/k) for k in ['core.py','prepare.py','train.py']},guards=['C/T3F and current RTMW TCN/web untouched','No private/user/mentor input','Unknown observations retained in denominator','Synthetic reversal is contrast only; never taught as correct sign','No HTML unless meaningful gain'])
 write(U/'FROZEN.json',cfg);write(U/'CURRENT_EXECUTION.json',dict(status='PREPARED',pid=None));event('prepared',**{k:v for k,v in read(U/'READY.json').items() if k!='rows'})
if __name__=='__main__':main()
