from core import *
from collections import Counter

def main():
 assert not (U/'FROZEN.json').exists()
 sources=read(J/'SOURCES.json');old={r['clip_id']:r for r in read(S/'READY.json')['rows']}
 drafts=read(O/'word100_shared_feature_bundle_20261027/DRAFT_LABELS.json')['rows']
 extras=[r for r in drafts if r['word_id'] not in WORDS]
 # Additional root RGB review used the original public contact sheets, without predictions.
 # Reuse only concordant visible shape cells. Omit low/resting hands and ambiguous edge-on views.
 review=[];accepted=[]
 for r in extras:
  attrs={};notes=[]
  for name,j in [('R_index_only',0),('R_open_four',1),('L_open_four',2)]:
   val=r['labels'][name]
   if r['status'][name]!='both_visible_agree' or val is None:continue
   if j==2 and r['word_id'] in ['WORD0678','WORD0949','WORD2551']:continue
   # The forward-pointing middle/ring/little configuration is ambiguous in this small view.
   if j==0 and r['word_id']=='WORD0949' and r['ordinal']!=5:continue
   if j==0 and r['word_id']=='WORD1139':continue
   # At the transition (ID02, frame53) only right index, not complete palm, is reliable.
   if j==2 and r['word_id']=='WORD1139' and r['recorded_person_id']==2 and r['frame_id']==53:continue
   attrs[str(j)]=int(val)
  review.append(dict(clip_id=r['clip_id'],frame=r['frame_id'],pts_us=r['pts_us'],image=r['image'],image_sha256=sha(r['image']),attributes=attrs,provenance='prior two-Codex visible agreement + current root full-image review; not expert gold'))
  if attrs:accepted.append(review[-1])
 rows=[];checks=[];positive=Counter();negative=Counter();conflicts=[]
 for src in sources:
  cid=src['clip_id'];recpath=J/'records'/f'{cid}.json';rec=read(recpath);assert rec['status']=='OK';assert sha(rec['channel_path'])==rec['channel_sha256']
  z=np.load(rec['channel_path'],allow_pickle=False);n=len(z['frame_id']);x_full=normalize(z['full']);x_upper=normalize(z['feedback'] if 'feedback' in z.files else z['upper_mask'])
  data=dict(x_full=x_full,x_upper=x_upper,raw_x=z['full'].copy(),frame_id=z['frame_id'].copy(),timestamp_us=z['timestamp_us'].copy(),y=np.zeros((n,7),np.float32),ym=np.zeros((n,7),np.float32),sy=np.full(n,-1,np.int64),sw=np.zeros(n,np.float32),ci=np.full(n,-1,np.int64),stage_gold=np.full(n,-1,np.int64),direct_ym=np.zeros((n,7),np.float32),old_ym=np.zeros((n,7),np.float32),old_y=np.zeros((n,7),np.float32))
  if cid in old:
   q=old[cid];assert sha(q['path'])==q['sha256'];zz=np.load(q['path'],allow_pickle=False)
   assert np.array_equal(zz['frame_id'],data['frame_id'])
   for k in ['y','ym','sy','sw','ci','stage_gold','direct_ym','old_ym','old_y']:data[k]=zz[k].copy()
   assert np.allclose(x_full,zz['x_full'],atol=1e-5)
  for a in [a for a in accepted if a['clip_id']==cid]:
   ix=np.flatnonzero(data['frame_id']==a['frame']);assert len(ix)==1;i=int(ix[0]);assert abs(int(data['timestamp_us'][i])-a['pts_us'])<=2
   for j,v in a['attributes'].items():
    j=int(j)
    if data['ym'][i,j] and data['y'][i,j]!=v:conflicts.append(dict(cid=cid,frame=a['frame'],feature=j));continue
    data['y'][i,j]=v;data['ym'][i,j]=1
    (positive if v else negative)[str(j)]+=1
  assert np.isfinite(x_full).all() and np.isfinite(x_upper).all()
  dst=U/'inputs'/f'{cid}.npz';dst.parent.mkdir(exist_ok=True);np.savez_compressed(dst,**data)
  rows.append(dict(clip_id=cid,word=WORDS.index(src['word_id']) if src['word_id'] in WORDS else -1,word_id=src['word_id'],class_index=src['label_index'],person=src['person'],role=src['role'],source_sha256=src['source_sha256'],path=str(dst),sha256=sha(dst),channel=rec['channel_path'],channel_sha256=rec['channel_sha256'],has_labels=bool(data['ym'].sum()),pairs=old.get(cid,{}).get('pairs',[]),orders=old.get(cid,{}).get('orders',[])))
 assert not conflicts
 fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development']
 assert len(fit)==899 and len(dev)==300 and {r['person'] for r in fit}==set(range(1,10)) and {r['person'] for r in dev}=={11,12,13}
 assert not ({r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev})
 write(U/'ANNOTATION_REVIEW.json',dict(status='VISIBLE_DRAFTS_REVIEWED_FOR_RESEARCH',rows=review,new_positive_cells=dict(positive),new_negative_cells=dict(negative),conflicts=conflicts,not_expert=True,scope='Three shared shape attributes at listed frames only; no extra word-level guide or order correctness assigned',historical_split='ID01/02/03 belonged to older draft hold groups; all are current 100-word FIT. They are not a new independent evaluation set.'))
 write(U/'READY.json',dict(rows=rows,private_used=False,words=100,fit=899,development=300))
 # Same labelled supervision for both conditions isolates the auxiliary vocabulary scope.
 cfg=dict(status='FROZEN_BEFORE_TRAINING_AND_NEW_SCORES',conditions=['labels_only','shared100'],seeds=[42,43,44],primary_seed=42,epochs=24,steps_per_epoch=38,supervised_batch=6,aux_batch=24,lr_base=.0001,lr_head=.001,weight_decay=.01,word_loss_weight=.2,mask_every_steps=4,mask_frames=2,final_epoch_only=True,
  comparison='Both use identical reviewed shape/stage supervision across up to12 expressions. shared100 adds word-label auxiliary loss across899FIT clips; labels_only performs identical extra forward compute but zero auxiliary loss. This is not proof of broad anatomical understanding.',
  phase_labels='Existing weak DTW-transferred stage spans retain weight0.2; direct sparse labels weight1; no new pseudo-stage gold.',
  normal_examples='Only original public temporal order for word/phase learning. No reversed action or omitted event is taught as normal. Missing-sensor masks only; no joint reconstruction targets.',
  split='No model choices from DEV11-13. Fixed architecture/schedule/seeds. Historical DEV, no fresh-independent claim.',
  evaluation=['original_shape_labels','reviewed_shape','semantic_stage','normal_order','word_aux_100','short_hand_missing','long_hand_missing','speed','reversed_main','tail_reappearance_alignment_diagnostic'],
  deploy='Offline candidates only; changed latent cannot be combined with old runtime evidence heads without retraining/full regression.',
  source_pins={str(p):sha(p) for p in [BASE,V/'FEEDBACK_NORMALIZATION.json',S/'READY.json',O/'word100_shared_feature_bundle_20261027/DRAFT_LABELS.json',J/'SOURCES.json',D/'consumer.py',D/'app.js']},
  code_pins={str(U/f):sha(U/f) for f in ['core.py','prepare.py','train.py']},gpu='Installed torch is CPU; RTMW OpenVINO GPU extraction already cached. No new device packages.')
 write(U/'FROZEN.json',cfg)
 write(U/'PRETRAIN_CHECK.json',dict(status='PASS',fit=899,development=300,disjoint_people=True,disjoint_source_sha=True,finite_arrays=True,frame_pts_checked=True,additional_cells=sum(positive.values())+sum(negative.values()),source_pins_unchanged=True,frozen_sha256=sha(U/'FROZEN.json')))
 event('prepared',fit=899,development=300,additional_positive=dict(positive),additional_negative=dict(negative),labelled_fit=sum(r['has_labels'] for r in fit))
if __name__=='__main__':main()
