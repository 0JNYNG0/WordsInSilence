from common_split import *
from collections import Counter
v=read(U/'LABELS_DRAFT.json');rows=v['rows'];G=[1]*4;E=[1]*4;I=[1,0,0,0];F=[0]*4
for r in rows:
 if r['role']=='confirmation_reserved'and r['word']=='WORD2700'and r['index']in [7,8,9]:
  r['note']='Full source-resolution RGB inspected. Curled little tip is occluded; cannot distinguish precise own-chain placement from ring-base association. Quality remains unknown; no forced error label.'
for r in read(U/'controls_REVIEW_INPUTS.json')['rows']:
 q=G if r['hand']=='R'or(r['person']==2 and r['index']==3)else [-1]*4
 rows.append(r|dict(quality=q,shape=E,note='Parallel open control. Edge-on left individual chains unresolved except early visible palm.',annotator='Codex RGB/overlay review'))
def add(group,p,ids,quality,shape):
 src=read(Q/f'{group}_REVIEW_INPUTS.json')['rows']
 for k in ids:
  r=next(r for r in src if r['person']==p and r['index']==k)
  rows.append(r|dict(quality=quality,shape=shape,note='Additional new posture labels on existing public FIT source RGB; no diagnostic fitting',annotator='Codex RGB/overlay review'))
add('fit_left',3,[6],G,E);add('fit_left',3,[7,8,9,10,11],G,I)
add('fit_left',6,[6,7,8,9,10],G,I);add('fit_left',6,[11],G,F)
add('fit_left',9,[7,8],G,E);add('fit_left',9,[9,10,11],G,I)
add('fit_front',3,[6,7,8,9],G,I);add('fit_front',6,[6,7,8,9,10],G,I)
add('fit_front',9,[6,7,9],G,I);add('fit_front',9,[8],[0,1,1,1],I)
for r in rows:
 assert (1<=r['person']<=9)if r['role']=='fit'else(r['person']==10 if r['role']=='diagnostic'else r['person']==16)
 for key,path in [('source_sha256','path'),('raw_sha256','raw_path'),('channel_sha256','channel_path')]:assert sha(r[path])==r[key]
 assert len(r['quality'])==len(r['shape'])==4
 assert all(a in [-1,0,1]for a in r['quality']+r['shape'])
keys=[(r['clip_id'],r['hand'],r['frame'])for r in rows];assert len(set(keys))==len(keys)
v['rows']=rows;v['review_status']='All listed images inspected, before new performer model predictions';write(U/'LABELS.json',v)
summary={role:dict(anchors=len(rr),clips=len({r['clip_id']for r in rr}),shape_known=sum(a>=0 for r in rr for a in r['shape']),quality_good=sum(a==1 for r in rr for a in r['quality']),quality_bad=sum(a==0 for r in rr for a in r['quality']),quality_unknown=sum(a<0 for r in rr for a in r['quality']))for role in ['fit','diagnostic','confirmation_reserved']for rr in [[r for r in rows if r['role']==role]]}
write(U/'LABEL_VERIFIED.json',dict(summary=summary,source_hashes_checked=True,annotator='Codex, no independent human expert',candidate_predictions_not_yet_run=True,quality_holdout_positive_support=summary['confirmation_reserved']['quality_bad']))
write(U/'PLAN_CLARIFICATION.json',dict(reason='No confirmed corrupt-chain positive in reserved16. Do not invent a quality label. This was found before candidate training or diagnostic predictions.',change='Proceed with requested shape fine-tuning and research quality branch separately. Quality branch cannot be promoted without reserved positive support. A shape-only candidate may be assessed independently against all posture and replay gates.',unchanged=['No reserve tuning','No private footage','C/T3F/common100/base DTW preserved','Normal regression gates and source unknowns retained']))
print(json.dumps(summary))
