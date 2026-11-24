"""Record the completed RGB review, before any learner results are inspected."""
from review_support import *
def main():
 assert not (U/'ADMISSION.json').exists()
 inp=read(U/'INPUTS.json');rows=[]
 proto={'WORD2633':[0,1,1,0,0],'WORD2635':[0,1,1,1,1],'WORD2637':[1,1,0,0,0]}
 for r in inp['rows']:
  if r['word'] in proto:
   # All 35 clips' three original RGB hand crops were explicitly viewed.
   for f in r['anchors']:
    labels=proto[r['word']].copy();note='Extended vs folded coarse state visible; no exact joint angles/contact asserted.'
    if r['word']=='WORD2637' and r['person']==8:
     labels[3:]=[-1,-1];note='Thumb/index extension and middle flexion visible. Ring/little individually obscured; masked.'
    rows.append(dict(clip_id=r['clip_id'],word=r['word'],person=r['person'],role=r['role'],frame=f,hand='right',finger_states=labels,shape_class=list(proto).index(r['word']),review_note=note,source_sha256=r['source_sha256'],channels_sha256=r['channels_sha256']))
 excluded=['NIA_SL_WORD0099_REAL05_F']
 write(U/'ADMISSION.json',dict(status='LIMITED_RGB_ANCHOR_ADMISSION',reviewer='Codex direct original RGB/crop review; not independent KSL expert',finger_order=['thumb','index','middle','ring','little'],states={'1':'extended','0':'folded','-1':'not resolved'},rows=rows,
  classifier_admitted=[r['clip_id'] for r in inp['rows'] if r['word']!='WORD1290' and r['clip_id'] not in excluded],
  excluded_semantic_variants=[dict(clip_id=excluded[0],reason='Sampled action differs from common lip/index then opposite-hand wipe. Fruit/apple meaning not independently confirmed; not labeled incorrect.')],
  source_failures=inp['failures'],
  limitations=['No labels propagated to unsampled frames.','Three reviewed moments are not full hold interval boundaries.','All three shapes extend the index; no index-fold contrast is learned.','Right-hand/front-view scope; no bilateral/occlusion transfer proof.','Numeral six and character ㄴ are not merged.']))
 write(U/'LEARNING_CONTRACT.json',dict(original_freeze_sha256=hashlib.sha256((U/'FROZEN.json').read_bytes()).hexdigest(),
  before_any_learning=True,shape=dict(conditions=['class_only','class_plus_fingers'],seeds=[42,43,44],steps=600,hidden=32,lr=.001,optimizer='Adam',batch='all FIT anchors',finger_loss_weight=1.0,class_loss_weight=1.0,normalization='FIT mean/std only',input='right hand local xy42 + point mask21; no wordID, target or person',confidence=.85,primary_unit='9 development videos; 27 correlated anchors secondary',selection='none; all fixed seeds reported'),
  classifier=dict(seed=42,epochs=80,lr=.003,optimizer='Adam',batch_size=64,views='full only, both existing and added classes',new_classes=['WORD2633','WORD2635','WORD2637','WORD0099'],method='frozen backbone and old100 linear logits, appended four trained rows',replay='all original FIT full-view features; 34 new source-admitted FIT clips',regression='all300 original development full view, plus12 new development; no DEV threshold/rule/model selection'),
  adoption_gate='Not automatic. No overall feedback improvement claim from numeral probe; existing100 top1/top3 regression blocks classifier promotion.',
  protection='C/T3-F/current web/100 common feedback learning unchanged',
  direct_user_answer='Guide structured attributes are useful supervision candidates, not invisible-finger truth. This ablation checks limited auxiliary supervision only.'))
 print('ADMITTED',len(rows),'anchors',len({r['clip_id'] for r in rows}),'videos')
if __name__=='__main__':main()
