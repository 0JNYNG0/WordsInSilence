from prepare_review import V,P,read,save
import hashlib
import numpy as np
# Manual image review entries: face / bilateral long-finger convergence / right hand at left arm.
# Each character is one viewed center image W0,W1,W2. '?' is unknown, never a negative.
ENTRIES={
 'WORD0739':{i:('100','000','000') for i in [1,2,3,4,5,6,7,8,9,11,12,13]},
 'WORD1133':{1:('000','000','111'),2:('000','000','111'),3:('000','000','111'),
  4:('000','000','111'),5:('000','000','111'),6:('000','000','110'),
  7:('000','000','111'),8:('000','000','111'),9:('000','000','1?0'),
  11:('000','000','111'),12:('000','000','110'),13:('000','000','111')},
 'WORD1567':{1:('0??','000','000'),2:('0?0','000','000'),3:('010','000','000'),
  4:('01?','000','000'),5:('010','000','000'),6:('010','000','000'),
  7:('000','000','000'),8:('01?','000','000'),9:('0?1','000','000'),
  11:('1?1','000','000'),12:('010','000','000'),13:('011','000','000')},
 'WORD2523':{1:('100','100','000'),2:('000','000','000'),3:('110','100','000'),
  4:('100','100','000'),5:('100','100','000'),6:('100','100','000'),
  7:('100','100','000'),8:('100','100','000'),9:('100','100','000'),
  11:('?00','100','000'),12:('100','100','000'),13:('100','100','000')}
}
def main():
 rows=[]
 for r in read(P/'INPUT_MANIFEST.json')['rows']:
  entries=ENTRIES[r['word_id']][r['person']]
  for k,l in enumerate(r['labels']):
   values=[e[k] for e in entries]
   rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],role=r['role'],
    center_frame=l['center_frame'],timestamp_us=l['center_timestamp_us'],image=l['center_image'],
    image_sha256=hashlib.sha256(__import__('pathlib').Path(l['center_image']).read_bytes()).hexdigest(),
    y=[int(x) if x!='?' else 0 for x in values],mask=[x!='?' for x in values],
    provenance='Codex direct saved RGB center review before relation-model predictions; no independent human/expert verification'))
 definitions=read(V/'FROZEN_RELATION_PLAN.json')
 definitions.update(definition_clarification_before_training=[
  'Bilateral upper ends means long-finger ends approaching a common meeting point; stacked separated thumbs are not this relation.',
  'Right hand at left arm includes upper arm/elbow/forearm as a broad physical relation; endpoint, touch and stroke correctness remain unsupported.',
  'Face boundary cases near chin have unknown targets. The left hand near face cannot label the right hand positive.'],
  limits='Relation recognition, not correctness, movement direction, exact repetition count, or full guide comprehension.',
  same_word_negatives_required=True,training_note='Some positives occur mainly within one word; feature model has no word input but appearance shortcut risk remains.')
 save('TRAINING_CONTRACT.json',definitions)
 save('RELATION_TARGETS.json',dict(targets=definitions['targets'],rows=rows,reviewed_images=144,
  labels_are_independent_human_truth=False,annotation_model_predictions_seen=False,
  notes=['WORD2523 ID02 sampled centers already separated: all convergence negatives, not copied from the word label.',
   'WORD1133 ID06/09/12 later center lowering or separated: no arm relation assumed across whole word.',
   'WORD1567 face relation follows anatomical right hand, not screen position or high-hand identity.']))
 counts={}
 for role in ['fit','development']:
  rr=[r for r in rows if r['role']==role];y=np.array([r['y'] for r in rr]);m=np.array([r['mask'] for r in rr])
  counts[role]=dict(images=len(rr),positive=(y*m).sum(0).tolist(),negative=((1-y)*m).sum(0).tolist(),unknown=(~m).sum(0).tolist())
 save('LABEL_COUNTS.json',counts);print(counts)
if __name__=='__main__':main()
