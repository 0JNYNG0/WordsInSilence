from pathlib import Path
import json,hashlib,collections,datetime
V=Path(__file__).resolve().parent;U=V.parent/'word100_core_feature_split_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def main():
 d=read(U/'DATASET.json');rows={r['sample_id']:r for r in d['rows']};fit=[r for r in d['rows'] if r['role']=='fit']
 mandatory={'S002_W1':['crossing'],'S005_W1':['crossing'],'N008_W1':['crossing'],
  'S001_W2':['hold'],'S004_W2':['hold'],'N003_W1':['hold'],
  'S011_W1':['rotation'],'S024_W2':['rotation','boundary'],
  'S017_W2':['boundary','short_change'],'N047_W1':['short_change'],
  'N007_W1':['short_change','crossing'],'S016_W1':['slow_translation']}
 assert all(rows[k]['role']=='fit' for k in mandatory)
 selected=list(mandatory);used={rows[k]['clip_id'] for k in selected}
 for person in (1,2,3):
  candidates=sorted([r for r in fit if r['person']==person and r['clip_id'] not in used],key=lambda r:hashlib.sha256(('center_motion_pilot_v1|'+r['sample_id']).encode()).hexdigest())
  chosen=0
  for r in candidates:
   if r['clip_id'] in used:continue
   selected.append(r['sample_id']);used.add(r['clip_id']);chosen+=1
   if chosen==4:break
  assert chosen==4
 assert len(selected)==24 and len(set(selected))==24
 chosen=sorted([rows[k] for k in selected],key=lambda r:hashlib.sha256(('blind_order|'+r['sample_id']).encode()).hexdigest())
 private=[];public=[]
 for i,r in enumerate(chosen,1):
  alias=f'P{i:03d}';p=dict(alias=alias,sample_id=r['sample_id'],frame_ids=r['frame_ids'],timestamps_us=r['timestamps_us'],center_frame=r['center_frame'],center_timestamp_us=r['center_timestamp_us'],support_frames=r['frame_ids'][4:9],image=r['image'],image_sha256=sha(r['image']))
  public.append(p);private.append(dict(p,clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],role=r['role'],strata=mandatory.get(r['sample_id'],['hash_ordinary']),selection='pre-score fit-note challenge' if r['sample_id'] in mandatory else 'deterministic hash ordinary'))
 (V/'review').mkdir(exist_ok=True)
 write(V/'PILOT_MANIFEST.json',dict(status='FROZEN_BEFORE_PILOT_READING',rows=private,source=str(U/'DATASET.json'),source_sha256=sha(U/'DATASET.json'),people=dict(collections.Counter(r['person'] for r in private)),words=len({r['word_id'] for r in private}),selection='12 balanced named physical challenges using fit notes, plus12 hash-ranked ordinary windows with no duplicate ordinary clip; no model scores, no dev/eval images',coverage_strata_not_ground_truth=True))
 write(V/'BLIND_PILOT.json',dict(status='READ_IMAGES_AND_RUBRIC_ONLY',rows=public,word_role_oldlabel_scores_hidden=True))
 write(V/'CURRENT_EXECUTION.json',dict(stage='PROTOCOL_PRE_ANNOTATION_REVIEW',active_execution=None,training_started=False,pilot_windows=24,all_fit=True))
 print(json.dumps({'windows':24,'people':dict(collections.Counter(r['person'] for r in private)),'words':len({r['word_id'] for r in private}),'input_reextraction':0}))
if __name__=='__main__':main()
