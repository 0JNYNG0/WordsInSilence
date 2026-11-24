from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,random
import cv2,numpy as np
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_shubert_feasibility_20261028'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(U/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 if (U/'FROZEN.json').exists():raise RuntimeError('Frozen preparation already exists')
 m=read(P/'inputs/MANIFEST.json');old={r['alias']:r for r in read(O/'word100_shared_feature_bundle_20261027/visual_inputs/INDEX.json')['rows']}
 order=list(m['rows']);random.Random(20261028).shuffle(order)
 rubric={'purpose':'Blind AI visual observation drafts, not independent human truth or sign correctness.',
  'instructions':'Read center image for static fields. Use before/center/after only for motion. No word name, model scores, guides, prior labels. Preserve uncertain as null. Anatomical R/L use visible arm connections; never assume screen halves when crossing.',
  'fields':['R_index_only','L_index_only','R_open_four','L_open_four','R_face_overlap','L_face_overlap','hands_overlap','R_vertical','L_vertical','hands_separation'],
  'binary':'1 visible support, 0 visible evidence against, null hidden/ambiguous. Not open_four is not necessarily fist.',
  'index_only':'Index visibly broadly extended, middle/ring/little visibly bent. Thumb unconstrained. Foreshortened/hidden finger=null.',
  'open_four':'Four nonthumb fingers visibly broadly extended, together or spread. Slight natural bends allowed.',
  'face_overlap':'Hand projection overlaps actual face forehead-to-chin, excludes hair/merely nearby. 2D overlap is not physical touch.',
  'hands_overlap':'Both hands identifiable and their visible projections overlap/touch in 2D, not verified contact.',
  'vertical':'up/down/still/null. Clear wrist/hand center movement over shown time window, image upward=up. If direction reverses, hidden, or not determinable use null; still only adequately visible and stable.',
  'hands_separation':'closer/farther/still/null based on visible hand-center separation over window. Reversal or uncertain identity=null.',
  'output':'JSON {reviewer, scope, rows:[{sample_id, values:[10 values in fields order], note:brief optional}], images_viewed:[...]}',
  'scope_limits':['No depth contact','No exact repetition count','No linguistic allowed/error judgement','No dense semantic stage boundaries']}
 frozen={'at':datetime.now(timezone.utc).isoformat(),'authorization':'User asks immediate KSL handshape/stage comparison; keep C/T3F and private recordings untouched.',
  'scope':'Exploratory blinded AI visual agreement and reference correspondence, not educational accuracy. No new training.',
  'input':'same native fullframe observations for all conditions; source30fps cached SHuBERT and DINO reused',
  'selection':'V01~V12 fixed prior trial; all9 anchors at10,20,...90% original frame count; context +/-6 original frames',
  'reference_ids':[1,2],'query_ids':[3],'guide_not_input':True,'training':False,
  'shape_methods':['native finger-angle mean absolute difference','per-hand DINO cosine','whole-body SHuBERT lastFFN cosine'],
  'shape_task':'same anatomical hand, other-word references only; reference/current observed shape available for both. 1NN no fitted threshold. Class-support assessed separately.',
  'stage_methods':['native previous69-feature grouped local cost + DTW','SHuBERT cosine + same DTW','normalized-time index alignment'],
  'stage_task':'each same-word ID03 query vs ID01/02 reference separately,9-anchor monotonic DTW; no invented stage labels. Compare visible features and motion at matched anchors.',
  'dtw':'steps diagonal/queryadvance/refadvance, tie priority diagonal; query-to-ref median mapping; every9 reference anchors retained',
  'missing':'reference requirements/counts never shrink when query missing; copied/blank crops marked not current. For shape NN allow only current query/ref crops; retain all excluded slots and denominators.',
  'evaluation':'2 new blind Codex RGB drafts. Equal nonnull labels become provisional consensus; disagreements unknown; no human truth claim. Report each method on common assessable support and all unknown outcomes. No p-value or independent-frame sample claims.',
  'no_tuning':'no changing weights/threshold/anchors from results; adverse results retained; no selected best model adoption',
  'hardware':'local inference speed depends on hardware/runtime; GPU candidate not executed in this unit',
  'source_pins':{str(P/'inputs/MANIFEST.json'):sha(P/'inputs/MANIFEST.json'),str(P/'PUBLIC_FEATURE_RESULT.json'):sha(P/'PUBLIC_FEATURE_RESULT.json'),str(O/'word100_guide_video_feedback_20261028/motion_observer.py'):sha(O/'word100_guide_video_feedback_20261028/motion_observer.py'),str(O/'word100_guide_video_feedback_20261028/compare_pipeline.py'):sha(O/'word100_guide_video_feedback_20261028/compare_pipeline.py')},
  'rows':[]}
 # Freeze selection and rubric before feature distances or visual review.
 for i,r in enumerate(order):
  n=next(z['frames'] for z in m['summary'] if z['alias']==r['alias'])
  anchors=[int(round((n-1)*k/10)) for k in range(1,10)]
  frozen['rows'].append({'blind_clip':f'P{i+1:02d}',**r,'frames':n,'anchors':anchors,'display_bbox':old[r['alias']]['display_only_crop_xyxy'],'annotation_intervals':old[r['alias']]['annotation_intervals']})
 save('FROZEN.json',frozen);save('RUBRIC.json',rubric)
 (U/'blind').mkdir(exist_ok=False);samples=[];sheets=[]
 for r in frozen['rows']:
  folder=P/'inputs'/r['alias'];done=read(folder/'DONE.json');assert sha(r['source_video'])==done['source_video_sha256']
  cap=cv2.VideoCapture(r['source_video']);wanted={max(0,min(r['frames']-1,a+d)) for a in r['anchors'] for d in [-6,0,6]};frames={};idx=0
  while True:
   ok,im=cap.read()
   if not ok:break
   if idx in wanted:frames[idx]=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
   idx+=1
  cap.release();assert idx==r['frames'] and set(frames)==wanted
  x0,y0,x1,y1=r['display_bbox']
  for part in range(3):
   sheet=Image.new('RGB',(1080,1575),'white');draw=ImageDraw.Draw(sheet)
   for local in range(3):
    k=part*3+local;a=r['anchors'][k];sid=f"{r['blind_clip']}_{k+1:02d}"
    inds=[max(0,min(r['frames']-1,a+d)) for d in [-6,0,6]]
    for col,fi in enumerate(inds):
     crop=Image.fromarray(frames[fi][y0:y1,x0:x1]);crop.thumbnail((352,492))
     px=col*360+(360-crop.width)//2;py=local*525+28;sheet.paste(crop,(px,py))
     draw.text((col*360+5,local*525+5),f'{sid} | {[-0.2,0,0.2][col]:+.1f}s',fill='black')
    samples.append({'sample_id':sid,'alias':r['alias'],'frame_index':a,'context_indices':inds,'person':r['recorded_person_id'],'word_id':r['word_id']})
   filename=f"blind/{r['blind_clip']}_{part+1}.jpg";sheet.save(U/filename,quality=93);sheets.append(filename)
 save('SAMPLES.json',{'samples':samples,'sheets':sheets,'rubric_sha256':sha(U/'RUBRIC.json'),'frozen_sha256':sha(U/'FROZEN.json')})
 save('CURRENT_EXECUTION.json',{'stage':'BLIND_RGB_REVIEW_PENDING','samples':108,'sheets':36,'training_runs':0,'private_access':False})
 print('Prepared',len(samples),'samples',len(sheets),'blind sheets')
if __name__=='__main__':main()
