from pathlib import Path
import json,hashlib,cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
M=Path(__file__).resolve().parent;P=M.parent/'word100_learned_guide_feedback_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(n,x):
 p=M/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 rows=read(P/'INPUT_MANIFEST.json')['rows'];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);items=[]
 for w in ['WORD0739','WORD1133','WORD1567','WORD2523']:
  for group in [[1,2,3,4],[5,6,7,8],[9,11,12,13]]:
   selected=[r for r in rows if r['word_id']==w and r['person'] in group]
   canvas=Image.new('RGB',(1800,880),'white');draw=ImageDraw.Draw(canvas)
   for y,r in enumerate(selected):
    centers=[l['center_frame'] for l in r['labels']];start=max(0,min(centers)-12);end=min(r['frames']-1,max(centers)+6)
    ids=np.unique(np.rint(np.linspace(start,end,9)).astype(int));cap=cv2.VideoCapture(r['source']);assert cap.isOpened();images={};i=0
    while True:
     ok,bgr=cap.read()
     if not ok:break
     if i in ids:images[i]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
     i+=1
     if i>end:break
    cap.release();assert len(images)==len(ids)
    for x,fid in enumerate(ids):
     im=Image.fromarray(images[fid]);W,H=im.size;im=im.crop((int(W*.25),0,int(W*.75),int(H*.88)));im.thumbnail((198,190));canvas.paste(im,(x*200+(200-im.width)//2,y*220+26))
     draw.text((x*200+2,y*220+2),f"ID{r['person']:02d} f{fid}",font=font,fill='black')
    items.append(dict(clip_id=r['clip_id'],word_id=w,person=r['person'],role=r['role'],source=r['source'],source_sha256=r['source_sha256'],sampled_frames=ids.tolist(),center_frames=centers))
   path=M/'review'/f'{w}_{group[0]:02d}.jpg';path.parent.mkdir(exist_ok=True);canvas.save(path,quality=94)
 save('REVIEW_MANIFEST.json',dict(rows=items,review='Position/motion contacts, no model outputs, no semantic requiredness labels'))
 save('PLAN.json',dict(status='BEFORE_NEW_MOTION_PREDICTIONS',previous_goal_turn='progress: preserved shape, actual24 video controls, six relation trainings and working8806 demo',
  goal='Learn goal-blind directed physical movements and ordered episodes, attach limited source guide claims without erasing uncertainty.',
  targets=['right_hand_toward_left_wrist_along_arm','hands_move_apart','right_hand_rises_relative_left','right_hand_descends_face_to_torso'],
  inputs='Existing goal-blind physical features + body/arm-relative positions; no goal word, stage ID, source time, or performer ID in observation network.',
  supervision='Public RGB chronological review; specific native frame intervals, direction per event, stable controls. Reversal is physical order control, not automatically wrong sign.',
  conditions=['orderless_pool','ordered_gru'],seeds=[42,43,44],epochs=120,final_checkpoint_only=True,
  fit_people=list(range(1,10)),development_people=[11,12,13],no_private_inputs=True,
  requirements=['Do not reuse rejected coarse13frame motion labels as truth','Keep original timestamps and missing masks','Preserve existing10feature models and learned DTW','Report same-Codex review distinctly from independent human truth'],
  next='Read actual contacts, scope motion windows, verify supervision before training. Actual edited video tests use already extracted or newly extracted public pixels, not fake coordinate restoration.'))
 print('12 chronological public review sheets generated')
if __name__=='__main__':main()
