from pathlib import Path
import json,hashlib,collections,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
S=O/'word100_shared_motion_learning_20261109';B=O/'word100_broad_factorized_learning_20261110';T=O/'word100_handshape_transfer_20261113'
images={}
for r in read(S/'ANNOTATION_REVIEW.json')['rows']:images[r['clip_id'],r['frame']]=r['image']
for r in read(B/'REVIEW_NEW_INPUTS.json'):images[r['clip_id'],r['frame_id']]=r['source']
for r in read(S/'EXTRA_EVAL_LABELS.json')['rows']:
 if 'image'in r:images[r['clip_id'],r['frame']]=r['image']
mode='development'if '--development'in sys.argv else'fit';prefix='dev_'if mode=='development'else''
rows=read(T/'INPUT_MANIFEST.json')['ordinary'];selected=[];seen=set()
for r in rows:
 key=r['word'],r['person'],r['side']
 if r['role']!=mode or r['person']not in ([11,12,13]if prefix else[1,2,3])or key in seen:continue
 if (r['clip_id'],r['frame'])not in images:continue
 if not any(v==1 for v in r['y']) or r['word']=='WORD1133':selected.append(r);seen.add(key)
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16);review=[]
for n,r in enumerate(selected):
 p=Path(images[r['clip_id'],r['frame']]);im=Image.open(p).convert('RGB');rec=read(O/'word100_rtmw_joint_learning_20261108/records'/f"{r['clip_id']}.json");z=np.load(rec['raw_path']);ix=int(np.flatnonzero(z['frame_id']==r['frame'])[0]);sl=slice(112,133)if r['side']=='R'else slice(91,112);xy=z['xy_px'][ix,sl].copy();xy[:,0]*=im.width/int(z['source_width']);xy[:,1]*=im.height/int(z['source_height']);mid=xy.mean(0);rad=max(40,float(np.ptp(xy,axis=0).max())*.85);box=tuple(np.r_[np.maximum(mid-rad,0),np.minimum(mid+rad,[im.width,im.height])].astype(int));crop=im.crop(box);crop.thumbnail((270,235))
 if n%12==0:board=Image.new('RGB',(1120,870),'white');draw=ImageDraw.Draw(board)
 k=n%12;col=k%4;row=k//4;board.paste(crop,(col*280,row*290+52));draw.text((col*280+4,row*290+3),f"{n}: {r['word']} ID{r['person']} {r['side']}\nf{r['frame']} old:{r['y']}",font=font,fill='#12243c')
 review.append(dict(index=n,**r,image=str(p),image_sha256=sha(p),crop_box=[int(v)for v in box],reviewer='pending'))
 if n%12==11 or n==len(selected)-1:board.save(U/f'{prefix}review_{n//12:02}.jpg',quality=96)
write(U/f'{prefix}RGB_REVIEW_INPUTS.json',dict(rows=review,selection='First existing ambiguous sparse frame per word/person/side; no model-score selection',split=mode,independent_expert=False))
if prefix:print('development review',len(selected));sys.exit(0)
cards=read(O/'word100_action_feedback_demo_20261108/CARDS.json');families=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words'];out=[]
finger=['thumb','index','middle','ring','little']
for word,cc in cards['words'].items():
 for c in cc:
  if not c.get('required',True):continue
  req=[]
  for k,v in c['features'].items():
   j=int(k)
   if v!=1 or j not in[1,2,3,4]:continue
   hand='R'if j in[1,3]else'L';state='extended'if j in[1,2]else'flexed'
   req.extend(dict(hand=hand,finger=f,expected=state,scope=c.get('shape_scope','action'),source_feature=j)for f in finger[1:])
  f=next(f for f in families[word]if f['id']==c['id'])
  # Preserve source descriptions separately; canonical group IDs are global, not word-local.
  guides=[g for g in cards['guides']if g['word_id']==word]
  out.append(dict(word=word,action_id=c['id'],display_instruction=c['instruction'],instruction_origin='existing reviewed project guide, not verbatim official sentence',
    common_families=f['families'],observed_motion_axis=f.get('axis'),expected_direction=f.get('sign'),hands=c['hands'],finger_requirements=req,
    shape_scope=c.get('shape_scope','action'),source_description_context=guides,
    bindings='All simultaneous finger/body/movement clauses belong to this action; sequence is between actions; no guessed new action per comma',
    automatic_alignment=True,raw_sentence_language_model=False,unsupported=c.get('unverified')or c.get('coverage_message'),
    forbid=['Do not infer thumb from fist label','Do not infer individual finger from four_open=0','Do not locate action by requiring correct shape/direction','Do not claim exact contact from 2D proximity']))
write(U/'CLAUSE_CONTRACT.json',dict(actions=out,source_cards_sha256=sha(O/'word100_action_feedback_demo_20261108/CARDS.json'),whole_sentence_autonomous_understanding=False))
print(json.dumps(dict(review_frames=len(selected),sheets=(len(selected)+11)//12,actions=len(out),finger_requirements=sum(len(a['finger_requirements'])for a in out))))
