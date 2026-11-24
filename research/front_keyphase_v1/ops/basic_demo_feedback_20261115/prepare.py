"""Read-only public source review; private captures never enter this unit."""
from pathlib import Path
import json, hashlib, shutil
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

U=Path(__file__).resolve().parent; O=U.parent; R=O.parents[2]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 for d in ['before','review','runs','results']: (U/d).mkdir(exist_ok=True)
 if not (U/'PLAN.json').exists():
  for n in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md']: shutil.copy2(R/n,U/'before'/n)
  shutil.copy2(O/'STATE.json',U/'before/STATE.json')
  pins=read(O/'word100_game_acceptance_20261115/PROTECTED.json')
  write(U/'PREVIOUS_PROTECTED_FORMAT.json',{'type':type(pins).__name__})
  write(U/'PLAN.json',dict(goal='Basic game vocabulary: observed actions, localized correction, completion and game progression',
   actor='Codex implementation and separate verification; not independent KSL expert',
   graph=['F2 source/guide/action truth','F2 supervised phase/shape/relation candidate','F2 held-person normal/error/partial/occlusion comparison','F3 only admitted completion and correction -> quest','F4 user-started capture','F5 working demonstration evidence'],
   scope={'gratitude':'WORD1290 first, existing100 recognition; new fine-grained labels required','apple':'WORD0099 existing extra classifier unadmitted; person05 variant unresolved','hello':'SEN0354 needs adequate source; never replace with NIKL 인사'},
   protected={str(p):sha(p) for p in [O/'word100_live_g6_20261114/engine.py',O/'word100_rtmw_joint_learning_20261108/tcn_runs/finetune_C_20261013/model.pt',O/'word100_body_boundary_20261017/training_C_20261013_best.pt',O/'word100_frozen_c_temporal_20261019/T3F_20261013_RESULT.json']},
   split={'fit':list(range(1,10)),'development':[11,12,13],'reserved':[10,14,15,16,17,18]},
   contract=['No private recordings, downloads or cameras','Source timestamps and native action durations retained','No word-specific hand thresholds','No score or repeated count claims without public evidence','Natural public correctness versus engineered diagnostic errors reported separately','Full100 observer retained; no forced matching to requested action','No live promotion merely because TCN name matches'],
   training='Freeze architecture and comparison before results, keep all failed outputs; only source-reviewed observable labels'))
 rows=[r for r in read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows'] if r['word']=='WORD1290']
 for r in rows:
  for field in ['source','raw','channels']:
   p=r['video' if field=='source' else field];assert sha(p)==r[field+'_sha256']
 write(U/'INPUTS.json',dict(rows=rows,scope='public gratitude, previous split retained; reserved not opened'))
 font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
 for r in rows:
  z=np.load(r['channels']); t=z['timestamp_us']/1e6; lo,hi=r['source_interval']
  a=max(0,np.searchsorted(t,lo)-6); b=min(len(t)-1,np.searchsorted(t,hi)+6)
  # Dense source contact sheet, three consecutive native frames apart.
  frames=list(range(a,b+1,3)); pages=[frames[i:i+24] for i in range(0,len(frames),24)]
  cap=cv2.VideoCapture(r['video'])
  for page,ff in enumerate(pages):
   out=U/'review'/f'{r["person"]:02}_{page}.jpg'
   if out.exists():continue
   im=Image.new('RGB',(1440,270*((len(ff)+5)//6)),'white');d=ImageDraw.Draw(im)
   for k,f in enumerate(ff):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,bgr=cap.read();assert ok
    h,w=bgr.shape[:2];cut=bgr[int(h*.15):int(h*.65),int(w*.28):int(w*.70)]
    crop=Image.fromarray(cv2.cvtColor(cut,cv2.COLOR_BGR2RGB));crop.thumbnail((236,238))
    x=k%6*240;y=k//6*270;im.paste(crop,(x,y));d.text((x+3,y+239),f'P{r["person"]:02} f{f} / {t[f]:.2f}s',font=font,fill='#152c46')
   im.save(out,quality=94)
  cap.release()
 print('PUBLIC REVIEW READY',len(rows))
if __name__=='__main__':main()
