import json, hashlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 out=U/'hand_review';out.mkdir(exist_ok=True)
 font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
 for word in ['WORD2633','WORD2635','WORD2637']:
  rows=sorted([read(p) for p in (U/'records').glob(word.replace('WORD','*WORD')+'*.json')],key=lambda r:r['person'])
  for offset in range(0,len(rows),3):
   rr=rows[offset:offset+3];imout=Image.new('RGB',(660,220*len(rr)),'white');d=ImageDraw.Draw(imout)
   for i,r in enumerate(rr):
    raw=np.load(r['raw']);cap=cv2.VideoCapture(r['video'])
    for j,f in enumerate(r['anchors']):
     cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
     pts=raw['xy_px'][f,112:133];h,w=im.shape[:2]
     lo=pts.min(0)-35;hi=pts.max(0)+35
     x0,y0=np.maximum(lo,0).astype(int);x1,y1=np.minimum(hi,[w,h]).astype(int)
     crop=Image.fromarray(cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));crop.thumbnail((210,185))
     imout.paste(crop,(j*220+(220-crop.width)//2,i*220))
     d.text((j*220+3,i*220+190),f'{word} ID{r["person"]:02} f{f}',fill='#182d48',font=font)
    cap.release()
   imout.save(out/f'{word}_{offset//3}.jpg',quality=95)
 write(U/'SOURCE_REVIEW.json',{
  'checked':'2026-11-13','reviewer':'Codex source/RGB review, not independent KSL expert',
  'official_example':{'entry':'ㄴ','url':'https://kli.korean.go.kr/term/trgtWord/indexTrgtWord.do?trgtWordNo=2000013','quote':'오른 주먹의 1·5지를 펴서 1지 끝이 왼쪽으로 손등이 밖으로 향하게 한다.',
   'derived_expected_attributes':{'hand':'right','extended':['index','thumb'],'index_direction':'signer-left','dorsal_orientation':'outward'},
   'purpose':'Demonstrates guide structure. NOT the dictionary label for numeral six; NOT admitted as video training data.'},
  'numbering_caution':'Legacy dictionary digit numbers are not MediaPipe point indices or anatomical digit numbering. Preserve original wording and named fingers separately.',
  'research':[{'url':'https://aclanthology.org/2023.findings-emnlp.198/','finding':'Handshape-supervised SLR beats its baseline on PHOENIX14T-HS. Supports auxiliary targets, not KSL feedback effectiveness proof.'},{'url':'https://aclanthology.org/2025.naacl-long.19/','finding':'ASL translation gains from character-level tokenization; mixing fingerspelling data has mixed effects. More static examples alone not a guarantee.'}],
  'pipeline':['dictionary expected named attributes','RGB-confirmed observed anchors and masks','pose-only supervised model predicts observed attributes','target guide compares evidence; invisible fingers unknown'],
  'not_claimed':['word text alone produces visual understanding','numeric 2/4/6 covers all finger combinations','static numbers teach closing motion or order','dictionary text gives frame boundaries or visibility'],
  'ablation_required':'Same pose architecture/data, class-only CE vs CE+masked finger supervision, fixed seeds and split; development not used for selection.',
  'downloaded_new_model_or_data':False})
 write(U/'CURRENT_EXECUTION.json',{'status':'SOURCE_PREPARATION_AND_DICTIONARY_REVIEW','prepare_session':55932,'prepare_resume':'Completed records reused, one original container invalid without moov; failure kept in denominator.','unity_build':'exit198: no valid Unity Editor license; no build produced','next':'Finish source review; admit RGB anchors; compare fixed isolated heads and appended classifier. Existing runtime unchanged.','protected_prior_research':'C7 not replaced or declared passed'})
if __name__=='__main__':main()
