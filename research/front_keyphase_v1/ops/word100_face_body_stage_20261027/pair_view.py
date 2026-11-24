"""Public paired-view illustrations, annotated middle frames fixed independently of scores."""
import sys,json,html
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[4];LAB=ROOT/'research/front_keyphase_v1';U=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB/'src'))
from wis_lab.core.observation import Observation
from wis_lab.capture.face_landmarks import join_face_observation
from wis_lab.core.face_body_relations import region_geometry
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
F=read(U/'FROZEN.json');font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
colors=['#f5ad42','#bf8bff','#73b5ff','#f2769e','#70d3c1','#d5dd83','#d9dce4','#84bdaa'];outputs=[]
for code in ('WORD0949','WORD2551','WORD1290'):
 q=next(r for r in F['rows'] if r['word']==code and r['signer']==2);ref=next(r for r in F['rows'] if r['word']==code and r['signer']==1)
 qo=Observation.load(q['observation']);target=int(np.argmin(abs(qo.timestamp_us-np.mean(q['intervals'][0])*1e6)))
 with np.load(U/'data'/q['clip_id']/'cross_paths.npz') as z:
  rframes={c:int(np.median(z[c+'_reference'][np.abs(z[c+'_query']-target)==np.abs(z[c+'_query']-target).min()])) for c in 'AB'}
 canvas=Image.new('RGB',(1440,725),'#101923');draw=ImageDraw.Draw(canvas)
 draw.text((20,12),q['label']+' · 같은 공개 비교 시점에 연결된 참고 장면',font=font,fill='white')
 for col,(row,t,title) in enumerate(((q,target,'비교 영상 · 공개 ID02'),(ref,rframes['A'],'A · 범위가 달랐던 참고'),(ref,rframes['B'],'B · 상반신 범위를 맞춘 참고'))):
  p=U/'data'/row['clip_id'];d=read(p/'DONE.json');obs=Observation.load(str(p/'masked.npz'));face=join_face_observation(p/'face_roi.npz',obs)
  cap=cv2.VideoCapture(d['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,bgr=cap.read();assert ok;pts=cap.get(cv2.CAP_PROP_POS_MSEC);cap.release()
  assert abs(pts*1000-obs.timestamp_us[t])<1000
  im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));di=ImageDraw.Draw(im)
  centers,axes,radii,valid=region_geometry(face['xy_px'][t],obs.xy_px[t],obs.usable[t])
  for j,(x,y) in enumerate(centers):
   if valid[j]:di.ellipse((x-5,y-5,x+5,y+5),fill=colors[j])
  for side,base in enumerate((33,54)):
   for k in range(21):
    if obs.usable[t,base+k]:
     x,y=obs.xy_px[t,base+k];di.ellipse((x-3,y-3,x+3,y+3),fill=('#6da8ff','#ffbd67')[side])
  x=col*480+10;draw.text((x,58),title,font=font,fill='white')
  draw.text((x,91),f'{obs.timestamp_us[t]/1e6:.3f}초 · 프레임 {t}',font=small,fill='#b1c2d5')
  canvas.paste(im.resize((460,259)),(x,125))
  fxy=face['xy_px'][t];lo=np.floor(np.nanmin(fxy,axis=0)-45).astype(int);hi=np.ceil(np.nanmax(fxy,axis=0)+45).astype(int)
  crop=im.crop((max(0,lo[0]),max(0,lo[1]),min(im.width,hi[0]),min(im.height,hi[1])));crop.thumbnail((265,280));canvas.paste(crop,(x+90,400))
 draw.text((20,695),'주석 동작 구간의 가운데를 선택한 예시 · 정확한 접촉/세부 동작 일치를 보증하지 않음',font=small,fill='#b1c2d5')
 path='images/paired_'+code+'.jpg';canvas.save(U/path,quality=90);outputs.append(dict(word=code,label=q['label'],query_frame=target,reference_frames=rframes,path=path))
(U/'PAIRED_VIEW.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8')
p=U/'RESULT_KO.html';s=p.read_text(encoding='utf-8')
block='<section id="paired"><h2>같은 시점에서 얼굴 부위 관계도 함께 확인</h2><p>공개 ID02의 주석 동작 가운데 시점을 고정하고, A와 B가 ID01의 어느 장면을 연결했는지 나란히 표시했습니다. 얼굴 부위 정보는 현재 <b>정렬 후 비교 근거</b>로 연결했습니다. 시간 정렬 비용에 얼굴 특징을 새로 가중한 실험은 아닙니다.</p>'+''.join(f'<figure><img loading="lazy" src="{x["path"]}"><figcaption>{x["label"]} · 파랑/주황: 기존 왼손/오른손 관측 · 색 점: 얼굴 부위 기준</figcaption></figure>' for x in outputs)+'</section>'
s=s.replace('<section><h2>실제 장면으로 확인하기</h2>',block+'<section><h2>실제 장면으로 확인하기</h2>')
s=s.replace('손이 얼굴을 가리면 비교 근거를 보류할 수 있습니다.','검출된 손이 얼굴을 가리면 해당 부위에 경고를 표시합니다. 손 자체를 놓친 가림은 이 경고에서도 놓칠 수 있어 별도 검증이 필요합니다.')
p.write_text(s,encoding='utf-8')
# Contact sheet for the model-visible visual audit.
items=read(U/'VISUAL_INDEX.json');sheet=Image.new('RGB',(1440,6*280),'#101923');ds=ImageDraw.Draw(sheet)
for i,v in enumerate(items):
 im=Image.open(U/v['image']);crop=im.crop((770,63,1190,500));crop.thumbnail((230,240));x=(i%6)*240;y=(i//6)*280
 sheet.paste(crop,(x,y+25));ds.text((x+4,y+2),f'{v["label"]} {v["signer"]:02} f{v["frame"]}',font=small,fill='white')
sheet.save(U/'VISUAL_CONTACT_SHEET.jpg',quality=95)
print('PAIRED',len(outputs))
