"""Render already-public source frames only; no inference or training."""
from pathlib import Path
import json, hashlib
import cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
O=U.parent
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
rows=[]
for person,span,sources,expected in [
 (13,[59,79],[69,67,65,63,61],'4a8466ee2394c9312e06e7a073b99b0011f9764de98085f0565eb4b45c2f4b63'),
 (12,[53,77],[63,61,59,57,55],'6868ab57eeb13d7d21fb343c0eff4f748bfb7031f00d494efd8662ac408b4254')]:
 clip=f'NIA_SL_WORD2523_REAL{person}_F'
 p=O/'word100_multipeople_live_feedback_20261101/data_audit/videos'/f'{clip}.mp4'
 h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==expected
 cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS)
 im=Image.new('RGB',(1500,390),'white');draw=ImageDraw.Draw(im)
 draw.text((10,5),f'공개 {person}번 · 동작 순서를 뒤집은 시험 · 시험 시간 순서 →',font=font,fill='#172538')
 for j,src in enumerate(sources):
  cap.set(cv2.CAP_PROP_POS_FRAMES,src);ok,bgr=cap.read();assert ok
  rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));w,h=rgb.size
  # Upper torso and both arms; all panels use identical source crop.
  rgb=rgb.crop((int(w*.2),int(h*.12),int(w*.8),int(h*.75)))
  rgb.thumbnail((294,310));im.paste(rgb,(j*300+(300-rgb.width)//2,48))
  draw.text((j*300+8,354),f'시험 {sum(span)-src} / 원본 {src}',font=font,fill='#172538')
 cap.release()
 out=U/f'causal_public_{person}.jpg';im.save(out,quality=92)
 rows.append(dict(clip_id=clip,source=str(p),source_sha256=expected,fps=fps,reverse_span=span,source_frame_order=sources,analysis_frame_order=[sum(span)-v for v in sources],sheet=str(out),sheet_sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
(U/'CAUSAL_RGB_SOURCES.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),'utf-8')
print('Rendered 2 source strips; no model executed.')
