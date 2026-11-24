from pathlib import Path
import json,cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
old=read(O/'word100_feature_expansion_20261113/annotation_review/ORDINARY_LABELS.json')['rows']
rr=[];panels=[];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
for record,ids in [(2,[52,54,56,58,60,62,64,66,68,70,72,74]),(3,[62,64,66,68,69,70,71,72,74,76,78,80])]:
 source=next(r for r in old if r['word']=='WORD2551'and r['person']==record);cap=cv2.VideoCapture(source['source_video'])
 for fid in ids:
  cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok;h,w=im.shape[:2]
  # Broad upper-body crop retains actual hand contour; not a neural input.
  crop=im[:int(.62*h),int(.22*w):int(.78*w)]
  rgb=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));rgb.thumbnail((360,280))
  tile=Image.new('RGB',(370,315),'white');tile.paste(rgb,((370-rgb.width)//2,30));ImageDraw.Draw(tile).text((5,4),f'별 기록{record:02} f{fid}',font=font,fill='#17283f');panels.append(tile)
  rr.append(dict(clip_id=source['clip_id'],record_id=record,frame=fid,source=source['source_video'],source_sha256=source['source_sha256'],hand='R',label=None))
 cap.release()
 sheet=Image.new('RGB',(370*4,315*3),'white')
 for j,p in enumerate(panels[-12:]):sheet.paste(p,((j%4)*370,(j//4)*315))
 sheet.save(U/f'crossword_record{record:02}.jpg',quality=95)
with(U/'CROSSWORD_REVIEW_INPUTS.json').open('x',encoding='utf-8')as f:json.dump(dict(rows=rr,selection='Adjacent native frames around previously RGB-reviewed bend/extend anchors; no prediction selection'),f,ensure_ascii=False,indent=2)
