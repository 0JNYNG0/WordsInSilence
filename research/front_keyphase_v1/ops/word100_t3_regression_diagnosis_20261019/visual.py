"""Decode public reference frames with saved landmark overlay. No detector/model."""
from analyze import U,O,B,read,save,sha,load,now
from pathlib import Path
import zipfile,cv2,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
scope=read(U/'SCOPE.json');meta={r['clip_id']:r for r in read(O/'word100_assignment_scale_factorial_20261019/PREP_CONTEXT.json')['rows']}
focus=scope['focus_cases'];targets=list(focus)
for cid in focus[:3]:targets.append(cid.split('_REAL')[0]+'_REAL01_F')
targets.append('NIA_SL_WORD1519_REAL13_F')
cache=U/'public_video_cache';cache.mkdir(exist_ok=True);folder=U/'visual';folder.mkdir(exist_ok=True)
oldcache=O/'word100_confusion_late_training_diagnosis_20261019/public_video_cache'
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
out=[]
for cid in targets:
 rec=read(B/'data/records'/f'{cid}.json');row=meta[cid];src=rec['source'];video=oldcache/(cid+'.mp4')
 if not video.exists():
  video=cache/(cid+'.mp4');assert not video.exists()
  archive=Path(src['archive'])
  if not archive.exists():archive=Path(row['source']['archive'])
  with zipfile.ZipFile(archive) as z:data=z.read(src['member'])
  assert hashlib.sha256(data).hexdigest()==src['sha256'];video.write_bytes(data)
 assert sha(video)==src['sha256'];cap=cv2.VideoCapture(str(video));assert cap.isOpened()
 fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));lo=min(x[0] for x in row['intervals']);hi=max(x[1] for x in row['intervals'])
 times=[max(0,lo-.2)]+np.linspace(lo,hi,6).tolist()+[min((n-1)/fps,hi+.2)]
 sheet=Image.new('RGB',(1600,640),'white');dr=ImageDraw.Draw(sheet)
 dr.text((10,8),f"{row['label']} · 공개 촬영자 {row['signer_id']} · 원본 화면 + 저장된 sensor 관측",font=font,fill='black')
 dr.text((10,39),f'{cid} | 좌:초록 / 우:자홍 / 노랑:하단 경계 | 표본 장면, 정오 판정 아님',font=small,fill='black')
 obs=load(rec['views']['sensor']['observation']);used=[]
 for k,t in enumerate(times):
  f=min(n-1,max(0,round(t*fps)));cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok
  j=int(np.argmin(abs(obs['timestamp_us']/1e6-f/fps)));mask=obs['usable'][j];xy=obs['xy_px'][j]
  for start,color in [(33,(40,255,40)),(54,(240,40,240))]:
   for a in range(21):
    if mask[start+a]:cv2.circle(img,tuple(np.rint(xy[start+a]).astype(int)),4,color,-1)
  bottom=rec['calibration']['bottom'];cv2.line(img,(0,bottom),(img.shape[1]-1,bottom),(0,220,255),3)
  im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));im.thumbnail((400,225));x=k%4*400;y=80+k//4*280;sheet.paste(im,(x,y))
  dr.text((x+4,y+228),f'f{f} {f/fps:.2f}s L{mask[33:54].sum()} R{mask[54:75].sum()}',font=small,fill='black')
  used.append({'frame':f,'time':f/fps,'left_points':int(mask[33:54].sum()),'right_points':int(mask[54:75].sum())})
 cap.release();p=folder/(cid+'.jpg');sheet.save(p,quality=90)
 out.append({'clip_id':cid,'word':row['label'],'video':str(video),'video_sha256':sha(video),'sheet':str(p),'sheet_sha256':sha(p),'frames':used})
save('VISUAL_MANIFEST.json',{'at':now(),'public_videos':len(out),'sampled_RGB_frames':sum(len(x['frames']) for x in out),'sheets':out,'new_detector_inference':0,'new_training':0})
print('Prepared',len(out),'public video contact sheets')
