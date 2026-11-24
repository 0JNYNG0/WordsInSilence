"""Public source RGB with stored points; no new detector or camera."""
import json,sys,zipfile,hashlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
c=read(U/'CONTEXT.json');rows={r['clip_id']:r for r in c['rows']};ex=read(U/'EXAMPLES.json')
outdir=U/'evidence';outdir.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
paths=[];detail=[];clips={};archives={}
labels={'observed_points':'손바닥 관절 관측','partial_points':'일부 관절 관측','wrist_interior':'손목 안쪽 추정','wrist_boundary_or_outside':'손목 경계/밖 추정','assignment_unresolved':'좌우 배정 보류','unknown':'근거 부족'}
try:
 for k,e in enumerate(ex):
  cid=e['clip_id'];r=rows[cid];s=r['source'];dest=outdir/(cid+'.mp4')
  if cid not in clips:
   if s['archive'] not in archives:archives[s['archive']]=zipfile.ZipFile(s['archive'])
   z=archives[s['archive']];info=z.getinfo(s['member']);assert info.CRC==s['crc32'] and info.file_size==s['bytes'];data=z.read(s['member']);assert hashlib.sha256(data).hexdigest()==s['sha256']
   assert not dest.exists();dest.write_bytes(data);clips[cid]=dest
  cap=cv2.VideoCapture(str(dest));fps=cap.get(cv2.CAP_PROP_FPS);cap.set(cv2.CAP_PROP_POS_FRAMES,e['frame_id']);ok,img=cap.read();cap.release();assert ok
  h0,w0=img.shape[:2];w,h=r['box'][2:];assert w==w0 and h<=h0
  canvas=Image.new('RGB',(1280,860),'#101b2b');draw=ImageDraw.Draw(canvas)
  draw.text((20,12),f"{k+1:02}  {e['label']} / 촬영자 {e['signer_id']} / 프레임 {e['frame_id']} / {'왼손' if e['side']==0 else '오른손'}",font=font,fill='white')
  draw.text((20,44),e['stratum'],font=small,fill='#bed7f2')
  meta={'example':k+1,'clip_id':cid,'frame_id':e['frame_id'],'fps':fps,'stratum':e['stratum'],'views':{}}
  for vi,v in enumerate(['masked','sensor']):
   obs,raw,ass=[load(r['views'][v][q]) for q in ['observation','raw','assignment']];i=int(np.flatnonzero(obs['frame_id']==e['frame_id'])[0]);picture=img.copy();picture[h:]=(picture[h:]*.23).astype(np.uint8)
   cv2.line(picture,(0,h),(w-1,h),(0,220,255),4)
   xy=obs['xy_px'][i];valid=obs['usable'][i];praw=raw['pose_xy_px'][i]
   # Observation policy differs: yellow hollow circles are Pose estimates,
   # not counted as recovered Hand21. Always show stored raw Pose for audit.
   for sh,el,wr in [(11,13,15),(12,14,16)]:
    for a,b in [(sh,el),(el,wr)]:
     if np.isfinite(praw[[a,b]]).all():cv2.line(picture,tuple(praw[a].astype(int)),tuple(praw[b].astype(int)),(0,200,255),3)
    if np.isfinite(praw[wr]).all():cv2.circle(picture,tuple(praw[wr].astype(int)),13,(0,200,255),3)
   for side,color in [(0,(110,255,80)),(1,(255,190,80))]:
    for idx in range(33+side*21,54+side*21):
     if valid[idx] and np.isfinite(xy[idx]).all():cv2.circle(picture,tuple(xy[idx].astype(int)),5,color,-1)
   rgb=Image.fromarray(cv2.cvtColor(picture,cv2.COLOR_BGR2RGB)).resize((620,349));x=20+vi*640;canvas.paste(rgb,(x,108))
   draw.text((x,78),'전체 검출 후 범위 제한' if vi==0 else '잘린 화면에서 실제 검출',font=font,fill='white')
   draw.text((x,474),labels[e['states'][v]],font=font,fill='#76ddff')
   draw.text((x,508),f"화면 안 사용 가능 손 관절: {e['visible_hand_counts'][vi]} / 21",font=font,fill='white')
   wr=15+e['side'];rawside=ass['sides'][i,:int(raw['raw_count'][i])]
   meta['views'][v]={'usable_hand_points':e['visible_hand_counts'][vi],'pose_wrist_px':praw[wr].tolist(),'pose_wrist_confidence':float(raw['pose_conf'][i,wr]),'wrist_inside':bool(0<=praw[wr,0]<=w and 0<=praw[wr,1]<=h),'raw_hand_count':int(raw['raw_count'][i]),'assigned_sides':rawside.tolist()}
   draw.text((x,542),f"Pose 손목 y={praw[wr,1]:.1f} / 하단 경계={h}",font=small,fill='#bed7f2')
  draw.text((20,596),'노란 선: 입력 하단 경계 · 노란 선/원: Pose 팔/손목 추정 · 점: 사용 가능한 Hand21',font=small,fill='white')
  draw.text((20,638),'동일 sensor 관측의 처리 규칙만 바꾼 진단 결과 (모델 입력 수정 아님)',font=font,fill='white')
  for yy,key in enumerate(['sensor_wrist_outside_blocked','sensor_visible_assignment','sensor_common_policy']):
   title={'sensor_wrist_outside_blocked':'화면 밖 손목 추정 제한','sensor_visible_assignment':'화면 안 관절 기반 배정 상태','sensor_common_policy':'두 정책을 모두 화면 안 근거로 제한'}[key]
   draw.text((30,678+yy*42),title+' → '+labels[e['states'][key]],font=font,fill='#bed7f2')
  target=outdir/f'{k+1:02}.png';canvas.save(target);paths.append(str(target));detail.append(meta)
 for start in range(0,len(paths),4):
  tiles=paths[start:start+4];contact=Image.new('RGB',(1280,860*len(tiles)),'white')
  for j,p in enumerate(tiles):contact.paste(Image.open(p),(0,860*j))
  contact.save(outdir/f'contact_{start//4+1}.jpg',quality=88)
 (U/'VISUAL_METADATA.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Rendered',len(paths),'public examples',len(clips),'source videos with member SHA verified.')
finally:
 for z in archives.values():z.close()
