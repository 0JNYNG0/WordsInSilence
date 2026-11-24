"""Render saved public TRAIN changes; no camera, scoring, or learning."""
from common import *
import cv2
from PIL import Image,ImageDraw,ImageFont
def main():
 rows={r['clip_id']:r for r in read(U/'PREP_CONTEXT.json')['rows'] if r['role']=='train_candidate'}
 prior={(x['clip_id'],x['view'],x['index']):x for x in read(F/'RESULTS.json')}
 font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
 directory=U/'review';directory.mkdir(exist_ok=True);index=[]
 for rp in sorted((U/'records').glob('*.json')):
  rec=read(rp)
  if rec['role']!='train_candidate' or not rec['changes']:continue
  r=rows[rec['clip_id']]
  for change in rec['changes']:
   v=change['view'];t=change['index'];arm=change['arm'];key=(r['clip_id'],v,t)
   known=prior.get(key);already=known is not None and known['sides']==change['after']
   if already:continue
   path=directory/f"{r['clip_id']}_{v}_{t}_{arm}.jpg"
   timestamp=int(load(r['views'][v]['raw'])['timestamp_us'][t])/1e6
   item={'clip_id':r['clip_id'],'word':r['label'],**change,'image':str(path),'at_seconds':timestamp,'in_core':r['intervals'][0][0]<=timestamp<=r['intervals'][0][1]};index.append(item)
   if path.exists():continue
   z=load(r['views'][v]['raw']);b=load(OBS/'results'/f"{r['clip_id']}_{v}.npz")
   video=rec['source_video']
   if not video:
    matches=list(O.glob(f"*/videos/{r['clip_id']}.mp4"));video=next((str(p) for p in matches if sha(p)==r['source']['sha256']),None)
   assert video,'No saved public source path for review'
   cap=cv2.VideoCapture(video);panels=[]
   for offset in [-2,0,2]:
    k=max(0,min(len(z['frame_id'])-1,t+offset));cap.set(cv2.CAP_PROP_POS_FRAMES,int(z['frame_id'][k]));ok,frame=cap.read();assert ok
    width=int(z['width'][k]);height=int(z['height'][k]);frame=frame[:height,:width].copy()
    scale=560/width;frame=cv2.resize(frame,(560,round(height*scale)))
    if offset==0:
     for side,joints in enumerate([[11,13,15],[12,14,16]]):
      color=[(230,230,0),(0,0,255)][side]
      for a,bb in zip(joints,joints[1:]):
       if z['pose_usable'][k,a] and z['pose_usable'][k,bb]:
        pts=z['pose_xy_px'][k,[a,bb]]*scale
        if np.isfinite(pts).all():cv2.line(frame,tuple(pts[0].astype(int)),tuple(pts[1].astype(int)),color,2)
     for i in range(int(z['raw_count'][k])):
      for j in np.flatnonzero(b['point_mask'][k,i]):
       pt=tuple((z['hand_xy_px'][k,i,j]*scale).astype(int));cv2.circle(frame,pt,3,[(0,180,255),(0,255,80)][i],-1)
      if b['point_mask'][k,i,0]:cv2.putText(frame,f'raw{i}',tuple((z['hand_xy_px'][k,i,0]*scale).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    canvas=Image.new('RGB',(560,410),'white');canvas.paste(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),(0,30));draw=ImageDraw.Draw(canvas);draw.text((8,4),f'frame {int(z["frame_id"][k])} / {z["timestamp_us"][k]/1e6:.3f}s',font=small,fill='black');panels.append(canvas)
   cap.release();sheet=Image.new('RGB',(1680,480),'white');draw=ImageDraw.Draw(sheet);draw.text((10,6),f"{r['label']} {r['clip_id']} {v} {arm} | 0=왼손 1=오른손 -1=보류 | {change['before']} → {change['after']}",font=font,fill='black')
   for i,panel in enumerate(panels):sheet.paste(panel,(560*i,55))
   sheet.save(path,quality=92)
 write(directory/'INDEX.json',index);print('REVIEW_NEW_EVENTS',len(index))
if __name__=='__main__':main()
