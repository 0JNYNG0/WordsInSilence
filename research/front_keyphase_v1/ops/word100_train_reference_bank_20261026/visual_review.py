"""Qualitative audit of completed public records; no source correction or rule tuning."""
import prepare as R
import numpy as np,cv2,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def main():
 f=R.read(R.U/'FROZEN.json');catalog={x['clip_id']:x for x in f['catalog']};todo={x['clip_id']:x for x in f['rows']};candidates={};bestgap=float('inf');completed=sorted((R.U/'records').glob('*.json'));first=completed[:min(300,len(completed))]
 for rp in first:
  cid=rp.stem;z=R.load(catalog[cid]['input']);o=R.load(R.U/'outputs'/(cid+'.npz'));xx=z['xy_px'][:,33:].reshape(-1,2,21,2);av=z['available'][:,:,:15];raw=av.sum(-1);good=o['pixel_eligible'].sum(-1)
  tests={'well_supported':(good>=13)&o['requested'],'shape_disagreement':(raw>=10)&(good<=2)&(o['pixel_unresolved'].sum(-1)>=5)&o['requested'],'identity_uncertain':o['mp_identity_conflict']&o['requested']}
  for name,mask in tests.items():
   if name not in candidates and mask.any():i,h=np.argwhere(mask)[0];candidates[name]=(cid,int(i),int(h))
  for i in np.flatnonzero(o['requested'].any(-1)&z['core']):
   if np.isfinite(xx[i,:,0]).all() and np.isfinite(z['xy_px'][i,[11,12]]).all():
    width=np.linalg.norm(z['xy_px'][i,11]-z['xy_px'][i,12]);gap=np.linalg.norm(xx[i,0,0]-xx[i,1,0])/max(width,1)
    if gap<bestgap:bestgap=gap;candidates['closest_hand_wrist_pair']=(cid,int(i),0)
  if len(candidates)==4 and len(first)>=300:break
 dest=R.U/'review_visual';dest.mkdir();font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15);board=Image.new('RGB',(1500,len(candidates)*410),'#f0f3f7');draw=ImageDraw.Draw(board);rows=[]
 edges=[(chain[k],chain[k+1]) for chain in [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)] for k in range(4)]
 for k,(reason,(cid,i,h)) in enumerate(candidates.items()):
  z=R.load(catalog[cid]['input']);o=R.load(R.U/'outputs'/(cid+'.npz'));record=R.read(R.U/'records'/(cid+'.json'));source=record['source_video'];fid=int(z['frame_id'][i]);cap=cv2.VideoCapture(source);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,img=cap.read();assert ok;assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(fid+1))<.1;cap.release();xy=z['xy_px'][i];hand=xy[33:].reshape(2,21,2);points=hand[np.isfinite(hand).all(-1)];points=np.concatenate([points,xy[[13,14,15,16]]]);points=points[np.isfinite(points).all(-1)];lo=np.maximum([0,0],np.floor(points.min(0)-70)).astype(int);hi=np.minimum([img.shape[1],img.shape[0]],np.ceil(points.max(0)+70)).astype(int);overlay=img.copy()
  for side,col in [(0,(80,230,20)),(1,(255,220,20))]:
   for a,b in edges:
    if np.isfinite(hand[side,[a,b]]).all():cv2.line(overlay,tuple(hand[side,a].astype(int)),tuple(hand[side,b].astype(int)),col,2)
   for point in hand[side]:
    if np.isfinite(point).all():cv2.circle(overlay,tuple(point.astype(int)),3,col,-1)
  for a,b in [(11,13),(13,15),(12,14),(14,16)]:
   if np.isfinite(xy[[a,b]]).all():cv2.line(overlay,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),(180,180,180),2)
  y=k*410;draw.text((12,y+8),f'{reason} | {cid} | frame {fid} | target {"L" if h==0 else "R"}',font=font,fill='#142e45')
  for j,a in enumerate([img,overlay]):
   tile=Image.fromarray(cv2.cvtColor(a[lo[1]:hi[1],lo[0]:hi[0]],cv2.COLOR_BGR2RGB));tile.thumbnail((720,335));board.paste(tile,(15+j*750,y+40))
  stats=dict(raw_angles=int(z['available'][i,h,:15].sum()),image_corroborated=int(o['pixel_eligible'][i,h].sum()),image_disagreement=int(o['pixel_unresolved'][i,h].sum()),identity_conflict=bool(o['mp_identity_conflict'][i,h]));draw.text((12,y+381),'Original RGB left | unchanged masked coordinates right | '+str(stats),font=small,fill='#142e45');rows.append(dict(reason=reason,clip_id=cid,input_index=i,frame_id=fid,side=h,source_sha256=record['source_sha256'],record_sha256=R.sha(R.U/'records'/(cid+'.json')),stats=stats))
 p=dest/'public_audit.jpg';board.save(p,quality=94);R.write(dest/'INDEX.json',dict(at=R.now(),image=str(p),image_sha256=R.sha(p),completed_available=len(completed),scanned_limit=len(first),selected=rows,selection='First qualifying completed public case per evidence type and closest hand pair among scanned records; no model scores, private recordings or tuning. Coordinates unchanged. Anatomical finger truth not established.',note='Crop is visualization only; full-frame detection and fixed input ROI remain unchanged. Green anatomical left,cyan right. The overlay is original retained input, not a newly accepted recovery.'));print('VISUAL_AUDIT_READY',len(rows),len(first),str(p),flush=True)
if __name__=='__main__':main()
