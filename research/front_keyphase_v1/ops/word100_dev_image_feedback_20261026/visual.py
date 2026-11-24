import prepare as R
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
def main():
 f=R.read(R.U/'FROZEN.json');candidates=[]
 for outer in f['rows'][:10]:
  row=outer['source'];cid=row['clip_id'];o=R.load(R.U/'outputs'/(cid+'.npz'));d=R.load(outer['baseline_input']);rr=R.read(R.U/'records'/(cid+'.json'));n=R.load(row['input']['info']['source_arrays']['capture']);xx=n['xy_px'][:,33:].reshape(-1,2,21,2);vv=n['usable'][:,33:].reshape(-1,2,21)
  for z in rr['frames']:
   i=z['i']
   if not d['core'][i]:continue
   for s in z['requested']:
    e=next(x for x in z['mp_checks'] if x['side']==s);cross=False
    if vv[i,0].any() and vv[i,1].any():
     aa=xx[i,0][vv[i,0]];bb=xx[i,1][vv[i,1]];cross=bool((np.minimum(aa.max(0),bb.max(0))>np.maximum(aa.min(0),bb.min(0))).all())
    candidates.append(dict(row=row,frame=z,side=s,confirmed=int(o['pixel_eligible'][i,s].sum()),conflict=int(o['pixel_unresolved'][i,s].sum()),identity=bool(o['mp_identity_conflict'][i,s]),cross=cross,control=not cross))
 choices=[];seen=set()
 for name,score in [('identity',lambda z:(z['identity'],z['conflict'])),('pixel_conflict',lambda z:(z['conflict'],z['confirmed'])),('crossing',lambda z:(z['cross'],z['confirmed'])),('separate_hands',lambda z:(z['control'],z['confirmed']))]:
  for z in sorted(candidates,key=score,reverse=True):
   if z['row']['clip_id'] not in seen:choices.append((name,z));seen.add(z['row']['clip_id']);break
 out=R.U/'visual';out.mkdir();font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15);results=[];chains=[(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]
 for j,(name,z) in enumerate(choices):
  row=z['row'];rr=z['frame'];i=rr['i'];s=z['side'];n=R.load(row['input']['info']['source_arrays']['capture']);x=n['xy_px'][i,33+21*s:54+21*s].astype(float);cap=cv2.VideoCapture(row['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,rr['frame_id']);ok,bgr=cap.read();cap.release();assert ok;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);e=next(a for a in rr['mp_checks'] if a['side']==s);a=np.asarray(rr['mp_full'][e['full_index']]['xy']) if e['full_index'] is not None else None;b=np.asarray(rr['mp_crop'][e['crop_index']]['xy']) if e['crop_index'] is not None else None;pts=np.concatenate([a for a in [x,a,b] if a is not None]);good=pts[np.isfinite(pts).all(-1)];lo=good.min(0)-100;hi=good.max(0)+100;w,h=row['input']['info']['canvas'];box=[max(0,int(lo[0])),max(0,int(lo[1])),min(w,int(hi[0])),min(h,int(hi[1]))];crop=Image.fromarray(rgb).crop(box);canvas=Image.new('RGB',(1200,510),'#101929');draw=ImageDraw.Draw(canvas);draw.text((20,10),f'{name}: {row["clip_id"]} frame {rr["frame_id"]} side {s}',font=font,fill='white');draw.text((20,40),f'MP image corroborated angles {z["confirmed"]}/15; pixel uncertainty {z["conflict"]}/15. Not anatomical truth.',font=small,fill='#dde5ee')
  for k,(title,xy,col) in enumerate([('Original coordinates',x,'#61b3ff'),('MP full image',a,'#6ae8bd'),('MP hand crop',b,'#ffb16b')]):
   factor=min(380/crop.width,380/crop.height);nw,nh=int(crop.width*factor),int(crop.height*factor);ox=k*400+(400-nw)//2;oy=115+(380-nh)//2;canvas.paste(crop.resize((nw,nh)),(ox,oy));draw.text((k*400+20,80),title,font=font,fill=col)
   if xy is not None:
    ps=(xy-np.asarray(box[:2]))*factor+[ox,oy]
    for ch in chains:
     for u,v in zip(ch[:-1],ch[1:]):
      if np.isfinite(ps[[u,v]]).all():draw.line([tuple(ps[u]),tuple(ps[v])],fill=col,width=2)
    for xx,yy in ps:
     if np.isfinite([xx,yy]).all():draw.ellipse((xx-2,yy-2,xx+2,yy+2),fill=col)
   else:draw.text((k*400+30,470),'No uniquely matched image pair',font=small,fill='white')
  pp=out/f'{j+1:02d}_{name}.png';canvas.save(pp);results.append(dict(kind=name,clip_id=row['clip_id'],frame_id=rr['frame_id'],side=s,source_video=row['video'],image=str(pp),image_sha256=R.sha(pp),corroborated=z['confirmed'],pixel_uncertain=z['conflict'],anatomical_ground_truth=False))
 R.write(R.U/'VISUAL_INDEX.json',dict(at=R.now(),rows=results,selection='First10 independently verified DEV clips only: identity/pixel uncertainty/box-overlap/box-separation audit samples; not anatomical or semantic labels.'))
 print('VISUAL4_READY',flush=True)
if __name__=='__main__':main()

