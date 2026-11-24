import prepare as R
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
U=R.U
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
chains=[(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]
def main():
 f=R.read(U/'IMAGE_FROZEN.json');pool=[];rows={r['clip_id']:r for r in f['rows']}
 for row in f['rows']:
  p=U/'records'/(row['clip_id']+'.json')
  if not p.exists():continue
  r=R.read(p);o=R.load(U/'outputs'/(row['clip_id']+'.npz'))
  for z in r['frames']:
   i=z['i']
   for s in z['requested']:
    flags={'control':z['control'],'crossing_request':bool(z.get('reason',[0,0])[s]&4),'identity_conflict':bool(o['mp_identity_conflict'][i,s] or o['yolo_identity_conflict'][i,s]),'mp_confirmed':bool(o['mp_eligible'][i,s].sum()>=12),'shape_disagreement':bool(o['mp_unresolved'][i,s].any()),'yolo_added':bool((o['fused_eligible'][i,s]&~o['mp_eligible'][i,s]).any()),'yolo_conflict':bool(o['yolo_unresolved'][i,s].any())}
    pool.append(dict(cid=row['clip_id'],index=i,side=s,flags=flags,z=z,video=r['source_video']))
 selected=[]
 for name in ['identity_conflict','shape_disagreement','yolo_added','yolo_conflict','crossing_request','mp_confirmed','control']:
  candidates=sorted([x for x in pool if x['flags'][name]],key=lambda x:R.hashlib.sha256((x['cid']+str(x['index'])+str(x['side'])).encode()).hexdigest());seen=set()
  for c in candidates:
   if c['cid'] in seen:continue
   if any((c['cid'],c['index'],c['side'])==(a['cid'],a['index'],a['side']) for a in selected):continue
   selected.append(dict(c,category=name));seen.add(c['cid'])
   if len(seen)==3:break
 outdir=U/('visual_'+str(len(list((U/'records').glob('*.json')))));outdir.mkdir(exist_ok=True);index=[]
 for no,c in enumerate(selected):
  row=rows[c['cid']];n=R.load(row['input']['info']['source_arrays']['capture']);i,s=c['index'],c['side'];w,h=row['input']['info']['canvas'];x=n['xy_px'][i,33+21*s:54+21*s].astype(float);v=n['usable'][i,33+21*s:54+21*s]&np.isfinite(x).all(-1);p=x[v];lo=p.min(0);hi=p.max(0);mid=(lo+hi)/2;span=max(180.,float(max(hi-lo))*2.8);b=np.array([max(0,int(mid[0]-span/2)),max(0,int(mid[1]-span/2)),min(w,int(mid[0]+span/2)),min(h,int(mid[1]+span/2))]);cap=cv2.VideoCapture(c['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,int(n['frame_id'][i]));ok,frame=cap.read();cap.release();assert ok;rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);crop=Image.fromarray(rgb[b[1]:b[3],b[0]:b[2]]);cw,ch=crop.size;scale=min(490/cw,360/ch);crop=crop.resize((round(cw*scale),round(ch*scale)));canvas=Image.new('RGB',(1560,870),'#101824');d=ImageDraw.Draw(canvas)
  d.text((18,10),f"{no+1:02d} {row['label']} / {c['cid']} / 프레임 {i}, {'왼손' if s==0 else '오른손'} / {c['category']}",font=font,fill='white')
  z=c['z'];panels=[('원본 손 주변 (표시만 확대)',[]),('기존 좌표 (변경 없음)',[dict(xy=x,valid=v)]),('MediaPipe 전체 장면 재검출',z['mp_full']),('MediaPipe 손 주변 재검출',z['mp_crop']),('YOLO 전체 장면 보조',z['yolo_full']),('YOLO 손 주변 보조',z['yolo_crop'])]
  for k,(title,ds) in enumerate(panels):
   xx=10+(k%3)*520;yy=48+(k//3)*400;canvas.paste(crop,(xx,yy+32));d.text((xx,yy),title,font=font,fill='#c6dcff')
   if not ds and k>1:d.text((xx+10,yy+60),'검출 없음 또는 미호출',font=font,fill='#ffca82')
   for di,det in enumerate(ds):
    pts=np.asarray(det['xy'],float);vv=np.asarray(det.get('valid',np.isfinite(pts).all(-1)),bool);color=['#54edd2','#ffb658','#ff70a0'][di%3]
    for chn in chains:
     for a,bb in zip(chn[:-1],chn[1:]):
      if vv[a] and vv[bb]:
       aa=(pts[a]-b[:2])*scale+[xx,yy+32];pp=(pts[bb]-b[:2])*scale+[xx,yy+32]
       if ((aa>=[xx,yy+32])&(aa<=[xx+crop.width,yy+32+crop.height])).all() and ((pp>=[xx,yy+32])&(pp<=[xx+crop.width,yy+32+crop.height])).all():d.line([tuple(aa),tuple(pp)],fill=color,width=2)
    for j,point in enumerate(pts):
     if not vv[j]:continue
     q=(point-b[:2])*scale+[xx,yy+32]
     if not ((q>=[xx,yy+32])&(q<=[xx+crop.width,yy+32+crop.height])).all():continue
     d.ellipse((q[0]-2,q[1]-2,q[0]+2,q[1]+2),fill=color)
     if j in (0,4,8,12,16,20):d.text((q[0]+3,q[1]),str(j),font=small,fill=color)
  d.text((15,849),'색 선은 각 검출기의 예측. 확대 화면은 검토용이며 전체 영상의 검출 범위를 바꾸지 않음.',font=small,fill='#c4cdd9');name=f'{no+1:02d}_{c["category"]}.jpg';canvas.save(outdir/name,quality=94)
  index.append(dict(number=no+1,clip_id=c['cid'],label=row['label'],index=i,side=s,category=c['category'],flags=c['flags'],image=str(outdir/name),mp_checks=z['mp_checks'],yolo_checks=z['yolo_checks']))
 R.write(outdir/'INDEX.json',dict(at=R.now(),clips_available=len(list((U/'records').glob('*.json'))),scenes=index,selection='Up to3 distinct clips/category by fixed hash; descriptive audit not tuning or independent manual ground truth.'))
 print('VISUAL_READY',len(index),str(outdir),flush=True)
if __name__=='__main__':main()
