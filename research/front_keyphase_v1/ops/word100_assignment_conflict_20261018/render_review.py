"""Render public source RGB and actual candidate assignments for human review."""
import sys,json,hashlib,zipfile
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from run import read,write,load,sha
U=Path(__file__).resolve().parent;O=U.parent
F=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',21);S=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
def main():
 c=read(U/'CONTEXT.json');rows={r['clip_id']:r for r in c['rows']};changes=read(U/'CHANGES.json');cross=read(U/'CROSSING_SELECTION.json');selection=[];videos={};archives={};out=[]
 for x in changes:
  if not any(e['clip_id']==x['clip_id'] and e['view']==x['view'] and e['index']==x['index'] for e in selection):selection.append({**x,'kind':'all_changed'})
 selection.extend({**x,'kind':'crossing_projection'} for x in cross)
 try:
  for num,e in enumerate(selection,1):
   r=rows[e['clip_id']];s=r['source'];cid=r['clip_id'];view=e['view'];raw=load(r['views'][view]['raw']);old=load(r['views'][view]['assignment'])['sides'];new=load(U/'results'/f'{cid}_{view}.npz')['sides']
   if cid not in videos:
    reuse=O/'word100_evidence_axes_20261018/evidence'/f'{cid}.mp4'
    if reuse.exists():data=reuse.read_bytes()
    else:
     if s['archive'] not in archives:archives[s['archive']]=zipfile.ZipFile(s['archive'])
     z=archives[s['archive']];zi=z.getinfo(s['member']);assert zi.CRC==s['crc32'] and zi.file_size==s['bytes'];data=z.read(s['member'])
    assert hashlib.sha256(data).hexdigest()==s['sha256'];dest=U/'evidence'/f'{cid}.mp4';assert not dest.exists();dest.write_bytes(data);videos[cid]=dest
   cap=cv2.VideoCapture(str(videos[cid]));frames=sorted(set(max(0,min(len(raw['raw_count'])-1,e['index']+off)) for off in (-2,0,2)));ims=[]
   for f in frames:
    fid=int(raw['frame_id'][f]);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
    panels=[]
    for title,sides in [('기존 배정',old[f]),('보완 후보 배정',new[f])]:
     p=im.copy();h=int(raw['height'][f]);w=int(raw['width'][f]);p[h:]=(p[h:]*.22).astype(np.uint8)
     cv2.line(p,(0,h),(w-1,h),(0,220,255),3)
     for a,b in [(11,13),(13,15),(12,14),(14,16)]:
      if raw['pose_usable'][f,[a,b]].all():cv2.line(p,tuple(raw['pose_xy_px'][f,a].astype(int)),tuple(raw['pose_xy_px'][f,b].astype(int)),(220,220,220),3)
     for i in range(int(raw['raw_count'][f])):
      xy=raw['hand_xy_px'][f,i];color=(240,150,55) if sides[i]==0 else (55,180,255) if sides[i]==1 else (90,90,240)
      if sides[i]>=0:
       for a,b in EDGES:
        if np.isfinite(xy[[a,b]]).all():cv2.line(p,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),color,2)
       for v in xy:
        if np.isfinite(v).all():cv2.circle(p,tuple(v.astype(int)),5,color,-1)
      elif np.isfinite(xy[0]).all():cv2.drawMarker(p,tuple(xy[0].astype(int)),color,cv2.MARKER_TILTED_CROSS,32,4)
     panel=Image.new('RGB',(768,520),'#132233');d=ImageDraw.Draw(panel);d.text((16,12),f'{title} · 프레임 {fid}',font=F,fill='white');panel.paste(Image.fromarray(cv2.cvtColor(p,cv2.COLOR_BGR2RGB)).resize((768,432)),(0,52))
     d.text((16,488),'파랑: 왼손 / 주황: 오른손 / 빨강 ×: 후보 있으나 배정 보류',font=S,fill='white');panels.append(panel)
    canvas=Image.new('RGB',(1536,560),'#132233');canvas.paste(panels[0],(0,40));canvas.paste(panels[1],(768,40));ImageDraw.Draw(canvas).text((16,7),f'{r["label"]} | {cid} | {view} | 촬영자의 좌우',font=F,fill='white');ims.append(canvas)
   cap.release();target=U/'evidence'/f'{num:02}_{e["kind"]}.jpg';page=Image.new('RGB',(1536,560*len(ims)))
   for i,im in enumerate(ims):page.paste(im,(0,560*i))
   page.save(target,quality=92);out.append({**e,'frames':[int(raw['frame_id'][f]) for f in frames],'image':str(target),'image_sha256':sha(target),'source_sha256':s['sha256']})
  write(U/'REVIEW_IMAGES.json',out);print('Rendered review sequences',len(out),flush=True)
 finally:
  for z in archives.values():z.close()
if __name__=='__main__':main()
