from pathlib import Path
import json,hashlib,zipfile,sys
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parents[1]
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def wr(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for d in ['media','rgb','panels']:(U/d).mkdir(parents=True,exist_ok=True)
src={r['clip_id']:r for r in rd(O/'word100_rtmw_joint_learning_20261108/SOURCES.json') if r['role']=='fit' and 1<=r['person']<=9}
F=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',21)
archives={}
def video(cid):
 r=src[cid]
 candidates=[O/'word100_feature_expansion_20261113/annotation_review/media'/f'{cid}.mp4',O/'demo_handshape_learning_20261113/media'/f'{cid}.mp4',U/'media'/f'{cid}.mp4']
 p=next((x for x in candidates if x.exists()),candidates[-1])
 if not p.exists():
  if r['archive'] not in archives:archives[r['archive']]=zipfile.ZipFile(r['archive'])
  p.write_bytes(archives[r['archive']].read(r['member']))
 assert sha(p)==r['source_sha256'],cid
 return p
def make(cid,frames,tag):
 source=src[cid];vp=video(cid);rec=rd(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json');assert sha(rec['raw_path'])==rec['raw_sha256']
 z=np.load(rec['raw_path']);cap=cv2.VideoCapture(str(vp));out=[];cells=[]
 for f in frames:
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,bgr=cap.read();assert ok,(cid,f)
  im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));ip=U/'rgb'/f'{cid}_{f}.jpg';im.save(ip,quality=98)
  ii=int(np.flatnonzero(z['frame_id']==f)[0])
  cell=Image.new('RGB',(560,380),'white');d=ImageDraw.Draw(cell);d.text((8,4),f"{source['label']} ID{source['person']:02} f{f}",font=F,fill='#17283f')
  w,h=im.size;ctx=im.crop((int(w*.29),int(h*.08),int(w*.72),int(h*.88)));ctx.thumbnail((210,335));cell.paste(ctx,(2,40))
  boxes={}
  for side,sl,x in [('R',slice(112,133),215),('L',slice(91,112),387)]:
   pts=z['xy_px'][ii,sl];good=np.isfinite(pts).all(1)&(z['score'][ii,sl]>.2)
   if good.sum()>=5:
    q=pts[good];lo=np.quantile(q,.03,0);hi=np.quantile(q,.97,0);center=(lo+hi)/2;size=max(float(max(hi-lo))*1.6,120.)
   else:center=np.array([w*(.43 if side=='R' else .57),h*.45]);size=300
   l=max(0,int(center[0]-size/2));t=max(0,int(center[1]-size/2));r=min(w,int(center[0]+size/2));b=min(h,int(center[1]+size/2));box=(l,t,r,b);boxes[side]=box
   cut=im.crop(box);cut.thumbnail((170,285));cell.paste(cut,(x+(170-cut.width)//2,70+(285-cut.height)//2));d.text((x,42),side,font=F,fill='#17283f')
  cp=U/'rgb'/f'{cid}_{f}_review.jpg';cell.save(cp,quality=98);cells.append(cell)
  out.append(dict(clip_id=cid,word=source['word_id'],person=source['person'],role='fit',frame=int(f),pts_us=int(z['timestamp_us'][ii]),source_video=str(vp),source_sha256=source['source_sha256'],image=str(ip),image_sha256=sha(ip),review=str(cp),crop_boxes=boxes,raw_path=rec['raw_path'],raw_sha256=rec['raw_sha256']))
 cap.release();board=Image.new('RGB',(560*3,380*((len(cells)+2)//3)),'white')
 for j,c in enumerate(cells):board.paste(c,((j%3)*560,(j//3)*380))
 bp=U/'panels'/f'{cid}_{tag}.jpg';board.save(bp,quality=98)
 return out,str(bp)
def main():
 mode=sys.argv[1]
 if mode=='screen':
  words=['WORD2523','WORD2551','WORD1486','WORD0196','WORD0783','WORD2348','WORD1140']
  tasks=[]
  for word in words:
   cid=f'NIA_SL_{word}_REAL01_F';n=src[cid]['frames_header'];frames=np.linspace(n*.23,n*.78,9).round().astype(int).tolist()
   tasks.append(dict(clip_id=cid,frames=frames))
  wr(U/'SCREEN_SELECTION.json',dict(tasks=tasks,basis='Existing FIT RGB B evidence (Egypt/Star) plus fixed ordinary-expression list selected for potential bilateral handshape contrasts; uniform nine native frames across 23%-78% of video. No new DEV outputs or numeric identity labels used. All labels require direct RGB inspection.',screen_only=True))
 elif mode in ['anchors','followup']:tasks=rd(U/('ANCHOR_SELECTION.json' if mode=='anchors' else 'FOLLOWUP_SELECTION.json'))['tasks']
 else:raise ValueError(mode)
 rows=[];boards=[]
 for t in tasks:
  rr,b=make(t['clip_id'],t['frames'],mode);rows+=rr;boards.append(b);print(b,flush=True)
  wr(U/f'{mode.upper()}_INPUTS.json',dict(rows=rows,panels=boards,complete=False))
 wr(U/f'{mode.upper()}_INPUTS.json',dict(rows=rows,panels=boards,complete=True))
 for a in archives.values():a.close()
if __name__=='__main__':main()

