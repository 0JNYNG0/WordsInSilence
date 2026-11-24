"""Offline evidence only. No detector/assigner/model execution."""
from audit import *

def main():
 c=read(U/'CONTEXT.json');events=read(U/'EVENTS.json');draft=read(U/'VISUAL_DRAFT.json');rows={r['clip_id']:r for r in c['rows']}
 out=[]
 for eid in draft['needs_detail']:
  e=events[eid-1];r=rows[e['clip_id']];raw=load(r['views'][e['view']]['raw']);tr=read(Path(r['views'][e['view']]['assignment']).with_name('trace.json'))
  cap=cv2.VideoCapture(str(U/'videos'/f"{e['clip_id']}.mp4"));page=Image.new('RGB',(1800,1320),'#122334');facts=[]
  for n,(delta,plain) in enumerate([(-4,False),(0,True),(4,False),(-2,False),(0,False),(2,False)]):
   idx=max(0,min(len(raw['frame_id'])-1,e['index']+delta));fid=int(raw['frame_id'][idx]);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
   pic=im.copy();h=int(raw['height'][idx]);w=im.shape[1];cv2.line(pic,(0,h),(w-1,h),(0,210,255),3)
   labels=[]
   for i in range(int(raw['raw_count'][idx])):
    side=tr[idx]['assigned_sides'][i];why=tr[idx]['reasons'][i];labels.append(f"raw{i}:{['L','R'][side] if side in (0,1) else '?'}")
    if not plain:
     pts=raw['hand_xy_px'][idx,i];col=(0,255,80) if (delta==0 and i==e['raw_index']) else [(255,180,30),(230,80,255)][i%2]
     for q in pts:
      if np.isfinite(q).all():cv2.circle(pic,tuple(q.astype(int)),4,col,-1)
     cv2.circle(pic,tuple(pts[0].astype(int)),12,col,2)
   sh=raw['pose_xy_px'][idx,[11,12]];cx=int(np.nanmean(sh[:,0])) if np.isfinite(sh).all() else w//2;x0=max(0,min(w-1080,cx-540))
   tile=Image.new('RGB',(600,660),'#122334');tile.paste(Image.fromarray(cv2.cvtColor(pic[:,x0:x0+1080],cv2.COLOR_BGR2RGB)).resize((600,600)),(0,60));draw=ImageDraw.Draw(tile)
   draw.text((8,4),f"#{eid} {e['word']} f{fid} {'원본' if plain else '관측'}",font=FONT,fill='white');draw.text((8,30),' '.join(labels),font=FONT,fill='#d4ecf7');page.paste(tile,(n%3*600,n//3*660))
   facts.append({'delta_index':delta,'plain':plain,'index':idx,'frame_id':fid,'trace':tr[idx]})
  cap.release();path=U/'details'/f'{eid:03}.jpg';assert not path.exists();page.save(path,quality=96);out.append({'id':eid,'image':str(path),'frames':facts})
 write(U/'DETAILS.json',out);print('Detail evidence:',len(out))
if __name__=='__main__':main()
