"""Matched original RGB / supplied keypoints / frozen full-frame MediaPipe views."""
import study as S
import numpy as np,cv2,json,html
from PIL import Image,ImageDraw,ImageFont

U=S.U
FONT='C:/Windows/Fonts/malgun.ttf'
def font(n):return ImageFont.truetype(FONT,n)
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
COLORS=[(255,75,160),(255,220,40)]
def overlay(rgb,xy,valid):
    im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
    for side in range(2):
        for a,b in EDGES:
            if valid[side,a] and valid[side,b]:d.line([tuple(xy[side,a]),tuple(xy[side,b])],fill=COLORS[side],width=3)
        for j,p in enumerate(xy[side]):
            if valid[side,j]:
                x,y=p;d.ellipse((x-4,y-4,x+4,y+4),fill=COLORS[side],outline=(10,10,10))
                if j in [0,4,8,12,16,20]:d.text((x+4,y+2),str(j),font=font(13),fill=COLORS[side],stroke_width=1,stroke_fill='black')
    return im

def frames_for(r,z):
    n=r['frames'];timestamps=z['timestamp_us']/1e6
    f=S.read(U/'FROZEN.json');row=next(x for x in f['rows'] if x['clip_id']==r['clip_id'])
    a,b=row['intervals'][0];xy=z['baseline_xy'];v=z['baseline_valid']
    overlap=np.linalg.norm(xy[:,0,0]-xy[:,1,0],axis=-1);overlap[~v[:,:,0].all(axis=1)]=np.inf
    step=np.linalg.norm(np.diff(xy[:,:,0],axis=0),axis=-1);step[~np.isfinite(step)]=-1
    missing=(~v[:,:,0]).sum(axis=1);core=(timestamps>=a)&(timestamps<=b)
    missing[~core]=-1
    selected=[(0,'시작'),(n-1,'끝')]+[(int(np.argmin(np.abs(timestamps-(a+(b-a)*q)))),f'동작 {int(q*100)}%') for q in [.25,.5,.75]]
    if np.isfinite(overlap).any():selected.append((int(np.argmin(overlap)),'가장 가까운 양손'))
    if np.max(missing)>0:selected.append((int(np.argmax(missing)),'기존 MP 손 누락'))
    if np.max(step)>0:selected.append((int(np.unravel_index(np.argmax(step),step.shape)[0]+1),'가장 큰 손목 이동'))
    out={}
    for t,why in selected:out.setdefault(t,[]).append(why)
    return sorted(out.items())

def run():
    f=S.read(U/'FROZEN.json');done=S.read(U/'DONE.json');lookup={r['clip_id']:r for r in done['rows']};visual=[]
    (U/'images').mkdir(exist_ok=True)
    for cid in f['visual_ids']:
        r=lookup[cid];z=np.load(r['arrays']);chosen=frames_for(r,z);cap=cv2.VideoCapture(r['video']);imgs={};ti=0
        targets={t for t,_ in chosen}
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if ti in targets:imgs[ti]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            ti+=1
        cap.release();assert len(imgs)==len(targets)
        sheet=Image.new('RGB',(1440,80+len(chosen)*310),(245,248,252));d=ImageDraw.Draw(sheet)
        d.text((18,10),r['label']+' | '+cid,font=font(24),fill=(20,35,55))
        d.text((18,46),'분홍: 촬영자 왼손 / 노랑: 촬영자 오른손 · 번호 0=손목, 4/8/12/16/20=손끝',font=font(17),fill=(45,60,80))
        details=[]
        for row,(t,why) in enumerate(chosen):
            rgb=imgs[t];ah=z['official'][t,:,:,:2];av=z['official_valid'][t];mp=z['mp_xy'][t];mv=z['mp_valid'][t]
            points=np.concatenate([ah[av],mp[mv]],axis=0)
            w,h=rgb.shape[1],rgb.shape[0]
            if len(points):
                x0=max(0,int(points[:,0].min())-65);x1=min(w,int(points[:,0].max())+65);y0=max(0,int(points[:,1].min())-65);y1=min(h,int(points[:,1].max())+65)
            else:x0,y0,x1,y1=0,0,w,h
            if x1<=x0 or y1<=y0:x0,y0,x1,y1=0,0,w,h
            box=(x0,y0,x1,y1);panels=[Image.fromarray(rgb),overlay(rgb,ah,av),overlay(rgb,mp,mv)]
            full=Image.new('RGB',(w*3,h+55),(18,25,35));fd=ImageDraw.Draw(full)
            titles=['원본 RGB','AI Hub 제공 좌표','MediaPipe 전체 화면 관측 (저장된 배정 보완 포함)']
            for col,p in enumerate(panels):
                full.paste(p,(col*w,55));fd.text((col*w+10,8),titles[col],font=font(26),fill='white')
                crop=p.crop(box);crop.thumbnail((464,242),Image.Resampling.LANCZOS)
                ox=col*480+8+(464-crop.width)//2;oy=80+row*310+50
                sheet.paste(crop,(ox,oy));d.text((col*480+10,80+row*310+5),titles[col].split(' (')[0],font=font(18),fill=(20,35,55))
            d.text((12,80+row*310+29),f'frame {t} · {z["timestamp_us"][t]/1e6:.3f}s · '+', '.join(why),font=font(15),fill=(70,85,105))
            fn=f'{cid}_{t:04d}.jpg';full.save(U/'images'/fn,quality=92)
            details.append(dict(frame=t,time_s=float(z['timestamp_us'][t]/1e6),reason=why,crop_box=list(box),full_image='images/'+fn))
        sheet.save(U/'images'/(cid+'_sheet.jpg'),quality=94)
        # Always include full source overview, not just a hand crop.
        m=chosen[len(chosen)//2][0];Image.fromarray(imgs[m]).resize((960,540)).save(U/'images'/(cid+'_overview.jpg'),quality=90)
        visual.append(dict(clip_id=cid,label=r['label'],frames=details,sheet='images/'+cid+'_sheet.jpg',overview='images/'+cid+'_overview.jpg',selection=f['visual_selection_reasons'][cid]))
    S.write('VISUAL_MANIFEST.json',dict(status='MATERIAL_READY_NOT_QUALITY_APPROVAL',clips=visual,frame_count=sum(len(v['frames']) for v in visual)))
    print('VISUAL_READY',len(visual),'frames',sum(len(v['frames']) for v in visual),flush=True)

if __name__=='__main__':run()
