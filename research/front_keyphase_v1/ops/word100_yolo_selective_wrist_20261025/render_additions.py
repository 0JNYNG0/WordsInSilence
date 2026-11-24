from common import *
import cv2
vis=module('selective_old_plot',OLD/'render_report.py')
f=read(U/'FROZEN.json');mr={r['clip_id']:r for r in f['old_rows']+f['new_rows']};ix={r['clip_id']:r for r in read(U/'INDEX.json')}
(U/'additions_visual').mkdir();add=read(U/'ADDITIONS.json')
def draw(im,p,v):
    im=im.copy()
    for s in (0,1):
        co=(50,240,60) if s==0 else (30,170,255)
        for a,b in vis.LINKS:
            if v[s,a] and v[s,b]:cv2.line(im,tuple(p[s,a].astype(int)),tuple(p[s,b].astype(int)),co,2)
        for j in np.flatnonzero(v[s]):cv2.circle(im,tuple(p[s,j].astype(int)),5 if j==0 else 2,co,2)
    return im
for k,a in enumerate(add):
    cid=a['clip_id'];i=a['frame_id'];s=a['side'];r=mr[cid];z=load(r['arrays']);o=load(ix[cid]['arrays']);rec=next(q for q in read(ix[cid]['records'])['frames'] if q['i']==i)
    cap=cv2.VideoCapture(r['rgb_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();cap.release();assert ok
    p=o['p_xy'][i,s,0];cx,cy=p;x0=max(0,min(1920-640,int(cx)-320));y0=max(0,min(1080-440,int(cy)-220))
    base=z['redetected_roi_xy_px'][i,33:].reshape(2,21,2);bv=z['redetected_roi_usable'][i,33:].reshape(2,21)
    q=next(d for d in rec['decisions'] if d['accepted'] and d['side']==s);full=points(rec['full'][q['index']]);crop=points(rec['crop'][q['crop_index']]);allp=np.stack([full,crop]);allv=np.isfinite(allp).all(-1)
    panels=[('Original pixels',im),('M1: MP redetection',draw(im,base,bv)),('P: position only' if a['position_only'] else 'P: full supported hand',draw(im,o['p_xy'][i],o['p_valid'][i])),('YOLO full(green) / crop(orange)',draw(im,allp,allv))]
    tiles=[]
    for title,im2 in panels:
        tile=im2[y0:y0+440,x0:x0+640];tile=cv2.copyMakeBorder(tile,36,0,0,0,cv2.BORDER_CONSTANT,value=(20,25,35));cv2.putText(tile,title,(10,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);tiles.append(tile)
    cv2.imwrite(str(U/'additions_visual'/f'A{k+1:02d}.jpg'),np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
print('IMAGES',len(add))
