from prepare import *
from guard import quarantine
import cv2,av
from PIL import Image,ImageDraw,ImageFont
def draw(im,xy):
    im=im.copy()
    for side,start,arm,col in [('L',91,[5,7,9],(180,70,180)),('R',112,[6,8,10],(80,180,0))]:
        edges=list(zip(arm[:-1],arm[1:]))
        for d in [1,5,9,13,17]:edges += [(start,start+d)]+[(start+k,start+k+1) for k in range(d,d+3)]
        for a,b in edges:
            p,q=xy[a],xy[b]
            if np.isfinite([p,q]).all() and 0<=p[0]<640 and 0<=q[0]<640 and 0<=p[1]<480 and 0<=q[1]<480:cv2.line(im,tuple(p.astype(int)),tuple(q.astype(int)),col,2)
    return im
def main():
    plan=read(U/'INPUT_PLAN.json')['rows'];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    selected=[('NIA_SL_WORD0739_REAL11_F',63),('NIA_SL_WORD0739_REAL11_F',80),('NIA_SL_WORD0003_REAL06_F',66)]
    panels=[]
    for cid,fi in selected:
        r=next(x for x in plan if x['clip_id']==cid);z=np.load(r['cache'] or U/'crops'/(cid+'.npz'));g=quarantine(z['xy_px'],z['score'],z['timestamp_us'],640,480);l,t,rr,b=r['crop']
        with av.open(r['source']) as c:
            for i,fr in enumerate(c.decode(video=0)):
                if i==fi:im=cv2.resize(fr.to_ndarray(format='bgr24')[t:b,l:rr],(640,480));break
        imgs=[im,draw(im,z['xy_px'][fi]),draw(im,g['xy_px'][fi])]
        sheet=Image.new('RGB',(1152,342),'white');d=ImageDraw.Draw(sheet)
        for k,(part,title) in enumerate(zip(imgs,['실제 공개 영상','이전 관절 입력','보완 후 관절 입력'])):
            sheet.paste(Image.fromarray(cv2.cvtColor(part,cv2.COLOR_BGR2RGB)).resize((384,288)),(k*384,45));d.text((k*384+12,8),title,font=font,fill='#1b293b')
        panels.append(sheet)
    allimg=Image.new('RGB',(1152,1026),'white')
    for k,im in enumerate(panels):allimg.paste(im,(0,k*342))
    allimg.save(U/'BEFORE_AFTER.jpg',quality=92);panels[0].save(U/'BEFORE_AFTER_SINGLE.jpg',quality=92)
    write(U/'VISUAL_INDEX.json',[dict(clip_id=c,frame=f,source='Public original RGB and unchanged raw RTMW versus frozen guard output',note='Left purple/right green; absent points excluded, no repaired coordinates') for c,f in selected])
if __name__=='__main__':main()
