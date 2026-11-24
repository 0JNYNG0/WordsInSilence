"""Source RGB + newly accepted wrist markers, every smoke wrist; no anatomical label inference."""
from common import *
import cv2,math

def main():
    vr=read(U/'SMOKE_VERIFIED.json');items=vr['review'];(U/'visual').mkdir(exist_ok=True)
    W,H=460,275;perpage=12;rows=[]
    for k,item in enumerate(items):
        im=cv2.imread(item['image']);assert im is not None
        x,y=item['wrist'];uid=item['uid'];out=load(U/'outputs'/(uid+'.npz'));i=item['index'];s=item['side']
        # Upper-body context includes the actual current camera canvas; close-up shows raw and marked pixels.
        panel=np.full((H,W,3),245,np.uint8);small=cv2.resize(im,(230,130));panel[30:160,:230]=small
        cv2.circle(panel,(int(x*230/im.shape[1]),30+int(y*130/im.shape[0])),4,(0,0,255),1)
        x0=max(0,int(x)-100);y0=max(0,int(y)-100);x1=min(im.shape[1],int(x)+100);y1=min(im.shape[0],int(y)+100)
        crop=im[y0:y1,x0:x1];raw=cv2.resize(crop,(210,210));marked=raw.copy()
        q=(int((x-x0)*210/(x1-x0)),int((y-y0)*210/(y1-y0)));cv2.circle(marked,q,5,(0,0,255),2)
        # Pose wrist anchor is independent saved body observation, cyan, not ground truth.
        pj=15+s
        if out['usable'][0,i,pj]:
            px,py=out['xy_px'][0,i,pj];qp=(int((px-x0)*210/(x1-x0)),int((py-y0)*210/(y1-y0)))
            cv2.circle(marked,qp,4,(255,255,0),1)
        panel[30:240,240:450]=marked
        # Unmarked crop below context, preserved aspect is unnecessary for a reviewer seeing both versions.
        panel[165:265,:210]=cv2.resize(crop,(210,100))
        text=f'{k:03d} {uid.replace("NIA_SL_","")} f{i} {"L" if s==0 else "R"} src{item["origin"]}'
        cv2.putText(panel,text,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.36,(20,20,20),1,cv2.LINE_AA)
        if k%perpage==0:sheet=np.full((H*6,W*2,3),255,np.uint8)
        pos=k%perpage;sheet[(pos//2)*H:(pos//2+1)*H,(pos%2)*W:(pos%2+1)*W]=panel
        rows.append(dict(review_index=k,**item))
        if k%perpage==perpage-1 or k==len(items)-1:
            path=U/'visual'/f'wrists_{k//perpage:03d}.jpg';assert cv2.imwrite(str(path),sheet,[cv2.IMWRITE_JPEG_QUALITY,95])
    write(U/'VISUAL_INDEX.json',dict(items=rows,pages=math.ceil(len(items)/perpage),key='red recovered wrist, cyan saved pose wrist; neither model is anatomical ground truth'))
    print(len(items),'wrists',math.ceil(len(items)/perpage),'pages')
if __name__=='__main__':main()
