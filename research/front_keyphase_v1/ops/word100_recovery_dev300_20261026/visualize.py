from common import *
import cv2
from PIL import Image,ImageDraw,ImageFont
def main():
    q=read(U/'SMOKE_VERIFIED.json')['queue'];dest=U/'smoke_visual';dest.mkdir()
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    edges=[(ch[i],ch[i+1]) for ch in ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)) for i in range(4)]
    pages=[]
    for start in range(0,len(q),8):
        board=Image.new('RGB',(1440,1440),'#edf1f6');draw=ImageDraw.Draw(board)
        for ii,a in enumerate(q[start:start+8]):
            z=load(U/'outputs'/(a['clip_id']+'.npz'));i=a['frame'];s=a['side'];k=1 if a['stage']=='mp' else 2
            img=cv2.imread(a['image']);assert img is not None
            hands=z['xy_px'][k,i,33:].reshape(2,21,2);v=z['usable'][k,i,33:].reshape(2,21);pose=z['xy_px'][0,i,:33]
            validpoints=hands[v];center=hands[s][v[s]].mean(0)
            pts=np.vstack([validpoints,pose[[13,14,15,16]]]);pts=pts[np.isfinite(pts).all(-1)]
            lo=np.maximum(0,np.floor(pts.min(0)-80)).astype(int);hi=np.minimum([1920,1080],np.ceil(pts.max(0)+80)).astype(int)
            if (hi-lo<50).any():raise ValueError('Empty visual crop')
            overlay=img.copy()
            for side,color in ((1-s,(235,150,40)),(s,(30,150,250))):
                for x,y in edges:
                    if v[side,x] and v[side,y]:cv2.line(overlay,tuple(hands[side,x].astype(int)),tuple(hands[side,y].astype(int)),color,2)
                for xy,ok in zip(hands[side],v[side]):
                    if ok:cv2.circle(overlay,tuple(xy.astype(int)),3,color,-1)
            x=(ii%2)*720;y=(ii//2)*360
            for j,im in enumerate((img,overlay)):
                crop=Image.fromarray(cv2.cvtColor(im[lo[1]:hi[1],lo[0]:hi[0]],cv2.COLOR_BGR2RGB));crop.thumbnail((354,300))
                board.paste(crop,(x+j*360,y+40))
            draw.text((x+5,y+5),f"{start+ii:02d} {a['clip_id'][7:22]} f{i} {'L' if s==0 else 'R'}",font=font,fill='#132a44')
            draw.text((x+5,y+334),f"{a['stage']} points={a['points']} angle proxy={a['comparable_angle_count']}",font=font,fill='#132a44')
        path=dest/f'page_{start//8:02d}.jpg';board.save(path,quality=95);pages.append(str(path))
    write(dest/'INDEX.json',dict(pages=pages,queue=q,note='Raw and overlay side by side; orange newly recovered side, blue existing other side. All first3 added hand frames shown. Appearance is not precise21joint truth.'))
    print(json.dumps(pages))
if __name__=='__main__':main()
