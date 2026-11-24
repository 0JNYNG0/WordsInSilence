from motion_review import *
import av,numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
    source={r['person']:r for r in read(U/'INPUTS.json')['rows']}
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20)
    for person,frames in [(11,list(range(55,65))),(12,list(range(37,49)))]:
        r=source[person];raw=dict(np.load(r['raw']));ch=dict(np.load(r['channels']))
        print(person,'raw/channel',len(raw['frame_id']),len(ch['frame_id']),'ids',ch['frame_id'][frames].tolist())
        out=Image.new('RGB',(1800,340*((len(frames)+4)//5)),'white');dr=ImageDraw.Draw(out)
        with av.open(r['video']) as c:
            for i,fr in enumerate(c.decode(video=0)):
                if i not in frames:continue
                rgb=fr.to_ndarray(format='rgb24');h,w=rgb.shape[:2]
                l,t,rr,b=int(w*.3),int(h*.24),int(w*.69),int(h*.56)
                cut=Image.fromarray(rgb[t:b,l:rr]);cut.thumbnail((355,296));k=frames.index(i);x=k%5*360;y=k//5*340
                # Unmodified native source pixels: no overlaid prediction hides the hand.
                out.paste(cut,(x,y));dr.text((x+4,y+298),f'P{person} native {i}',font=font,fill='#152c46')
        out.save(D/f'failure_{person}.jpg',quality=96)

if __name__=='__main__':main()
