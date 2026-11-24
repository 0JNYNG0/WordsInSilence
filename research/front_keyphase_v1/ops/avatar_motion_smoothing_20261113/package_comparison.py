import json,hashlib,av,cv2,numpy as np
from pathlib import Path
from fractions import Fraction
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;P=U.parent/'avatar_motion_pilot_20261113'
font=lambda n:ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
navy='#1b304b';gray='#536273'
source=[]
with av.open(str(P/'video_only/NIA_SL_WORD1567_REAL02_F.mp4'))as c:
    for f in c.decode(video=0):source.append(f.to_ndarray(format='rgb24'))
assert len(source)==127
def cropped_source(f,size):
    im=source[f];h,w=im.shape[:2]
    return Image.fromarray(im[:1000,w//2-500:w//2+500]).resize((size,size))
def avatar_crop(path,size):
    im=Image.open(path).convert('RGB');w,h=im.size
    return im.crop((round(w*70/720),round(h*110/720),round(w*650/720),round(h*690/720))).resize((size,size))
def encode(path,images,fps,size):
    count=0
    with av.open(str(path),'w')as c:
        s=c.add_stream('libx264',rate=fps);s.width,s.height=size;s.pix_fmt='yuv420p';s.options={'crf':'19','preset':'fast'}
        for n,im in enumerate(images):
            f=av.VideoFrame.from_ndarray(np.asarray(im),format='rgb24');f.pts=n;f.time_base=Fraction(1,fps)
            for pkt in s.encode(f):c.mux(pkt)
            count+=1
        for pkt in s.encode():c.mux(pkt)
    with av.open(str(path))as c:
        frames=list(c.decode(video=0));assert len(frames)==count
        assert max(abs(float(f.pts*f.time_base)-i/fps)for i,f in enumerate(frames))<1e-6
    return {'frames':count,'fps':fps,'duration_seconds':count/fps,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
titles=['실제 공개 동작','기존 캐릭터 동작','손목 회전 보완']
def combined():
    for i in range(254):
        canvas=Image.new('RGB',(1536,598),'white');d=ImageDraw.Draw(canvas)
        for x,title in enumerate(titles):d.text((x*512+20,12),title,font=font(27),fill=navy)
        canvas.paste(cropped_source(i//2,512),(0,56))
        for x,label in enumerate(['baseline','candidate'],1):canvas.paste(avatar_crop(U/(label+'_frames60')/('frame%03d.png'%i),512),(x*512,56))
        d.text((20,571),'같은 시간 · %.2f초     |     얼굴 부위 위치와 일부 손가락 회전은 아직 다릅니다.'%(i/60),font=font(17),fill=gray)
        yield canvas
records={}
records['SOURCE_BASELINE_CANDIDATE_60FPS.mp4']=encode(U/'SOURCE_BASELINE_CANDIDATE_60FPS.mp4',combined(),60,(1536,598))
records['CANDIDATE_60FPS.mp4']=encode(U/'CANDIDATE_60FPS.mp4',(Image.open(U/'candidate_frames60'/('frame%03d.png'%i)).convert('RGB')for i in range(254)),60,(512,512))
# Direct before/after native frame crops, no visual magnification difference between conditions.
canvas=Image.new('RGB',(1560,1160),'white');d=ImageDraw.Draw(canvas)
d.text((28,14),'기존 Mixamo 뼈대에서 회전 흐름 보완',font=font(38),fill=navy)
for x,title in enumerate(titles):d.text((x*512+28,78),title,font=font(27),fill=navy)
for row,sf in enumerate([64,96]):
    y=122+row*476
    ims=[cropped_source(sf,468)]+[avatar_crop(U/(label+'_source%03d.png'%sf),468)for label in ['baseline','candidate']]
    for x,im in enumerate(ims):canvas.paste(im,(x*512+22,y))
    d.rounded_rectangle((30,y+430,192,y+461),radius=5,fill=navy);d.text((39,y+431),'같은 순간 %.2f초'%(sf/30),font=font(18),fill='white')
d.text((28,1090),'팔 위치와 손가락 굽힘은 유지하고, 양손목의 작은 회전 변화만 제한적으로 줄였습니다.',font=font(23),fill=navy)
d.text((28,1124),'추정 3D의 손·얼굴 위치 차이와 접촉 정확도는 별도 보완이 필요합니다.',font=font(21),fill=gray)
canvas.save(U/'SOURCE_BASELINE_CANDIDATE_KO.png')
records['SOURCE_BASELINE_CANDIDATE_KO.png']={'bytes':(U/'SOURCE_BASELINE_CANDIDATE_KO.png').stat().st_size,'sha256':hashlib.sha256((U/'SOURCE_BASELINE_CANDIDATE_KO.png').read_bytes()).hexdigest()}
(U/'PACKAGE_VERIFIED.json').write_text(json.dumps({'status':'COMPLETE_VIDEO_TIMING_VERIFIED','source_native_frames':127,'source_fps':30,'output_fps':60,'source_display':'each actual native source frame held twice, no frame synthesis','avatar_display':'native keys plus half-frame interpolation, final native pose held on last half-frame','records':records},indent=2)+'\n','utf-8')
print('PACKAGED',list(records))
