"""Source-frame visual audit, unchanged RGB and matched crops across conditions."""
import study as S
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
U=S.U
def font(n):return ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
colors=[(255,80,165),(255,220,40)]
def drawhand(rgb,xy,valid):
    im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
    for s in range(2):
        for a,b in edges:
            if valid[s,a] and valid[s,b]:d.line([tuple(xy[s,a]),tuple(xy[s,b])],fill=colors[s],width=3)
        for j,(x,y) in enumerate(xy[s]):
            if valid[s,j]:d.ellipse((x-4,y-4,x+4,y+4),fill=colors[s],outline='black')
    return im
def run():
    f=S.read(U/'FROZEN.json');records={r['clip_id']:r for r in S.read(U/'DONE.json')['rows']};result=[];(U/'images').mkdir(exist_ok=True)
    titles=['원본 RGB','A · 제공 좌표 그대로','B · 초반 팔 기준, 일정 이동','C · 매 순간 팔 손목에 맞춤']
    for row in f['visuals']:
        cid=row['clip_id'];r=records[cid];a=np.load(r['array']);targets=row['frames'];cap=cv2.VideoCapture(row['video']);frames={};n=0
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if n in targets:frames[n]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            n+=1
        cap.release();assert len(frames)==len(targets)
        sheet=Image.new('RGB',(1920,100+len(targets)*350),(245,248,252));d=ImageDraw.Draw(sheet)
        d.text((18,12),row['label']+' | '+cid,font=font(26),fill=(20,35,55))
        d.text((18,53),'분홍: 촬영자 왼손 · 노랑: 오른손 | B 이동값은 시작 15프레임의 어깨·팔꿈치로만 계산. 손가락 모양은 바꾸지 않음.',font=font(20),fill=(40,55,75))
        detail=[]
        for i,t in enumerate(targets):
            rgb=frames[t];v=a['source_valid'][t];cv=v&a['c_available'][t,:,None];panels=[Image.fromarray(rgb),drawhand(rgb,a['A'][t],v),drawhand(rgb,a['B'][t],v),drawhand(rgb,a['C'][t],cv)]
            pts=np.concatenate([a['A'][t][v],a['B'][t][v],a['C'][t][cv],a['mp_xy'][t][a['mp_valid'][t]]])
            box=(max(0,int(pts[:,0].min())-65),max(0,int(pts[:,1].min())-65),min(1920,int(pts[:,0].max())+65),min(1080,int(pts[:,1].max())+65))
            for j,panel in enumerate(panels):
                cut=panel.crop(box);cut.thumbnail((464,280));sheet.paste(cut,(j*480+8+(464-cut.width)//2,100+i*350+50))
                d.text((j*480+10,100+i*350+8),titles[j],font=font(19),fill=(20,35,55))
            d.text((18,100+i*350-15),f'frame {t} · {t/30:.3f}s',font=font(17),fill=(50,70,90))
            full=Image.new('RGB',(1920*4,1135),(245,248,252));fd=ImageDraw.Draw(full)
            for j,panel in enumerate(panels):full.paste(panel,(j*1920,55));fd.text((j*1920+10,10),titles[j],font=font(26),fill=(20,35,55))
            fn='images/'+cid+f'_{t:04d}.jpg';full.save(U/fn,quality=92)
            detail.append(dict(frame=t,full=fn,crop=list(box)))
        fn='images/'+cid+'_sheet.jpg';sheet.save(U/fn,quality=94);result.append(dict(clip_id=cid,label=row['label'],sheet=fn,frames=detail))
    S.write('VISUALS.json',dict(clips=result,total_frames=sum(len(r['frames']) for r in result),status='REVIEW_REQUIRED'));print('VISUALS',len(result),flush=True)
if __name__=='__main__':run()
