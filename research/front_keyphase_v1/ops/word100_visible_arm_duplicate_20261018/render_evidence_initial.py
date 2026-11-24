"""Public RGB evidence, display only. No detector execution or model access."""
from run import *
import cv2
from PIL import Image,ImageDraw,ImageFont
FONT=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22)
def main():
    rows={r['clip_id']:r for r in read(U/'CONTEXT.json')['rows']};events=read(U/'ARM_CONFLICTS.json');(U/'evidence').mkdir()
    evidence=[]
    for ix,ch in enumerate(events,1):
        r=rows[ch['clip_id']];video=A/'videos'/f"{ch['clip_id']}.mp4";assert sha(video)==r['source']['sha256']
        raw=load(r['views'][ch['view']]['raw']);old=load(P/'results'/f"{ch['clip_id']}_{ch['view']}.npz");new=load(U/'results'/f"{ch['clip_id']}_{ch['view']}.npz")
        cap=cv2.VideoCapture(str(video));page=Image.new('RGB',(2400,1020),'#132638');facts=[]
        for j,(delta,plain) in enumerate([(-2,False),(0,True),(2,False),(-1,False),(0,False),(1,False)]):
            f=max(0,min(len(raw['frame_id'])-1,ch['index']+delta));fid=int(raw['frame_id'][f]);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
            cv2.line(im,(0,int(raw['height'][f])-1),(im.shape[1]-1,int(raw['height'][f])-1),(0,220,255),3)
            words=[]
            for i in range(int(raw['raw_count'][f])):
                name=lambda s:'L' if s==0 else 'R' if s==1 else '보류'
                words.append(f"raw{i}:{name(int(old['sides'][f,i]))}→{name(int(new['sides'][f,i]))}")
                if not plain:
                    color=(0,255,90) if i==ch['raw_index'] else (255,190,30)
                    for q in raw['hand_xy_px'][f,i]:cv2.circle(im,tuple(q.astype(int)),5,color,-1)
                    cv2.circle(im,tuple(raw['hand_xy_px'][f,i,0].astype(int)),13,color,2)
            if not plain:
                p=raw['pose_xy_px'][f];u=raw['pose_usable'][f]
                for s in (0,1):
                    for a,b in [(11+s,13+s),(13+s,15+s)]:
                        if u[a] and u[b] and all(0<=p[k,0]<raw['width'][f] and 0<=p[k,1]<raw['height'][f] for k in (a,b)):
                            cv2.line(im,tuple(p[a].astype(int)),tuple(p[b].astype(int)),(255,255,255),4)
            tile=Image.new('RGB',(800,510),'#132638');tile.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((800,450)),(0,60));draw=ImageDraw.Draw(tile)
            draw.text((10,2),f"{ch['word']} {ch['view']} f{fid} {'원본' if plain else '저장 좌표'}",font=FONT,fill='white');draw.text((10,30),' · '.join(words),font=FONT,fill='#c8e7fa');page.paste(tile,(j%3*800,j//3*510));facts.append({'index':f,'frame_id':fid,'plain':plain})
        cap.release();path=U/'evidence'/f'conflict_{ix:02}.jpg';page.save(path,quality=97)
        evidence.append({'id':ix,'case':ch,'image':str(path),'video_sha256':sha(video),'frames':facts})
    write(U/'VISUAL_EVIDENCE.json',evidence);print('Rendered',len(evidence),'public arm-conflict cases')
if __name__=='__main__':main()
