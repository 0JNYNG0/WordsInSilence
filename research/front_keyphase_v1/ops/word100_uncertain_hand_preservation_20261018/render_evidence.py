from run import *
from PIL import Image,ImageDraw,ImageFont
import cv2,zipfile,zlib
F=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22)
def main():
    rows={r['clip_id']:r for r in read(U/'CONTEXT.json')['rows']};changes=read(U/'CHANGES.json');(U/'evidence').mkdir();evidence=[]
    for index,x in enumerate(changes,1):
        r=rows[x['clip_id']];video=next((q/'videos'/f"{x['clip_id']}.mp4" for q in (A,G,P,O/'word100_track_revalidation_20261018') if (q/'videos'/f"{x['clip_id']}.mp4").exists()),None)
        if video is None:
            (U/'videos').mkdir(exist_ok=True);video=U/'videos'/f"{x['clip_id']}.mp4";src=r['source']
            if not video.exists():
                with zipfile.ZipFile(src['archive']) as z:info=z.getinfo(src['member']);blob=z.read(info)
                assert len(blob)==src['bytes'] and hashlib.sha256(blob).hexdigest()==src['sha256'] and zlib.crc32(blob)&0xffffffff==info.CRC;video.write_bytes(blob)
        assert sha(video)==r['source']['sha256'];raw=load(r['views'][x['view']]['raw']);old=load(P/'results'/f"{x['clip_id']}_{x['view']}.npz");new=load(U/'results'/f"{x['clip_id']}_{x['view']}.npz");cap=cv2.VideoCapture(str(video));page=Image.new('RGB',(2400,1020),'#132638');frames=[]
        for j,(delta,plain) in enumerate([(-2,True),(0,True),(2,True),(-1,False),(0,False),(1,False)]):
            f=max(0,min(len(raw['frame_id'])-1,x['index']+delta));fid=int(raw['frame_id'][f]);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
            cv2.line(im,(0,int(raw['height'][f])-1),(im.shape[1]-1,int(raw['height'][f])-1),(0,220,255),3);desc=[]
            for k in range(int(raw['raw_count'][f])):
                name=lambda s:'L' if s==0 else 'R' if s==1 else '보류';desc.append(f"raw{k}: {name(int(old['sides'][f,k]))}→{name(int(new['sides'][f,k]))}")
                if not plain:
                    color=(0,255,90) if k==x['raw_index'] else (255,190,30)
                    for pt in raw['hand_xy_px'][f,k]:cv2.circle(im,tuple(pt.astype(int)),5,color,-1)
                    cv2.circle(im,tuple(raw['hand_xy_px'][f,k,0].astype(int)),12,color,2)
            tile=Image.new('RGB',(800,510),'#132638');tile.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((800,450)),(0,60));draw=ImageDraw.Draw(tile);draw.text((10,0),f"{x['word']} {x['view']} f{fid} {'원본' if plain else '관측 보존·중복 검토'}",font=F,fill='white');draw.text((10,30),' / '.join(desc),font=F,fill='#d0e8fc');page.paste(tile,(j%3*800,j//3*510));frames.append({'index':f,'frame_id':fid,'before':old['sides'][f].tolist(),'after':new['sides'][f].tolist()})
        cap.release();path=U/'evidence'/f'change_{index:02}.jpg';page.save(path,quality=97);evidence.append({'id':index,'change':x,'frames':frames,'image':str(path),'video':str(video),'source_sha256':sha(video)})
    write(U/'VISUAL_EVIDENCE.json',evidence);print('Rendered',len(evidence),'changes')
if __name__=='__main__':main()
