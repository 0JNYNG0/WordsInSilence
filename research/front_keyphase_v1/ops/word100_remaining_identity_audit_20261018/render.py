from audit import *
from PIL import Image,ImageDraw,ImageFont
import cv2,zipfile,zlib
F=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',23)
A=O/'word100_label_only_audit_20261018'
def main():
    (U/'evidence').mkdir(exist_ok=True);rows={r['clip_id']:r for r in read(U/'CONTEXT.json')['rows']};a=read(U/'ANNOTATED.json');events=[dict(x,kind='remaining') for x in read(U/'REMAINING9.json')]
    events += [dict(x,kind='elbow_counterexample') for x in a if x['verdict']=='normal_identity' and x['elbow_only'] is not None and x['elbow_only']['side']!=x['actual_side']]
    events += [dict(x,kind='delayed_change',id='delay') for x in read(U/'PROPOSALS.json') if x['side']!=x['old_hint']]
    records=[]
    for x in events:
        r=rows[x['clip_id']];video=next((q/'videos'/f"{x['clip_id']}.mp4" for q in (A,G,P,S,O/'word100_track_revalidation_20261018') if (q/'videos'/f"{x['clip_id']}.mp4").exists()),None)
        if video is None:
            (U/'videos').mkdir(exist_ok=True);video=U/'videos'/f"{x['clip_id']}.mp4";src=r['source']
            if not video.exists():
                with zipfile.ZipFile(src['archive']) as z:info=z.getinfo(src['member']);blob=z.read(info)
                assert len(blob)==src['bytes'] and hashlib.sha256(blob).hexdigest()==src['sha256'] and zlib.crc32(blob)&0xffffffff==info.CRC;video.write_bytes(blob)
        assert sha(video)==r['source']['sha256'];raw=load(r['views'][x['view']]['raw']);p=load(P/'results'/f"{x['clip_id']}_{x['view']}.npz");cap=cv2.VideoCapture(str(video));page=Image.new('RGB',(2400,1040),'#132638');fs=[]
        for j,delta in enumerate([-1,0,1,2,3,7]):
            f=max(0,min(len(raw['frame_id'])-1,x['index']+delta));fid=int(raw['frame_id'][f]);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
            h=int(raw['height'][f]);cv2.line(im,(0,h-1),(im.shape[1]-1,h-1),(0,220,255),4)
            # full original image remains visible; yellow line is SENSOR edge.
            for k in range(int(raw['raw_count'][f])):
                color=(80,255,70) if k==0 else (255,180,70)
                for pt,good in zip(raw['hand_xy_px'][f,k],p['point_mask'][f,k]):
                    if good:cv2.circle(im,tuple(pt.astype(int)),5,color,-1)
                pt=raw['hand_xy_px'][f,k,0];cv2.putText(im,f'raw{k}',tuple(pt.astype(int)),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
            for k in (11,12,13,14):
                pt=raw['pose_xy_px'][f,k]
                if np.isfinite(pt).all():cv2.circle(im,tuple(pt.astype(int)),8,(200,50,255),2)
            tile=Image.new('RGB',(800,520),'#132638');tile.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((800,450)),(0,70));d=ImageDraw.Draw(tile);d.text((8,0),f"{x['word']} {x['view']} f{fid} · 손후보 {int(raw['raw_count'][f])}개",font=F,fill='white')
            d.text((8,32),'관측점 '+str(p['point_mask'][f,:int(raw['raw_count'][f])].sum(1).tolist())+' · 기존제안 '+str(p['sides'][f,:int(raw['raw_count'][f])].tolist()),font=F,fill='#d6e8fc');page.paste(tile,(j%3*800,j//3*520));fs.append(f)
        cap.release();name=f"{x['kind']}_{x['id']}.jpg";page.save(U/'evidence'/name,quality=95);records.append({'event':x,'image':str(U/'evidence'/name),'frames':fs,'source_video':str(video),'source_sha256':sha(video)})
    write(U/'VISUAL_EVIDENCE.json',records);print('Rendered',len(records))
if __name__=='__main__':main()
