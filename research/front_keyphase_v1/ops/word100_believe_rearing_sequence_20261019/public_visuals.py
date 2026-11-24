"""Predeclared six public TRAIN clips; local RGB evidence, no user recordings."""
from pathlib import Path
import json,hashlib,zipfile
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;B=U.parent/'word100_body_boundary_20261017'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (U/'VISUAL_EVIDENCE.json').exists()
    out=U/'public_visuals';out.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20)
    small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
    evidence=[]
    for code in ('WORD1139','WORD0196'):
        for signer in (1,5,9):
            clip=f'NIA_SL_{code}_REAL{signer:02}_F';r=read(B/'data/records'/f'{clip}.json');assert r['role']=='train_candidate'
            p=out/f'{clip}.mp4';src=r['source'];assert not p.exists()
            with zipfile.ZipFile(src['archive']) as z:
                data=z.read(src['member']);assert hashlib.sha256(data).hexdigest()==src['sha256'];p.write_bytes(data)
            cap=cv2.VideoCapture(str(p));assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            span=r['views']['masked']['feature_info']['whole_span'];frameids=np.linspace(*span,16).round().astype(int)
            canvas=Image.new('RGB',(1280,4*208+64),'#111b2b');d=ImageDraw.Draw(canvas)
            d.text((18,14),f"{r['canonical_label']}  공개 TRAIN ID{signer:02} | 전체 {count/fps:.2f}s | 활동 입력 프레임 {span[0]}–{span[1]}",font=font,fill='white')
            for j,fid in enumerate(frameids):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(fid));ok,fr=cap.read();assert ok
                im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((318,179))
                x=(j%4)*320;y=64+(j//4)*208;canvas.paste(im,(x,y))
                d.text((x+6,y+181),f'{fid/fps:.2f}s · frame {fid}',font=small,fill='white')
            cap.release();jpg=out/f'{clip}.jpg';canvas.save(jpg,quality=90)
            evidence.append({'clip_id':clip,'label':r['canonical_label'],'role':'train_candidate','video':str(p),'video_sha256':sha(p),
                'source_archive':src['archive'],'source_member':src['member'],'source_sha256':src['sha256'],
                'montage':str(jpg),'frame_ids':frameids.tolist(),'fps':fps,'video_frames':count,'video_seconds':count/fps})
    (U/'VISUAL_EVIDENCE.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Six public TRAIN videos and 96 timestamped evidence frames ready.')
if __name__=='__main__':main()
