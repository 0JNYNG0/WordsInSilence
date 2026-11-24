"""Public RGB contact sheets only; no detector/inference; preserve original IDs."""
from pathlib import Path
import json,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    auth=json.loads((U/'AUTHORIZATION.json').read_text(encoding='utf8'));out=[]
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17)
    for row in auth['rows']:
        dest=U/'visuals'/row['alias'];dest.mkdir(parents=True,exist_ok=False)
        a=np.load(row['input'],allow_pickle=False);ids=a['frame_id'];times=a['timestamp_us'];cap=cv2.VideoCapture(row['source']);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        assert cap.isOpened() and count==len(ids),(row['alias'],count,len(ids))
        idx=np.unique(np.linspace(0,len(ids)-1,48).round().astype(int));chosen=set(idx.tolist());frames=[];i=0
        canvas=Image.new('RGB',(6*256,8*266),'#f6f7fb');draw=ImageDraw.Draw(canvas)
        while True:
            ok,frame=cap.read()
            if not ok:break
            if i in chosen:
                h,w=frame.shape[:2];crop=(int(w*.18),0,int(w*.82),h)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);im=Image.fromarray(rgb);im.save(dest/f'f{int(ids[i]):04d}.jpg',quality=92)
                thumb=im.crop(crop);thumb.thumbnail((256,240))
                k=len(frames);x=k%6*256;y=k//6*266;canvas.paste(thumb,(x+(256-thumb.width)//2,y+26))
                draw.text((x+5,y+3),f'f{int(ids[i])} {times[i]/1e6:.3f}s',font=font,fill='#111111')
                frames.append(dict(index=i,frame_id=int(ids[i]),timestamp_us=int(times[i]),image=str(dest/f'f{int(ids[i]):04d}.jpg')))
            i+=1
        cap.release();assert i==len(ids)
        canvas.save(dest/'sequence_full.jpg',quality=95)
        meta=dict(alias=row['alias'],source=row['source'],source_sha256=sha(row['source']),frames=count,fps=fps,view_only_crop=crop,input_unchanged=True,selected=frames)
        (dest/'MANIFEST.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8');out.append(meta)
        print(row['alias'],'48 visual samples ready',flush=True)
    (U/'VISUAL_MANIFEST.json').write_text(json.dumps(dict(rows=out,model_inference=0,all_source_frame_ids_preserved=True),ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
