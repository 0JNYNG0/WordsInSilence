from prepare_ordinary import U,O,read,sha,write
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import cv2,numpy as np
def main():
    parent=U.parent;tasks=[]
    for r in read(parent/'SOURCES_NEW.json')['rows']:
        if r['person'] not in [1,4,7,11,12,13]:continue
        rec=read(parent/'records'/f'{r["clip_id"]}.json');a,b=r['source_interval']
        cap=cv2.VideoCapture(r['video']);fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        rec['selected_frames']=[round((a+(b-a)*q)*fps) for q in [.35,.65]];tasks.append(rec)
    for rec in read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']:
        if rec['word']=='WORD0099' and rec['person'] in [1,2,3]:
            rec['selected_frames']=[rec['anchors'][0],rec['anchors'][-1]];tasks.append(rec)
    write(U/'EXTERNAL_SELECTION_FROZEN.json',dict(tasks=tasks,selection='numbers FIT01/04/07, DEV11/12/13; two fixed .35/.65 interval frames; apple FIT01/02/03 first and last existing anchors; before model predictions'))
    output=[];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20)
    for rec in tasks:
        assert sha(rec['video'])==rec['source_sha256'];assert sha(rec['raw'])==rec['raw_sha256']
        raw=np.load(rec['raw']);cap=cv2.VideoCapture(rec['video'])
        for frame in rec['selected_frames']:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
            im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));w,h=im.size
            ip=U/'rgb'/f'{rec["clip_id"]}_{frame}.jpg';im.save(ip,quality=97)
            i=int(np.flatnonzero(raw['frame_id']==frame)[0]);cell=Image.new('RGB',(600,350),'white');d=ImageDraw.Draw(cell)
            d.text((8,4),f'{rec["label"]} ID{rec["person"]:02} f{frame}  R / L',font=font,fill='#152a45')
            ctx=im.crop((int(w*.23),int(h*.03),int(w*.76),int(h*.9)));ctx.thumbnail((112,280));cell.paste(ctx,(1,40));boxes={}
            for side,sl,x in [('R',slice(112,133),118),('L',slice(91,112),358)]:
                pts=raw['xy_px'][i,sl];good=np.isfinite(pts).all(1)&(raw['score'][i,sl]>.2)
                if good.sum()>=5:
                    q=pts[good];lo=np.quantile(q,.05,0);hi=np.quantile(q,.95,0);center=(lo+hi)/2;size=max(float(max(hi-lo))*1.7,100.)
                else:center=np.array([w*(.42 if side=='R' else .58),h*.4]);size=300
                box=(max(0,int(center[0]-size/2)),max(0,int(center[1]-size/2)),min(w,int(center[0]+size/2)),min(h,int(center[1]+size/2)))
                cut=im.crop(box);cut.thumbnail((235,285));cell.paste(cut,(x+(235-cut.width)//2,40+(285-cut.height)//2));boxes[side]=box
            cp=U/'rgb'/f'{rec["clip_id"]}_{frame}_review.jpg';cell.save(cp,quality=96)
            output.append(dict(**{k:rec[k] for k in ['clip_id','word','person','role']},frame=frame,pts_us=int(raw['timestamp_us'][i]),source_video=rec['video'],source_sha256=rec['source_sha256'],image=str(ip),image_sha256=sha(ip),review=str(cp),crop_boxes=boxes,raw_path=rec['raw'],raw_sha256=rec['raw_sha256']))
        cap.release()
    for word in sorted({r['word'] for r in output}):
        for role in ['fit','development']:
            rr=[r for r in output if r['word']==word and r['role']==role]
            if not rr:continue
            board=Image.new('RGB',(1800,700),'white')
            for j,r in enumerate(rr):board.paste(Image.open(r['review']),((j%3)*600,(j//3)*350))
            board.save(U/'panels'/f'{word}_{role}.jpg',quality=97)
    write(U/'EXTERNAL_REVIEW_INPUTS.json',dict(rows=output,complete=True,review_scope='native original RGB; no overlay; RTMW only locates crops'))
    print('PREPARED',len(output))
if __name__=='__main__':main()
