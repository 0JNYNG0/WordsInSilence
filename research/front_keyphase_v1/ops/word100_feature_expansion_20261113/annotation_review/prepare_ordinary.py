from pathlib import Path
import json,hashlib,zipfile,cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parents[1]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
WORDS=['WORD0003','WORD0739','WORD1133','WORD1290','WORD1567','WORD2523','WORD0071','WORD0500','WORD0678','WORD0949','WORD1139','WORD2551']
def main():
    for d in ['media','rgb','panels']: (U/d).mkdir(exist_ok=True)
    manifest=read(O/'word100_handshape_transfer_20261113/INPUT_MANIFEST.json')['ordinary']
    sources={r['clip_id']:r for r in read(O/'word100_rtmw_joint_learning_20261108/SOURCES.json')}
    tasks=[]
    for code in WORDS:
        fits=[1,4,7] if code in ['WORD0739','WORD1133','WORD1567','WORD2523'] else [1,2,3]
        for person in fits+[11,12,13]:
            cid=f'NIA_SL_{code}_REAL{person:02}_F'; rr=[r for r in manifest if r['clip_id']==cid]
            frames=sorted({r['frame'] for r in rr});assert len(frames)>=2,(cid,frames)
            selected=[frames[round((len(frames)-1)*.25)],frames[round((len(frames)-1)*.75)]]
            assert len(set(selected))==2
            tasks.append(dict(clip_id=cid,word=code,person=person,role='fit' if person<=9 else 'development',frames=selected))
    write(U/'ORDINARY_SELECTION_FROZEN.json',dict(tasks=tasks,selection='12 predefined expressions; fixed recorded IDs and25/75 percentile of existing reviewed frame indices; no model predictions or new DEV performance used',schema=dict(finger_order=['thumb','index','middle','ring','little'],states={'0':'extended','1':'bent','2':'folded','-1':'unknown'})))
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20);archives={};output=[]
    for task in tasks:
        cid=task['clip_id'];src=sources[cid]
        local=O/'demo_handshape_learning_20261113/media'/f'{cid}.mp4'
        if not local.exists():local=U/'media'/f'{cid}.mp4'
        if not local.exists():
            arc=src['archive']
            if arc not in archives:archives[arc]=zipfile.ZipFile(arc)
            local.write_bytes(archives[arc].read(src['member']))
        assert sha(local)==src['source_sha256'],cid
        rec=read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json');raw=np.load(rec['raw_path']);assert sha(rec['raw_path'])==rec['raw_sha256']
        cap=cv2.VideoCapture(str(local))
        for frame in task['frames']:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok,(cid,frame)
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);im=Image.fromarray(rgb);ip=U/'rgb'/f'{cid}_{frame}.jpg';im.save(ip,quality=97)
            i=int(np.flatnonzero(raw['frame_id']==frame)[0]);w,h=im.size
            cell=Image.new('RGB',(600,350),'white');d=ImageDraw.Draw(cell);d.text((8,4),f'{src["label"]} ID{task["person"]:02} f{frame}   R / L',font=font,fill='#152a45')
            ctx=im.crop((int(w*.23),int(h*.03),int(w*.76),int(h*.9)));ctx.thumbnail((112,280));cell.paste(ctx,(1,40))
            boxes={}
            for side,sl,x in [('R',slice(112,133),118),('L',slice(91,112),358)]:
                pts=raw['xy_px'][i,sl];good=np.isfinite(pts).all(1)&(raw['score'][i,sl]>.2)
                if good.sum()>=5:
                    q=pts[good];lo=np.quantile(q,.05,0);hi=np.quantile(q,.95,0);center=(lo+hi)/2;size=max(float(max(hi-lo))*1.7,100.)
                else:
                    center=np.array([w*(.42 if side=='R' else .58),h*.40]);size=300
                l=max(0,int(center[0]-size/2));t=max(0,int(center[1]-size/2));r=min(w,int(center[0]+size/2));b=min(h,int(center[1]+size/2));box=(l,t,r,b)
                cut=im.crop(box);cut.thumbnail((235,285));cell.paste(cut,(x+(235-cut.width)//2,40+(285-cut.height)//2));boxes[side]=box
            cp=U/'rgb'/f'{cid}_{frame}_review.jpg';cell.save(cp,quality=96)
            output.append(dict(**{k:task[k] for k in ['clip_id','word','person','role']},frame=frame,pts_us=int(raw['timestamp_us'][i]),source_video=str(local),source_sha256=src['source_sha256'],image=str(ip),image_sha256=sha(ip),review=str(cp),crop_boxes=boxes,raw_path=rec['raw_path'],raw_sha256=rec['raw_sha256']))
        cap.release()
        write(U/'ORDINARY_REVIEW_INPUTS.json',dict(rows=output,complete=False))
        print('RGB',cid,flush=True)
    for code in WORDS:
        for role in ['fit','development']:
            rr=[r for r in output if r['word']==code and r['role']==role];assert len(rr)==6
            board=Image.new('RGB',(1800,700),'white')
            for j,r in enumerate(rr):board.paste(Image.open(r['review']),((j%3)*600,(j//3)*350))
            board.save(U/'panels'/f'{code}_{role}.jpg',quality=97)
    write(U/'ORDINARY_REVIEW_INPUTS.json',dict(rows=output,complete=True,review_scope='Native RGB only; RTMW used only to propose crop position, not ground truth'))
    for arc in archives.values():arc.close()
    print('COMPLETE',len(output),flush=True)
if __name__=='__main__':main()
