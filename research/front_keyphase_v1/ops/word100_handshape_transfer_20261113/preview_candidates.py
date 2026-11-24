"""Select source candidates by missing feature coverage, never by DEV model errors."""
from prepare import *
import zipfile, cv2
from PIL import Image, ImageDraw, ImageFont

def main():
    dst=U/'candidate_preview';dst.mkdir(exist_ok=True)
    manifest_path=O/'word_acquisition_inventory_20261011/FRONT_MANIFEST.jsonl'
    allrows=[json.loads(s) for s in manifest_path.read_text('utf-8-sig').splitlines()]
    codes=['WORD2631','WORD2632','WORD2634','WORD2636','WORD2640','WORD2641']
    archive=Path('W:/WordsInSilence/data/AIHub_103/videos/1.Training/[원천]01_real_word_video.zip')
    output=[];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20)
    board=Image.new('RGB',(1800,830),'white');draw=ImageDraw.Draw(board)
    with zipfile.ZipFile(archive) as z:
        for col,code in enumerate(codes):
            meta=next(r for r in allrows if r['word_code']==code and r['clip_id'].endswith('REAL01_F'))
            member=meta['video']['member'];blob=z.read(member);path=dst/(meta['clip_id']+'.mp4');path.write_bytes(blob)
            cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));a,b=meta['intervals'][0];selected=[]
            for row,t in enumerate([a+.35*(b-a),a+.65*(b-a)]):
                f=min(frames-1,round(t*fps));cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok
                h,w=img.shape[:2];crop=img[int(h*.03):int(h*.85),int(w*.28):int(w*.72)]
                im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));im.thumbnail((290,360));x=col*300+(300-im.width)//2;y=35+row*405
                board.paste(im,(x,y));draw.text((col*300+8,y+365),f'{code} {meta["labels"][0]} f{f}',font=font,fill='#172b49')
                selected.append(dict(frame=f,time=t))
            cap.release()
            output.append(dict(code=code,labels=meta['labels'],clip_id=meta['clip_id'],record_id=1,source=str(archive),member=member,video=str(path),sha256=sha(path),frames=frames,fps=fps,selected=selected,scope='FIT01 source preview only, no annotation admission or model evaluation'))
    board.save(dst/'CONTACT_SHEET.jpg',quality=94)
    write(U/'EXPANSION_SOURCE_PREVIEW.json',dict(rows=output,metadata_sha256=sha(manifest_path),purpose='Preview diverse numeric hand configurations, not fixed game vocabulary; further multi-record review required'))
    print([(r['code'],r['labels']) for r in output])
if __name__=='__main__':main()
