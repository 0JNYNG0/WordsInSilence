import json,math
from pathlib import Path
from collections import defaultdict
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf8'))
FONT='C:/Windows/Fonts/malgun.ttf'
font=ImageFont.truetype(FONT,17)
small=ImageFont.truetype(FONT,14)
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
def skeleton(img,pts,color):
    for a,b in EDGES:cv2.line(img,tuple(np.int32(pts[a])),tuple(np.int32(pts[b])),color,2,cv2.LINE_AA)
    for p in pts:cv2.circle(img,tuple(np.int32(p)),3,color,-1)
def fit(img,size):
    im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));im.thumbnail(size)
    canvas=Image.new('RGB',size,'#eceff2');canvas.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return canvas
def main():
    (U/'evidence').mkdir(exist_ok=True)
    rows={r['clip_id']:r for r in read(U/'FROZEN.json')['rows']}
    adds=read(U/'ADDITIONS.json');groups=defaultdict(list)
    for a in adds:groups[(a['clip_id'],a['index'],a['side'])].append(a)
    ordered=sorted(groups,key=lambda k:(k[0],k[1],k[2]));cards=[];targets=[];caps={}
    for no,key in enumerate(ordered):
        cid,i,side=key;r=rows[cid];data=read(U/'records'/f'{cid}.json');e=data['frames'][i]
        if cid not in caps:caps[cid]=cv2.VideoCapture(r['video'])
        cap=caps[cid];cap.set(cv2.CAP_PROP_POS_FRAMES,e['frame_id']);ok,bgr=cap.read();assert ok
        bgr=bgr[:r['box'][3],:r['box'][2]];base=bgr.copy()
        for a in groups[key]:
            pts=np.array(e[a['mode']][a['candidate_index']]['xy']);skeleton(base,pts,(0,210,0) if a['mode']=='image' else (240,40,240))
        pts=np.array([e[a['mode']][a['candidate_index']]['xy'] for a in groups[key]]).reshape(-1,2)
        mn=pts.min(0)-60;mx=pts.max(0)+60
        x0,y0=np.maximum(mn,0).astype(int);x1,y1=np.minimum(mx,[bgr.shape[1],bgr.shape[0]]).astype(int)
        card=Image.new('RGB',(650,236),'white');draw=ImageDraw.Draw(card)
        modes='+'.join('전체' if a['mode']=='image' else '국소' for a in groups[key])
        draw.text((5,2),f'{no:03d} {r["label"]} {r["signer_id"]:02d} f{i} {"좌" if side==0 else "우"} {modes} {"본동작" if e["core"] else "구간밖"}',font=font,fill='black')
        card.paste(fit(base,(345,205)),(0,28));card.paste(fit(base[y0:y1,x0:x1],(300,205)),(350,28))
        cards.append(card)
        targets.append(dict(id=no,clip_id=cid,index=i,side=side,additions=groups[key],page=no//10+1,cell=no%10))
    for cap in caps.values():cap.release()
    for page in range(math.ceil(len(cards)/10)):
        canvas=Image.new('RGB',(1300,1180),'#dddddd')
        for c in range(10):
            idx=page*10+c
            if idx<len(cards):canvas.paste(cards[idx],((c%2)*650,(c//2)*236))
        canvas.save(U/'evidence'/f'added_{page+1:02d}.jpg',quality=94)
    (U/'REVIEW_TARGETS.json').write_text(json.dumps(targets,ensure_ascii=False,indent=2),encoding='utf8')
    # Representative first frames: whole source includes resting hands, sensor boundary hides them.
    initial=read(U/'INITIAL_AUDIT.json');pick=[initial['clips'][k] for k in [0,3,8]]
    out=Image.new('RGB',(1200,390*len(pick)),'white');d=ImageDraw.Draw(out)
    for j,item in enumerate(pick):
        r=rows[item['clip_id']];cap=cv2.VideoCapture(r['video']);ok,bgr=cap.read();cap.release();assert ok
        y=r['box'][3];cv2.line(bgr,(0,y),(bgr.shape[1]-1,y),(0,255,255),5)
        out.paste(fit(bgr,(600,350)),(0,j*390+40));out.paste(fit(bgr[:y],(600,350)),(600,j*390+40))
        d.text((5,j*390+5),f'{r["label"]} {r["signer_id"]:02d} | 첫 프레임 원본 전체 손 후보2 / 제한 화면 후보{item["views"]["sensor"]["first_frame_raw_hands"]}',font=font,fill='black')
    out.save(U/'evidence/initial.jpg',quality=95)
    print('Rendered',len(cards),'unique hand-frames;',math.ceil(len(cards)/10),'sheets')
if __name__=='__main__':main()
