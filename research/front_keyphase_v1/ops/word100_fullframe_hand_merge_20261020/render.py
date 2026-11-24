"""Source image evidence for every added/identity-conflict hand, saved coordinates only."""
import json,math
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf8'))
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',13)
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
def skeleton(img,p,c,mask=None):
    good=np.isfinite(p).all(-1) if mask is None else np.array(mask)&np.isfinite(p).all(-1)
    for a,b in edges:
        if good[a] and good[b]:cv2.line(img,tuple(np.int32(p[a])),tuple(np.int32(p[b])),c,2,cv2.LINE_AA)
    for j,xy in enumerate(p):
        if good[j]:cv2.circle(img,tuple(np.int32(xy)),3,c,-1)
def fit(bgr,size):
    im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail(size)
    out=Image.new('RGB',size,'#ecf0f5');out.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return out
def main():
    (U/'evidence').mkdir(exist_ok=True)
    rows={r['clip_id']:r for r in read(U/'FROZEN.json')['rows']}
    targets=[dict(a,kind='added') for a in read(U/'ADDITIONS.json')]+[dict(a,kind='identity_conflict') for a in read(U/'CONFLICTS.json')]
    targets.sort(key=lambda x:(x['clip_id'],x['index'],x['decision'],x['kind']))
    caps={};cache={};cards=[]
    for n,a in enumerate(targets):
        cid=a['clip_id'];r=rows[cid];i=a['index'];j=a['decision']
        if cid not in cache:
            with np.load(r['raw']) as z:raw={k:z[k] for k in z.files}
            cache[cid]=(read(U/'records'/f'{cid}.json'),raw)
            caps[cid]=cv2.VideoCapture(r['video'])
        d,z=cache[cid];e=d['frames'][i];p=np.array(e['full'][j]['xy']);q=e['decisions'][j]
        cap=caps[cid];cap.set(cv2.CAP_PROP_POS_FRAMES,e['frame_id']);ok,bgr=cap.read();assert ok
        old=z['hand_xy_px'][i,:int(z['raw_count'][i])];drawn=bgr.copy()
        for h in old:skeleton(drawn,h,(255,160,0))
        skeleton(drawn,p,(0,240,40),q['joint_supported'] if a['kind']=='added' else None)
        if a['kind']=='added' and q['crop_index'] is not None:
            cp=np.array(e['crop'][q['crop_index']]['xy'])
            # Tiny magenta points expose scale agreement without obscuring the full image.
            for xy in cp:
                if np.isfinite(xy).all():cv2.circle(drawn,tuple(np.int32(xy)),1,(220,30,220),-1)
        lo=np.maximum(p.min(0)-65,0).astype(int);hi=np.minimum(p.max(0)+65,[bgr.shape[1],bgr.shape[0]]).astype(int)
        cv2.line(drawn,(0,r['box'][3]),(bgr.shape[1]-1,r['box'][3]),(0,240,255),3)
        card=Image.new('RGB',(700,270),'white');dd=ImageDraw.Draw(card)
        label=f"{n:03d} {r['label']} {r['signer_id']:02d} f{i} {'좌' if a['side']==0 else '우'}"
        dd.text((5,3),label,font=font,fill='black')
        status=(f"추가 · 원본 {a['native_points']}점 → 학습범위 {a['roi_points']}점 · {'본동작' if a['core'] else '주석밖'}" if a['kind']=='added' else '기존/새 좌우 충돌 · 새 손 추가 안 함')
        dd.text((5,26),status,font=small,fill='#315473')
        card.paste(fit(drawn,(310,213)),(0,55))
        card.paste(fit(bgr[lo[1]:hi[1],lo[0]:hi[0]],(185,213)),(315,55))
        card.paste(fit(drawn[lo[1]:hi[1],lo[0]:hi[0]],(195,213)),(503,55))
        cards.append(card);a.update(id=n,page=n//10+1,cell=n%10)
    for cap in caps.values():cap.release()
    for page in range(math.ceil(len(cards)/10)):
        sheet=Image.new('RGB',(1400,1350),'#ddd')
        for c in range(10):
            if page*10+c<len(cards):sheet.paste(cards[page*10+c],((c%2)*700,(c//2)*270))
        sheet.save(U/'evidence'/f'check_{page+1:02d}.jpg',quality=96)
    (U/'REVIEW_TARGETS.json').write_text(json.dumps(targets,ensure_ascii=False,indent=2),encoding='utf8')
    print('RENDERED',len(targets),'targets',math.ceil(len(cards)/10),'sheets')
if __name__=='__main__':main()
