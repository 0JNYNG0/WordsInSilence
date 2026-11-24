import json,hashlib,zipfile,zlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent;E=U/'evidence';E.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def font(n):return ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
rows={r['clip_id']:r for r in read(U/'CONTEXT.json')['rows']};events=read(U/'CHANGES.json');index=[]
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
for j,x in enumerate(events,1):
    r=rows[x['clip_id']];src=None
    for name in ['word100_label_only_audit_20261018','word100_image_arm_connectivity_20261018','word100_uncertain_hand_preservation_20261018','word100_remaining_identity_audit_20261018','word100_visible_arm_duplicate_20261018','word100_single_arm_recheck_20261018','word100_track_revalidation_20261018']:
        p=O/name/'videos'/f"{r['clip_id']}.mp4"
        if p.exists():assert sha(p)==r['source']['sha256'];src=p;break
    if src is None:
        (U/'videos').mkdir(exist_ok=True);src=U/'videos'/f"{r['clip_id']}.mp4";assert not src.exists()
        z=r['source']
        with zipfile.ZipFile(z['archive']) as archive:blob=archive.read(z['member'])
        assert len(blob)==z['bytes'] and hashlib.sha256(blob).hexdigest()==z['sha256'] and zlib.crc32(blob)&0xffffffff==z['crc32'];src.write_bytes(blob)
    n=np.load(r['views'][x['view']]['raw']);b=np.load(O/'word100_uncertain_hand_preservation_20261018/results'/f"{r['clip_id']}_{x['view']}.npz")
    cap=cv2.VideoCapture(str(src));assert cap.isOpened();panels=[];rawimages=[]
    for t in range(max(0,x['index']-1),min(len(n['raw_count']),x['index']+2)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(n['frame_id'][t]));ok,im=cap.read();assert ok
        w,h=int(n['width'][t]),int(n['height'][t]);im=im[:h,:w];rawimages.append(im)
        whole=Image.fromarray(cv2.cvtColor(cv2.resize(im,(640,round(h*640/w))),cv2.COLOR_BGR2RGB));ch=int(n['raw_count'][t]);q=n['hand_xy_px'][t,:ch];m=b['point_mask'][t,:ch];finite=q[m]
        if len(finite):
            low=finite.min(0)-60;high=finite.max(0)+60;cx,cy=(low+high)/2;span=max(*(high-low),150.)
        else:cx,cy=w/2,h/2;span=300
        x0=max(0,int(cx-span/2));y0=max(0,int(cy-span/2));x1=min(w,int(cx+span/2));y1=min(h,int(cy+span/2));zoom=Image.fromarray(cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)).resize((640,whole.height));d=ImageDraw.Draw(zoom)
        sx=640/(x1-x0);sy=whole.height/(y1-y0)
        for hand in range(ch):
            col=['#ff9461','#54d8ff'][hand];points=[((p[0]-x0)*sx,(p[1]-y0)*sy) for p in q[hand]]
            for a,v in edges:
                if m[hand,a] and m[hand,v]:d.line((points[a],points[v]),fill=col,width=2)
            for k,p in enumerate(points):
                if m[hand,k]:
                    px,py=p;d.ellipse((px-3,py-3,px+3,py+3),fill=col)
            if m[hand,0]:
                px,py=points[0];d.rectangle((px-7,py-7,px+7,py+7),outline='white',width=2);d.text((max(3,min(480,px)),max(3,py-30)),f'후보 {hand+1} 손목',font=font(18),fill=col,stroke_width=1,stroke_fill='black')
        panel=Image.new('RGB',(1280,whole.height+58),'#132238');pd=ImageDraw.Draw(panel);pd.text((16,12),f"{r['label']} · {x['view']} · 프레임 {t} {'(변경 장면)' if t==x['index'] else '(앞뒤 문맥)'}",font=font(23),fill='white');panel.paste(whole,(0,58));panel.paste(zoom,(640,58));panels.append(panel)
    cap.release();n.close();b.close()
    canvas=Image.new('RGB',(1280,sum(p.height for p in panels)+110),'#0b1422');y=0
    for p in panels:canvas.paste(p,(0,y));y+=p.height
    draw=ImageDraw.Draw(canvas);draw.text((18,y+12),'왼쪽: 실제 입력 전체 / 오른쪽: 손 확대·관절 후보 (흰 네모=손목)',font=font(21),fill='white')
    desc=' · '.join(f"후보{i+1} 손목-손바닥 비율 {g.get('ratio',0):.3f}" for i,g in enumerate(x['out']['geometry']))
    draw.text((18,y+48),desc,font=font(20),fill='#a9d4ef');draw.text((18,y+78),'앞뒤 장면은 사람의 검토용입니다. 후보 계산은 현재와 이전 프레임만 사용합니다.',font=font(17),fill='#c6cfda')
    path=E/f'change_{j:02}.jpg';canvas.save(path,quality=95)
    # Exact-size current sensor frame kept for unobstructed visual review.
    plain=E/f'original_{j:02}.png';cv2.imwrite(str(plain),rawimages[1 if x['index'] else 0])
    index.append({'display_id':j,'clip_id':x['clip_id'],'word':x['word'],'view':x['view'],'index':x['index'],'changed_raw':x['changed_raw'],'file':str(path),'original':str(plain),'source_video':str(src),'source_sha256':sha(src)})
(U/'EVIDENCE_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('RENDERED',len(index),'changes')
