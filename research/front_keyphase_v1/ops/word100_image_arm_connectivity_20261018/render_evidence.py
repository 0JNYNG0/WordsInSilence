"""Display saved observations; no candidate rerun or parameter changes."""
import json,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

U=Path(__file__).resolve().parent
OUT=U/'evidence';OUT.mkdir(exist_ok=True)
FONT='C:/Windows/Fonts/malgun.ttf'
def font(n):return ImageFont.truetype(FONT,n)
def read(p):return json.loads(p.read_text('utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
R=read(U/'RESULTS.json');A=read(U/'ANNOTATED.json')

def panel(x,mask=False):
    im=Image.open(x['image']).convert('RGB');arr=np.array(im)
    z=np.load(x['map']);support=z['support'];cc=z['components']
    if mask:
        arr=(arr*.45).astype(np.uint8)
        arr[support]=(arr[support]*.35+np.array([30,195,140])*.65).astype(np.uint8)
        im=Image.fromarray(arr)
    d=ImageDraw.Draw(im)
    for j,p in enumerate(x['elbows']):
        if p is None:continue
        col=['#55c7ff','#ffb15b'][j];xx,yy=p;d.ellipse((xx-5,yy-5,xx+5,yy+5),fill=col,outline='white',width=1)
        d.text((xx+7,max(1,yy-22)),['왼팔','오른팔'][j],font=font(17),fill=col,stroke_width=1,stroke_fill='black')
    for j,p in enumerate(x['wrists']):
        if p is None:continue
        xx,yy=p;d.rectangle((xx-4,yy-4,xx+4,yy+4),fill='#fff179',outline='black')
        d.text((max(2,xx-28),max(1,yy-25)),f'손{j}',font=font(16),fill='#fff179',stroke_width=1,stroke_fill='black')
        if mask and x['paths'][j]:d.line([tuple(q) for q in x['paths'][j]],fill='#ff75df',width=2)
    return im

def card(x,title):
    a=panel(x);b=panel(x,True);w,h=a.size
    im=Image.new('RGB',(1280,h+110),'#132032');d=ImageDraw.Draw(im)
    d.text((18,8),title,font=font(23),fill='white')
    d.text((18,43),'실제 입력 + 팔꿈치·손목 표시',font=font(18),fill='#bacbe0')
    d.text((658,43),'초록: 색 연결 영역 / 분홍: 통과한 연결 경로',font=font(18),fill='#bacbe0')
    im.paste(a,(0,72));im.paste(b,(640,72))
    z=np.load(x['map']);ec=x['elbow_components'];common=ec[0]>0 and ec[0]==ec[1]
    frac=float((z['components']==ec[0]).mean()) if common else 0.
    if common:tail=f'양쪽 팔이 같은 영역으로 연결됨 · 해당 영역이 화면의 {frac*100:.1f}% → 좌우 확정 보류'
    else:tail='색 연결은 보조 근거일 뿐, 실제 팔 연결과 같은지 별도 확인이 필요함'
    d.text((18,h+78),tail,font=font(18),fill='#ffdf9a')
    return im,frac

rem=[x for x in A if x['remaining9']];metrics=[];cards=[]
for a in rem:
    x=R[a['record_index']];im,frac=card(x,f"#{a['id']} {a['word']} · {x['view']} · 프레임 {x['index']} · 관측점 {a['usable_points']}개")
    path=OUT/f"remaining_{a['id']}.png";im.save(path);cards.append(im)
    metrics.append({'id':a['id'],'word':a['word'],'observed_points':a['usable_points'],'shared_component_frame_fraction':frac,'file':str(path),'record_index':a['record_index']})
for page in range(3):
    group=cards[page*3:page*3+3];sheet=Image.new('RGB',(1280,sum(x.height for x in group)+16*(len(group)-1)),'#07101c');y=0
    for im in group:sheet.paste(im,(0,y));y+=im.height+16
    sheet.save(OUT/f'remaining_sheet_{page+1}.jpg',quality=92)
pos=[x for x in R if any(v>=0 for v in x['rgb_side'])]
for k in range(0,len(pos),3):
    group=[card(x,f"{x['word']} · {x['clip_id']} · {x['index']} · 既존 배정 유지".replace('既존','기존'))[0] for x in pos[k:k+3]]
    sheet=Image.new('RGB',(1280,sum(im.height for im in group)+16*(len(group)-1)),'#07101c');y=0
    for im in group:sheet.paste(im,(0,y));y+=im.height+16
    sheet.save(OUT/f'positive_sheet_{k//3+1}.jpg',quality=92)
controls=[x for x in A if x['id'] in [37,38,39,90,291] and x['set']=='original456']
for a in controls:
    im,_=card(R[a['record_index']],f"정상 교차 반례 #{a['id']} · {a['word']} · 프레임 {a['index']}")
    im.save(OUT/f"normal_{a['id']}.png")
save(U/'EVIDENCE_INDEX.json',{'remaining9':metrics,'positive_support_frames':len(pos),'positive_support_candidates':sum(sum(s>=0 for s in x['rgb_side']) for x in pos),'pixel_reasons':dict(collections.Counter(v for x in R for v in x['pixel_reason'])),'note':'Saved cropped input pixels and computed mask only. Green is color similarity, not a semantic arm mask.'})
print('EVIDENCE',len(rem),'remaining,',len(pos),'positive frames,',len(controls),'normal crossing counterexamples')
