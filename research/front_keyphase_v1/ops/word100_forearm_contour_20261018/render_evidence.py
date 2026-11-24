import json,gzip
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;E=U/'evidence';E.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text('utf8'))
def font(n):return ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',n)
R=read(U/'RESULTS.json');A=read(U/'ANNOTATED.json');C=read(U/'CHANGES.json')
SIDES={-1:'미확정',0:'왼손',1:'오른손'}
def card(k,raw,title):
    x=R[k];im=Image.open(x['image']).convert('RGB');a=im.copy();b=im.convert('L').convert('RGB');d=ImageDraw.Draw(a);db=ImageDraw.Draw(b)
    for s,p in enumerate(x['elbows']):
        if p is None:continue
        col=['#57d3ff','#ffba68'][s];px,py=p
        for dd in [d,db]:
            dd.ellipse((px-4,py-4,px+4,py+4),fill=col,outline='black');dd.text((px+6,max(0,py-22)),['왼팔','오른팔'][s],font=font(16),fill=col,stroke_width=1,stroke_fill='black')
    for i,p in enumerate(x['wrists']):
        if p is None:continue
        px,py=p
        for dd in [d,db]:dd.rectangle((px-4,py-4,px+4,py+4),fill='#ffe66a',outline='black');dd.text((px-25,max(0,py-26)),f'손{i}',font=font(16),fill='#fff179',stroke_width=1,stroke_fill='black')
    with gzip.open(x['trials_file'],'rt',encoding='utf8') as f:trials=json.load(f)[raw]
    desc=[]
    for side,t in enumerate(trials):
        col='#53ee9d' if t['pass'] else '#ff996b';last=None
        for sl in t['slices']:
            pair=sl['pair']
            if pair is None:last=None;continue
            cx,cy=sl['center'];nx,ny=sl['normal'];p=[(cx+v*nx,cy+v*ny) for v in pair[:2]]
            for px,py in p:db.ellipse((px-2,py-2,px+2,py+2),fill=col)
            if last:
                for i in [0,1]:db.line((last[i],p[i]),fill=col,width=2)
            last=p
        desc.append(f"{['왼팔','오른팔'][side]} {t.get('supported',0)}/12 {'통과' if t['pass'] else '미충족'}")
    panel=Image.new('RGB',(1280,im.height+122),'#122033');q=ImageDraw.Draw(panel)
    q.text((18,8),title,font=font(22),fill='white');q.text((18,43),'실제 입력 · 팔꿈치와 손목 표시',font=font(17),fill='#b9cbe1');q.text((658,43),'윤곽: 초록=조건 통과 / 주황=불충분',font=font(17),fill='#b9cbe1')
    panel.paste(a,(0,71));panel.paste(b,(640,71));q.text((18,im.height+80),' · '.join(desc)+f"  |  임시 좌우 {SIDES[x['old_sides'][raw]]} → {SIDES[x['sides'][raw]]}",font=font(19),fill='#ffe1a5')
    return panel

cards=[];index=[]
for j,x in enumerate(C):
    k=x['record_index'];r=R[k];im=card(k,x['raw_index'],f"변경 {j+1} · {r['word']} · 프레임 {r['index']} · raw{x['raw_index']}")
    p=E/f'change_{j+1:02}.png';im.save(p);cards.append(im);index.append({**x,'word':r['word'],'file':str(p),'display_id':j+1})
for page in range((len(cards)+2)//3):
    g=cards[page*3:page*3+3];out=Image.new('RGB',(1280,sum(im.height for im in g)+16*(len(g)-1)),'#07101c');y=0
    for im in g:out.paste(im,(0,y));y+=im.height+16
    out.save(E/f'changes_sheet_{page+1}.jpg',quality=94)
for a in A:
    if a['remaining9'] or (a['set']=='original456' and a['id'] in [37,38,39,90,291]):
        im=card(a['record_index'],a['raw_index'],f"#{a['id']} · {a['word']} · 프레임 {a['index']}")
        im.save(E/f"case_{a['id']}.png")
(U/'EVIDENCE_INDEX.json').write_text(json.dumps({'changes':index,'remaining_ids':[a['id'] for a in A if a['remaining9']],'normal_crossing_controls':[37,38,39,90,291]},ensure_ascii=False,indent=2),encoding='utf8')
print('RENDER',len(C),'changes and remaining9/5controls')
conflicts=[];representatives={}
for k,x in enumerate(R):
    for i,(q,old) in enumerate(zip(x['evidence_side'],x['old_sides'])):
        if q>=0 and old>=0 and q!=old and x['reason'][i]!='changed_hint':
            conflicts.append({'record_index':k,'raw_index':i,'reason':x['reason'][i],'old_side':old,'evidence_side':q})
            representatives.setdefault((x['clip_id'],x['view']),(k,i))
g=[]
for j,(k,i) in enumerate(representatives.values()):
    x=R[k];im=card(k,i,f"차단된 충돌 · {x['word']} · {x['view']} · 프레임 {x['index']}")
    im.save(E/f'conflict_{k}.png');g.append(im)
for start in range(0,len(g),3):
    group=g[start:start+3];im=Image.new('RGB',(1280,sum(x.height for x in group)+16*(len(group)-1)),'#07101c');y=0
    for x in group:im.paste(x,(0,y));y+=x.height+16
    im.save(E/f'conflict_sheet_{start//3+1}.jpg',quality=93)
(U/'BLOCKED_CONFLICTS.json').write_text(json.dumps({'blocked':conflicts,'representatives':[{'record_index':k,'raw_index':i} for k,i in representatives.values()],'note':'Conflict with previous assignment is not by itself a new visual error label. Known normal #38 is confirmed wrong standalone support. Final guarded assignments unchanged.'},ensure_ascii=False,indent=2),encoding='utf8')
