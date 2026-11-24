"""Readable project-native PNG/SVG workflow; status comes from saved task graph."""
from pathlib import Path
import json,math
from xml.sax.saxutils import escape
from PIL import Image,ImageDraw,ImageFont
W=Path(__file__).resolve().parent;U=W.parent;OUT=U/'presentation';OUT.mkdir(exist_ok=True)
g=json.loads((U/'TASK_GRAPH.json').read_text(encoding='utf-8-sig'));assert next(n for n in g['nodes'] if n['id']=='G2')['status']=='REWORK'
S=2;WW=1920;HH=1080
im=Image.new('RGB',(WW*S,HH*S),'#FFFFFF');d=ImageDraw.Draw(im)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{WW}" height="{HH}" viewBox="0 0 {WW} {HH}"><rect width="{WW}" height="{HH}" fill="white"/>']
ink='#173047';muted='#53677B';green='#187B64';orange='#B66013';blue='#285E9D';line='#A5B6C8'
def rect(x,y,w,h,fill,stroke='none',r=18,sw=2):
 d.rounded_rectangle((x*S,y*S,(x+w)*S,(y+h)*S),r*S,fill=fill,outline=None if stroke=='none' else stroke,width=sw*S)
 svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def text(x,y,t,size=28,c=ink,b=False,center=False):
 f=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf' if b else 'C:/Windows/Fonts/malgun.ttf',size*S)
 if center:x-=d.textlength(t,font=f)/(2*S)
 d.text((x*S,y*S),t,font=f,fill=c)
 svg.append(f'<text x="{x:.1f}" y="{y+size*.96:.1f}" font-family="Malgun Gothic,sans-serif" font-size="{size}" font-weight="{700 if b else 400}" fill="{c}">{escape(t)}</text>')
def arrow(points,c=line,width=4):
 d.line([(int(x*S),int(y*S)) for x,y in points],fill=c,width=width*S,joint='curve');x,y=points[-1];a,b=points[-2];ang=math.atan2(y-b,x-a);p=[(x,y),(x-12*math.cos(ang)+6*math.sin(ang),y-12*math.sin(ang)-6*math.cos(ang)),(x-12*math.cos(ang)-6*math.sin(ang),y-12*math.sin(ang)+6*math.cos(ang))];d.polygon([(int(x*S),int(y*S)) for x,y in p],fill=c)
 svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{c}" stroke-width="{width}"/>');svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in p)+f'" fill="{c}"/>')

text(60,38,'웹 피드백 완성까지 · 현재 위치와 다음 작업',47,b=True)
text(60,108,'현재는 ② 행동 구간 연결을 보완 중입니다. 아래 단계는 검증을 통과해야 넘어갑니다.',27,muted)
steps=[('① 근거 정리','기존 범위 통과','자료·설명·불확실성',green,'#EAF6F1'),('② 행동 구간 연결','보완 중','이 행동은 어디부터 어디까지?',orange,'#FFF3E3'),('③ 차이 판단','다음 단계','모양·방향·순서·누락',muted,'#F2F5F8'),('④ 별도 공개 검증','이후 단계','정상 보존 + 새 자료',muted,'#F2F5F8'),('⑤ 웹 결과 연결','이후 단계','손·팔 강조 + 짧은 안내',muted,'#F2F5F8'),('⑥ 촬영 흐름 확인','이후 단계','자동 종료·속도·재시도',muted,'#F2F5F8')]
for i,(title,status,sub,c,bg) in enumerate(steps):
 x=60+306*i;rect(x,178,274,192,bg,c);text(x+20,200,title,28,b=True);text(x+20,252,status,27,c,True);text(x+20,306,sub,18,muted)
 if i<5:arrow([(x+280,272),(x+300,272)],blue)
text(60,414,'②를 해결하는 세부 작업 그래프',36,b=True)
text(60,466,'학습 정답·모델 판단·구간 연결을 나눠 확인하고, 마지막에는 함께 검증합니다.',26,muted)
boxes=[('A','학습 근거 보강','같은 행동 / 반복 회차 / 다른 행동','단어 하나에 몰린 정답을 보강'),('B','행동 자체를 구분','신체 상대위치 + 앞뒤 움직임','순서가 바뀌어도 행동을 알아보기'),('C','구간을 함께 연결','반복과 동시 동작은 보존','다른 행동을 한 구간으로 합치지 않기'),('D','전체 피드백 검증','정상 속도·교차 유지','방향·순서·누락이 겹쳐도 확인')]
for i,(letter,title,a,b) in enumerate(boxes):
 x=60+i*460;rect(x,545,420,208,'#F1F6FC',blue);rect(x+20,563,44,42,blue,r=10);text(x+42,566,letter,25,'white',True,True);text(x+82,566,title,29,b=True);text(x+22,631,a,23,muted);text(x+22,679,b,21,muted)
 if i<3:arrow([(x+426,646),(x+454,646)],blue)
# Explicit loop outcomes, with no expected-order or all-overlap ban.
arrow([(1650,761),(1650,826),(270,826),(270,762)],orange)
rect(566,798,820,57,'white',r=0);text(976,807,'미통과 → 원인에 해당하는 A·B·C로 돌아가 보완',25,orange,True,True)
arrow([(1868,650),(1889,650),(1889,919),(1596,919)],green)
rect(1197,880,384,82,'#EAF6F1',green);text(1389,904,'통과 → 상위 ③으로',27,green,True,True)
text(60,903,'반복 원칙',26,b=True);text(226,903,'실패 근거 보존 → 가설·비교 기준 고정 → 새 수정만 검증',24,muted)
text(60,1006,'기존 C·T3-F와 공통 100단어 학습 유지  ·  웹 완성 후 게임 연결',25,muted)
svg.append('</svg>');(OUT/'web_completion_graph_20261112.svg').write_text('\n'.join(svg),encoding='utf-8');im.save(OUT/'web_completion_graph_20261112.png',dpi=(192,192));im.resize((1920,1080),Image.Resampling.LANCZOS).save(OUT/'web_completion_graph_20261112_preview.png');print(OUT/'web_completion_graph_20261112_preview.png')
