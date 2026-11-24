"""Standalone presentation diagram, paired PNG/SVG. Planned work, not deployed architecture."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from xml.sax.saxutils import escape
import math,json
U=Path(__file__).resolve().parent;OUT=U/'presentation';OUT.mkdir(exist_ok=True)
S=2;W,H=1920,1080
im=Image.new('RGB',(W*S,H*S),'white');d=ImageDraw.Draw(im);svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="1920" height="1080" fill="white"/>']
F='C:/Windows/Fonts/malgun.ttf';FB='C:/Windows/Fonts/malgunbd.ttf'
ink='#182C45';gray='#596B80';line='#B7C4D2';orange='#A95B0D';amber='#FFF3DF';green='#147862';blue='#225A9C'
def rect(x,y,w,h,fill,stroke='none',r=20,sw=2):
 d.rounded_rectangle([x*S,y*S,(x+w)*S,(y+h)*S],r*S,fill=fill,outline=None if stroke=='none' else stroke,width=sw*S)
 svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def text(x,y,txt,size=30,color=ink,bold=False,center=False):
 font=ImageFont.truetype(FB if bold else F,size*S)
 if center:x-=d.textlength(txt,font=font)/S/2
 d.text((x*S,y*S),txt,font=font,fill=color,stroke_width=0)
 svg.append(f'<text x="{x:.2f}" y="{y+size*.96:.2f}" fill="{color}" font-family="Malgun Gothic, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}">{escape(txt)}</text>')
def arrow(points,color=line,width=4):
 d.line([(int(x*S),int(y*S)) for x,y in points],fill=color,width=width*S,joint='curve')
 x,y=points[-1];u,v=points[-2];a=math.atan2(y-v,x-u);pts=[(x,y),(x-14*math.cos(a)+7*math.sin(a),y-14*math.sin(a)-7*math.cos(a)),(x-14*math.cos(a)-7*math.sin(a),y-14*math.sin(a)+7*math.cos(a))]
 d.polygon([(int(x*S),int(y*S)) for x,y in pts],fill=color)
 svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
 svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in pts)+f'" fill="{color}"/>')

text(64,55,'한국수어 피드백 · 웹 완성 작업 흐름',54,bold=True)
text(66,141,'현재: 학습 근거와 실제 행동 구간의 연결을 보완하는 단계',30,color=gray)
rect(1436,143,420,54,amber,r=12);text(1646,152,'주황 = 현재 보완 중',25,orange,True,True)
stages=[
 ('학습 근거','정리·보강','설명과 영상 연결','관측 / 불확실 구분'),
 ('행동 구간','찾고 연결','준비·주요·마무리','같은 행동끼리 비교'),
 ('중요한 차이','판단','손모양·방향·순서','누락 / 관측 부족 구분'),
 ('공개 자료로','검증','정상 동작 보존','새 자료·오류 조건 비교'),
 ('웹 피드백','시각화','차이가 난 손·팔 강조','해당 순간 + 짧은 설명'),
 ('웹 테스트','완료 확인','촬영 → 분석 → 재시도','자동 종료·대기시간')]
for i,(a,b,c,e) in enumerate(stages):
 x=56+i*306;active=i<2
 rect(x,274,278,280,amber if active else '#F3F6FA',orange if active else '#CCD6E2',22,3 if active else 2)
 rect(x+22,295,48,43,orange if active else '#DCE4ED',r=12)
 text(x+46,299,str(i+1),27,'white' if active else ink,True,True)
 text(x+24,356,a,34,bold=True);text(x+24,402,b,34,bold=True)
 text(x+24,470,c,23,color=gray);text(x+24,508,e,23,color=gray)
 if i<5:arrow([(x+282,414),(x+302,414)],color=blue,width=4)

text(64,620,'각 단계는 검증을 통과해야 다음으로 넘어갑니다.',34,bold=True)
rect(78,735,300,132,'#EDF3FB','#BDCEE2',r=20);text(228,773,'해당 단계 작업',32,blue,True,True)
arrow([(386,801),(497,801)],blue)
# Explicit decision diamond.
pts=[(635,711),(775,801),(635,891),(495,801)]
d.polygon([(x*S,y*S) for x,y in pts],fill='#EDF7F3',outline=green,width=3*S)
svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in pts)+f'" fill="#EDF7F3" stroke="{green}" stroke-width="3"/>')
text(635,758,'검토·검증',29,green,True,True);text(635,805,'통과?',29,green,True,True)
arrow([(785,801),(1067,801)],green);text(921,750,'통과',28,green,True,True)
rect(1080,735,300,132,'#EDF7F3','#A8CDBF',r=20);text(1230,773,'다음 단계',32,green,True,True)
arrow([(635,900),(635,975),(228,975),(228,878)],orange)
text(433,929,'미통과 → 원인 분리·보완',23,orange,True,True)
text(1482,747,'실패 근거 보존',28,gray,True)
text(1482,793,'수정 후 다시 검증',28,gray,True)
text(64,1021,'범위: 웹 테스트 완성  |  게임·Unity 연동은 이후 단계',25,gray)
svg.append('</svg>');(OUT/'web_completion_graph.svg').write_text('\n'.join(svg),encoding='utf-8')
im.save(OUT/'web_completion_graph.png',dpi=(192,192))
Image.open(OUT/'web_completion_graph.png').resize((1920,1080),Image.Resampling.LANCZOS).save(OUT/'web_completion_graph_preview.png')
(OUT/'GRAPH_NOTE_KO.txt').write_text('발표용 연구·개발 작업 흐름도. 현재1~2단계보완중, 이후단계는완료를주장하지않음. 1~6은웹완료순서이며이미구현된기능의존재유무를뜻하지않음. PNG3840×2160,16:9;SVG확대가능. 실행상태근거는상위TASK_GRAPH.json. 실제시스템아키텍처와구분해사용.\n',encoding='utf-8')
print(str(OUT/'web_completion_graph.png'))
