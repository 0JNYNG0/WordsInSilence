"""Standalone project progress diagram, not an experiment accuracy plot."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
im=Image.new('RGB',(1680,1240),'#f5f7fa');d=ImageDraw.Draw(im)
def font(n,bold=False):return ImageFont.truetype('C:/Windows/Fonts/'+('malgunbd.ttf'if bold else'malgun.ttf'),n)
navy='#18314f';ink='#20344c';muted='#53677b';line='#94a4b6'
d.text((75,48),'수어 피드백 — 현재 작업 위치',font=font(47,True),fill=navy)
d.text((78,118),'기존 웹 기능은 유지 · 이번 후보는 검증 후 미반영 · 전체 완료는 아직 아님',font=font(26),fill=muted)
xs=[75,625,1175];w=430;h=270;ys=[210,630]
nodes=[
 (0,0,'구현 · 공개 검증','촬영과 단어 인식',['촬영 중 RTMW 관절 추출','TCN으로 단어 후보 확인'],True),
 (1,0,'부분 구현','확인된 특징 교정',['손모양·팔 이동 방향 비교','해당 순간 강조 + 짧은 문장'],True),
 (2,0,'이번 학습 · 검증 완료','설명과 행동 연결',['시작·진행·끝 정답 연결','부분·다른 행동 구분 학습','부작용이 있어 웹 미반영'],False),
 (2,1,'현재 보완 중','구간과 완료 판단 분리',['찾은 행동은 비교에서 유지','끝까지 했는지는 따로 확인','정상 차이·관측 부족 구분'],False),
 (1,1,'남은 검증','정상과 오류 함께 비교',['정상 속도·교차 유지','방향·순서·누락 오류 확인','선행 통과 후 새 공개 평가'],False),
 (0,1,'남은 마무리','웹 교정 기능 최종 시험',['통과한 기능만 연결','실제 촬영 → 교정 → 재시도','사용자가 바로 이해하는지 확인'],False)]
for col,row,tag,title,lines,dark in nodes:
 x,y=xs[col],ys[row];fill=navy if dark else '#ffffff';border=navy if(col,row)==(2,1)else '#c8d2de'
 d.rounded_rectangle((x,y,x+w,y+h),22,fill=fill,outline=border,width=4 if(col,row)==(2,1)else 2)
 d.text((x+24,y+24),tag,font=font(22,True),fill='#c4d5e8'if dark else muted)
 d.text((x+24,y+70),title,font=font(31,True),fill='white'if dark else navy)
 for i,t in enumerate(lines):d.text((x+24,y+133+i*37),t,font=font(23),fill='#eef3f8'if dark else ink)
def arrow(a,b):
 d.line((a,b),fill=line,width=5)
 if a[0]<b[0]:pts=[b,(b[0]-14,b[1]-9),(b[0]-14,b[1]+9)]
 elif a[0]>b[0]:pts=[b,(b[0]+14,b[1]-9),(b[0]+14,b[1]+9)]
 else:pts=[b,(b[0]-9,b[1]-14),(b[0]+9,b[1]-14)]
 d.polygon(pts,fill=line)
arrow((505,345),(610,345));arrow((1055,345),(1160,345));arrow((1390,490),(1390,615))
arrow((1160,765),(1070,765));arrow((610,765),(520,765))
d.text((1410,535),'다음',font=font(24,True),fill=muted)
# Failure loop: review labels/observations instead of repeating the same training.
d.line([(840,915),(840,972),(1608,972),(1608,600),(1580,600)],fill=navy,width=3)
d.polygon([(1580,600),(1594,591),(1594,609)],fill=navy)
d.text((910,929),'검증 실패 → 원인에 맞는 단계만 보완',font=font(23),fill=navy)
d.rounded_rectangle((75,1040,1605,1170),16,fill='#e5ebf2')
d.text((102,1060),'이번에 확인한 핵심',font=font(25,True),fill=navy)
d.text((102,1104),'일부 동작 구분은 개선됐지만 정상·오류 동작도 놓침 → 구간 찾기와 완료 판단을 분리',font=font(25),fill=ink)
d.text((78,1190),'공개 자료만 사용 · 기존 C·T3-F 및 공통100 모델 보존 · 사용자 촬영은 학습에 사용하지 않음',font=font(21),fill=muted)
im.save(U/'PROGRESS_GRAPH.png')
print(U/'PROGRESS_GRAPH.png')
