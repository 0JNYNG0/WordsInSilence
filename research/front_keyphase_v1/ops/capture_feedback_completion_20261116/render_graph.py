from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
NAVY='#182F4C';GRAY='#536174';LIGHT='#F1F4F8'
font=lambda s,b=False:ImageFont.truetype('C:/Windows/Fonts/malgun'+('bd'if b else'')+'.ttf',s)
im=Image.new('RGB',(1400,1430),'white');d=ImageDraw.Draw(im)
d.text((80,45),'촬영 후 피드백 완성 경로',font=font(47,True),fill=NAVY)
d.text((80,115),'일시 중단 · 재개 요청 후 B·C부터 이어서 진행',font=font(27),fill=GRAY)
d.rounded_rectangle((80,173,1320,251),16,fill=LIGHT)
d.text((106,194),'이어 쓰는 기반: RTMW · TCN · 시간 합성곱+GRU · DTW · 손가락 판정',font=font(25),fill=NAVY)
rows=[
 ('A','기존 기능과 빈칸 확인','확인 완료','정상 수행을 모아 퀘스트를 통과시키는 연결이 미완성'),
 ('B','설명 → 필수 행동·정답 연결','재개 지점','손모양 · 손 방향 · 상대 위치 · 반복 · 순서'),
 ('C','원인별 보완·필요한 학습','일부 완료','수행한 움직임은 보존하고, 끝 확인과 따로 표시'),
 ('D','정상·오류·관측 부족 비교','일부 확인','이번 공개 파생 검사 통과 / 전체 과제 검증은 남음'),
 ('E','피드백과 퀘스트 연결','다음 단계','차이는 손·팔 강조 + 한 문장 / 근거가 모이면 통과'),
 ('F','실제 촬영·시연 확인','마지막 단계','자동 종료 · 대기시간 · 교정 후 재시도 · HDD 없이 실행')]
for i,(num,title,status,desc)in enumerate(rows):
 y=294+i*164
 d.rounded_rectangle((120,y,1235,y+125),16,fill=LIGHT,outline=NAVY,width=2)
 d.rounded_rectangle((141,y+20,201,y+80),10,fill=NAVY)
 d.text((157,y+25),num,font=font(34,True),fill='white')
 d.text((225,y+17),title,font=font(30,True),fill=NAVY)
 d.text((995,y+20),status,font=font(23,True),fill=GRAY)
 d.text((225,y+72),desc,font=font(24),fill=GRAY)
 if i<5:
  d.line((675,y+126,675,y+153),fill=NAVY,width=3)
  d.polygon([(666,y+147),(684,y+147),(675,y+158)],fill=NAVY)
# Explicit failure route, without implying a completed downstream stage.
d.line([(1245,846),(1295,846),(1295,520),(1245,520)],fill=GRAY,width=3)
d.polygon([(1245,520),(1256,514),(1256,526)],fill=GRAY)
d.text((82,1303),'문제가 있으면 B·C로 돌아가 공개 자료로 보완 → D에서 다시 확인',font=font(26,True),fill=NAVY)
d.text((82,1353),'실시간 화면은 확장 보류 · 사용자 촬영은 학습에 사용하지 않음 · C·T3-F 유지',font=font(23),fill=GRAY)
im.save(HERE/'TASK_GRAPH.png')
print(HERE/'TASK_GRAPH.png')
