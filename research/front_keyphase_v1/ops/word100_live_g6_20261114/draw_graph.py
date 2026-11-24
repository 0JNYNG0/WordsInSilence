from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
U=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
navy='#1b304c';muted='#617086';light='#edf1f6';line='#aab7c8'
fig,ax=plt.subplots(figsize=(16,9),dpi=150);fig.patch.set_facecolor('white');ax.set(xlim=(0,16),ylim=(0,9));ax.axis('off')
def text(x,y,s,size=16,c=navy,weight='normal',ha='left'):
 ax.text(x,y,s,fontsize=size,color=c,fontweight=weight,ha=ha,va='center',linespacing=1.55)
def box(x,y,w,h,title,sub,filled=False):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=.14',facecolor=navy if filled else light,edgecolor=navy if filled else line,linewidth=1.2))
 text(x+w/2,y+h*.67,title,16,'white' if filled else navy,'bold','center');text(x+w/2,y+h*.28,sub,12,'#dce5ef' if filled else muted,ha='center')
def arrow(x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=17,color=line,linewidth=1.8))
text(.45,8.45,'웹 피드백 연결 · 현재 작업 위치',26,weight='bold')
text(.45,7.92,'기존 단계는 확인한 범위에서 통과했습니다. G6의 실제 웹캠 확인이 남아 있습니다.',14,muted)
stages=[('G0','범위·기준'),('G1','설명·정답'),('G2','행동 연결'),('G3','통합 판정'),('G4','공개 검증'),('G5','시각 피드백')]
for i,(gid,label) in enumerate(stages):
 x=.45+i*2.61;box(x,6.4,2.13,1.0,gid+' '+label,'확인 범위 통과',True)
 if i<5:arrow(x+2.16,6.9,x+2.54,6.9)
text(.45,5.83,'G6  새 촬영 → 분석 → 피드백',21,weight='bold')
for i,(title,sub) in enumerate([('최신 판정 연결','공개 입력으로 기존 결과와 비교'),('촬영·자동 종료 보완','관측 대기열 제어 · 짧은 가림 검사'),('피드백·대기시간 확인','차이 강조 / 회색 보류 / 로딩 화면')]):
 x=.45+i*5.22;box(x,4.15,4.6,1.13,title,sub,True)
 if i<2:arrow(x+4.64,4.72,x+5.14,4.72)
text(.6,3.66,'공개 영상을 촬영 입력으로 재생한 시험:  순종 2.87초  ·  잘하다 1.55초',16,weight='bold')
text(.6,3.27,'자동 종료 → 결과 화면 표시 확인까지. 실제 사용자 촬영의 보장 시간은 아닙니다.',12,muted)
ax.plot([13.2,13.2,4.45],[4.1,3.10,3.10],color=line,linewidth=1.8)
arrow(4.45,3.10,4.45,3.0)
box(.45,1.5,8.0,1.5,'남은 확인 · 실제 웹캠 촬영','순종·잘하다 직접 촬영 → 자동 종료 · 표시 · 대기시간 확인')
arrow(8.55,2.25,9.35,2.25);box(9.5,1.5,6.0,1.5,'웹 테스트 마무리','실제 촬영 확인을 통과한 뒤 완료 처리')
text(.45,.86,'남은 판정 한계',14,weight='bold')
text(2.55,.86,'정상 동작에서도 판단 충돌이 남음 · 불확실한 부분은 오답으로 표시하지 않음',13,muted)
text(.45,.38,'C·T3-F / 공통 100단어 유지  ·  이번 새 학습 없음  ·  사용자 촬영 학습·튜닝 제외',12,muted)
fig.savefig(U/'PROGRESS_GRAPH.png',bbox_inches='tight',pad_inches=.15);fig.savefig(U/'PROGRESS_GRAPH.svg',bbox_inches='tight',pad_inches=.15)
plt.close(fig)
