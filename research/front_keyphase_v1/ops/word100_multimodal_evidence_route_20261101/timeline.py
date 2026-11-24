from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
U=Path(__file__).resolve().parent
r=json.loads((U/'MISSINGNESS.json').read_text(encoding='utf8'))['rows']['P065']
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(12,3.9),dpi=155)
fig.patch.set_facecolor('#f7f9fc'); ax.set_facecolor('#f7f9fc')
ts=[t/1e6 for t in r['timestamp_us']]
for s,y in [('L',1),('R',0)]:
    counts=r['sides'][s]['observed_points']
    for i,(t,c) in enumerate(zip(ts,counts)):
        width=(ts[i+1]-t) if i+1<len(ts) else 1/30
        ax.broken_barh([(t,width)],(y-.14,.28),facecolors='#278a70' if c else '#e27b56',linewidth=0)
ax.axvspan(1.28,2.825,color='#658ad6',alpha=.10,zorder=-1)
ax.plot(1.6,1,'o',color='#1e2b45',ms=8,zorder=5)
ax.annotate('보여드린 사진: 1.60초\n왼손 누락 5프레임 구간 안 (약 0.17초)',xy=(1.6,1),xytext=(2.05,1.5),
            arrowprops={'arrowstyle':'->','color':'#1e2b45'},fontsize=10,color='#1e2b45')
ax.set_yticks([1,0],['왼손','오른손']); ax.set_xlim(0,4); ax.set_ylim(-.55,1.85)
ax.set_xlabel('영상 시간 (초)',fontsize=10); ax.grid(axis='x',alpha=.13)
for s in ax.spines.values(): s.set_visible(False)
ax.legend(handles=[Patch(color='#278a70',label='원래 관측에 손 점이 있음'),Patch(color='#e27b56',label='원래 관측에 해당 손 점이 없음')],
          loc='lower left',ncol=2,frameon=False,fontsize=9)
fig.suptitle('‘조상’ 공개 영상 전체 120프레임 — 계속 손을 놓친 것은 아닙니다',fontsize=14,fontweight='bold',x=.09,ha='left')
fig.text(.09,.035,'파란 배경: 기존 자료의 동작 주석 시간. 원래 검출 기록이며, 뒤의 1프레임 재확인 결과는 합산하지 않았습니다.\n초록색도 좌표의 정확함이나 수어의 정답을 뜻하지 않습니다.',fontsize=8.8,color='#43516a')
fig.subplots_adjust(left=.09,right=.98,top=.84,bottom=.25)
path=U/'P065_TIMELINE_KO.png'
assert not path.exists()
fig.savefig(path,facecolor=fig.get_facecolor())
print(str(path))
