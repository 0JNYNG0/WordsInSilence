"""Presentation only: render two saved aggregate figures with unclipped labels."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
U=Path(__file__).resolve().parent
a=json.loads((U/'ANALYSIS.json').read_text('utf8'))
archive=U/'visuals'/'layout_before';archive.mkdir(exist_ok=False)
for name in ('cross.png','regression.png'):shutil.copy2(U/'visuals'/name,archive/name)
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False})
fig,axs=plt.subplots(1,2,figsize=(13,6));fig.subplots_adjust(left=.065,right=.985,bottom=.28,top=.91,wspace=.23)
groups=('angle','palm','elbow','wrist');gnames=['각도','손바닥','팔꿈치','팔 손목'];x=np.arange(4)
g=a['cross']['identity']['groups']
for cond,shift,color,label in [('old',-.18,'#b88b5d','기존'),('new',.18,'#3b8792','보조')]:
    vals=[g[k][cond]['warning'] for k in groups];axs[0].bar(x+shift,vals,.36,color=color,label=label)
    for i,val in enumerate(vals):axs[0].text(i+shift,val+2,str(val),ha='center',fontsize=9)
axs[0].set(xticks=x,xticklabels=gnames,ylim=(0,250),ylabel='지속 차이가 있는 비교 / 216',title='다른 사람 · 원래 속도');axs[0].legend(fontsize=9)
for key,shift,color,label in [('resolved',-.25,'#3b8792','공통 관측 · 차이 감소'),('added',0,'#b66662','공통 관측 · 새 차이'),('old_comparable_now_unknown',.25,'#9a9fa9','비교 가능 → 근거 부족')]:axs[1].bar(x+shift,[g[k]['comparison'][key] for k in groups],.25,color=color,label=label)
axs[1].set(xticks=x,xticklabels=gnames,ylabel='비교 칸 수',title='경고 변화와 관측 손실을 구분');axs[1].legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.12))
fig.savefig(U/'visuals'/'cross.png',dpi=145,bbox_inches='tight',pad_inches=.2);plt.close(fig)
ex=a['examples']['regression']
with np.load(U/'known'/(ex['name']+'.npz'),allow_pickle=False) as zz:z={k:zz[k] for k in zz.files}
ix=z['source_index'];pred={c:np.array([np.median(z['base_path' if c=='old' else 'path'][z['base_path' if c=='old' else 'path'][:,0]==i,1]) for i in range(len(ix))]) for c in ('old','new')}
fig,axs=plt.subplots(2,1,figsize=(11,7));fig.subplots_adjust(left=.1,right=.98,bottom=.1,top=.92,hspace=.25);tt=np.arange(len(ix))/30
axs[0].plot(tt,ix,c='#668b63',lw=2,label='생성 과정에서 아는 실제 원본 시점')
axs[0].plot(tt,pred['old'],c='#b88b5d',label='기존 정렬');axs[0].plot(tt,pred['new'],c='#3b8792',label='위치·방향 보조')
axs[0].set(ylabel='원본 프레임',title=f'{ex["label"]} · {ex["query"].split("_")[-2]} · {ex["kind"]} · 오차 변화 {ex["delta_new_minus_old"]:+.3f}프레임');axs[0].legend(fontsize=9)
axs[1].plot(tt,z['old_error'],c='#b88b5d',label='기존 오차');axs[1].plot(tt,z['new_error'],c='#3b8792',label='보조 오차')
axs[1].fill_between(tt,0,1,where=z['moving'],color='#dbe6ed',alpha=.5,transform=axs[1].get_xaxis_transform(),label='원본에서 변화가 관측된 구간')
axs[1].set(xlabel='시험 영상 시간 (초)',ylabel='절대 시간 오차 (원본 프레임)');axs[1].legend(fontsize=9)
fig.savefig(U/'visuals'/'regression.png',dpi=145,bbox_inches='tight',pad_inches=.2);plt.close(fig)
print('Two figures re-rendered; saved numeric results unchanged.')
