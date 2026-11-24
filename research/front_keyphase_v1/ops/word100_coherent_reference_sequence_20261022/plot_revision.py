"""Revise one report figure from immutable saved ANALYSIS; preserve previous figure."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

U=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'PLOT_REVISION.json').exists()
    src=U/'ANALYSIS.json';source_hash=sha(src)
    x=json.loads(src.read_text('utf-8'))
    s=x['cohorts']['cross']['speed_only'];a=x['cohorts']['cross']['normal_all']
    paired=s['incremental_between']['point_to_levels'];seq=s['incremental_between']['point_to_sequence']
    assert paired['common_four']==seq['common_four']==567457
    left=[paired['a_new_bad_common_four'],paired['b_new_bad_common_four'],seq['b_new_bad_common_four']]
    assert left==[8213,11866,12118]
    modes=('point','levels','sequence')
    right=[sum(a['groups'][g]['modes'][m]['comparable'] for g in a['groups']) for m in modes]
    total=sum(g['expected'] for g in a['groups'].values())
    assert right==[1148965,718558,718558] and total==1638000
    dest=U/'visuals'/'normal_tradeoff.png';before=dest.parent/'before_report'
    before.mkdir(exist_ok=True);backup=before/dest.name;assert not backup.exists()
    old_hash=sha(dest);shutil.copyfile(dest,backup);assert sha(backup)==old_hash
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,
      'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,6.2));colors=['#64748b','#268b9b','#7461bb'];ticks=['기존 시점별','3시점','3시점+변화량']
    for ax,values in zip(axes,(left,right)):
        bars=ax.bar(np.arange(3),values,color=colors,width=.65)
        ax.set_xticks(np.arange(3),ticks);ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'));ax.margins(y=.24)
        ax.set_ylabel('정보 칸 수 (시간 × 손 × 각도/좌표)')
        labels=[f'{v:,}' for v in values] if ax is axes[0] else [f'{v:,}\n({100*v/total:.1f}%)' for v in values]
        ax.bar_label(bars,labels=labels,padding=5,fontsize=11)
    axes[0].set_title('속도 변화로 생긴 새로운 차이\n세 조건 모두 비교 가능한 567,457칸에서',pad=16)
    axes[1].set_title('정상 원본·속도 변형의 비교 가능 총량\n고정된 전체 1,638,000칸 중',pad=16)
    fig.suptitle('같은 근거의 차이 변화와 전체 관측량을 분리해서 확인',fontsize=15,y=1.02)
    fig.subplots_adjust(wspace=.3,bottom=.24)
    fig.text(.5,.08,'기존안 → 새 조건은 여러 변경이 함께 적용된 비교입니다. 3시점 → 3시점+변화량이 통제된 추가 비교입니다.',ha='center',fontsize=10,color='#475569')
    fig.text(.5,.035,'정보 칸의 수는 영상 개수나 수어 오류 정확도가 아닙니다. 비교 불가능한 칸은 정상으로 처리하지 않습니다.',ha='center',fontsize=10,color='#475569')
    fig.savefig(dest,dpi=160,bbox_inches='tight',pad_inches=.25);plt.close(fig)
    assert sha(src)==source_hash
    record={'source_analysis_sha256':source_hash,'before_sha256':old_hash,'after_sha256':sha(dest),'backup':str(backup),
      'left_common_information_cells':567457,'left_new_difference_cells':dict(zip(modes,left)),
      'right_expected_information_cells':total,'right_comparable_cells':dict(zip(modes,right)),
      'unchanged':'All result arrays and ANALYSIS.json preserved; only authorized report figure revised.'}
    with (U/'PLOT_REVISION.json').open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print('PLOT REVISION PASS')

if __name__=='__main__':main()
