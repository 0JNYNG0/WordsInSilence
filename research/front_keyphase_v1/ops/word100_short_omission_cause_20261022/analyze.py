"""Summarize verified saved diagnostics; never fit, align, infer, or train."""
import json
from collections import defaultdict,Counter
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

U=Path(__file__).resolve().parent;O=U.parent
P=O/'word100_motion_phase_alignment_20261022';L=O/'word100_lopo_reference_range_20261022'
I=O/'word100_cross_person_motion_support_20261022'/'inputs'
FACTORS=('alignment','range','neighbors')
FACTORTITLES={'alignment':'시간 연결 민감성','range':'다중 범위 민감성','neighbors':'앞뒤 이웃 확장 민감성'}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def ratio(n,d):return n/d if d else None

def source_summary(ss):
    keys=('expected_hand_frames','raw_observed_hand_frames','location_seen_hand_frames','positive_rest_hand_frames',
          'expected_angle_cells','supported_angle_cells','raw_observed_but_angle_unsupported',
          'explicitly_missing_angle_cells','observed_motion_hand_frames')
    out=dict(n=len(ss),totals={k:sum(s[k] for s in ss) for k in keys},groups={},
        provenance='Input field raw_observed_* inherits e.observed=(usable & observed).any(point). It means at least one preserved usable observed handpoint in prepared input, not original RGB visibility or all21 raw MediaPipe points. Angle valid requires prior state4 and finite interval.')
    for group in ('angle','palm','elbow','wrist'):
        fields=('expected','observed','endpoint_eligible','outside_endpoint_hull','outside_linear_chord')
        out['groups'][group]={k:sum(s['groups'][group][k] for s in ss) for k in fields}
        out['groups'][group]['coverage']=ratio(out['groups'][group]['observed'],out['groups'][group]['expected'])
        for k in ('max_endpoint_gap','max_chord_gap'):
            vals=[s['groups'][group][k] for s in ss if s['groups'][group][k] is not None]
            out['groups'][group][k]=max(vals) if vals else None
    t=out['totals'];out['raw_hand_observation_fraction']=ratio(t['raw_observed_hand_frames'],t['expected_hand_frames'])
    out['flags']={
      'all_expected_raw_hands_absent':sum(s['expected_hand_frames']>0 and s['raw_observed_hand_frames']==0 for s in ss),
      'some_expected_raw_hands_absent':sum(s['raw_observed_hand_frames']<s['expected_hand_frames'] for s in ss),
      'some_reliable_angles_unavailable':sum(s['supported_angle_cells']<s['expected_angle_cells'] for s in ss),
      'no_reliable_angle':sum(s['expected_angle_cells']>0 and s['supported_angle_cells']==0 for s in ss),
      'no_usable_angle_or_position':sum(sum(g['observed'] for g in s['groups'].values())==0 for s in ss),
      'some_observed_channel_outside_endpoint_hull':sum(any(g['outside_endpoint_hull']>0 for g in s['groups'].values()) for s in ss),
      'some_observed_channel_outside_frame_linear_chord':sum(any(g['outside_linear_chord']>0 for g in s['groups'].values()) for s in ss),
      'no_endpoint_eligible_channel':sum(sum(g['endpoint_eligible'] for g in s['groups'].values())==0 for s in ss),
    }
    out['self_source_new_difference']={method:sum(bool(s['self_source'][method]['inside_raw']) for s in ss) for method in ('old','new')}
    return out

def variant_record(s,rr,method):
    assert len(rr)==8
    factor_refs={k:sum(r['factors'][k]>0 for r in rr) for k in FACTORS}
    factor_cells={k:sum(r['factors'][k] for r in rr) for k in FACTORS}
    combos=Counter('+'.join(k for k in FACTORS if r['factors'][k]>0) or 'none' for r in rr)
    configs={}
    for k in rr[0]['config']:
        fields=('new_bad','lost_observation','comparable','bad','unknown','within','original_bad')
        configs[k]={f:sum(r['config'][k][f] for r in rr) for f in fields}
        configs[k]['references_with_new_bad']=sum(r['config'][k]['new_bad']>0 for r in rr)
    active=rr[0]['config']['actual_pool_1']
    assert factor_refs.keys()==factor_cells.keys()
    return dict(query=s['query'],kind=s['kind'],label=s['label'],person=s['query'].split('_')[-2],method=method,
      missed_incremental=bool(s['previous'][method]['inside_raw']==0),references=8,
      current_references_with_new_bad=sum(r['local_new_bad']>0 for r in rr),
      factor_references=factor_refs,factor_cells=factor_cells,reference_factor_overlap=dict(combos),
      references_with_original_difference=sum(r['original_bad']>0 for r in rr),
      references_with_any_difference_after=sum(r['config']['actual_pool_1']['bad']>0 for r in rr),
      references_with_unknown=sum(r['local_unknown']>0 for r in rr),
      all_8_without_any_difference_after=all(r['config']['actual_pool_1']['bad']==0 for r in rr),
      original_difference_cells=sum(r['original_bad'] for r in rr),
      local_unknown_cells=sum(r['local_unknown'] for r in rr),local_expected_cells=sum(r['local_expected'] for r in rr),
      convex_gap_cells=sum(r['convex_gap_cells'] for r in rr),references_with_convex_gap=sum(r['convex_gap_cells']>0 for r in rr),
      rows_far_from_deleted_neighbors=sum(r['reference_rows_far_from_deleted_neighbors'] for r in rr),
      configs=configs,source=s)

def cohort(records):
    ss=[r['source'] for r in records]
    combos=Counter('+'.join(k for k in FACTORS if r['factor_references'][k]>0) or 'none' for r in records)
    return dict(n=len(records),source=source_summary(ss),
      factors={k:dict(variants=sum(r['factor_references'][k]>0 for r in records),
                         references=sum(r['factor_references'][k] for r in records),
                         cells=sum(r['factor_cells'][k] for r in records)) for k in FACTORS},
      factor_overlap=dict(combos),no_factor_witness=sum(all(v==0 for v in r['factor_references'].values()) for r in records),
      any_original_difference=sum(r['references_with_original_difference']>0 for r in records),
      all_8_without_any_difference_after=sum(r['all_8_without_any_difference_after'] for r in records),
      any_unknown=sum(r['references_with_unknown']>0 for r in records),
      local_unknown_cells=sum(r['local_unknown_cells'] for r in records),
      local_expected_cells=sum(r['local_expected_cells'] for r in records),
      convex_gap_variants=sum(r['convex_gap_cells']>0 for r in records),
      convex_gap_cells=sum(r['convex_gap_cells'] for r in records),
      rows_far_from_deleted_neighbors=sum(r['rows_far_from_deleted_neighbors'] for r in records))

def source_channel(e,h,a):
    if a<15:return e['angle_low'][:,h,a],e['angle_high'][:,h,a],e['angle_valid'][:,h,a],'손가락 각도 (도)'
    if a<17:
        axis=a-15;v=e['centre'][:,h,axis];r=e['radius'][:,h]
        return v-r,v+r,e['palm_valid'][:,h],'손바닥 위치 (어깨너비 단위)'
    joint=(a-17)//2;axis=(a-17)%2;v=e['arm_xy'][:,h,joint,axis]
    return v,v,e['arm_valid'][:,h,joint],('팔꿈치' if joint==0 else '팔 손목')+' 위치 (어깨너비 단위)'

def run():
    assert not (U/'RUNNING.json').exists()
    verified=read(U/'VERIFY.json');assert verified['status'].startswith('PASS')
    assert not (U/'ANALYSIS.json').exists()
    result=read(U/'RESULT.json');f=read(U/'FROZEN.json')
    assert len(result['sources'])==81 and len(result['pairs'])==1296 and len(result['normal'])==2160
    lookup={(s['query'],s['kind']):s for s in result['sources']}
    bins=defaultdict(list)
    for r in result['pairs']:bins[r['query'],r['kind'],r['method']].append(r)
    records={method:[variant_record(s,bins[s['query'],s['kind'],method],method) for s in result['sources']] for method in ('old','new')}
    misses={method:[r for r in records[method] if r['missed_incremental']] for method in records}
    keysets={m:{(r['query'],r['kind']) for r in rr} for m,rr in misses.items()}
    assert len(misses['old'])==12 and len(misses['new'])==9
    common=keysets['old']&keysets['new'];union=keysets['old']|keysets['new'];assert len(common)==3 and len(union)==18
    cohorts={m+'_all':cohort(records[m]) for m in records}
    cohorts.update({m+'_missed':cohort(misses[m]) for m in records})
    for m in records:cohorts[m+'_both_missed']=cohort([r for r in records[m] if (r['query'],r['kind']) in common])
    cohorts['source_union_missed']=source_summary([lookup[k] for k in sorted(union)])
    normal={}
    for method in ('old','new'):
        normal[method]={}
        for kind in ('identity','fast075','slow15','pace','pause'):
            rr=[r for r in result['normal'] if r['method']==method and r['kind']==kind];assert len(rr)==216
            out={}
            for config in ('pool_1','pool_0','anchor_1','anchor_0'):
                keys=('comparable','bad','unknown','within','any_bad')
                sums={k:sum(r['config'][config][k] for r in rr) for k in keys}
                sums.update(n=len(rr),added_any_difference=sum(r['config'][config]['any_bad'] and not r['config']['pool_1']['any_bad'] for r in rr),
                  removed_any_difference=sum(r['config']['pool_1']['any_bad'] and not r['config'][config]['any_bad'] for r in rr),
                  comparable_change=sums['comparable']-sum(r['config']['pool_1']['comparable'] for r in rr))
                out[config]=sums
            normal[method][kind]=out
    # Explain one current miss with a reliable single-anchor-vs-pool witness.
    candidates=sorted([r for r in result['pairs'] if r['method']=='new' and (r['query'],r['kind']) in keysets['new'] and r['factors']['range']>0],key=lambda r:(-r['factors']['range'],r['name']))
    example=None
    for row in candidates:
        z=load(U/'cases'/('new__'+row['name']+'.npz'));e=load(I/(row['query']+'.npz'));deleted=z['deleted']
        for j,h,a in np.argwhere(z['range_witness']):
            lo,hi,valid,unit=source_channel(e,int(h),int(a))
            if valid[deleted].any():
                example=dict(query=row['query'],reference=row['reference'],kind=row['kind'],name=row['name'],reference_frame=int(j),hand=int(h),channel=int(a),unit=unit,deleted=deleted.tolist(),factor='range',selection='After results, most range-witness cells then first originally observed source channel; explanatory only, no conditions tuned.')
                break
        if example:break
    (U/'visuals').mkdir()
    plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False})
    cols=['#8b9caa','#4e8e98','#b98b58']
    fig,axs=plt.subplots(1,2,figsize=(12,5));fig.subplots_adjust(bottom=.24,wspace=.28)
    for ax,m,n in [(axs[0],'old',12),(axs[1],'new',9)]:
        vals=[cohorts[m+'_missed']['factors'][k]['variants'] for k in FACTORS]
        ax.bar(np.arange(3),vals,color=cols)
        for x,val in enumerate(vals):ax.text(x,val+.15,str(val),ha='center')
        ax.set(xticks=np.arange(3),xticklabels=['기존 시점 연결\n유지하면 새 차이','단일 참고이면\n새 차이','이웃 확장 없으면\n새 차이'],ylim=(0,n+1),ylabel='새 차이를 놓친 변형 수',title=f'{"기존 정렬" if m=="old" else "위치·방향 보조"} · {n}개 놓침')
    fig.suptitle('여러 요인이 겹칠 수 있는 진단 · 원인별 분할이나 검출 정확도가 아님',fontsize=12,y=1.01)
    fig.savefig(U/'visuals'/'factor_cohorts.png',dpi=145,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5));fig.subplots_adjust(bottom=.2,wspace=.28)
    configs=('pool_1','anchor_1','pool_0','anchor_0');labels=['다중 참고\n이웃 사용','단일 참고\n이웃 사용','다중 참고\n이웃 제외','단일 참고\n이웃 제외']
    for i,method in enumerate(('old','new')):
        totals={k:{v:sum(normal[method][kind][k][v] for kind in normal[method]) for v in ('bad','comparable','unknown')} for k in configs}
        vals=[100*totals[k]['bad']/totals[k]['comparable'] if totals[k]['comparable'] else 0 for k in configs]
        axs[i].bar(np.arange(4),vals,color=['#4e8e98','#b98b58','#8b9caa','#a36f70'])
        for x,val in enumerate(vals):axs[i].text(x,val+.3,f'{val:.1f}%',ha='center',fontsize=10)
        axs[i].set(xticks=np.arange(4),xticklabels=labels,ylabel='비교 가능한 칸 중 차이 표시 (%)',title=('기존 정렬' if method=='old' else '위치·방향 보조')+' · 정상 시간변형 1,080쌍',ylim=(0,max(vals)*1.2+1))
    fig.suptitle('단일 참고·좁은 비교는 정상 수행자의 차이 표시도 늘릴 수 있음 · 의미 오류율 아님',fontsize=12,y=1.01)
    fig.savefig(U/'visuals'/'normal_tradeoff.png',dpi=145,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(3);width=.18
    cs=[source_summary(result['sources']),cohorts['old_missed']['source'],cohorts['new_missed']['source']]
    metrics=[('raw','입력의 손 관측',lambda c:c['raw_hand_observation_fraction']),('angle','각도 비교 근거',lambda c:c['groups']['angle']['coverage']),('palm','손바닥 위치 근거',lambda c:c['groups']['palm']['coverage']),('arm','팔 손목 위치 근거',lambda c:c['groups']['wrist']['coverage'])]
    for i,(k,title,fn) in enumerate(metrics):
        vals=[100*(fn(c) or 0) for c in cs];pos=x+(i-1.5)*width
        ax.bar(pos,vals,width,label=title,color=['#748597','#4e8e98','#b98b58','#9aa66f'][i])
        for j,val in enumerate(vals):ax.text(pos[j],val+1,f'{val:.1f}',ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=['전체 81개 제거 구간','기존 정렬 놓침 12개','보조 정렬 놓침 9개'],ylim=(0,120),ylabel='삭제 전 원본에서 확보된 근거 (%)',title='손이 관측됐어도 모든 손가락 각도를 믿을 수 있다는 뜻은 아님')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.11),ncol=2,fontsize=9)
    fig.savefig(U/'visuals'/'source_coverage.png',dpi=145,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    if example:
        ex=example;e=load(I/(ex['query']+'.npz'));j,h,a=ex['reference_frame'],ex['hand'],ex['channel'];deleted=np.array(ex['deleted'])
        lo,hi,valid,unit=source_channel(e,h,a);lo=np.where(valid,lo,np.nan);hi=np.where(valid,hi,np.nan)
        left,right=deleted[0]-1,deleted[-1]+1;window=np.arange(max(0,left-8),min(len(lo),right+9));z=load(P/'cross'/(ex['name']+'.npz'));t=load(L/'templates'/(ex['query']+'__'+ex['reference']+'.npz'))
        fold=next(fold for fold in f['folds'] if fold['query']==ex['query']);anchor_index=fold['references'].index(ex['reference'])
        refwin=np.arange(max(0,j-8),min(len(t['low']),j+9))
        fig,axs=plt.subplots(2,1,figsize=(11,8));fig.subplots_adjust(hspace=.4,bottom=.1,top=.91)
        axs[0].fill_between(window,lo[window],hi[window],color='#4e8e98',alpha=.25,label='원본 관측의 범위')
        axs[0].plot(window,(lo[window]+hi[window])/2,color='#4e8e98',marker='.',label='원본 관측의 가운데 값')
        axs[0].axvspan(left+.5,right-.5,color='#b96865',alpha=.15,label='시험에서 지운 3프레임')
        if valid[left] and valid[right]:axs[0].plot([left,right],[(lo[left]+hi[left])/2,(lo[right]+hi[right])/2],c='#a9874f',ls='--',marker='o',label='남은 양끝의 직선 연결 (진단)')
        axs[0].set(xlabel='원본 프레임',ylabel=unit,title=f'{lookup[ex["query"],ex["kind"]]["label"]} · {ex["query"].split("_")[-2]} · {"왼손" if h==0 else "오른손"} · 항목 {a+1}')
        axs[0].legend(fontsize=9)
        axs[1].fill_between(refwin,t['low'][refwin,h,a],t['high'][refwin,h,a],color='#9aa3ae',alpha=.3,label='8명 다중 참고 범위')
        axs[1].fill_between(refwin,t['member_low'][anchor_index,refwin,h,a],t['member_high'][anchor_index,refwin,h,a],color='#b96865',alpha=.4,label='한 참고의 관측 범위')
        axs[1].fill_between(refwin,z['query_low'][refwin,h,a],z['query_high'][refwin,h,a],color='#4e8e98',alpha=.3,label='삭제 후 정렬된 시험 범위')
        axs[1].plot(refwin,(z['query_low'][refwin,h,a]+z['query_high'][refwin,h,a])/2,c='#4e8e98')
        axs[1].axvline(j,c='#333333',ls=':',label='범위 민감성 확인 시점')
        axs[1].set(xlabel='참고 프레임 (사람 간 단계 정답은 아님)',ylabel=unit,title='같은 시간 연결에서도 단일 참고 차이가 넓은 범위에 들어갈 수 있음')
        axs[1].legend(fontsize=9)
        fig.savefig(U/'visuals'/'range_example.png',dpi=145,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    out=dict(at=now(),cohorts=cohorts,misses=misses,records=records,cohort_sets={'both_missed':[list(k) for k in sorted(common)],'union_missed':[list(k) for k in sorted(union)]},normal=normal,example=example,
      preserved_controls=result['unchanged_controls'],
      interpretation=['Factor witnesses overlap; counts are not a unique causal partition.','No incremental mismatch does not mean no mismatch or a correct sign.','Single-anchor or no-neighbor sensitivity is not a recommended tighter acceptance rule; normal controls show tradeoffs.','Prepared-input preserved hand observation and reliable angle/position evidence are distinct. Field raw_observed is inherited naming, not full raw21-point or RGB visibility truth.','Endpoint/chord differences indicate measurable geometry, not certified linguistic essentiality.','Held identity mapping is evaluator-only sensitivity, not ground-truth cross-person alignment.','Convex-gap cells may lie between observed individuals; this is descriptive, not semantic error proof.','All data and outputs are research-only; no model/runtime/user/mentor changes.'])
    write(U/'ANALYSIS.json',out)
    print('ANALYSIS COMPLETE',flush=True)
    print('MISSED_COHORTS',json.dumps({k:{j:v for j,v in cohorts[k].items() if j!='source'} for k in ('old_missed','new_missed')},ensure_ascii=False),flush=True)
    print('CURRENT_MISSES',json.dumps([{k:v for k,v in r.items() if k in ('query','kind','factor_references','references_with_original_difference','references_with_any_difference_after','references_with_unknown','all_8_without_any_difference_after')} for r in misses['new']],ensure_ascii=False),flush=True)

if __name__=='__main__':run()
