import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import cv2
import study as d
U=d.U;f=d.read(U/'FROZEN.json');result=d.read(U/'RESULT.json');rows=result['rows'];assert d.read(U/'VERIFY.json')['status']=='PASS'
byid={r['clip_id']:r for r in f['rows']};base={(r['query'],r['reference']):r for r in rows if r['kind']=='identity'}
groups={};perquery=[]
def average(xs):
    xs=[x for x in xs if x is not None];return float(np.mean(xs)) if xs else None
for kind in d.KINDS:
    rr=[r for r in rows if r['kind']==kind];exp=sum(r['expected'] for r in rr)
    groups[kind]=dict(n=len(rr),old=sum(r['old_warning'] for r in rr),new=sum(r['new_warning'] for r in rr),
      support=sum(r['expected']-r['unknown'] for r in rr)/exp,strict=sum(r['strict'] for r in rr)/exp,
      expected=exp,unknown=sum(r['unknown'] for r in rr),persistent_fraction=sum(r['persistent'] for r in rr)/exp,
      mean_gap_deg=average(r['gap_deg'] for r in rr),
      added=sum(r['new_warning'] and not base[r['query'],r['reference']]['new_warning'] for r in rr),
      removed=sum(not r['new_warning'] and base[r['query'],r['reference']]['new_warning'] for r in rr),
      geometry=[dict(support=1-sum(r['geometry'][j]['unknown'] for r in rr)/sum(r['geometry'][j]['expected'] for r in rr),
                     strict=sum(r['geometry'][j]['strict'] for r in rr)/sum(r['geometry'][j]['expected'] for r in rr),
                     mean_pair_distance_sw=average(r['geometry'][j]['distance_sw'] for r in rr)) for j in range(3)])
    for cid in byid:
        qq=[r for r in rr if r['query']==cid];assert len(qq)==8
        def quartiles(values):
            values=[v for v in values if v is not None]
            return np.percentile(values,[25,50,75]).tolist() if values else [None]*3
        perquery.append(dict(query=cid,kind=kind,references=8,warning_refs=sum(r['new_warning'] for r in qq),
           angle_gap_quartiles=quartiles(r['gap_deg'] for r in qq),
           geometry_distance_quartiles=[quartiles(r['geometry'][j]['distance_sw'] for r in qq) for j in range(3)],
           whole_action_correct=None))

lost=[r for r in rows if r['kind']!='identity' and not r['new_warning'] and base[r['query'],r['reference']]['new_warning']]
detail=[]
for r in lost:
    b=base[r['query'],r['reference']];z=d.load(U/'cases'/(r['name']+'.npz'));old=d.load(U/'cases'/(b['name']+'.npz'))
    previously=old['persistent'];detail.append(dict(case=r['name'],previous=b['name'],baseline=b,changed=r,
        previously_flagged_cells=int(previously.sum()),now_unknown=int((previously&z['unknown']).sum()),
        now_mismatch_short=int((previously&z['mismatch']).sum()),now_comparable_in_range=int((previously&~z['unknown']&~z['mismatch']).sum())))

(U/'visuals').mkdir();plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False})
labels=['원래 속도','빠르게','느리게','중간 속도 변화','잠깐 정지'];x=np.arange(5)
fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
axs[0].bar(x-.18,[groups[k]['old'] for k in d.KINDS],.36,color='#c99550',label='이전 시간 압축 경고')
axs[0].bar(x+.18,[groups[k]['new'] for k in d.KINDS],.36,color='#b84b52',label='주변 각도 범위 경고')
axs[0].set(xticks=x,xticklabels=labels,ylim=(0,250),ylabel='경고가 하나라도 있는 비교 / 216',title='다른 사람 비교에서는 경고가 거의 모두 발생');axs[0].legend(fontsize=9)
for i,k in enumerate(d.KINDS):
    for dx,key in [(-.18,'old'),(.18,'new')]:axs[0].text(i+dx,groups[k][key]+3,str(groups[k][key]),ha='center',fontsize=9)
g=groups['identity'];cv=[g['support']]+[v['support'] for v in g['geometry']]
axs[1].barh(range(4),np.array(cv)*100,color=['#6b65a8','#32a398','#3b72a1','#3b72a1'])
axs[1].set(yticks=range(4),yticklabels=['손가락 각도','손바닥 위치','팔꿈치 위치','팔 손목 위치'],xlim=(0,110),xlabel='참고가 요구하는 칸 중 비교 가능 비율 (%)',title='항목별 관측 근거는 다르다 — 원래 속도')
for i,v in enumerate(cv):axs[1].text(v*100+1,i,f'{v*100:.1f}%',va='center')
fig.savefig(U/'visuals'/'summary.png',dpi=145);plt.close(fig)

# Deterministic examples after outcomes, explicitly illustrative and not evaluation selection.
example=lost[0] if lost else rows[0];identity=base[example['query'],example['reference']]
z=d.load(U/'cases'/(identity['name']+'.npz'));q=d.load(U/'inputs'/(example['query']+'.npz'));r=d.load(U/'inputs'/(example['reference']+'.npz'))
h=1;times=r['timestamp_us']/1e6
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for k,title in [(0,'손바닥 이동'),(1,'팔꿈치·팔 손목 이동')]:
    ax=axs[0,k]
    for sample,color,label in [(q,'#27709b','비교 영상'),(r,'#cb734c','다른 사람 참고')]:
        if k==0:values=[(sample['centre'][:,h],sample['palm_valid'][:,h],'')]
        else:values=[(sample['arm_xy'][:,h,a],sample['arm_valid'][:,h,a],[' 팔꿈치',' 손목'][a]) for a in range(2)]
        for xy,valid,suffix in values:
            xy=xy.copy();xy[~valid|~sample['core']]=np.nan
            ax.plot(xy[:,0],xy[:,1],color=color,label=label+suffix,linestyle='--' if suffix==' 팔꿈치' else '-')
    ax.invert_yaxis();ax.set(title=title+' — 오른쪽',xlabel='어깨 중앙 기준 가로 / 어깨너비',ylabel='세로 / 어깨너비');ax.legend(fontsize=8)
for k,label in enumerate(['손바닥','팔꿈치','팔 손목']):axs[1,0].plot(times,z['geom_distance'][:,h,k],label=label)
axs[1,0].set(title='DTW로 연결한 시점의 위치 차이',xlabel='참고 시간(초)',ylabel='어깨너비 단위');axs[1,0].legend()
unknown=z['unknown'][:,h].sum(1);flags=z['persistent'][:,h].sum(1);domain=z['expected'][:,h].sum(1)
axs[1,1].stackplot(times,unknown,flags,np.maximum(domain-unknown-flags,0),labels=['근거 부족','범위 차이 지속','기타 비교 가능'],colors=['#b5bcc7','#ba5559','#6cab9f'])
axs[1,1].set(title='오른손 각도 15개의 상태 — 맞음 점수 아님',xlabel='참고 시간(초)',ylabel='각도 수',ylim=(0,15));axs[1,1].legend(fontsize=8)
fig.suptitle(byid[example['query']]['label']+' / '+example['query'].split('_')[-2]+' ↔ '+example['reference'].split('_')[-2]+' (시간 변형 전)',fontsize=15)
fig.savefig(U/'visuals'/'position_and_evidence.png',dpi=140);plt.close(fig)

if lost:
    changed=d.load(U/'cases'/(example['name']+'.npz'));mask=z['persistent'];hit=np.argwhere(mask);j,h,a=map(int,hit[0])
    fig,axs=plt.subplots(2,1,figsize=(11,6),layout='constrained')
    for ax,zz,title in zip(axs,[z,changed],['원래 속도: 일부 구간 경고','빠르게 재표본: 경고 없음 ≠ 잘 맞음']):
        ax.fill_between(times,zz['local_low'][:,h,a],zz['local_high'][:,h,a],alpha=.4,color='#438e83',label='비교 가능한 주변 모양 범위')
        ax.plot(times,(r['angle_low'][:,h,a]+r['angle_high'][:,h,a])/2,color='#365789',label='참고 각도')
        for t in np.flatnonzero(zz['unknown'][:,h,a]):ax.axvspan(times[t]-.016,times[t]+.016,color='#b4bac4',alpha=.4)
        hit=zz['persistent'][:,h,a];ax.scatter(times[hit],((r['angle_low'][:,h,a]+r['angle_high'][:,h,a])/2)[hit],color='#b34c51',s=20,label='지속 차이 경고')
        ax.set(title=title,xlabel='참고 시간(초), 회색=관측 근거 부족',ylabel='각도(도)');ax.legend(fontsize=8)
    fig.savefig(U/'visuals'/'warning_disappears.png',dpi=140);plt.close(fig)

# Two people, original frames at three DTW-related times. No semantic labels.
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16);page=Image.new('RGB',(1440,650),'#183148');draw=ImageDraw.Draw(page);raw=[]
active=np.flatnonzero(r['core']);chosen=[int(active[int((len(active)-1)*p)]) for p in (.25,.5,.75)]
for rowi,cid in enumerate([example['query'],example['reference']]):
    e=q if rowi==0 else r;cap=cv2.VideoCapture(byid[cid]['video']);frames=[]
    for col,j in enumerate(chosen):
        idx=int(np.median(z['path'][z['path'][:,1]==j,0])) if rowi==0 else j
        frame=int(e['frame_id'][idx]);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok,(cid,frame)
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((480,270))
        page.paste(im,(col*480+(480-im.width)//2,rowi*325+35))
        draw.text((col*480+8,rowi*325+5),f'{byid[cid]["label"]} {cid.split("_")[-2]} / 원본 f{frame}',font=font,fill='white');frames.append(frame)
    cap.release();raw.append(dict(clip=cid,frames=frames))
page.save(U/'visuals'/'public_pair_raw.jpg',quality=92)

d.write(U/'ANALYSIS.json',dict(at=d.now(),groups=groups,perquery=perquery,lost_warnings=detail,example=identity,raw=raw,
 conclusions=['Other-person warning reduction not demonstrated; even originals216/216flagged. Do not auto-correct user.',
 'No new warning at clip level is uninformative because baseline already saturated.',
 'Two-detector agreement envelope is not allowed variation across people; motion/angle/detector errors combine.',
 'Palm/arm paths and differences now separated with unknown masks. Arm support cannot certify finger shape.',
 'Time resampling valid endpoints AND can reduce support; disappearing warning is not improved match.',
 'Nearest local observed support is descriptive and may not form a globally monotone point trajectory; plotted trajectories are raw masked paths, not reconstructed motion.',
 'Cross-person times lack independent truth; report consistency/warning counts only, not timing accuracy or semantic pass.']))
print(json.dumps(groups,ensure_ascii=False,indent=2),flush=True)
