import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import cv2
import study as d
U=d.U;f=d.read(U/'FROZEN.json');v=d.read(U/'VERIFY.json');rows=d.read(U/'RESULT.json')['rows'];assert v['status']=='PASS'
(U/'visuals').mkdir();byid={r['clip_id']:r for r in f['rows']}
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':12})
groups={};examples={}
for pref in ('skip3','skip20pct'):
    sub=[r for r in rows if r['kind'].startswith(pref)];cc=Counter();members={}
    for r in sub:
        z=d.load(U/'cases'/(r['kind']+'__'+r['clip']+'.npz'));ids=z['deleted'];available=z['expected'][ids]&~z['unknown'][ids]
        key='localized_warning' if r['hit_deleted'] else 'no_comparable_evidence' if not available.any() else 'mismatch_not_three_contiguous' if z['mismatch'][ids].any() else 'within_permitted_envelope'
        cc[key]+=1;members.setdefault(key,[]).append(dict(clip=r['clip'],kind=r['kind']))
    groups[pref]=dict(counts=dict(cc),members=members)
    for k,l in members.items():examples[pref+'_'+k]=l[0]
normal=[r for r in rows if r['kind'] in d.NORMAL and r['kind']!='identity'];normalflag=[r for r in normal if r['local_warning']]
assert len(normalflag)==1

fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
labels=['일정하게 빠르게','일정하게 느리게','중간 빠르기 변화','중간 0.5초 정지'];kinds=['fast075','slow15','pace','pause'];x=np.arange(4)
old=[v['conditions'][k]['previous_warning'] for k in kinds];new=[v['conditions'][k]['local_warning'] for k in kinds]
axs[0].bar(x-.18,old,width=.36,color='#c88c43',label='이전 연결 횟수 경고');axs[0].bar(x+.18,new,width=.36,color='#1d8c80',label='주변 모양 범위 확인')
axs[0].set(xticks=x,xticklabels=labels,ylim=(0,31),ylabel='경고 시험 수 / 조건별 27개',title='정상 변형에 대한 경고 감소');axs[0].tick_params(axis='x',labelsize=9);axs[0].legend(fontsize=10)
for i,(a,b) in enumerate(zip(old,new)):axs[0].text(i-.18,a+.4,str(a),ha='center');axs[0].text(i+.18,b+.4,str(b),ha='center')
axs[1].bar([0,1],[16,48],color='#466fa5');axs[1].bar([0,1],[65,33],bottom=[16,48],color='#e4e9ef')
axs[1].set(xticks=[0,1],xticklabels=['3프레임 제거','동작 구간의 20% 제거'],ylim=(0,90),ylabel='시험 수 / 각 81개',title='뺀 시간대에서 지속 경고가 난 경우')
for i,n in enumerate([16,48]):axs[1].text(i,n/2,f'{n} / 81',ha='center',va='center',color='white',fontsize=15)
fig.savefig(U/'visuals'/'summary.png',dpi=150);plt.close(fig)

def graph(case,filename):
    cid,kind=case['clip'],case['kind'];z=d.load(U/'cases'/(kind+'__'+cid+'.npz'));e=d.load(d.P/'evidence'/(cid+'.npz'));path=z['path'];ix=z['source_index'];deleted=z['deleted'];times=e['timestamp_us']/1e6
    qi=np.arange(len(ix))/30;target=np.interp(ix,np.arange(len(times)),times);pred=np.array([np.median(times[path[path[:,0]==i,1]]) for i in range(len(ix))])
    flags=np.argwhere(z['persistent']);raw=np.argwhere(z['mismatch'])
    pick=flags[0] if len(flags) else raw[0] if len(raw) else np.array([int(deleted[0]) if len(deleted) else len(times)//2,1,0])
    j,h,a=map(int,pick)
    # For missed windows choose a comparable angular signal from within the deleted interval.
    if len(deleted) and not len(flags):
        hit=np.argwhere(z['mismatch'][deleted])
        if len(hit):jj,h,a=map(int,hit[0]);j=int(deleted[jj])
        else:
            available=np.argwhere(z['expected'][deleted]&~z['unknown'][deleted])
            if len(available):jj,h,a=map(int,available[0]);j=int(deleted[jj])
    fig,ax=plt.subplots(2,1,figsize=(11,7.2),layout='constrained')
    ax[0].plot(qi,target,color='#1a9981',label='시간 변환으로 아는 대응 시점',linewidth=3);ax[0].plot(qi,pred,'--',color='#405fa0',label='기존 DTW가 연결한 시점')
    ax[0].set(xlabel='변형 영상의 시간(초)',ylabel='원본 영상의 시간(초)',title=byid[cid]['label']+' / '+cid.split('_')[-2]+' / '+kind);ax[0].legend(fontsize=10)
    low=e['angle_low'][:,h,a];high=e['angle_high'][:,h,a]
    ax[1].fill_between(times,z['local_low'][:,h,a],z['local_high'][:,h,a],color='#9dcfc6',alpha=.6,label='연결 장면 + 바로 이웃의 모양 범위')
    ax[1].plot(times,(low+high)/2,color='#243d67',label='원본에서 확인된 각도');ax[1].fill_between(times,low,high,color='#425d81',alpha=.2)
    warn=z['persistent'][:,h,a];ax[1].scatter(times[warn],((low+high)/2)[warn],color='#c95050',label='3프레임 이상 범위 밖 경고',zorder=4)
    if len(deleted):
        for at in deleted:ax[1].axvspan(times[at]-.01667,times[at]+.01667,color='#edbc57',alpha=.2)
        ax[1].set_xlim(times[max(0,deleted[0]-9)],times[min(len(times)-1,deleted[-1]+9)])
    else:ax[1].set_xlim(times[max(0,j-10)],times[min(len(times)-1,j+12)])
    ax[1].set(xlabel='원본 촬영 시간(초) · 노란색=인위적으로 제거한 부분',ylabel='투영 각도(도)',title=('왼손' if h==0 else '오른손')+f' 각도 {a+1}: 모양 범위 비교');ax[1].legend(fontsize=9)
    fig.savefig(U/'visuals'/filename,dpi=140);plt.close(fig)
    return dict(clip=cid,kind=kind,side=h,angle=a,highlight_frame=int(e['frame_id'][j]),deleted_frames=e['frame_id'][deleted].tolist())

selections={}
selections['normal_false_warning']=graph(normalflag[0],'normal_false_warning.png')
selections['deletion_detected']=graph(examples['skip20pct_localized_warning'],'deletion_detected.png')
selections['deletion_undetected']=graph(examples['skip20pct_within_permitted_envelope'],'deletion_undetected.png')

# Current raw frames for context, not ground-truth semantic labeling.
raw=[];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
for key,s in selections.items():
    row=byid[s['clip']];video=row['video'];frames=[max(0,s['highlight_frame']-3),s['highlight_frame'],s['highlight_frame']+3]
    if s['deleted_frames']:frames=[max(0,s['deleted_frames'][0]-1),s['deleted_frames'][len(s['deleted_frames'])//2],s['deleted_frames'][-1]+1]
    cap=cv2.VideoCapture(video);imgs=[]
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read()
        if not ok:imgs=[];break
        imgs.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((480,270)))
    cap.release()
    if not imgs:raw.append(dict(case=key,status='source_unavailable',path=video));continue
    page=Image.new('RGB',(1440,320),'#122d49');draw=ImageDraw.Draw(page)
    for i,(im,frame) in enumerate(zip(imgs,frames)):
        page.paste(im,(480*i,25));draw.text((480*i+10,3),f'{row["label"]} {s["clip"].split("_")[-2]} / 원본 f{frame}',font=font,fill='white')
    draw.text((10,297),'원본의 시간상 앞·중간·뒤 장면. 의미상 필수 동작이라는 판정은 하지 않음.',font=font,fill='white')
    filename=key+'_raw.jpg';page.save(U/'visuals'/filename,quality=92);raw.append(dict(case=key,status='read',frames=frames,image=filename))
d.write(U/'ANALYSIS.json',dict(at=d.now(),normal_total=len(normal),normal_previous=sum(r['previous_warning'] for r in normal),normal_local=sum(r['local_warning'] for r in normal),
 groups=groups,normal_false_warnings=normalflag,selections=selections,raw_context=raw,
 interpretation=['Existing DTW normal same-source timing is accurate; previous plateau alarm is not a timing accuracy metric.',
 'Local envelope greatly reduces normal alarms in this controlled benchmark, but detects only16/81short48/81longdeleted windows.',
 'No proof removed portions are mandatory sign units; no universal correctness claim.',
 'No-warning can mean permissive envelope or insufficient evidence, never automatic full-match.',
 'Same-source matched samples and errors are correlated; outside-person test remains unproved.',
 'Local convex envelopes permit intermediate values across matched valid samples; they do not reconstruct unseen poses or guarantee temporal order.',
 'All results/parameters preserved; no post-hoc retuning.']))
print('ANALYZED',groups.keys(),raw)
