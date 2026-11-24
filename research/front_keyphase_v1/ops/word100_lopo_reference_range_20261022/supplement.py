import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import cv2
import study as d
U=d.U;a=d.read(U/'ANALYSIS.json');f=d.read(U/'FROZEN.json');lookup={r['clip_id']:r for r in f['rows']}
plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False})
# Summary layout repair only, no outcomes changed.
g=a['groups']['identity']['groups'];keys=list(g);labels=['손가락 각도','손바닥 위치','팔꿈치 위치','팔 손목 위치'];x=np.arange(4)
fig,axs=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
axs[0].bar(x-.18,[g[k]['base_warning'] for k in keys],.36,color='#bb9161',label='단일 참고 · 엄격 관측');axs[0].bar(x+.18,[g[k]['warning'] for k in keys],.36,color='#458e98',label='다중 사람 범위')
axs[0].set(xticks=x,xticklabels=labels,ylabel='지속 차이가 하나라도 있는 비교 / 216',ylim=(0,280),title='원래 속도 · 다른 사람끼리 비교');axs[0].legend(fontsize=9,ncol=2)
for i,k in enumerate(keys):
    axs[0].text(i-.18,g[k]['base_warning']+3,str(g[k]['base_warning']),ha='center');axs[0].text(i+.18,g[k]['warning']+3,str(g[k]['warning']),ha='center')
rr=[100*g[k]['resolved_observed']/max(1,g[k]['base_mismatch']) for k in keys];uu=[100*g[k]['old_difference_now_unknown']/max(1,g[k]['base_mismatch']) for k in keys]
axs[1].barh(x,rr,color='#4c9e91',label='관측 유지 · 다중 범위 안');axs[1].barh(x,uu,left=rr,color='#b9bec8',label='근거 부족으로 보류');axs[1].barh(x,100-np.array(rr)-uu,left=np.array(rr)+uu,color='#be6864',label='범위 밖 차이 남음')
axs[1].set(yticks=x,yticklabels=labels,xlim=(0,100),xlabel='단일 참고 범위 밖 칸을 나눈 비율 (%)',title='차이가 줄어든 이유를 분리');axs[1].legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=1)
fig.savefig(U/'visuals'/'summary.png',dpi=145);plt.close(fig)

miss=[r for r in a['omission']['skip3']['members'] if r['new_raw_refs']==0];selection=sorted(miss,key=lambda r:(r['new_unknown_cells'],r['query'],r['kind']))[0]
cid=selection['query'];kind=selection['kind'];fold=next(fold for fold in f['folds'] if fold['query']==cid);rid=fold['references'][0]
z=d.load(U/'cases'/(kind+'__'+cid+'__'+rid+'.npz'));b=d.load(U/'cases'/('identity__'+cid+'__'+rid+'.npz'));t=d.load(U/'templates'/(cid+'__'+rid+'.npz'))
q,e=d.prev.data(lookup[cid]);r,re=d.prev.data(lookup[rid]);affected=np.zeros(len(re['core']),bool)
for i,j in b['path']:
    if i in z['deleted']:affected[j]=True
delta=np.abs((z['query_low']+z['query_high'])/2-(b['query_low']+b['query_high'])/2)
score=np.where(affected[:,None,None]&np.isfinite(delta),delta,-1.)[:,:,:15];j,h,ch=np.unravel_index(np.argmax(score),score.shape)
ts=re['timestamp_us']/1e6;fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
ax.fill_between(ts,t['low'][:,h,ch],t['high'][:,h,ch],color='#93cabc',alpha=.5,label='시험 대상 제외한 다른8명 범위')
ax.plot(ts,(b['query_low'][:,h,ch]+b['query_high'][:,h,ch])/2,color='#386a97',label='제거 전 연결 각도');ax.plot(ts,(z['query_low'][:,h,ch]+z['query_high'][:,h,ch])/2,'--',color='#bd7952',label='3프레임 제거 후 연결 각도')
for j in np.flatnonzero(affected):ax.axvspan(ts[j]-.0167,ts[j]+.0167,color='#e4bb64',alpha=.25)
ax.set(title=lookup[cid]['label']+' / '+cid.split('_')[-2]+' / 8참고 모두 추가 차이 없음',xlabel='기준 참고 시간(초), 노란색=기존 정렬로 추정한 제거 대응 시간',ylabel='투영 각도(도)');ax.legend(fontsize=9)
fig.savefig(U/'visuals'/'missed_all_references.png',dpi=145);plt.close(fig)
ids=z['deleted'];frames=[int(e['frame_id'][max(0,ids[0]-1)]),int(e['frame_id'][ids[len(ids)//2]]),int(e['frame_id'][min(len(e['core'])-1,ids[-1]+1)])]
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17);sheet=Image.new('RGB',(1440,310),'#143148');draw=ImageDraw.Draw(sheet);cap=cv2.VideoCapture(lookup[cid]['video'])
for i,frame in enumerate(frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok,(cid,frame)
    im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((480,270));sheet.paste(im,(480*i+(480-im.width)//2,35))
    draw.text((480*i+8,5),f'{lookup[cid]["label"]} {cid.split("_")[-2]} / f{frame} / '+['제거 직전','제거된 부분','제거 직후'][i],font=font,fill='white')
cap.release();sheet.save(U/'visuals'/'short_omission_raw.jpg',quality=93)
d.write(U/'SUPPLEMENT.json',dict(at=d.now(),missed_variants=miss,selected=selection,anchor=rid,raw_frames=frames,side=int(h),channel=int(ch),
  observations='All8references have no newly outside-range cell in original-path localized deleted window; original differences elsewhere can remain. Source core3frames not labeled linguistically indispensable. No new gap in selected example, so insufficiency alone does not explain all misses.',
  normal_range_warning_count_differences=[a['people'][str(p)]['groups']['angle']['base_warning']-a['people'][str(p)]['groups']['angle']['warning'] for p in range(1,10)]))
print('SUPPLEMENT',selection,frames,flush=True)
