"""Descriptive summaries and visual evidence; no parameter adjustment."""
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import study as d
U,P=d.U,d.P
f=d.read(U/'FROZEN.json');rows={r['clip_id']:r for r in f['rows']}
e={c:d.load(U/'evidence'/(c+'.npz')) for c in rows}
(U/'visuals').mkdir()
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':12})
counts=Counter();details=[]
for p in d.read(P/'DIAGNOSIS.json')['changed_pairs']:
    q,r=p['query'],p['reference'];name=q+'__'+r
    old=d.load(P/'pairs'/('baseline__'+name+'.npz'));new=d.load(P/'pairs'/('rest_alignment__'+name+'.npz'))
    z=e[q];flags=new['active_to_rest']&~old['active_to_rest']
    for i,h in np.argwhere(flags):
        rec=dict(query=q,reference=r,index=int(i),frame=int(z['frame_id'][i]),side=('L','R')[h],
            observed=bool(z['observed'][i,h]),position_change=bool(z['position_change'][i,h]),
            angle_change_count=int(z['angle_change'][i,h].sum()),rest_witness=bool(z['source_rest'][i,h]),
            valid_angles=int(z['angle_valid'][i,h].sum()),semantic_truth=None)
        details.append(rec);counts['pair_handframes']+=1;counts['observed']+=int(rec['observed']);counts['position_change']+=int(rec['position_change']);counts['angle_change']+=int(rec['angle_change_count']>0);counts['no_measured_change']+=int(not z['measured_change'][i,h]);counts['rest_witness']+=int(rec['rest_witness'])
known=[]
for signer,frames in [(4,[58,61,64]),(6,[28])]:
    q=f'NIA_SL_WORD0185_REAL{signer:02d}_F';r='NIA_SL_WORD0185_REAL09_F';z=e[q]
    for frame in frames:
        match=np.flatnonzero(z['frame_id']==frame);assert len(match)==1;i=int(match[0]);row=dict(query=q,reference=r,frame=frame,index=i,angle_evidence=int(z['angle_valid'][i,1].sum()),angle_change=int(z['angle_change'][i,1].sum()))
        for c in ('baseline','rest_alignment'):
            pair=d.load(U/'pairs'/(c+'__'+q+'__'+r+'.npz'));con=d.load(U/'queries'/(c+'__'+q+'.npz'))
            row[c]=dict(angle_warning=int(pair['angle_warning'][i,1].sum()),position_warning=bool(pair['position_warning'][i,1]),consensus_angle_warning=int((con['angle_votes'][i,1]>=5).sum()),insufficient=int(pair['insufficient_timeline'][i,1].sum()))
        known.append(row)

def timeline(cid,name):
    z=e[cid];t=z['timestamp_us']/1e6;gap=d.load(U/'pairs'/('baseline__'+cid+'__'+next(r for r in rows if r!=cid and rows[r]['word_code']==rows[cid]['word_code'])+'.npz'))
    fig,ax=plt.subplots(2,1,figsize=(12,6.8),sharex=True,layout='constrained')
    for h,side in enumerate(('왼손','오른손')):
        tracks=[z['observed'][:,h],z['position_change'][:,h],z['angle_change'][:,h].any(1),z['source_rest'][:,h],gap['insufficient_timeline'][:,h].any(1)]
        colors=['#2874ae','#159a80','#845ec2','#ddad2c','#c64f50'];labels=['관측됨','위치 변화 근거','각도 변화 근거','확인된 안정 손내림','비교 근거 부족']
        for j,(arr,col) in enumerate(zip(tracks,colors)):
            ax[h].bar(t[arr],np.ones(int(arr.sum()))*.66,bottom=4-j-.33,width=.03334,color=col)
        for seg in np.split(np.flatnonzero(z['core']),np.flatnonzero(np.diff(np.flatnonzero(z['core']))>1)+1):
            if len(seg):ax[h].axvspan(t[seg[0]],t[seg[-1]]+.03333,color='#a0afc5',alpha=.14,zorder=-1)
        ax[h].set(yticks=np.arange(5)[::-1],yticklabels=labels,ylim=(-.6,4.6),title=side+' · 겹치는 근거는 함께 표시');ax[h].grid(axis='x',alpha=.15)
    ax[1].set_xlabel('촬영 시간(초) · 회색 배경=공개 주석 동작 구간')
    fig.suptitle(rows[cid]['label']+' / '+cid.split('_')[-2]+' — 보인다는 사실과 변화 근거를 구분',fontsize=16)
    fig.savefig(U/'visuals'/name,dpi=140);plt.close(fig)
timeline('NIA_SL_WORD0185_REAL04_F','evidence_timeline.png')
timeline('NIA_SL_WORD0003_REAL01_F','crossing_timeline.png')
timeline('NIA_SL_WORD0949_REAL01_F','onehand_timeline.png')

original=[]
for c in d.prev.CONDITIONS:
    m=[d.read(p) for p in (U/'queries').glob(c+'__*.json')];original.append(dict(condition=c,queries=len(m),flagged=sum(x['angle_consensus_frames']>0 for x in m),cells=sum(x['angle_consensus_frames'] for x in m)))
controls=[];cr=d.read(U/'CONTROLS.json')['rows']
for kind in ['speed0.75','speed1.5','pause','repeat2','repeat3']:
    rr=[r for r in cr if r['kind']==kind];controls.append(dict(kind=kind,queries=len(rr),flagged=sum(r['has_consensus'] for r in rr),cells=sum(r['consensus_angle_frames'] for r in rr)))
fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
vals=[original[0]['flagged']]+[r['flagged'] for r in controls]
names=['원래 공개 영상','길이 0.75배\n(빠르게 재표본)','길이 1.5배\n(느리게 재표본)','중간 0.5초 정지','2회 이어붙임\n(연결도 복사)','3회 이어붙임\n(연결도 복사)']
ax.barh(names[::-1],vals[::-1],color=['#6c89a4']*2+['#dc9b4a']*3+['#376ba4']);ax.set(xlim=(0,30),xlabel='8명 중 5명 이상 참고에서 경고가 겹친 영상 수 / 각 27개',title='정상 변형에서도 경고가 많음 → 동작 오류 판정에 사용 불가')
for i,v in enumerate(vals[::-1]):ax.text(v+.25,i,str(v)+' / 27',va='center',fontsize=12)
fig.savefig(U/'visuals'/'normal_warnings.png',dpi=150);plt.close(fig)

# Fixed source-time gap display: demonstrate independently of any new matched path.
cid='NIA_SL_WORD0185_REAL04_F';rid='NIA_SL_WORD0185_REAL09_F';name='short_R__'+cid+'__'+rid
saved=d.load(P/'stress'/(name+'.npz'));out=d.load(U/'missing'/(name+'.npz'));z=e[cid];t=z['timestamp_us']/1e6;ids=saved['removed_query_rows']
fig,ax=plt.subplots(figsize=(10,3.8),layout='constrained')
ax.step(t,out['missing_timeline'][:,1].sum(1),where='mid',color='#ca5151',linewidth=2,label='비교에 필요한 시간 중 미관측 각도 수')
for i in ids:ax.axvspan(t[i]-.01667,t[i]+.01667,color='#f0c25b',alpha=.4)
ax.set(xlim=(max(0,t[ids[0]]-.45),t[ids[-1]]+.45),ylim=(-.5,17),yticks=[0,5,10,15],xlabel='원래 촬영 시간(초)',ylabel='오른손 각도 수',title='중간 3프레임을 지워도 사라진 시간은 기록에 남음')
ax.legend(loc='upper left',fontsize=10);fig.savefig(U/'visuals'/'missing_timeline.png',dpi=150);plt.close(fig)

result=dict(at=d.now(),old_visibility_flags=dict(counts),old_visibility_rows=details,known_transition_checks=known,original=original,controls=controls,
    short_gap_example=dict(query=cid,reference=rid,frames=z['frame_id'][ids].tolist(),all_15_angles_visible_as_missing=bool(out['missing_timeline'][ids,1].all())),
    decision='Diagnostic timelines useful. Plateau warnings not suitable for automatic scoring/abstention because normal controls frequently trigger. No alignment or recognition improvement established; no deployment or training.',
    caveats=['Motion evidence means separation between two detector estimates, not semantic movement truth or calibrated uncertainty.', 'No observed change can mean stationary hand or insufficient evidence, not proof of rest.', 'Rest requires positive stable lowering; a low motion hand still in ROI is not automatically rest.', 'Five-of-eight consensus is correlated geometric warning, not linguistic consensus.', 'Repeated paths were copied with known repeats, no automatic cycle inference validation.', 'Annotated query core used for diagnostic timeline; live-camera segmentation still separate.'])
d.write(U/'ANALYSIS.json',result)
print('ANALYSIS',dict(counts),known,flush=True)
