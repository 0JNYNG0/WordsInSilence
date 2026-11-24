import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import study as d
U=d.U;f=d.read(U/'FROZEN.json');result=d.read(U/'RESULT.json');rows=result['rows'];assert d.read(U/'VERIFY.json')['status']=='PASS'
lookup={r['clip_id']:r for r in f['rows']};base={(r['query'],r['anchor']):r for r in rows if r['kind']=='identity'}
def aggregate(rr):
    out={}
    for k in ('angle','palm','elbow','wrist'):
        gg=[r['groups'][k] for r in rr];tot={field:sum(g[field] for g in gg) for field in gg[0]}
        tot['coverage']=tot['comparable']/tot['expected'];tot['common_resolution']=tot['resolved_observed']/tot['common_old_difference'] if tot['common_old_difference'] else None
        out[k]=tot
    return dict(n=len(rr),groups=out)
groups={k:aggregate([r for r in rows if r['kind']==k]) for k in f['kinds']}
people={str(p):aggregate([r for r in rows if r['kind']=='identity' and lookup[r['query']]['signer_id']==p]) for p in range(1,10)}
words={w:aggregate([r for r in rows if r['kind']=='identity' and r['label']==w]) for w in sorted(set(r['label'] for r in rows))}
querycases=[];deletions=[]
for row in rows:
    if not row['kind'].startswith('skip'):continue
    z=d.load(U/'cases'/(row['name']+'.npz'));brow=base[row['query'],row['anchor']];b=d.load(U/'cases'/(brow['name']+'.npz'))
    removed=set(map(int,z['deleted']));affected=np.zeros(len(z['expected']),bool)
    for i,j in b['path']:
        if i in removed:affected[j]=True
    mask=affected[:,None,None]&z['expected'];newraw=z['mismatch']&~b['mismatch'];newlong=z['persistent']&~b['persistent']
    relevant=mask&b['comparable'];incomplete=relevant&z['unknown']
    deletions.append(dict(name=row['name'],query=row['query'],anchor=row['anchor'],kind=row['kind'],
        new_raw=bool((newraw&mask).any()),new_persistent=bool((newlong&mask).any()),
        angle_new_raw=bool((newraw[:,:,:15]&mask[:,:,:15]).any()),
        new_unknown_cells=int(incomplete.sum()),original_comparable=int(relevant.sum()),
        new_raw_cells=int((newraw&mask).sum()),new_persistent_cells=int((newlong&mask).sum()),
        affected_reference_frames=int(affected.sum()),linguistically_essential=None))
for cid in lookup:
    for kind in f['kinds']:
        rr=[r for r in rows if r['query']==cid and r['kind']==kind];assert len(rr)==8
        querycases.append(dict(query=cid,kind=kind,warning_references=sum(any(r['groups'][g]['warning'] for g in ('angle','palm','elbow','wrist')) for r in rr),
            angle_warning_references=sum(r['groups']['angle']['warning'] for r in rr),
            whole_action_correct=None))
omission={}
for size in ('skip3','skip20pct'):
    rr=[r for r in deletions if r['kind'].startswith(size)];blocks=[]
    for cid in lookup:
        for portion in (25,50,75):
            pairs=[r for r in rr if r['query']==cid and r['kind']==size+'_'+str(portion)];assert len(pairs)==8
            blocks.append(dict(query=cid,kind=size+'_'+str(portion),new_raw_refs=sum(p['new_raw'] for p in pairs),new_persistent_refs=sum(p['new_persistent'] for p in pairs),
                new_unknown_cells=sum(p['new_unknown_cells'] for p in pairs)))
    omission[size]=dict(n_pairs=len(rr),n_source_variants=len(blocks),pairs_added_raw=sum(r['new_raw'] for r in rr),pairs_added_persistent=sum(r['new_persistent'] for r in rr),
         variants_any_raw=sum(b['new_raw_refs']>0 for b in blocks),variants_five_raw=sum(b['new_raw_refs']>=5 for b in blocks),
         variants_any_persistent=sum(b['new_persistent_refs']>0 for b in blocks),variants_no_added_difference=sum(b['new_raw_refs']==0 for b in blocks),members=blocks)

# Example selection is for illustration only; all results are retained.
normal=sorted(base.values(),key=lambda r:(-r['groups']['angle']['resolved_observed'],r['name']))[0]
misses=[r for r in deletions if r['kind'].startswith('skip3') and not r['new_raw'] and r['original_comparable']>0]
miss=sorted(misses,key=lambda r:(-r['original_comparable'],r['name']))[0] if misses else deletions[0]
(U/'visuals').mkdir();plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False})
g=groups['identity']['groups'];labels=['손가락 각도','손바닥 위치','팔꿈치 위치','팔 손목 위치'];keys=list(g)
fig,axs=plt.subplots(1,2,figsize=(13,4.7),layout='constrained');x=np.arange(4)
axs[0].bar(x-.18,[g[k]['base_warning'] for k in keys],.36,color='#bb9161',label='단일 참고 · 엄격 관측')
axs[0].bar(x+.18,[g[k]['warning'] for k in keys],.36,color='#458e98',label='다중 사람 범위')
axs[0].set(xticks=x,xticklabels=labels,ylabel='지속 차이가 하나라도 있는 비교 / 216',ylim=(0,280),title='원래 속도 · 다른 사람끼리 비교');axs[0].legend(fontsize=9,ncol=2)
for i,k in enumerate(keys):
    axs[0].text(i-.18,g[k]['base_warning']+3,str(g[k]['base_warning']),ha='center');axs[0].text(i+.18,g[k]['warning']+3,str(g[k]['warning']),ha='center')
rr=[100*g[k]['resolved_observed']/max(1,g[k]['base_mismatch']) for k in keys];uu=[100*g[k]['old_difference_now_unknown']/max(1,g[k]['base_mismatch']) for k in keys]
remain=[100-a-b for a,b in zip(rr,uu)]
axs[1].barh(x,rr,color='#4c9e91',label='관측 유지 · 다중 범위 안');axs[1].barh(x,uu,left=rr,color='#b9bec8',label='근거 부족으로 보류');axs[1].barh(x,remain,left=np.array(rr)+uu,color='#be6864',label='범위 밖 차이 남음')
axs[1].set(yticks=x,yticklabels=labels,xlim=(0,100),xlabel='기존 범위 밖 칸을 나눈 비율 (%)',title='경고 감소의 이유를 분리');axs[1].legend(fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=1)
fig.savefig(U/'visuals'/'summary.png',dpi=145);plt.close(fig)

def graph(row,name):
    z=d.load(U/'cases'/(row['name']+'.npz'));cid=row['query'];rid=row.get('anchor');t=d.load(U/'templates'/(cid+'__'+rid+'.npz'));r=d.prev.data(lookup[rid])[1]
    candidates=np.argwhere(z['base_mismatch'][:,:,:15]&z['within_range'][:,:,:15])
    j,h,a=map(int,candidates[len(candidates)//2]) if len(candidates) else (len(r['core'])//2,1,0)
    ts=r['timestamp_us']/1e6;fig,axs=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    axs[0].fill_between(ts,t['low'][:,h,a],t['high'][:,h,a],color='#86c2b3',alpha=.55,label='시험 대상 제외 · 다른8명 관측 범위')
    axs[0].plot(ts,(z['query_low'][:,h,a]+z['query_high'][:,h,a])/2,color='#365a8a',label='시험 영상의 연결된 각도 범위 중앙')
    axs[0].plot(ts,(r['angle_low'][:,h,a]+r['angle_high'][:,h,a])/2,color='#c18a57',alpha=.8,label='단일 참고 각도')
    axs[0].set(title=f'{lookup[cid]["label"]} / 시험 {cid.split("_")[-2]} / 기준축 {rid.split("_")[-2]} / {row["kind"]}',ylabel='투영 각도(도)');axs[0].legend(fontsize=9)
    expected=z['expected'][:,:,:15].sum((1,2));unknown=z['unknown'][:,:,:15].sum((1,2));diff=z['mismatch'][:,:,:15].sum((1,2));within=z['within_range'][:,:,:15].sum((1,2))
    axs[1].stackplot(ts,unknown,diff,within,labels=['근거 부족','범위 밖 차이','관측 범위 안 (수어 정답 뜻 아님)'],colors=['#b9bec8','#bb6362','#5ba699'])
    axs[1].set(xlabel='기준 참고 시간(초)',ylabel='양손 각도 칸 수',title='모름을 정상으로 바꾸지 않는다');axs[1].legend(fontsize=9)
    fig.savefig(U/'visuals'/name,dpi=140);plt.close(fig)
    return dict(name=row['name'],side=h,angle=a,reference_frame=int(r['frame_id'][j]))
shown=graph(normal,'normal_range.png');shown_miss=graph(miss,'short_omission.png')
fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for ax,(size,title) in zip(axs, [('skip3','짧은 3프레임 구간 제거'),('skip20pct','동작 구간의 20% 제거')]):
    s=omission[size];vals=[s['variants_any_raw'],s['variants_five_raw'],s['variants_any_persistent']]
    ax.bar(range(3),vals,color=['#4d89aa','#527c93','#7399a1']);ax.set(xticks=range(3),xticklabels=['1개 이상 참고\n새 차이','5개 이상 참고\n새 차이','1개 이상 참고\n새 지속 차이'],ylim=(0,92),ylabel='시간 제거 원본 변형 / 81개',title=title)
    for i,v in enumerate(vals):ax.text(i,v+2,str(v),ha='center')
fig.savefig(U/'visuals'/'omissions.png',dpi=145);plt.close(fig)
d.write(U/'ANALYSIS.json',dict(at=d.now(),groups=groups,people=people,words=words,querycases=querycases,omission=omission,deletion_pairs=deletions,
 examples=dict(normal=shown,short=shown_miss),
 interpretation=['Minmax range describes existing observed public variation, not established correct-sign tolerance.',
 'Report common observed cells separately from insufficient-reference/query cells.',
 'Added deletion warnings use originalquery-anchor alignment as localization proxy, not exact crossperson time truth.',
 'No added difference does not make omitted execution normal; semantic pass always unset.',
 'Peraxis/time envelopes lose joint shape correlation and can absorb brief differences.','Reference-only LOPO is not unseen model-person testing.']))
print('IDENTITY',groups['identity'],flush=True)
print('OMISSIONS',{k:{kk:vv for kk,vv in value.items() if kk!='members'} for k,value in omission.items()},flush=True)
