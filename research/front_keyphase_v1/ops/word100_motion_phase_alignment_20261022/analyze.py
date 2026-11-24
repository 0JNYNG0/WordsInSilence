"""Saved-result evaluation only; no re-alignment, threshold fitting, or training.

Run once AFTER the separate verifier completes. The independent verifier owns
numerical correctness; this file separates timing, observed differences, losses
of evidence, and artificial temporal omissions for reporting.
"""
import re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import study as d

U=d.U

def counts_stats(x):
    x=np.asarray(x,float)
    return dict(n=int(len(x)),mae=float(x.mean()) if len(x) else None,
                p95=float(np.percentile(x,95)) if len(x) else None,
                within2=float((x<=2).mean()) if len(x) else None)

def run():
    assert not (U/'RUNNING.json').exists(), 'Execution is still active.'
    verify=d.read(U/'VERIFY.json');assert verify['status']=='PASS'
    assert not (U/'ANALYSIS.json').exists()
    f=d.read(U/'FROZEN.json');result=d.read(U/'RESULT.json')
    lookup={r['clip_id']:r for r in f['rows']}
    es={cid:d.load(d.K.P/'evidence'/(cid+'.npz')) for cid in lookup}
    groups=('angle','palm','elbow','wrist')
    primary=('expected','comparable','unknown','reference_insufficient','query_incomplete','query_partial','mismatch','persistent','within_range','warning','any_difference')
    paired=('common','resolved','added','old_difference_now_unknown','old_comparable_now_unknown')
    known={};known_examples=[]
    for kind in f['kinds']:
        rr=[r for r in result['known'] if r['kind']==kind]
        assert len(rr)==27
        pools={k:{c:[] for c in ('old','new')} for k in ('core','moving')}
        pairs=[]
        for row in rr:
            z=d.load(U/'known'/(row['name']+'.npz'));e=es[row['query']]
            ix=z['source_index'];lo=np.floor(ix).astype(int);hi=np.ceil(ix).astype(int)
            core=e['core'][lo]&e['core'][hi]
            moving=core&(e['measured_change'][lo].any(1)|e['measured_change'][hi].any(1))
            assert np.array_equal(moving,z['moving'])
            for maskname,mask in [('core',core),('moving',moving)]:
                for condition in ('old','new'):pools[maskname][condition].extend(z[condition+'_error'][mask].tolist())
            b=row['old_timing']['moving']['mae'];n=row['new_timing']['moving']['mae']
            delta=n-b if b is not None and n is not None else None
            p=dict(name=row['name'],query=row['query'],label=row['label'],kind=kind,
                   old_moving_mae=b,new_moving_mae=n,delta_new_minus_old=delta,path_changed=row['path_changed'])
            pairs.append(p)
            if kind in d.K.NORMAL and kind!='identity' and delta is not None:known_examples.append(p)
        rec=dict(n=len(rr),timing={mask:{condition:counts_stats(values) for condition,values in conditions.items()} for mask,conditions in pools.items()},
                 improved=sum(p['delta_new_minus_old'] is not None and p['delta_new_minus_old'] < -1e-12 for p in pairs),
                 worsened=sum(p['delta_new_minus_old'] is not None and p['delta_new_minus_old'] > 1e-12 for p in pairs),
                 unchanged=sum(p['delta_new_minus_old'] is not None and abs(p['delta_new_minus_old'])<=1e-12 for p in pairs),
                 pairs=pairs,path_changed=sum(r['path_changed'] for r in rr))
        known[kind]=rec

    cross={}
    for kind in f['kinds']:
        rr=[r for r in result['cross'] if r['kind']==kind];assert len(rr)==216
        rec=dict(n=len(rr),path_changed=sum(r['path_changed'] for r in rr),groups={},
                 old_proxy={k:sum(r['old_proxy'][k] for r in rr) for k in rr[0]['old_proxy']},
                 new_proxy={k:sum(r['new_proxy'][k] for r in rr) for k in rr[0]['new_proxy']},
                 support={},native_unknown={k:sum(r[k] for r in rr) for k in ('native_unknown_angles','native_unknown_palms','native_unknown_arms')})
        for group in groups:
            g={condition:{key:sum(r[condition][group][key] for r in rr) for key in primary} for condition in ('old','new')}
            for condition in ('old','new'):
                g[condition]['coverage']=g[condition]['comparable']/g[condition]['expected']
                assert g[condition]['expected']==g[condition]['unknown']+g[condition]['comparable']
            g['comparison']={key:sum(r['comparison'][group][key] for r in rr) for key in paired}
            rec['groups'][group]=g
        # Support is compared on both paths using the same saved cost-evidence matrices.
        support={k:{p:[] for p in ('old','new')} for k in ('position','direction')}
        for row in rr:
            z=d.load(U/'cross'/(row['name']+'.npz'))
            req=f['roles'].get(lookup[row['query']]['word_code'],{}).get('required_sides',['L','R'])
            for condition,path in [('old',z['base_path']),('new',z['path'])]:
                for key,denom in [('position',3*len(req)),('direction',6*len(req))]:
                    support[key][condition].append(float(z[key+'_count'][tuple(path.T)].mean()/denom))
        rec['support']={key:{condition:float(np.mean(values)) for condition,values in bycondition.items()} for key,bycondition in support.items()}
        for condition in ('old','new'):
            p=rec[condition+'_proxy'];p['query_outside_core_fraction']=p['query_outside_core_steps']/p['core_path_steps'] if p['core_path_steps'] else None
        cross[kind]=rec

    omission_pairs={'known':[],'cross':[]}
    identities={folder:{(r['query'],r['reference']):r for r in result[folder] if r['kind']=='identity'} for folder in ('known','cross')}
    identity_cache={folder:{key:d.load(U/folder/(row['name']+'.npz')) for key,row in entries.items()} for folder,entries in identities.items()}
    for folder in ('known','cross'):
        for row in result[folder]:
            if not row['kind'].startswith('skip'):continue
            z=d.load(U/folder/(row['name']+'.npz'));b=identity_cache[folder][row['query'],row['reference']]
            e=es[row['query']];_,deleted=d.K.mapping(len(e['core']),e['core'],row['kind'])
            affected=np.zeros(len(z['expected']),bool)
            if folder=='known':affected[deleted]=True
            else:
                # Fixed old original alignment; do not move the target using a new path.
                old=d.load(d.P/'cases'/('identity__'+row['query']+'__'+row['reference']+'.npz'))
                affected[np.unique(old['path'][np.isin(old['path'][:,0],deleted),1])]=True
            domain=z['expected'];inside=domain&affected[:,None,None];outside=domain&~affected[:,None,None]
            assert np.array_equal(domain,b['expected'])
            rec=dict(name=row['name'],query=row['query'],reference=row['reference'],kind=row['kind'],
                     removed_source_frames=deleted.tolist(),affected_reference_frames=int(affected.sum()),
                     locality='exact original source frames' if folder=='known' else 'fixed old original path proxy; no exact cross-person phase truth')
            for condition,prefix in [('old','old_'),('new','')]:
                bad=z[prefix+'mismatch'];origbad=b[prefix+'mismatch'];pers=z[prefix+'persistent'];origpers=b[prefix+'persistent']
                comparable=z['old_comparable'] if condition=='old' else z['comparable']
                origcomparable=b['old_comparable'] if condition=='old' else b['comparable']
                newbad=bad&~origbad;newpers=pers&~origpers
                rec[condition]=dict(inside_raw=bool((newbad&inside).any()),inside_persistent=bool((newpers&inside).any()),
                    inside_raw_cells=int((newbad&inside).sum()),inside_persistent_cells=int((newpers&inside).sum()),
                    outside_raw=bool((newbad&outside).any()),outside_persistent=bool((newpers&outside).any()),
                    outside_raw_cells=int((newbad&outside).sum()),outside_persistent_cells=int((newpers&outside).sum()),
                    angle_inside_raw=bool((newbad[:,:,:15]&inside[:,:,:15]).any()),
                    old_comparable_inside_now_unknown=int((origcomparable&~comparable&inside).sum()),
                    comparable_inside=int((comparable&inside).sum()))
            omission_pairs[folder].append(rec)
    omissions={}
    for folder,rr in omission_pairs.items():
        omissions[folder]={}
        for size in ('skip3','skip20pct'):
            selected=[x for x in rr if x['kind'].startswith(size)];blocks=defaultdict(list)
            for x in selected:blocks[x['query'],x['kind']].append(x)
            members=[]
            for (cid,kind),items in sorted(blocks.items()):
                assert len(items)==(1 if folder=='known' else 8)
                member=dict(query=cid,kind=kind,references=len(items))
                for condition in ('old','new'):
                    member[condition]={key:sum(x[condition][key] for x in items) for key in ('inside_raw','inside_persistent','outside_raw','outside_persistent','inside_raw_cells','outside_raw_cells','old_comparable_inside_now_unknown')}
                members.append(member)
            s=dict(n_pairs=len(selected),n_variants=len(members),members=members)
            for condition in ('old','new'):
                s[condition]=dict(variants_any_raw=sum(x[condition]['inside_raw']>0 for x in members),
                    variants_five_raw=sum(x[condition]['inside_raw']>=5 for x in members) if folder=='cross' else None,
                    variants_any_persistent=sum(x[condition]['inside_persistent']>0 for x in members),
                    variants_no_new_raw=sum(x[condition]['inside_raw']==0 for x in members),
                    variants_outside_raw=sum(x[condition]['outside_raw']>0 for x in members),
                    pairs_inside_raw=sum(x[condition]['inside_raw'] for x in selected),
                    pairs_outside_raw=sum(x[condition]['outside_raw'] for x in selected),
                    pairs_outside_persistent=sum(x[condition]['outside_persistent'] for x in selected))
            s['newly_detected_variants']=sum(x['old']['inside_raw']==0 and x['new']['inside_raw']>0 for x in members)
            s['lost_detection_variants']=sum(x['old']['inside_raw']>0 and x['new']['inside_raw']==0 for x in members)
            omissions[folder][size]=s

    missing=[]
    for row in result['missing']:
        z=d.load(U/'missing'/(row['name']+'.npz'));cid=row['query'];h=('L','R').index(row['side']);ids=np.asarray(row['removed'],int)
        orig=identity_cache['cross'][cid,row['reference']]
        assert np.array_equal(z['expected'],orig['expected'])
        assert np.all(z['query_expected']>=orig['query_expected'])
        target=ids[z['query_expected'][ids,h]]
        key='query_arm_unknown' if row['kind']=='wholearm' else 'query_angle_unknown'
        assert z[key][target,h].all()
        rec=dict(name=row['name'],kind=row['kind'],query=cid,side=row['side'],path_changed=row['path_changed'],
                 expected_unchanged=True,native_expected_not_reduced=True,native_forced_unknown_preserved=True,
                 forced_native_rows=len(target),native_unknown_angles=row['native_unknown_angles'],native_unknown_palms=row['native_unknown_palms'],native_unknown_arms=row['native_unknown_arms'],
                 old=row['old'],new=row['new'],comparison=row['comparison'],old_proxy=row['old_proxy'],new_proxy=row['new_proxy'])
        for condition,path in [('old',z['base_path']),('new',z['path'])]:
            mapped=np.unique(path[z['expected'][path[:,1],h,0],0])
            rec[condition+'_forced_native_rows_only_mapped_to_excluded_reference']=int((~np.isin(target,mapped)).sum())
            req=f['roles'].get(lookup[cid]['word_code'],{}).get('required_sides',['L','R'])
            rec[condition+'_position_support']=float(z['position_count'][tuple(path.T)].mean()/(3*len(req)))
            rec[condition+'_direction_support']=float(z['direction_count'][tuple(path.T)].mean()/(6*len(req)))
        missing.append(rec)
    missing_summary={}
    for kind in ('wholehand','shortgap','wholearm'):
        rr=[r for r in missing if r['kind']==kind];assert len(rr)==45
        keys=('forced_native_rows','native_unknown_angles','native_unknown_palms','native_unknown_arms','old_forced_native_rows_only_mapped_to_excluded_reference','new_forced_native_rows_only_mapped_to_excluded_reference')
        missing_summary[kind]=dict(n=len(rr),path_changed=sum(r['path_changed'] for r in rr),
            counts={k:sum(r[k] for r in rr) for k in keys},
            all_reference_domain_unchanged=all(r['expected_unchanged'] for r in rr),
            all_native_expected_not_reduced=all(r['native_expected_not_reduced'] for r in rr),
            all_forced_native_unknown_preserved=all(r['native_forced_unknown_preserved'] for r in rr),
            mean_support={k:float(np.mean([r[k] for r in rr])) for k in ('old_position_support','new_position_support','old_direction_support','new_direction_support')},
            groups={g:{condition:{k:sum(r[condition][g][k] for r in rr) for k in primary} for condition in ('old','new')} for g in groups})

    examples=dict(improvement=min(known_examples,key=lambda x:(x['delta_new_minus_old'],x['name'])),
                  regression=max(known_examples,key=lambda x:(x['delta_new_minus_old'],x['name'])))
    misses=[x for x in omissions['cross']['skip3']['members'] if x['new']['inside_raw']==0]
    examples['missed_omission']=sorted(misses,key=lambda x:(-x['old']['inside_raw'],x['query'],x['kind']))[0] if misses else None
    # Figures are explanatory output only; examples never feed back into conditions.
    (U/'visuals').mkdir();plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False})
    normal=[('identity','원래'),('fast075','0.75 길이'),('slow15','1.5 길이'),('pace','속도 변화'),('pause','정지 추가')]
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained');x=np.arange(len(normal))
    for cond,shift,color,label in [('old',-.18,'#b88b5d','기존 정렬'),('new',.18,'#3b8792','위치·방향 보조')]:
        vals=[known[k]['timing']['moving'][cond]['mae'] for k,_ in normal]
        axs[0].bar(x+shift,vals,.36,color=color,label=label)
    axs[0].set(xticks=x,xticklabels=[label for _,label in normal],ylabel='관측 변화 시점의 평균 오차 (원본 프레임)',title='같은 원본 · 실제 대응 시점을 아는 시험');axs[0].legend(fontsize=9)
    b=np.array([p['old_moving_mae'] for p in known_examples]);n=np.array([p['new_moving_mae'] for p in known_examples]);mx=max(float(b.max()),float(n.max()),.01)
    axs[1].scatter(b,n,c=np.where(n<b,'#3b8792',np.where(n>b,'#b66662','#8b949f')),s=25,alpha=.65)
    axs[1].plot([0,mx],[0,mx],color='#999999',ls='--');axs[1].set(xlabel='기존 평균 오차',ylabel='보조 적용 평균 오차',title='정상 시간변형 108개 · 대각선 아래가 개선')
    fig.savefig(U/'visuals'/'timing.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained');x=np.arange(4);gnames=['각도','손바닥','팔꿈치','팔 손목']
    for cond,shift,color,label in [('old',-.18,'#b88b5d','기존'),('new',.18,'#3b8792','보조')]:
        vals=[cross['identity']['groups'][g][cond]['warning'] for g in groups]
        axs[0].bar(x+shift,vals,.36,color=color,label=label)
        for i,val in enumerate(vals):axs[0].text(i+shift,val+2,str(val),ha='center',fontsize=9)
    axs[0].set(xticks=x,xticklabels=gnames,ylim=(0,250),ylabel='지속 차이가 있는 비교 / 216',title='다른 사람 · 원래 속도');axs[0].legend(fontsize=9)
    vals={key:[cross['identity']['groups'][g]['comparison'][key] for g in groups] for key in ('resolved','added','old_comparable_now_unknown')}
    for key,shift,color,label in [('resolved',-.25,'#3b8792','공통 관측 · 차이 감소'),('added',0,'#b66662','공통 관측 · 새 차이'),('old_comparable_now_unknown',.25,'#9a9fa9','비교 가능 → 근거 부족')]:axs[1].bar(x+shift,vals[key],.25,color=color,label=label)
    axs[1].set(xticks=x,xticklabels=gnames,ylabel='비교 칸 수',title='경고 변화와 관측 손실을 구분');axs[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=1)
    fig.savefig(U/'visuals'/'cross.png',dpi=145);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,size,title in [(axs[0],'skip3','3프레임 제거'),(axs[1],'skip20pct','동작 구간 약 20% 제거')]:
        vals={cond:[omissions['known'][size][cond]['variants_any_raw'],omissions['cross'][size][cond]['variants_any_raw'],omissions['cross'][size][cond]['variants_five_raw']] for cond in ('old','new')};x=np.arange(3)
        for cond,shift,color,label in [('old',-.18,'#b88b5d','기존'),('new',.18,'#3b8792','보조')]:
            ax.bar(x+shift,vals[cond],.36,color=color,label=label)
            for i,val in enumerate(vals[cond]):ax.text(i+shift,val+1.5,str(val),ha='center',fontsize=9)
        ax.set(xticks=x,xticklabels=['같은 원본\n새 차이','다른 사람\n1참고 이상 새 차이','다른 사람\n5참고 이상 새 차이'],ylim=(0,95),ylabel='81개 제거 변형 중',title=title);ax.legend(fontsize=9)
    fig.savefig(U/'visuals'/'omissions.png',dpi=145);plt.close(fig)
    for typ in ('improvement','regression'):
        ex=examples[typ];z=d.load(U/'known'/(ex['name']+'.npz'));ix=z['source_index'];pred={cond:np.array([np.median(z['base_path' if cond=='old' else 'path'][z['base_path' if cond=='old' else 'path'][:,0]==i,1]) for i in range(len(ix))]) for cond in ('old','new')}
        fig,axs=plt.subplots(2,1,figsize=(11,7),layout='constrained');tt=np.arange(len(ix))/30
        axs[0].plot(tt,ix,c='#668b63',lw=2,label='생성 과정에서 아는 실제 원본 시점')
        axs[0].plot(tt,pred['old'],c='#b88b5d',label='기존 정렬');axs[0].plot(tt,pred['new'],c='#3b8792',label='위치·방향 보조')
        axs[0].set(ylabel='원본 프레임',title=f'{ex["label"]} · {ex["query"].split("_")[-2]} · {ex["kind"]} · 오차 변화 {ex["delta_new_minus_old"]:+.3f}프레임');axs[0].legend(fontsize=9)
        axs[1].plot(tt,z['old_error'],c='#b88b5d',label='기존 오차');axs[1].plot(tt,z['new_error'],c='#3b8792',label='보조 오차')
        axs[1].fill_between(tt,0,1,where=z['moving'],color='#dbe6ed',alpha=.5,transform=axs[1].get_xaxis_transform(),label='원본에서 변화가 관측된 구간')
        axs[1].set(xlabel='시험 영상 시간 (초)',ylabel='절대 시간 오차 (원본 프레임)');axs[1].legend(fontsize=9)
        fig.savefig(U/'visuals'/(typ+'.png'),dpi=145);plt.close(fig)
    ex=examples['missed_omission']
    if ex is not None:
        rr=sorted([r for r in omission_pairs['cross'] if r['query']==ex['query'] and r['kind']==ex['kind']],key=lambda r:(-r['old']['inside_raw_cells'],r['reference']))
        row=rr[0];z=d.load(U/'cross'/(row['name']+'.npz'));orig=identity_cache['cross'][row['query'],row['reference']]
        def to_source(path,ix):return np.array([np.median(ix[path[path[:,1]==j,0]]) for j in range(len(z['expected']))])
        fig,axs=plt.subplots(2,1,figsize=(11,7),layout='constrained');x=np.arange(len(z['expected']))
        axs[0].plot(x,to_source(orig['path'],orig['source_index']),c='#668b63',label='제거 전 · 보조 정렬')
        axs[0].plot(x,to_source(z['base_path'],z['source_index']),c='#b88b5d',label='제거 후 · 기존 정렬')
        axs[0].plot(x,to_source(z['path'],z['source_index']),c='#3b8792',label='제거 후 · 보조 정렬')
        deleted=row['removed_source_frames'];axs[0].axhspan(min(deleted)-.5,max(deleted)+.5,color='#bd6260',alpha=.15,label='제거한 원본 시간대')
        axs[0].set(ylabel='연결된 시험 원본 프레임',title=f'새 차이를 못 찾은 짧은 제거 예 · {lookup[row["query"]]["label"]} · {ex["kind"]}');axs[0].legend(fontsize=8)
        axs[1].plot(x,orig['mismatch'].sum((1,2)),c='#668b63',label='제거 전 · 범위 밖 칸')
        axs[1].plot(x,z['mismatch'].sum((1,2)),c='#3b8792',label='제거 후 · 범위 밖 칸')
        axs[1].plot(x,z['unknown'].sum((1,2)),c='#949ba6',ls='--',label='제거 후 · 근거 부족 칸')
        axs[1].set(xlabel='참고 영상 프레임 (사람 간 실제 정답 시점은 아님)',ylabel='각도·위치 비교 칸 수');axs[1].legend(fontsize=9)
        fig.savefig(U/'visuals'/'missed_omission.png',dpi=145);plt.close(fig)
        examples['missed_omission_pair']=row
    d.write(U/'ANALYSIS.json',dict(at=d.now(),known=known,cross=cross,omissions=omissions,omission_pairs=omission_pairs,
        missing=missing_summary,missing_rows=missing,examples=examples,
        limits=['Timing truth exists only for same-source synthetic transformations, not linguistic phase correctness.',
        'Different-person original-path deletion localization is a fixed proxy; essential sign phases have no semantic labels.',
        'Different-person range templates are unchanged; reduced warnings cannot be explained by rebuilding a wider template.',
        'Source native missing timeline is independent of the selected path and reference rest exclusions.',
        'No whole-action correct verdict, classification inference, training, or user footage access.']))
    print('ANALYSIS COMPLETE',flush=True)
    print('KNOWN_NORMAL', {k:{'old':known[k]['timing']['moving']['old']['mae'],'new':known[k]['timing']['moving']['new']['mae'],'improved':known[k]['improved'],'worsened':known[k]['worsened']} for k,_ in normal},flush=True)
    print('OMISSIONS',{folder:{size:{cond:x[cond] for cond in ('old','new')} for size,x in ss.items()} for folder,ss in omissions.items()},flush=True)

if __name__=='__main__':run()
