"""Saved result aggregation; no threshold selection or new model evaluation."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
U=Path(__file__).resolve().parent;B=U.parent/'word100_coherent_reference_sequence_20261022'
MODES={'point':'point_bad','combined':'prior_combined_bad','levels':'level_bad','changes':'change_bad','split':'split_any_bad'}
NORMAL={'identity','fast075','slow15','pace','pause'}
GROUPS={'angle':slice(0,15),'palm':slice(15,17),'elbow':slice(17,19),'wrist':slice(19,21)}
def read(p):return json.loads(p.read_text('utf-8'))
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def mask(z,m):return z['point_comparable'] if m=='point' else z['sequence_comparable']
def inc(z,identity,domain,m):
    common=domain&mask(z,m)&mask(identity,m)
    return {'common':int(common.sum()),'new':int((common&z[MODES[m]]&~identity[MODES[m]]).sum()),
            'resolved':int((common&~z[MODES[m]]&identity[MODES[m]]).sum())}

def main():
    assert read(U/'VERIFY.json')['status']=='PASS';r=read(U/'RESULT.json');prior=read(B/'ANALYSIS.json')
    ident={}
    for folder in ('known','cross'):
        for row in r[folder]:
            if row['kind']=='identity':ident[row['query'],row['reference']]=load(U/folder/(row['name']+'.npz'))
    rows={};groups={};cohorts={};variants={}
    for folder in ('known','cross','missing'):
        records=[]
        for meta in r[folder]:
            z=load(U/folder/(meta['name']+'.npz'));base=ident[meta['query'],meta['reference']];dom=z['expected']
            rec={k:meta[k] for k in ('name','query','reference','kind','label','side','removed') if k in meta}
            rec['incremental']={m:inc(z,base,dom,m) for m in MODES}
            rec['groups']=meta['groups']
            # Shared original/variant sidecar support for fair old-combined vs component comparison.
            common=dom&z['sequence_comparable']&base['sequence_comparable']
            rec['shared_incremental']={m:{'common':int(common.sum()),'new':int((common & z[key] & ~base[key]).sum())} for m,key in MODES.items()}
            rec['component_new']={g:[int((common[:,:,sl] & z['component_bad'][:,:,sl,i] & ~base['component_bad'][:,:,sl,i]).sum()) for i in range(5)] for g,sl in GROUPS.items()}
            rec['normal_extra_absolute']={m:int((dom & z['sequence_comparable'] & z[MODES[m]] & ~z['prior_combined_bad']).sum()) for m in ('levels','changes','split')}
            if meta['kind'].startswith('skip'):
                sourcecount=len(base['source_index']);removed=np.setdiff1d(np.arange(sourcecount),z['source_index'].astype(int))
                ids=np.unique(base['path'][np.isin(base['path'][:,0],removed),1]);local=np.zeros_like(dom);local[ids]=dom[ids]
                rec['inside']={m:inc(z,base,local,m) for m in MODES}
                rec['outside']={m:inc(z,base,dom&~local,m) for m in MODES}
            records.append(rec)
        rows[folder]=records
        cohorts[folder]={}
        for tag,rr in [('all',records),('normal',[x for x in records if x['kind'] in NORMAL]),('speed',[x for x in records if x['kind'] in NORMAL-{'identity'}])]:
            g={}
            for name in GROUPS:
                g[name]={k:sum(x['groups'][name][k] for x in rr) for k in ('expected','point_comparable','point_unknown','point_bad','point_within','sequence_comparable','sequence_unknown','level_bad','change_bad','split_any_bad','prior_combined_bad')}
                g[name]['states']={str(i):sum(x['groups'][name]['states'][str(i)] for x in rr) for i in (1,2,3,4)}
                g[name]['components_bad']=[sum(x['groups'][name]['components_bad'][i] for x in rr) for i in range(5)]
            co={'cases':len(rr),'groups':g}
            for metric in ('incremental','shared_incremental'):
                co[metric]={m:{k:sum(x[metric][m][k] for x in rr) for k in ('common','new')} for m in MODES}
            co['extra_absolute']={m:sum(x['normal_extra_absolute'][m] for x in rr) for m in ('levels','changes','split')}
            co['component_new']={g:[sum(x['component_new'][g][i] for x in rr) for i in range(5)] for g in GROUPS}
            cohorts[folder][tag]=co
        if folder!='missing':
            variants[folder]={}
            for kind,prefix in [('short','skip3_'),('long','skip20pct_')]:
                vv={}
                for rr in records:
                    if not rr['kind'].startswith(prefix):continue
                    key=rr['query']+'__'+rr['kind'];v=vv.setdefault(key,{'query':rr['query'],'kind':rr['kind'],'modes':{m:0 for m in MODES}})
                    for m in MODES:v['modes'][m]+=int(rr['inside'][m]['new']>0)
                assert len(vv)==81
                variants[folder][kind]={'variants':list(vv.values()),'counts':{m:{'one':sum(v['modes'][m]>=1 for v in vv.values()),'five':sum(v['modes'][m]>=5 for v in vv.values())} for m in MODES}}
    # Exact reconciliation of retained prior instantaneous metrics.
    for g,v in cohorts['cross']['normal']['groups'].items():
        p=prior['cohorts']['cross']['normal_all']['groups'][g]
        assert v['expected']==p['expected'] and v['point_comparable']==p['modes']['point']['comparable'] and v['point_bad']==p['modes']['point']['bad']
    review=load(U/'known'/'identity__NIA_SL_WORD0185_REAL03_F.npz');frame_ids=list(range(101,112))+[113]
    cross={}
    for g,sl in GROUPS.items():
        cross[g]={k:int(review[k][frame_ids,:,sl].sum()) for k in ('expected','point_comparable','point_bad','sequence_comparable','sequence_unknown','split_any_bad')}
    a={'status':'PASS','rows':rows,'cohorts':cohorts,'variants':variants,'crossing_self_12frames':cross,
       'limits':['3 words 9 people27 already developed TRAIN clips; no fresh independent cohort','Point preservation is contractual, not accuracy gain','Split flags are diagnostic and never fused into correct/wrong','No new time alignment, classifier training, user data or semantic ground truth','One/five references are descriptive summaries, not selected decision thresholds','Shared strict descriptor support can still withhold sequence information near normal rest or missing frames']}
    (U/'ANALYSIS.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':12})
    v=U/'visuals';v.mkdir(exist_ok=True)
    g=cohorts['cross']['normal']['groups'];names=list(g);x=np.arange(4)
    fig,ax=plt.subplots(figsize=(11,5));ax.bar(x-.2,[100*g[n]['point_comparable']/g[n]['expected'] for n in names],.4,label='유지한 순간 비교',color='#397798');ax.bar(x+.2,[100*g[n]['sequence_comparable']/g[n]['expected'] for n in names],.4,label='추가 연속 비교',color='#9279bb');ax.set_xticks(x,['손가락 각도','손 위치','팔꿈치','팔 손목']);ax.set_ylim(0,110);ax.set_ylabel('고정 비교 범위 중 근거 있는 비율(%)');ax.set_title('순간 비교를 그대로 남기고 연속 비교 가능 여부를 따로 표시');ax.legend();fig.tight_layout();fig.savefig(v/'coverage.png',dpi=140);plt.close(fig)
    modes=['point','combined','levels','changes','split'];labels=['기존 순간','이전 합친 기준','모습만','변화량만','분리 항목 중 하나']
    fig,axes=plt.subplots(1,2,figsize=(13,5));counts=variants['cross']['short']['counts']
    for ax,key,title in zip(axes,['one','five'],['8개 참고 중 1개 이상 반응','8개 참고 중 5개 이상 반응']):
        vals=[counts[m][key] for m in modes];ax.bar(np.arange(5),vals,color=['#718096','#9983b8','#348b95','#d5a44b','#526a95']);ax.set_xticks(np.arange(5),labels,rotation=15);ax.set_ylim(0,90);ax.set_title(title);ax.set_ylabel('짧은 삭제 81변형 중 개수')
        for i,y in enumerate(vals):ax.text(i,y+1,str(y),ha='center')
    fig.suptitle('짧은 구간 삭제에 생긴 새 차이 — 의미상 오류 정답률 아님');fig.tight_layout();fig.savefig(v/'short_response.png',dpi=140);plt.close(fig)
    co=cohorts['cross']['speed'];vals=[co['shared_incremental'][m]['new'] for m in modes]
    fig,ax=plt.subplots(figsize=(11,5));ax.bar(np.arange(5),vals,color=['#718096','#9983b8','#348b95','#d5a44b','#526a95']);ax.set_xticks(np.arange(5),labels);ax.set_title('속도만 바꿨을 때도 나타난 새 차이');ax.set_ylabel('같은 원본·변형 공통 근거에서 차이 칸 수')
    for i,y in enumerate(vals):ax.text(i,y+max(vals)*.02,f'{y:,}',ha='center')
    ax.set_ylim(0,max(vals)*1.2);fig.text(.5,.01,'분리 표시는 원인 설명을 돕지만, 추가 표시를 바로 동작 오류로 취급하면 안 됩니다.',ha='center');fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(v/'normal_speed.png',dpi=140);plt.close(fig)
    print('ANALYSIS',json.dumps({'short':variants['cross']['short']['counts'],'speed':co['shared_incremental'],'crossing':cross},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
