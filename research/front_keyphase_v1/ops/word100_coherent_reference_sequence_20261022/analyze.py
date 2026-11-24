"""Aggregate verified saved sidecars only. No fitting, inference, or new alignment."""
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

U=Path(__file__).resolve().parent
I=U.parent/'word100_cross_person_motion_support_20261022'/'inputs'
MODES=('point','levels','sequence')
GROUPS={'angle':slice(0,15),'palm':slice(15,17),'elbow':slice(17,19),'wrist':slice(19,21)}
PAIRS=(('point','levels'),('point','sequence'),('levels','sequence'))
NORMAL=('identity','fast075','slow15','pace','pause')
WHY=('context_unavailable','anchor_incomplete','query_incomplete','peer_insufficient','query_span_bad','query_rest_bad')
TITLES={'point':'기존 시점별 비교','levels':'앞·중간·뒤 위치/각도','sequence':'앞·중간·뒤 + 변화량'}
GTITLES={'angle':'손가락 각도','palm':'손 위치','elbow':'팔꿈치','wrist':'팔 손목'}

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def n(x):return int(np.count_nonzero(x))
def div(a,b):return float(a/b) if b else None
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def masks(z,mode):
    return (z['point_comparable'],z['point_bad']) if mode=='point' else (z['comparable'],z[mode+'_bad'])
def sums(ds):
    out=Counter()
    for d in ds:
        for k,v in d.items():
            if isinstance(v,(int,np.integer)) and not isinstance(v,bool):out[k]+=int(v)
    return dict(out)
def stats(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    return {'n':len(a),'min':float(a.min()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'max':float(a.max()) if len(a) else None}

def basic(z):
    out={}
    for g,sl in GROUPS.items():
        d=z['expected'][:,:,sl]
        out[g]={'expected':n(d),'modes':{},'unknown_reasons':{k:n(z[k][:,:,sl]&d) for k in WHY}}
        for mode in MODES:
            c,b=masks(z,mode);c=c[:,:,sl]&d;b=b[:,:,sl]&c
            out[g]['modes'][mode]={'comparable':n(c),'unknown':n(d&~c),'bad':n(b),'within':n(c&~b),'coverage':div(n(c),n(d))}
        c=z['comparable'][:,:,sl]&d
        dl=z['levels_distance'][:,:,sl];ds=z['sequence_distance'][:,:,sl]
        tl=dl-z['levels_excess'][:,:,sl];ts=ds-z['sequence_excess'][:,:,sl]
        out[g]['sequence_vs_levels_numeric']={'comparable':n(c),
          'max_distance_unchanged':n(c&np.isclose(dl,ds,rtol=0,atol=1e-6)),
          'max_distance_increased':n(c&(ds>dl+1e-6)),
          'peer_bound_unchanged':n(c&np.isclose(tl,ts,rtol=0,atol=1e-6)),
          'peer_bound_increased':n(c&(ts>tl+1e-6)),
          'both_distance_and_bound_increased':n(c&(ds>dl+1e-6)&(ts>tl+1e-6)),
          'bad_state_unchanged':n(c&(z['levels_bad'][:,:,sl]==z['sequence_bad'][:,:,sl]))}
        out[g]['native_expected']=n(z['native_expected'][:,:,sl]);out[g]['native_unknown']=n(z['native_unknown'][:,:,sl])
    return out

def state_comparison(z,a,b,domain):
    ac,ab=masks(z,a);bc,bb=masks(z,b);c=domain&ac&bc
    return {'common':n(c),'a_bad':n(c&ab),'b_bad':n(c&bb),'removed_difference_common':n(c&ab&~bb),
      'added_difference_common':n(c&~ab&bb),'a_comparable_now_unknown':n(domain&ac&~bc),
      'a_bad_now_unknown':n(domain&ab&~bc),'newly_comparable':n(domain&~ac&bc)}

def incremental(z,identity,domain,mode):
    c,b=masks(z,mode);ic,ib=masks(identity,mode);common=domain&c&ic
    # Unknown identity is never treated as a valid normal baseline.
    return {'expected':n(domain),'common':n(common),'comparable':n(domain&c),
      'bad':n(domain&b),'original_bad':n(domain&ic&ib),'new_bad_common':n(common&b&~ib),
      'resolved_bad_common':n(common&~b&ib),'identity_comparable_now_unknown':n(domain&ic&~c),
      'original_bad_now_unknown':n(domain&ib&~c),'unknown':n(domain&~c),
      'raw_new_bad_without_identity_support_requirement':n(domain&b&~ib),
      'raw_new_bad_at_unknown_identity':n(domain&b&~ib&~ic)}

def incremental_between(z,identity,domain,a,b):
    ac,ab=masks(z,a);aic,aib=masks(identity,a);bc,bb=masks(z,b);bic,bib=masks(identity,b)
    common=domain&ac&aic&bc&bic
    anew=ab&~aib;bnew=bb&~bib
    return {'common_four':n(common),'a_new_bad_common_four':n(common&anew),'b_new_bad_common_four':n(common&bnew),
      'a_only_new_bad_common_four':n(common&anew&~bnew),'b_only_new_bad_common_four':n(common&~anew&bnew),
      'both_new_bad_common_four':n(common&anew&bnew),
      'a_pair_comparable_now_unknown':n(domain&ac&aic&~(bc&bic))}

def duration_metrics(z,source,reference):
    # Evaluation-only physical durations of the already fixed descriptors.
    # query_phase_index indexes the resampled query; source_index returns its original time.
    q=z['query_phase_index'];valid=np.isfinite(q).all(1)
    ix=z['source_index'];t=source['timestamp_us'].astype(float)/1000.
    sampled_t=np.interp(ix,np.arange(len(t)),t)
    qt=np.full_like(q,np.nan,dtype=float)
    for k in range(3):qt[valid,k]=np.interp(q[valid,k],np.arange(len(sampled_t)),sampled_t)
    rt=reference['timestamp_us'].astype(float)/1000.
    width=np.full(len(rt),np.nan);width[2:-2]=rt[4:]-rt[:-4]
    core=z['expected'].any((1,2));c=z['query_collapsed']&core
    return {'reference_window_ms':stats(width[core]),'mapped_original_source_equivalent_ms':stats((qt[:,2]-qt[:,0])[core]),
      'expected_frames':n(core),'collapsed_expected_frames':n(c),
      'collapsed_with_comparable_cells':n(c&z['comparable'].any((1,2))),
      'collapsed_reference_indices':np.flatnonzero(c).tolist(),
      'note':'Mapped durations return through source_index to original source timestamps; they are original-source-equivalent spans, not elapsed time of the synthetic speed-transformed query. Zero mapped phase gaps are diagnostics, not automatic repetition, omission, or error decisions.'}

def aggregate(rows):
    out={'rows':len(rows),'groups':{},'state_comparison':{}}
    for g in GROUPS:
        expected=sum(r['groups'][g]['expected'] for r in rows)
        out['groups'][g]={'expected':expected,'modes':{},'unknown_reasons':sums(r['groups'][g]['unknown_reasons'] for r in rows),
          'sequence_vs_levels_numeric':sums(r['groups'][g]['sequence_vs_levels_numeric'] for r in rows),
          'native_expected':sum(r['groups'][g]['native_expected'] for r in rows),'native_unknown':sum(r['groups'][g]['native_unknown'] for r in rows)}
        for mode in MODES:
            s=sums(r['groups'][g]['modes'][mode] for r in rows);s['coverage']=div(s.get('comparable',0),expected)
            s['rows_with_any_difference']=sum(r['groups'][g]['modes'][mode]['bad']>0 for r in rows)
            out['groups'][g]['modes'][mode]=s
    for a,b in PAIRS:
        key=a+'_to_'+b;out['state_comparison'][key]=sums(r['state_comparison'][key] for r in rows)
    if rows and 'whole' in rows[0]:
        out['incremental']={}
        for mode in MODES:
            s=sums(r['whole'][mode] for r in rows)
            s['rows_with_new_bad_common']=sum(r['whole'][mode]['new_bad_common']>0 for r in rows)
            s['rows_losing_identity_comparability']=sum(r['whole'][mode]['identity_comparable_now_unknown']>0 for r in rows)
            out['incremental'][mode]=s
        out['incremental_between']={a+'_to_'+b:sums(r['incremental_between'][a+'_to_'+b] for r in rows) for a,b in PAIRS}
    out['collapsed_expected_frames']=sum(r['durations']['collapsed_expected_frames'] for r in rows)
    out['rows_with_collapsed_expected_phase']=sum(r['durations']['collapsed_expected_frames']>0 for r in rows)
    out['physical_duration_row_medians_ms']={k:stats([r['durations'][k]['median'] for r in rows if r['durations'][k]['median'] is not None]) for k in ('reference_window_ms','mapped_original_source_equivalent_ms')}
    return out

def deletion_variants(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['query'],r['kind'])].append(r)
    out=[]
    for (cid,kind),rr in sorted(grouped.items()):
        v={'query':cid,'person':cid.split('_')[-2],'label':rr[0]['label'],'kind':kind,'references':len(rr),'modes':{},'between':{}}
        for mode in MODES:
            v['modes'][mode]={'references_new_local_common':sum(r['inside'][mode]['new_bad_common']>0 for r in rr),
              'references_raw_new_local':sum(r['inside'][mode]['raw_new_bad_without_identity_support_requirement']>0 for r in rr),
              'references_new_outside_common':sum(r['outside'][mode]['new_bad_common']>0 for r in rr),
              'references_with_local_unknown':sum(r['inside'][mode]['unknown']>0 for r in rr),
              'references_losing_local_comparability':sum(r['inside'][mode]['identity_comparable_now_unknown']>0 for r in rr),
              'inside':sums(r['inside'][mode] for r in rr),'outside':sums(r['outside'][mode] for r in rr)}
        for a,b in PAIRS:
            key=a+'_to_'+b
            s=sums(r['inside_between'][key] for r in rr)
            s['references_b_only_new_bad_common_four']=sum(r['inside_between'][key]['b_only_new_bad_common_four']>0 for r in rr)
            s['references_a_only_new_bad_common_four']=sum(r['inside_between'][key]['a_only_new_bad_common_four']>0 for r in rr)
            v['between'][key]=s
        out.append(v)
    summary={'variants':len(out),'references_per_variant':sorted(set(r['references'] for r in out)),'modes':{},'transitions':{}}
    for mode in MODES:
        summary['modes'][mode]={
          'any_reference_new_local_common':sum(r['modes'][mode]['references_new_local_common']>=1 for r in out),
          'at_least5_references_new_local_common':sum(r['modes'][mode]['references_new_local_common']>=5 for r in out),
          'no_reference_new_local_common':sum(r['modes'][mode]['references_new_local_common']==0 for r in out),
          'any_reference_raw_new_local':sum(r['modes'][mode]['references_raw_new_local']>=1 for r in out),
          'any_reference_new_outside_common':sum(r['modes'][mode]['references_new_outside_common']>=1 for r in out),
          'any_reference_losing_local_comparability':sum(r['modes'][mode]['references_losing_local_comparability']>=1 for r in out),
          'references_new_local_common':sum(r['modes'][mode]['references_new_local_common'] for r in out),
          'inside':sums(r['modes'][mode]['inside'] for r in out),'outside':sums(r['modes'][mode]['outside'] for r in out)}
    for a,b in PAIRS:
        key=a+'_to_'+b
        summary['transitions'][key]={}
        for threshold in (1,5):
            yes_a=[r['modes'][a]['references_new_local_common']>=threshold for r in out]
            yes_b=[r['modes'][b]['references_new_local_common']>=threshold for r in out]
            summary['transitions'][key][str(threshold)]={'both':sum(x and y for x,y in zip(yes_a,yes_b)),
              'a_only':sum(x and not y for x,y in zip(yes_a,yes_b)),'b_only':sum(y and not x for x,y in zip(yes_a,yes_b)),
              'neither':sum(not x and not y for x,y in zip(yes_a,yes_b)),
              'note':'Reference-count transitions can also reflect support changes; use common-four cell comparisons to separate them.'}
        summary['transitions'][key]['common_four']=sums(r['between'][key] for r in out)
    return {'summary':summary,'variants':out}

def crossing_records(result,inputs):
    cid='NIA_SL_WORD0185_REAL03_F';e=inputs[cid]
    src=np.flatnonzero(np.isin(e['frame_id'],list(range(101,112))+[113]))
    assert len(src)==12
    observed={'angle':n(e['angle_valid'][src]),'palm':n(e['palm_valid'][src]),'arm':n(e['arm_valid'][src])}
    assert observed=={'angle':0,'palm':0,'arm':48},observed
    rows=[]
    for folder in ('known','cross'):
        for m in result[folder]:
            if m['query']!=cid or m['kind']!='identity':continue
            z=load(U/folder/(m['name']+'.npz'));target=np.unique(z['path'][np.isin(z['path'][:,0],src),1]);domain=np.zeros_like(z['expected']);domain[target]=z['expected'][target]
            row={'folder':folder,'reference':m['reference'],'reference_indices':target.tolist(),'groups':{}}
            for g,sl in GROUPS.items():
                d=domain[:,:,sl];c=z['comparable'][:,:,sl]
                row['groups'][g]={'expected':n(d),'comparable':n(d&c),'unknown':n(d&~c),
                  'levels_within':n(d&z['levels_within'][:,:,sl]),'sequence_within':n(d&z['sequence_within'][:,:,sl])}
            # Native missing hand evidence must not turn into a within-range result.
            for g in ('angle','palm'):assert row['groups'][g]['comparable']==0,row
            rows.append(row)
    assert len(rows)==9
    return {'query':cid,'source_indices':src.tolist(),'frame_ids':e['frame_id'][src].tolist(),
      'native_supported':observed,'mapped_identity_cases':rows,
      'note':'Reviewed crossing has no supported hand-angle/palm evidence; it is preserved as unknown, not certified normal hand shape. Original arm evidence remains available, although strict neighborhood/calibration can reduce its sidecar comparability.'}

def plots(out):
    dest=U/'visuals';dest.mkdir()
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    colors={'point':'#64748b','levels':'#268b9b','sequence':'#7461bb'}
    def save(fig,name):
        fig.savefig(dest/name,dpi=160,bbox_inches='tight',pad_inches=.25);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(4);w=.24
    agg=out['cohorts']['cross']['normal_all']
    for i,mode in enumerate(MODES):
        vals=[100*(agg['groups'][g]['modes'][mode]['coverage'] or 0) for g in GROUPS]
        bars=ax.bar(x+(i-1)*w,vals,w,label=TITLES[mode],color=colors[mode])
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ax.set_xticks(x,list(GTITLES.values()));ax.set_ylim(0,110);ax.set_ylabel('정해 둔 비교 범위 중 비교 가능 비율 (%)')
    ax.set_title('정상 원본·속도 변형: 비교 범위를 유지한 채 근거가 남는 비율',pad=16)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,frameon=False);save(fig,'coverage.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    s=out['deletions']['cross']['short']['summary'];x=np.arange(3)
    for ax,key,title in zip(axes,('any_reference_new_local_common','at_least5_references_new_local_common'),('8개 참고 중 1개 이상 반응','8개 참고 중 5개 이상 반응')):
        v=[s['modes'][m][key] for m in MODES];bars=ax.bar(x,v,color=[colors[m] for m in MODES],width=.65);ax.bar_label(bars,padding=4)
        ax.set_xticks(x,['시점별','3시점','3시점+변화량']);ax.set_ylim(0,90);ax.set_title(title);ax.set_ylabel('짧은 삭제 81개 중 해당 개수')
    fig.suptitle('3프레임 삭제: 원본과 변형 모두 관측된 곳의 새로운 차이',fontsize=14,y=1.03)
    fig.text(.5,-.03,'이 수치는 삭제 시험 반응이며 수어 오류 판정 정확도가 아닙니다.',ha='center',fontsize=10,color='#475569');save(fig,'short_response.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5));a=out['cohorts']['cross']['speed_only'];x=np.arange(3)
    vals=[a['incremental'][m]['rows_with_new_bad_common'] for m in MODES]
    bars=axes[0].bar(x,vals,color=[colors[m] for m in MODES]);axes[0].bar_label(bars,padding=4)
    axes[0].set_ylabel('864개 비교 중 해당 개수');axes[0].set_title('속도만 바꿨을 때 새로운 차이가 생긴 비교')
    vals=[a['incremental'][m]['identity_comparable_now_unknown'] for m in MODES]
    bars=axes[1].bar(x,vals,color=[colors[m] for m in MODES]);axes[1].bar_label(bars,padding=4,fmt='%d')
    axes[1].set_ylabel('시점 × 손 × 각도/좌표 개수');axes[1].set_title('원본에서 가능했지만 비교가 보류된 정보')
    for ax in axes:ax.set_xticks(x,['시점별','3시점','3시점+변화량']);ax.margins(y=.18)
    fig.suptitle('경고 감소와 관측 손실을 따로 확인',fontsize=14,y=1.02);save(fig,'normal_tradeoff.png')
    fig,ax=plt.subplots(figsize=(10,5))
    keys=('context_unavailable','anchor_incomplete','query_incomplete','peer_insufficient')
    labels=('주변 시간대가\n고정 범위 밖','참고의 주변\n관측 부족','비교 영상의\n주변 관측 부족','다른 사람\n근거 5명 미만')
    totals=[sum(agg['groups'][g]['unknown_reasons'][k] for g in GROUPS) for k in keys]
    bars=ax.bar(np.arange(4),totals,color=['#8b9aad','#7b93b6','#6899a6','#957dab']);ax.bar_label(bars,padding=4,fmt='%d')
    ax.set_xticks(np.arange(4),labels);ax.set_ylabel('시점 × 손 × 각도/좌표 개수');ax.margins(y=.18)
    ax.set_title('연속 구간 비교를 보류하는 근거 — 항목끼리 겹칠 수 있음',pad=16);save(fig,'unknown_reasons.png')

def main():
    assert not (U/'RUNNING.json').exists(),'Study is still running'
    assert not (U/'ANALYSIS.json').exists(),'Do not overwrite saved analysis'
    verify=read(U/'VERIFY.json');assert verify['status']=='PASS',verify
    result=read(U/'RESULT.json');assert [len(result[k]) for k in ('known','cross','missing')]==[297,2376,135]
    frozen=read(U/'FROZEN.json');inputs={r['clip_id']:load(I/(r['clip_id']+'.npz')) for r in frozen['rows']}
    identities={}
    for folder in ('known','cross'):
        for m in result[folder]:
            if m['kind']=='identity':identities[m['query'],m['reference']]=load(U/folder/(m['name']+'.npz'))
    outputrows={};cohorts={};deletions={}
    for folder in ('known','cross','missing'):
        rr=[]
        for m in result[folder]:
            z=load(U/folder/(m['name']+'.npz'));identity=identities[m['query'],m['reference']];d=z['expected']
            assert np.array_equal(d,identity['expected'])
            r={k:m[k] for k in ('name','query','reference','kind','label','side','removed') if k in m}
            r['groups']=basic(z);r['state_comparison']={a+'_to_'+b:state_comparison(z,a,b,d) for a,b in PAIRS}
            r['whole']={mode:incremental(z,identity,d,mode) for mode in MODES}
            r['incremental_between']={a+'_to_'+b:incremental_between(z,identity,d,a,b) for a,b in PAIRS}
            r['durations']=duration_metrics(z,inputs[m['query']],inputs[m['reference']])
            if m['kind'].startswith('skip'):
                ix=z['source_index'];assert np.allclose(ix,np.rint(ix));count=len(inputs[m['query']]['core'])
                removed=np.setdiff1d(np.arange(count),np.rint(ix).astype(int));assert len(removed)>0
                oldpath=identity['path'];refids=np.unique(oldpath[np.isin(oldpath[:,0],removed),1])
                local=np.zeros_like(d);local[refids]=d[refids];outside=d&~local
                r['removed_source_indices']=removed.tolist();r['local_reference_indices']=refids.tolist()
                r['inside']={mode:incremental(z,identity,local,mode) for mode in MODES}
                r['outside']={mode:incremental(z,identity,outside,mode) for mode in MODES}
                r['inside_between']={a+'_to_'+b:incremental_between(z,identity,local,a,b) for a,b in PAIRS}
                r['inside_groups']={g:{mode:incremental(z,identity,local&np.broadcast_to(np.isin(np.arange(21),np.arange(21)[sl]),d.shape),mode) for mode in MODES} for g,sl in GROUPS.items()}
            rr.append(r)
        outputrows[folder]=rr
        cohorts[folder]={'all':aggregate(rr),'by_kind':{k:aggregate([r for r in rr if r['kind']==k]) for k in sorted({r['kind'] for r in rr})}}
        if folder!='missing':
            cohorts[folder]['normal_all']=aggregate([r for r in rr if r['kind'] in NORMAL]);cohorts[folder]['speed_only']=aggregate([r for r in rr if r['kind'] in NORMAL[1:]])
            deletions[folder]={name:deletion_variants([r for r in rr if r['kind'].startswith(prefix)]) for name,prefix in (('short','skip3_'),('long','skip20pct_'))}
            assert deletions[folder]['short']['summary']['variants']==81
    out={'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'status':'PASS','rows':outputrows,'cohorts':cohorts,'deletions':deletions,
      'crossing':crossing_records(result,inputs),'training':0,'new_dtw':0,'user_data_access':0,
      'interpretation':[
        'All findings describe saved public TRAIN sidecars; no new recognition score, semantic-error accuracy, or fresh independent evaluation is claimed.',
        'Point compares previous pooled interval hulls at individual times. Levels and sequence compare coherent whole descriptors of named people with separately calibrated bounds. A pooled hull may contain points supported by no individual member.',
        'Common new difference requires both original and variant comparable within the same condition. Raw new differences at unknown original cells are retained separately, never interpreted as regression from known normal.',
        'Between-condition incremental gains/regressions use four-way original/variant comparability. Row/reference response counts can additionally change with coverage.',
        'Expected reference range is fixed. Unknown, full-span loss, and insufficient peer evidence never become normal or correct; no whole-action correctness is emitted.',
        'All eight cross-person anchors remain represented. At least one/five reference counts are descriptive summaries, not tuned acceptance thresholds.',
        'Short omissions are synthetic deleted input frames; their semantic importance and cross-person phase truth are not established by this test.',
        'Unknown reasons overlap. Stronger full-span support can reduce comparisons; fewer warnings alone are not improvement.',
        'The time window is five input frames, not a certified sign phase. Collapsed sample indices can occur with normal pauses and do not automatically prove skipped motion.',
        'Native missing evidence remains in the original input timeline; descriptor unknown is additional. Angle reliability is not equivalent to visibility in raw RGB.',
        'Levels and sequence thresholds differ; adding deltas can enlarge the peer bound, so a change is not isolated proof that sequence understanding improved.'
        ,'Only levels versus sequence holds the new descriptor support and calibration recipe constant. Point versus new conditions changes temporal context, evidence support, representative sampling, and the reference-distance calibration together; it is not an isolated sequence ablation.'
      ]}
    plots(out);write(U/'ANALYSIS.json',out)
    print('ANALYSIS PASS',json.dumps({'short':{m:deletions['cross']['short']['summary']['modes'][m]['any_reference_new_local_common'] for m in MODES},'plots':4}),flush=True)

if __name__=='__main__':main()
