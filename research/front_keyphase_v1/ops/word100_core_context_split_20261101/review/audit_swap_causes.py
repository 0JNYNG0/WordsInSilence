"""Read frozen swap outputs; split costs on stored/known-index fixed paths only."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import numpy as np

U=Path(__file__).resolve().parents[1]
OLD=U.parent/'word100_context_sequence_match_20261031'
BASE=U.parent/'word100_relational_sequence_compare_20261031'
PH=U.parent/'word100_core_stage_evidence_20261031'/'PHASE_DATA.json'
pins={};checks=0
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):pins[str(p)]=sha(p);return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def arrays(p):
    pins[str(p)]=sha(p)
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
def check(cond):
    global checks
    checks+=1;assert cond

auth=read(OLD/'AUTHORIZATION.json');rows={r['alias']:r for r in auth['rows']}
phase={c['alias']:c for c in read(PH)['clips']}
evaluation=read(OLD/'CONTROL_EVALUATION.json')['controls']['swap_adjacent_core_blocks']
frozen=read(OLD/'FROZEN.json');manifest=read(OLD/'RESULT_MANIFEST.json');manifestpins={x['file']:x['sha256'] for x in manifest['results']}
relp=BASE/'relational.py';pins[str(relp)]=sha(relp);check(frozen['pins'][str(relp)]==pins[str(relp)])
spec=importlib.util.spec_from_file_location('swap_features_only',relp);rel=importlib.util.module_from_spec(spec);spec.loader.exec_module(rel)
def forbidden(*args,**kwargs):raise RuntimeError('New search/DP forbidden in audit')
rel.dtw=rel.compare=rel.free_order=forbidden

def cost_at(q,r,qi,ri):
    """Scalar frozen frame cost at one fixed pair, not an alignment search."""
    gs=[];missing=0;expected=0
    for g in rel.GROUPS:
        x,qm=q[g];y,rm=r[g];ids=np.flatnonzero(rm[ri])
        if not len(ids):continue
        vals=[]
        for k in ids:
            expected+=1
            if qm[qi,k]:vals.append(min(abs(float(x[qi,k])-float(y[ri,k])),1.))
            else:vals.append(1.);missing+=1
        gs.append(sum(vals)/len(vals))
    return (sum(gs)/len(gs) if gs else 0.),expected,missing

def decompose(q,r,path,core):
    pp=np.asarray(path,dtype=int);a,b=core
    weights=np.ones(len(pp),dtype=int);weights[0]=2
    for i in range(1,len(pp)):
        delta=tuple(pp[i]-pp[i-1]);check(delta in [(1,1),(1,0),(0,1)])
        if delta==(1,1):weights[i]=2
    check(int(weights.sum())==len(set(pp[:,0]))+len(set(pp[:,1])))
    values=np.asarray([cost_at(q,r,int(i),int(j))[0] for i,j in pp])
    parts={};total_weight=int(weights.sum())
    for name,which in [('entry',pp[:,1]<a),('core',(pp[:,1]>=a)&(pp[:,1]<=b)),('exit',pp[:,1]>b)]:
        ix=np.flatnonzero(which);w=weights[ix];vs=values[ix];den=int(w.sum())
        contribution=float((w*vs).sum())
        support={}
        refids=sorted(set(pp[ix,1].tolist())) if len(ix) else []
        for g in rel.GROUPS:
            x,qm=q[g];y,rm=r[g];expected=int(rm[refids].sum()) if refids else 0
            compared=np.zeros_like(rm);missing=0;qs=set()
            for i,j in pp[ix]:
                shared=qm[i]&rm[j];compared[j]|=shared;missing+=int((rm[j]&~qm[i]).sum())
                if shared.any():qs.add(int(i))
            count=int(compared[refids].sum()) if refids else 0
            refseen=int(rm[refids].any(1).sum()) if refids else 0
            support[g]={'expected':expected,'compared':count,'coverage':count/expected if expected else None,
              'path_missing_cells':missing,'distinct_shared_query_frames':len(qs),
              'sufficient':bool(expected and count/expected>=.75 and missing==0 and refseen>=2 and len(qs)>=2)}
        parts[name]={'path_pairs':len(ix),'query_frames':sorted(set(pp[ix,0].tolist())) if len(ix) else [],
          'reference_frames':refids,'objective_weight':den,'weighted_cost_sum':contribution,
          'weighted_phase_mean':contribution/den if den else None,'whole_objective_contribution':contribution/total_weight,
          'groups':support}
    return {'weighted_sum':float((weights*values).sum()),'objective_weight':total_weight,
      'whole_weighted_cost':float((weights*values).sum()/total_weight),'phases':parts}

cases=[];transition_counts=Counter();status_losses=Counter();detail=[]
for e in evaluation:
    a=e['alias'];r=rows[a];name=a+'_swap_adjacent_core_blocks'
    resultpath=OLD/'results'/f'{name}.json';new=read(resultpath)
    check(pins[str(resultpath)]==manifestpins[str(resultpath)])
    baselinepath=BASE/'controls'/f'{name}.json';old=read(baselinepath)
    check(pins[str(baselinepath)]==new['baseline_sha256'])
    original=arrays(r['input']);query=arrays(BASE/'controls'/f'{name}.npz')
    check(pins[r['input']]==r['input_sha256'])
    check(pins[str(BASE/'controls'/f'{name}.npz')]==old['array_sha256']==new['query_array_sha256'])
    indices=np.asarray(old['metadata']['original_indices'],dtype=int)
    check(len(indices)==len(original['x']) and sorted(indices.tolist())==list(range(len(original['x']))))
    check(np.array_equal(query['x'],original['x'][indices]) and np.array_equal(query['mask'],original['mask'][indices]))
    q=rel.features(query);ref=rel.features(original)
    oldmap={z['segment_id']:z for z in old['free_order']['anchors']};newmap={z['segment_id']:z for z in new['anchors']}
    emap={z['segment_id']:z for z in e['stages']}
    for s in [s for s in phase[a]['segments'] if s['role']=='core_candidate']:
        sid=s['segment_id'];lo,hi=s['core'];expected=set(range(lo,hi+1));olda=oldmap[sid];newa=newmap[sid];best=newa['best']
        trueq=np.flatnonzero(np.isin(indices,list(expected)))
        check(len(trueq)==len(expected) and np.array_equal(trueq,np.arange(trueq[0],trueq[-1]+1)))
        check(np.array_equal(indices[trueq],np.arange(lo,hi+1)))
        oldorig=int(indices[olda['query_center']]);oldhit=oldorig in expected
        newcenter=(sum(best['query_core'])//2) if best else None
        neworig=int(indices[newcenter]) if newcenter is not None else None;newhit=neworig in expected
        check(oldhit==emap[sid]['old_center_in_original_core'] and newhit==emap[sid]['center_in_original_core'])
        transition=('hit' if oldhit else 'miss')+'->'+('hit' if newhit else 'miss');transition_counts[transition]+=1
        check(emap[sid]['state']==newa['state'])
        row={'alias':a,'role':r['role'],'word_id':r['word_id'],'segment_id':sid,'source_core_original_frames':[lo,hi],
          'source_description_ko':s['description_ko'],'true_query_core_after_swap':[int(trueq[0]),int(trueq[-1])],
          'true_query_to_original_core':[[int(i),int(indices[i])] for i in trueq],
          'old_mean_query_window':olda['query_window'],'old_mean_center_query':olda['query_center'],
          'old_mean_center_original':oldorig,'old_mean_cost':olda['cost'],'old_mean_supported':olda['supported'],
          'old_center_hit':oldhit,'context_query_window':best['query_window'] if best else None,
          'context_query_core':best['query_core'] if best else None,'context_center_query':newcenter,
          'context_center_original':neworig,'context_center_hit':newhit,'context_state':newa['state'],
          'transition':transition,'known_frame_identity_only':True}
        # Every known transformed core is an unchanged original block. Evaluate fixed corresponding pairs only.
        true_core_path=np.column_stack([trueq,np.arange(lo,hi+1)])
        true_core=decompose(q,ref,true_core_path,[lo,hi])
        row['true_mapped_core_only_fixed_path_cost']=true_core['whole_weighted_cost']
        tg=true_core['phases']['core']['groups']
        row['true_mapped_core_support']={'expected_cells':sum(v['expected'] for v in tg.values()),
          'supported_groups':[g for g,v in tg.items() if v['expected'] and v['sufficient']],
          'unsupported_expected_groups':[g for g,v in tg.items() if v['expected'] and not v['sufficient']],
          'reference_unavailable_groups':[g for g,v in tg.items() if not v['expected']],
          'numeric_zero_alone_does_not_establish_support':True}
        check(true_core['whole_weighted_cost']==0.)
        if transition!='hit->hit' and best is not None:
            stored=decompose(q,ref,best['path'],[lo,hi])
            check(np.isclose(stored['whole_weighted_cost'],best['cost'],atol=1e-12,rtol=0))
            for phase_name,phase_stats in stored['phases'].items():
                if best['phases'][phase_name] is None:continue
                for g,calc in phase_stats['groups'].items():
                    prev=best['phases'][phase_name][g]
                    for k,v in calc.items():
                        check((v is None and prev[k] is None) or isinstance(v,float) and np.isclose(v,prev[k],atol=1e-12,rtol=0) or v==prev[k])
            ra,rb=newa['reference_context'];offset=int(trueq[0])-lo
            direct=np.column_stack([np.arange(ra,rb+1)+offset,np.arange(ra,rb+1)])
            available=bool(direct[:,0].min()>=0 and direct[:,0].max()<len(indices))
            known=decompose(q,ref,direct,[lo,hi]) if available else None
            row.update(context_chosen_path_decomposition=stored,true_core_direct_support=true_core['phases']['core']['groups'],
              true_anchored_context_fixed_diagonal_path=direct.tolist() if available else None,
              true_anchored_context_fixed_diagonal_decomposition=known,
              true_anchored_path_is_new_optimized_search=False,
              context_chosen_core_original_indices=indices[best['query_core'][0]:best['query_core'][1]+1].tolist(),
              fixed_window_changed_context_reference_frames=[int(j) for i,j in direct if indices[i]!=j] if available else None)
            if available:
                check(known['phases']['core']['weighted_phase_mean']==0.)
                check(best['cost']<=known['whole_weighted_cost']+1e-12)
                row['selected_whole_cost_advantage_vs_feasible_true_anchored_fixed_path']=known['whole_weighted_cost']-best['cost']
            detail.append(row)
        if transition=='hit->miss':status_losses[newa['state']]+=1
        cases.append(row)

check(len(evaluation)==14 and len(cases)==43)
check(sum(c['old_center_hit'] for c in cases)==40 and sum(c['context_center_hit'] for c in cases)==34)
check(transition_counts=={'hit->hit':33,'hit->miss':7,'miss->miss':2,'miss->hit':1})
check(all(sha(p)==h for p,h in pins.items()))
lost=[c for c in detail if c['transition']=='hit->miss']
out={'status':'PASS_SAVED_SWAP_CAUSE_ARITHMETIC','author':'Codex agent A; same-family arithmetic audit, not human KSL correctness labels',
 'scope':{'public_control_cases':14,'fixed_core_denominator':43,'new_searches':0,'new_DP_calls':0,'new_model_calls':0,'old_files_written':0,
   'saved_paths_reused':True,'known_original_index_fixed_paths_evaluated':True,'thresholds_changed':False},
 'counts':{'old_mean_center_hits':40,'context_center_hits':34,'transition_counts':dict(transition_counts),
   'lost_center_support_states':dict(status_losses),'lost_while_supported':sum(c['context_state']=='supported_candidate' for c in lost),
   'lost_while_observation_insufficient':sum(c['context_state']=='observation_insufficient' for c in lost)},
 'all_43_stage_correspondences':cases,'lost_or_previously_missed_detailed_cases':detail,
 'findings':[
   '40->34 is net six, composed of seven lost centers and one recovered center;33 retained hits and2 persistent misses preserve the43stage denominator.',
   'Four newly lost centers remain supported_candidate; three are observation_insufficient. Mislocalization is not explained solely by withholding.',
   'All swapped original core blocks and masks are bitwise unchanged. Every true mapped core fixed path has numeric cost0. This is a synthetic index fact, not a claim that the action is linguistically correct.',
   'A zero numeric cost is not support: unavailable reference cells/groups are excluded and fully unavailable rows cost0. Each true core also retains separate expected/support/unavailable counts.',
   'The core+context objective can select a different core location because neighboring entry/exit rows around the known moved block no longer match the original context. Saved chosen paths are decomposed into core/entry/exit costs below.',
   'A feasible same-length fixed diagonal path anchored on the true core has zero core cost and altered-context cost; it is compared with the stored selected objective without new DP or candidate search. It is not an optimized contextual cost at that location.',
   'Positive chosen core cost with support can coexist with a lower whole context objective. Core correspondence, observation support and context mismatch therefore need separate outputs.',
   'The published reverse count14->2 also reflects context-changed synthetic block permutations; it must not be called better semantic-order detection.'
 ],
 'recommended_generic_split':[
   'Locate core trajectory from its own observed features and preserve several competing occurrence candidates; report entry/exit fit separately rather than letting context silently replace the core identity.',
   'Record core support, location ambiguity and context support/difference independently. A supported numerical match need not have correct original-index correspondence.',
   'Keep unknown crossing/rest/low-hand frames in original scope; do not fill hidden hands or use a body-wrist estimate as handshape truth.',
   'Use known transform maps only for evaluating synthetic correspondence, never as an input to real candidate selection or as semantic mandatory-order labels.',
   'Retain the same43stages and all14cases in comparison. Report raw correspondence and supported correspondence separately, including the adverse results.'
 ],'checks':checks,'pins':pins,'all_read_artifacts_unchanged':True,
 'limits':['No new method was evaluated; only old outputs and two explicitly fixed-path arithmetic views were used.',
   '2D saved numerical features and synthetic reordered blocks do not establish actual KSL errors, necessary stages, contact or acceptable human variation.',
   'The known true core mapping is evaluation-only privileged information. Its zero cost cannot be promised for another person or real video.',
   'The fixed true-context diagonal path is one feasible witness, not the optimum over paths at that position; selected-path and witness costs do not have identical alignments.']}
(U/'review'/'SWAP_CAUSE_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['블록 교환 통제: 핵심 대응과 앞뒤 문맥 회귀 원인',
 '저장된14개 통제,43개 핵심 후보를 그대로 분모로 사용했다. 새 검색·DP·모델 호출은 없다.',
 '40→34는 유지33, 상실7, 회복1, 계속 실패2의 합이다. 상실7중4는 지원된 후보인데 위치가 틀리고3은 관측 부족이다.',
 '교환한 핵심 블록 자체의 숫자와 mask는 원본 그대로다. 실제 이동된 위치를 알고 고정 대응시키면43개 핵심 모두 수치비용0이다. 수어의 정오를 뜻하지는 않는다.',
 '비용0만으로 관측이 충분하다는 뜻도 아니다. 참고에서 보이지 않은 값은 비용에서 제외되므로 각 핵심의 관측 지원을 별도로 남겼다.',
 '반면 블록 앞뒤 이웃은 바뀐다. 이전 방법은 핵심과 앞뒤를 한 비용으로 고르므로, 핵심 자체보다 문맥이 비슷한 다른 위치를 선택할 수 있다.',
 '아래의 정답 위치 문맥 비용은 원본index로 고정한 대각 경로1개의 비용이다. 새DP로 찾은 최소 비용은 아니다.', '', '새로 중심 대응을 잃은7개:']
for c in lost:
    chosen=c['context_chosen_path_decomposition'];true=c['true_anchored_context_fixed_diagonal_decomposition']
    lines.append(f"{c['segment_id']} 원본core{c['source_core_original_frames']} / 실제교환위치{c['true_query_core_after_swap']} / 옛중심원본{c['old_mean_center_original']}→새중심원본{c['context_center_original']} / {c['context_state']}")
    lines.append(f"  실제핵심 고정비용0, 선택경로 핵심평균{chosen['phases']['core']['weighted_phase_mean']:.6f}, 전체{chosen['whole_weighted_cost']:.6f}; 실제핵심에 고정한 문맥포함 전체비용{true['whole_weighted_cost']:.6f}." if true else '  실제핵심 고정비용0; 비디오경계로 문맥 고정창 미구성.')
lines += ['', '후속 비교는 핵심의 위치·지원과 앞뒤 문맥의 지원·차이를 따로 내야 한다. 문맥을 무조건 무시하거나 기대 순서를 강제하자는 뜻이 아니다.',
 '숫자 블록 교환은 실제 사람의 잘못된 수어 영상이 아니다. 정상 교차·휴식·가림·미관측을 오류로 바꾸지 않고 전체 원본 범위를 남긴다.']
(U/'review'/'SWAP_CAUSE_AUDIT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'checks':checks,'counts':out['counts'],'lost':[
 {'segment':c['segment_id'],'state':c['context_state'],'chosen_core_cost':c['context_chosen_path_decomposition']['phases']['core']['weighted_phase_mean'],
  'chosen_whole_cost':c['context_chosen_path_decomposition']['whole_weighted_cost'],
  'true_anchored_fixed_context_cost':c['true_anchored_context_fixed_diagonal_decomposition']['whole_weighted_cost'] if c['true_anchored_context_fixed_diagonal_decomposition'] else None} for c in lost]},ensure_ascii=False))
