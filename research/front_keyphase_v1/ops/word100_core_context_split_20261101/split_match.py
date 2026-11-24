"""Word-independent core traces with separately anchored context evidence."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'word100_context_sequence_match_20261031'))
from context_match import frame_cost,windows,recover,window_dp,EPS,CONTEXT_SECONDS

def supported(groups):
    vals=[v for v in groups.values() if v['expected']]
    return bool(vals) and all(v['sufficient'] for v in vals)

def empty_evidence(r,ids,groups):
    out={}
    for g in groups:
        y,mask=r[g];expected=int(mask[ids].sum());possible=len(ids)*y.shape[1]
        out[g]=dict(expected=expected,compared=0,reference_possible=possible,reference_unobserved=possible-expected,path_missing_cells=None,unaligned_expected_cells=expected,coverage=0. if expected else None,mean_difference=None,peak_difference=None,sufficient=False,distinct_shared_query_frames=0,query_unobserved_frames=[],reason='no_complete_context_path')
    return out

def anchored_context(q,r,full_cost,core_span,query_span,qpts,rpts,groups,summary_fn):
    a,b=core_span;s,e=query_span;qt=np.asarray(qpts)/1e6;rt=np.asarray(rpts)/1e6
    scale=float((qt[e]-qt[s])/(rt[b]-rt[a])) if b>a and rt[b]>rt[a] else 1.
    ra=int(np.searchsorted(rt,rt[a]-CONTEXT_SECONDS,'left'));rb=min(len(rt)-1,int(np.searchsorted(rt,rt[b]+CONTEXT_SECONDS,'right'))-1)
    qa=int(np.searchsorted(qt,qt[s]-CONTEXT_SECONDS*scale,'left'));qb=min(len(qt)-1,int(np.searchsorted(qt,qt[e]+CONTEXT_SECONDS*scale,'right'))-1)
    out={};costs=[]
    for phase,qs,qe,rs,re in [('entry',qa,s-1,ra,a-1),('exit',e+1,qb,b+1,rb)]:
        refs=np.arange(rs,re+1);path=None;value=None
        if qe>=qs and re>=rs:
            cc=full_cost[:,rs:re+1];dp,_=window_dp(cc,qs,qe-qs+1)
            if np.isfinite(dp[-1,-1].min()):
                value=float(dp[-1,-1].min()/((qe-qs+1)+(re-rs+1)))
                path=recover(cc,qs,qe-qs+1);path[:,1]+=rs
        evidence=summary_fn(q,r,path,refs) if path is not None else empty_evidence(r,refs,groups)
        clipped=(rt[a]-CONTEXT_SECONDS<rt[0] or qt[s]-CONTEXT_SECONDS*scale<qt[0]) if phase=='entry' else (rt[b]+CONTEXT_SECONDS>rt[-1] or qt[e]+CONTEXT_SECONDS*scale>qt[-1])
        ok=path is not None and supported(evidence) and not clipped
        out[phase]=dict(query_range=[qs,qe] if qe>=qs else None,reference_range=[rs,re] if re>=rs else None,cost=value,path=path.tolist() if path is not None else [],groups=evidence,supported=bool(ok),clipped_at_video_edge=bool(clipped))
        if value is not None:costs.append(value)
    return dict(scale=scale,phases=out,complete=all(x['supported'] for x in out.values()),mean_cost=float(np.mean(costs)) if len(costs)==2 else None,query_context=[qa,qb],reference_context=[ra,rb],core_not_moved=True)

def intervals_disjoint(a,b):return a[1]<b[0] or b[1]<a[0]

def match(q,r,segments,qpts,rpts,old_context,groups,summary_fn):
    full=frame_cost(q,r,groups);anchors=[];previous={x['segment_id']:x for x in old_context['anchors']}
    for s in segments:
        if s['role']!='core_candidate':continue
        a,b=s['core'];cc=full[:,a:b+1];ranked=windows(cc);old=previous[s['segment_id']]
        if not ranked:
            anchors.append(dict(segment_id=s['segment_id'],reference_core=[a,b],state='no_candidate',selected=None,raw=None,core_ties=[],feedback_codes=['no_candidate'],old_context_core=old['best']['query_core'] if old.get('best') else None));continue
        ties=[x for x in ranked if x[0]<=ranked[0][0]+EPS]
        tie_records=[]
        for cost,start,length in ties:
            context=anchored_context(q,r,full,[a,b],[start,start+length-1],qpts,rpts,groups,summary_fn)
            tie_records.append(dict(core_cost=cost,query_core=[start,start+length-1],context=context))
        chosen=0;tie_resolved=False
        # Context is only allowed to choose from EXACT numeric core minima.
        if len(tie_records)>1 and all(x['context']['complete'] for x in tie_records):
            context_costs=[x['context']['mean_cost'] for x in tie_records];minimum=min(context_costs)
            context_ties=[i for i,v in enumerate(context_costs) if abs(v-minimum)<=EPS]
            if len(context_ties)==1:chosen=context_ties[0];tie_resolved=True
        def pack(record):
            start,end=record['query_core'];path=recover(cc,start,end-start+1);path[:,1]+=a
            evidence=summary_fn(q,r,path,np.arange(a,b+1))
            return dict(core_cost=record['core_cost'],query_core=record['query_core'],core_path=path.tolist(),groups=evidence,core_supported=supported(evidence),context=record['context'])
        raw=pack(tie_records[0]);selected=raw if chosen==0 else pack(tie_records[chosen])
        disjoint_tie=min(x['query_core'][1] for x in tie_records)<max(x['query_core'][0] for x in tie_records)
        ambiguous=disjoint_tie and not tie_resolved
        state='observation_insufficient' if not selected['core_supported'] else 'ambiguous_occurrence' if ambiguous else 'supported_core_candidate'
        old_span=old['best']['query_core'] if old.get('best') else None
        conflict=old_span is not None and intervals_disjoint(selected['query_core'],old_span)
        range_diff=old_span is not None and old_span!=selected['query_core']
        old_unsupported=old.get('state')!='supported_candidate'
        codes=[state]
        if not selected['context']['complete']:codes.append('context_incomplete')
        if conflict:codes.append('core_context_localization_disagreement')
        elif range_diff:codes.append('core_context_boundary_difference')
        if old_unsupported:codes.append('previous_context_unsupported')
        holds=[x for x in codes if x!='supported_core_candidate']
        anchors.append(dict(segment_id=s['segment_id'],reference_core=[a,b],raw=raw,selected=selected,core_ties=tie_records,selected_tie_index=chosen,context_tie_resolved=tie_resolved,disjoint_core_tie=disjoint_tie,state=state,old_context_core=old_span,old_context_state=old['state'],old_context_disjoint=conflict,old_context_span_equal=old_span==selected['query_core'],diagnostic_hold_reasons=holds,feedback_codes=codes,searched_windows=len(ranked),minimum_core_cost=float(ranked[0][0]),within_window_path_ties_enumerated=False))
    relations=[]
    for a,b in zip(anchors,anchors[1:]):
        if any(x['state'] in ('no_candidate','observation_insufficient') for x in [a,b]):state='insufficient'
        elif any(x['state']=='ambiguous_occurrence' for x in [a,b]):state='ambiguous'
        else:
            x,y=a['selected']['query_core'],b['selected']['query_core']
            state='forward_candidate' if x[1]<y[0] else 'reverse_candidate' if y[1]<x[0] else 'overlap'
        relations.append(dict(first=a['segment_id'],second=b['segment_id'],state=state,context_disagreement=a.get('old_context_disjoint',False) or b.get('old_context_disjoint',False),educational_verdict=None))
    return dict(anchors=anchors,relations=relations,query_frames=full.shape[0],reference_frames=full.shape[1],original_scope_preserved=True,forced_interstage_order=False,word_or_person_used_for_search=False,semantic_correctness=False)

def feedback(anchor):
    texts={'no_candidate':'정해 둔 비교 조건에서 대응 구간을 찾지 못했습니다.','observation_insufficient':'핵심 구간에서 필요한 관측이 부족해 자세한 비교를 보류합니다.','ambiguous_occurrence':'비슷한 동작 회차가 둘 이상 남아 어느 차례인지 확정하지 않습니다.','supported_core_candidate':'관측 근거가 있는 핵심 동작의 대응 후보를 찾았습니다.','context_incomplete':'앞이나 뒤의 관측 근거는 별도로 부족합니다. 핵심 구간의 근거는 남겨 둡니다.','core_context_localization_disagreement':'핵심 동작만 찾은 위치와 앞뒤를 합쳐 찾은 위치가 다릅니다. 주변 흐름 때문에 핵심 위치를 바꾸지 않았습니다.','core_context_boundary_difference':'두 비교에서 찾은 구간이 겹치지만 시작 또는 끝의 범위는 다릅니다. 경계 차이를 동작 오류로 확정하지 않습니다.','previous_context_unsupported':'이전 앞뒤 통합 검색의 관측 지원은 부족하거나 모호했습니다. 그 결과로 핵심 동작을 확정하지 않습니다.'}
    return dict(kind='research_evidence_only',messages=[texts[c] for c in anchor['feedback_codes']],sign_correct=None,sign_error=None,correction_instruction=None,source_guidance_is_target_not_query_observation=True)
