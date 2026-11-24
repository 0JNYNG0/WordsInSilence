"""Independent arithmetic on saved split paths; no search, DP or model execution."""
from pathlib import Path
from collections import Counter,defaultdict
from bisect import bisect_left,bisect_right
import datetime,hashlib,importlib.util,json,math
import numpy as np
U=Path(__file__).resolve().parents[1]
OLD=U.parent/'word100_context_sequence_match_20261031'
REL=U.parent/'word100_relational_sequence_compare_20261031'
PHASE=U.parent/'word100_core_stage_evidence_20261031'
ORACLE=REL/'review/verify_saved_relational.py'
spec=importlib.util.spec_from_file_location('independent_prior_review',ORACLE)
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
G=oracle.GROUPS;EPS=1e-9
counts=Counter();stats=Counter();failures=[];max_error=0.
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(name,actual,expected=True,category='arithmetic'):
    global max_error
    counts[category]+=1
    if isinstance(expected,(float,np.floating)):
        ok=isinstance(actual,(float,int)) and math.isfinite(actual) and math.isclose(actual,float(expected),rel_tol=3e-9,abs_tol=3e-11)
        if isinstance(actual,(float,int)) and math.isfinite(actual):max_error=max(max_error,abs(actual-float(expected)))
    else:ok=actual==expected
    if not ok:failures.append(dict(name=name,actual=actual,expected=expected,category=category))
def compare(name,actual,expected,category):
    if isinstance(expected,dict):
        ck(name+'/keys',set(actual)==set(expected),category=category)
        for k,v in expected.items():compare(name+'/'+k,actual.get(k),v,category)
    elif isinstance(expected,list):
        ck(name+'/length',len(actual),len(expected),category)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(name+'/'+str(i),a,b,category)
    else:ck(name,actual,expected,category)
def support(evidence):
    vals=[x for x in evidence.values() if x['expected']>0]
    return bool(vals) and all(v['sufficient'] for v in vals)
def cost(q,r,i,j):
    active=[]
    for g in G:
        x,xok=q[g];y,yok=r[g];dims=[k for k,ok in enumerate(yok[j]) if ok]
        if dims:active.append(math.fsum(min(abs(float(x[i,k])-float(y[j,k])),1.) if xok[i,k] else 1. for k in dims)/len(dims))
    return math.fsum(active)/len(active) if active else 0.
def path_check(label,path,q,r,qspan,rspan,stored_cost):
    stats['saved_paths']+=1
    p=np.asarray(path);n=len(q[G[0]][0]);m=len(r[G[0]][0]);qs,qe=qspan;rs,re=rspan
    ck(label+'/integer_path',bool(np.issubdtype(p.dtype,np.integer)),category='path')
    ck(label+'/shape',bool(p.ndim==2 and p.shape[1]==2 and len(p)>0),category='path')
    if p.ndim!=2 or p.shape[1]!=2 or not len(p):raise ValueError(label+' invalid path')
    p=p.astype(int);ck(label+'/bounds',bool(((p[:,0]>=0)&(p[:,0]<n)&(p[:,1]>=0)&(p[:,1]<m)).all()),category='path')
    ck(label+'/endpoints',[p[0].tolist(),p[-1].tolist()],[[qs,rs],[qe,re]],'path')
    ck(label+'/full_query_window',sorted(set(p[:,0].tolist())),list(range(qs,qe+1)),'path')
    ck(label+'/full_reference_part',sorted(set(p[:,1].tolist())),list(range(rs,re+1)),'path')
    weights=[2];prev=None;run=0;largest=0;legal=True
    for d in np.diff(p,axis=0):
        d=tuple(d);legal &= d in [(1,0),(0,1),(1,1)]
        if d==(1,1):weights.append(2);prev=None;run=0
        else:weights.append(1);run=run+1 if d==prev else 1;prev=d;largest=max(largest,run)
    ck(label+'/legal_steps',legal,category='path');ck(label+'/maximum_axis_run2',largest<=2,category='path')
    den=qe-qs+1+re-rs+1;ck(label+'/weight_denominator',sum(weights),den,'path')
    expected=math.fsum(w*cost(q,r,int(i),int(j)) for w,(i,j) in zip(weights,p))/den
    ck(label+'/scalar_cost',stored_cost,expected,'cost')
    return p
def empty_summary(r,ids):
    out={}
    for g in G:
        y,mask=r[g];expected=sum(bool(mask[j,k]) for j in ids for k in range(y.shape[1]));possible=len(ids)*y.shape[1]
        out[g]=dict(expected=expected,compared=0,reference_possible=possible,reference_unobserved=possible-expected,path_missing_cells=None,unaligned_expected_cells=expected,coverage=0. if expected else None,mean_difference=None,peak_difference=None,sufficient=False,distinct_shared_query_frames=0,query_unobserved_frames=[],reason='no_complete_context_path')
    return out
def context_check(label,c,q,r,core,query,qpts,rpts):
    stats['tied_candidate_contexts']+=1;a,b=core;s,e=query
    qt=(np.asarray(qpts)/1e6).tolist();rt=(np.asarray(rpts)/1e6).tolist()
    scale=(qt[e]-qt[s])/(rt[b]-rt[a]) if b>a and rt[b]>rt[a] else 1.
    ra=bisect_left(rt,rt[a]-.2);rb=min(len(rt)-1,bisect_right(rt,rt[b]+.2)-1)
    qa=bisect_left(qt,qt[s]-.2*scale);qb=min(len(qt)-1,bisect_right(qt,qt[e]+.2*scale)-1)
    ck(label+'/scale',c['scale'],scale,'context');ck(label+'/ref_context',c['reference_context'],[ra,rb],'context');ck(label+'/query_context',c['query_context'],[qa,qb],'context');ck(label+'/core_unchanged',c['core_not_moved'],True,'context')
    oks=[];phase_costs=[]
    for phase,qs,qe,rs,re in [('entry',qa,s-1,ra,a-1),('exit',e+1,qb,b+1,rb)]:
        v=c['phases'][phase];lab=label+'/'+phase;refs=list(range(rs,re+1))
        nq=qe-qs+1;nr=re-rs+1
        ck(lab+'/qrange',v['query_range'],[qs,qe] if nq>0 else None,'context');ck(lab+'/rrange',v['reference_range'],[rs,re] if nr>0 else None,'context')
        # With finite costs and max two axial moves, a complete grid path exists
        # iff both dimensions are positive and neither exceeds 3 times the other.
        feasible=nq>0 and nr>0 and nq<=3*nr and nr<=3*nq
        ck(lab+'/path_exists_dimension_bound',bool(v['path']),feasible,'path')
        if v['path']:
            p=path_check(lab,v['path'],q,r,[qs,qe],[rs,re],v['cost']);summary=oracle.summary(q,r,p,refs);phase_costs.append(v['cost'])
        else:
            summary=empty_summary(r,refs);ck(lab+'/no_path_cost_null',v['cost'],None,'cost');stats['no_path_context_parts']+=1
        compare(lab+'/groups',v['groups'],summary,'support')
        clipped=rt[a]-.2<rt[0] or qt[s]-.2*scale<qt[0] if phase=='entry' else rt[b]+.2>rt[-1] or qt[e]+.2*scale>qt[-1]
        ok=feasible and support(summary) and not clipped;oks.append(ok)
        ck(lab+'/clipped',v['clipped_at_video_edge'],clipped,'context');ck(lab+'/supported',v['supported'],ok,'context')
    ck(label+'/mean_phase_cost',c['mean_cost'],math.fsum(phase_costs)/2 if len(phase_costs)==2 else None,'cost')
    ck(label+'/complete',c['complete'],all(oks),'context')
    return all(oks)
def core_pack(label,p,q,r,core,record):
    ck(label+'/record_span',p['query_core'],record['query_core'],'tie')
    ck(label+'/record_cost',p['core_cost'],record['core_cost'],'tie')
    ck(label+'/record_context',p['context'],record['context'],'tie')
    path=path_check(label,p['core_path'],q,r,p['query_core'],core,p['core_cost'])
    evidence=oracle.summary(q,r,path,range(core[0],core[1]+1));compare(label+'/groups',p['groups'],evidence,'support')
    ok=support(evidence);ck(label+'/supported',p['core_supported'],ok,'support');return ok
def numeric(anchor,indices,key):
    lo,hi=anchor['reference_core'];expected=set(range(lo,hi+1));p=anchor[key];original=set(indices)
    d=dict(segment_id=anchor['segment_id'],state=anchor['state'],expected_original_frames=len(expected),present_unique_original_frames=len(expected&original),missing_original_frames=sorted(expected-original))
    if p is None:d.update(center_hit=False,center_original=None,recall=0.,precision=0.,path_mean_error=None)
    else:
        x,y=p['query_core'];got=set(indices[x:y+1]);center=indices[(x+y)//2];err=[abs(indices[i]-j) for i,j in p['core_path']]
        d.update(center_hit=lo<=center<=hi,center_original=center,recall=len(expected&got)/len(expected),precision=len(expected&got)/len(got),path_mean_error=math.fsum(err)/len(err))
    return d
def disjoint(x,y):return x[1]<y[0] or y[1]<x[0]
TEXT={'no_candidate':'정해 둔 비교 조건에서 대응 구간을 찾지 못했습니다.','observation_insufficient':'핵심 구간에서 필요한 관측이 부족해 자세한 비교를 보류합니다.','ambiguous_occurrence':'비슷한 동작 회차가 둘 이상 남아 어느 차례인지 확정하지 않습니다.','supported_core_candidate':'관측 근거가 있는 핵심 동작의 대응 후보를 찾았습니다.','context_incomplete':'앞이나 뒤의 관측 근거는 별도로 부족합니다. 핵심 구간의 근거는 남겨 둡니다.','core_context_localization_disagreement':'핵심 동작만 찾은 위치와 앞뒤를 합쳐 찾은 위치가 다릅니다. 주변 흐름 때문에 핵심 위치를 바꾸지 않았습니다.','core_context_boundary_difference':'두 비교에서 찾은 구간이 겹치지만 시작 또는 끝의 범위는 다릅니다. 경계 차이를 동작 오류로 확정하지 않습니다.','previous_context_unsupported':'이전 앞뒤 통합 검색의 관측 지원은 부족하거나 모호했습니다. 그 결과로 핵심 동작을 확정하지 않습니다.'}
def main():
    mf=read(U/'RESULT_MANIFEST.json');frozen=read(U/'FROZEN.json');progress=read(U/'STATUS.json')
    ck('manifest_complete',mf['status'],'COMPLETE_PENDING_VERIFICATION','provenance');ck('complete_progress',progress['done'],162,'provenance');ck('no_running_lock',not (U/'RUNNING.json').exists(),category='provenance')
    for p,h in frozen['pins'].items():ck('frozen_pin/'+p,sha(p),h,'provenance')
    stable_files=['FROZEN.json','RESULT_MANIFEST.json','SUMMARY.json','REPORT_DATA.json','CONTROL_EVALUATION.json','PAIR_EVALUATION.json']
    initial_hashes={n:sha(U/n) for n in stable_files};freezehash=initial_hashes['FROZEN.json']
    rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']};oldrows={r['alias']:r for r in read(REL/'AUTHORIZATION.json')['rows']};clips={x['alias']:x for x in read(PHASE/'PHASE_DATA.json')['clips']}
    raw={a:oracle.load_array(r['input']) for a,r in rows.items()};features={a:oracle.independent_features(raw[a],oldrows[a]['feature_names']) for a in rows}
    oldmanifest={x['name']:x for x in read(OLD/'RESULT_MANIFEST.json')['results']}
    ck('fixed_jobs',sorted(x['name'] for x in mf['results']),sorted(oldmanifest),'scope');ck('job_count',len(mf['results']),162,'scope');ck('kind_counts',dict(Counter(x['kind'] for x in mf['results'])),{'control':126,'person_pair':36},'scope')
    ck('no_unmanifested_results',sorted(p.stem for p in (U/'results').glob('*.json')),sorted(oldmanifest),'scope')
    for a,row in rows.items():
        ck(a+'/input_sha',sha(row['input']),row['input_sha256'],'provenance');ck(a+'/native_frames',raw[a]['frame_id'].tolist(),list(range(row['frames'])),'scope');ck(a+'/native_pts',bool((np.diff(raw[a]['timestamp_us'])>0).all()),category='scope')
    controls=defaultdict(list);pairs=[];allstates=Counter();holds=Counter();tiesizes=Counter();case_outputs=[]
    report=read(U/'REPORT_DATA.json');reportcases={x['id']:x for x in report['cases']}
    ck('report_case_ids',sorted(reportcases),sorted(oldmanifest),'report')
    for jobidx,item in enumerate(mf['results'],1):
        name=item['name'];d=read(item['file']);old=read(oldmanifest[name]['file']);a,b=d['query'],d['reference'];r=features[b]
        for label,actual,expected in [('result_hash',sha(item['file']),item['sha256']),('old_result',d['old_result'],oldmanifest[name]['file']),('old_hash',d['old_result_sha256'],oldmanifest[name]['sha256']),('actual_old_hash',sha(d['old_result']),d['old_result_sha256']),('frozen_hash',d['frozen_sha256'],freezehash),('query_hash',sha(d['query_array']),d['query_array_sha256'])]:ck(name+'/'+label,actual,expected,'provenance')
        for key in ['name','kind','query','reference']:ck(name+'/'+key,d[key],old[key],'scope')
        ck(name+'/role',d['role'],rows[a]['role'],'scope');ck(name+'/old_seconds',d['old_context_seconds_saved'],old['seconds'],'timing');ck(name+'/new_time_valid',math.isfinite(d['seconds_incremental_core_split']) and d['seconds_incremental_core_split']>=0,category='timing')
        if d['kind']=='control':
            previous=read(REL/'controls'/(name+'.json'));qraw=oracle.load_array(d['query_array']);q=oracle.independent_features(qraw,oldrows[a]['feature_names']);indices=previous['metadata'].get('original_indices',list(range(len(qraw['x']))));meananchors={x['segment_id']:x for x in previous['free_order']['anchors']}
            ck(name+'/control_array_sha',d['query_array_sha256'],previous['array_sha256'],'provenance');ck(name+'/self_reference',a,b,'scope');ck(name+'/control',d['control'],previous['control'],'scope')
        else:
            previous=None;qraw=raw[a];q=features[a];indices=list(range(len(qraw['x'])))
            ck(name+'/pair_query_path',str(Path(d['query_array'])),str(Path(rows[a]['input'])),'scope');ck(name+'/different_people',rows[a]['person']!=rows[b]['person'],category='scope');ck(name+'/reference_fit',rows[b]['role'],'fit','scope');ck(name+'/same_word',rows[a]['word_id'],rows[b]['word_id'],'scope')
        n=len(qraw['x']);m=len(raw[b]['x']);segments=[s for s in clips[b]['segments'] if s['role']=='core_candidate'];olds={x['segment_id']:x for x in old['anchors']}
        for key,value in [('query_frames',n),('reference_frames',m),('original_scope_preserved',True),('forced_interstage_order',False),('word_or_person_used_for_search',False),('semantic_correctness',False),('outside_is_not_normal_or_discarded',True)]:ck(name+'/'+key,d[key],value,'scope')
        ck(name+'/fixed_core_ids',[x['segment_id'] for x in d['anchors']],[x['segment_id'] for x in segments],'scope');ck(name+'/query_pts_monotonic',bool((np.diff(qraw['timestamp_us'])>0).all()),category='scope')
        stage_evidence=[];covered=set();expected_raw=[];expected_selected=[];derived_states=[];ui_anchors=[]
        for seg,anc in zip(segments,d['anchors']):
            sid=seg['segment_id'];label=name+'/'+sid;core=seg['core'];oldanc=olds[sid];stats['anchors']+=1;ck(label+'/reference_core',anc['reference_core'],core,'scope')
            nr=core[1]-core[0]+1;lo=max(2,math.ceil(nr*.5));hi=min(n,math.floor(nr*2));eligible=sum(n-length+1 for length in range(lo,hi+1));oldspan=oldanc['best']['query_core'] if oldanc.get('best') else None
            if anc['selected'] is None:
                ck(label+'/no_candidate_windows',eligible,0,'scope');ck(label+'/no_candidate_state',anc['state'],'no_candidate','state');ck(label+'/empty_ties',anc['core_ties'],[],'tie');codes=['no_candidate'];state='no_candidate';context_complete=False;core_ok=False;conflict=False
            else:
                ts=anc['core_ties'];tiesizes[len(ts)]+=1;ck(label+'/window_count',anc['searched_windows'],eligible,'scope');ck(label+'/some_ties',bool(ts),category='tie');ck(label+'/tie_count_bound',len(ts)<=eligible,category='tie')
                ranked=[];context_oks=[]
                for ti,t in enumerate(ts):
                    qs,qe=t['query_core'];length=qe-qs+1;ranked.append((t['core_cost'],qs,length))
                    ck(label+'/tie'+str(ti)+'/range',0<=qs<=qe<n and lo<=length<=hi,category='tie');ck(label+'/tie'+str(ti)+'/eps_minimum',math.isfinite(t['core_cost']) and anc['minimum_core_cost']<=t['core_cost']<=anc['minimum_core_cost']+EPS,category='tie')
                    context_oks.append(context_check(label+'/tie'+str(ti),t['context'],q,r,core,[qs,qe],qraw['timestamp_us'],raw[b]['timestamp_us']))
                ck(label+'/sorted_ties',ranked,sorted(ranked),'tie');ck(label+'/unique_windows',len({(s,l) for _,s,l in ranked}),len(ts),'tie');ck(label+'/raw_minimum',anc['minimum_core_cost'],ranked[0][0],'tie')
                chosen=0;resolved=False
                if len(ts)>1 and all(context_oks):
                    context_costs=[t['context']['mean_cost'] for t in ts];cm=min(context_costs);winners=[i for i,x in enumerate(context_costs) if abs(x-cm)<=EPS]
                    if len(winners)==1:chosen=winners[0];resolved=True
                ck(label+'/selected_index',anc['selected_tie_index'],chosen,'tie');ck(label+'/tie_resolved',anc['context_tie_resolved'],resolved,'tie')
                core_pack(label+'/raw',anc['raw'],q,r,core,ts[0]);core_ok=core_pack(label+'/selected',anc['selected'],q,r,core,ts[chosen])
                ck(label+'/unique_core_never_moves',len(ts)!=1 or anc['raw']==anc['selected'],category='tie')
                split=min(t['query_core'][1] for t in ts)<max(t['query_core'][0] for t in ts);ck(label+'/any_pair_disjoint',anc['disjoint_core_tie'],split,'tie');ck(label+'/within_window_limit',anc['within_window_path_ties_enumerated'],False,'tie')
                state='observation_insufficient' if not core_ok else 'ambiguous_occurrence' if split and not resolved else 'supported_core_candidate';ck(label+'/state',anc['state'],state,'state')
                selected=anc['selected'];span=selected['query_core'];context_complete=context_oks[chosen];covered.update(range(span[0],span[1]+1));conflict=oldspan is not None and disjoint(span,oldspan);different=oldspan is not None and span!=oldspan
                ck(label+'/old_span',anc['old_context_core'],oldspan,'diagnostic');ck(label+'/old_state',anc['old_context_state'],oldanc['state'],'diagnostic');ck(label+'/old_disjoint',anc['old_context_disjoint'],conflict,'diagnostic');ck(label+'/old_exact_equal',anc['old_context_span_equal'],oldspan==span,'diagnostic')
                codes=[state]
                if not context_complete:codes.append('context_incomplete')
                if conflict:codes.append('core_context_localization_disagreement')
                elif different:codes.append('core_context_boundary_difference')
                if oldanc['state']!='supported_candidate':codes.append('previous_context_unsupported')
                compare(label+'/holds',anc['diagnostic_hold_reasons'],[x for x in codes if x!='supported_core_candidate'],'diagnostic');holds.update(anc['diagnostic_hold_reasons'])
            ck(label+'/codes',anc['feedback_codes'],codes,'diagnostic')
            expected_feedback=dict(kind='research_evidence_only',messages=[TEXT[c] for c in codes],sign_correct=None,sign_error=None,correction_instruction=None,source_guidance_is_target_not_query_observation=True)
            compare(label+'/feedback',anc['feedback'],expected_feedback,'diagnostic');compare(label+'/reference_only_guidance',anc['reference_guidance'],dict(description=seg.get('description_ko'),source_links=seg.get('source_links',[]),query_label_inferred=False),'diagnostic')
            derived_states.append(state)
            nrw=numeric(anc,indices,'raw');nsel=numeric(anc,indices,'selected');expected_raw.append(nrw);expected_selected.append(nsel)
            if previous:
                ma=meananchors[sid];hit_mean=core[0]<=indices[ma['query_center']]<=core[1]
                context_hit=False if oldspan is None else core[0]<=indices[(oldspan[0]+oldspan[1])//2]<=core[1]
                stage_evidence.append(dict(segment_id=sid,mean_hit=hit_mean,context_hit=context_hit,raw_core=nrw,selected=nsel,core_observed=core_ok,supported_unambiguous=state=='supported_core_candidate',context_complete=context_complete,conflict=conflict,tie_resolved=anc.get('context_tie_resolved',False),state=state))
            selected=anc['selected']
            if selected:
                c=selected['context'];packed=dict(cost=selected['core_cost'],query_core=selected['query_core'],query_window=c['query_context'],path=c['phases']['entry']['path']+selected['core_path']+c['phases']['exit']['path'],core_supported=selected['core_supported'],context_support={k:v['supported'] for k,v in c['phases'].items()},phases=dict(core=selected['groups'],**{k:v['groups'] for k,v in c['phases'].items()}))
            else:c=None;packed=None
            ui_num=dict(nsel,center_in_original_core=nsel['center_hit'],path_mean_original_frame_error=nsel['path_mean_error']) if previous else None
            ui_anchors.append(dict(segment_id=sid,reference_core=core,reference_context=c['reference_context'] if c else core,old_center=(oldspan[0]+oldspan[1])//2 if oldspan else None,old_window=oldspan,state=state,best=packed,second_disjoint_gap=None,numerical=ui_num,feedback=anc['feedback'],reference_guidance=anc['reference_guidance'],raw_core=anc['raw']['query_core'] if anc['raw'] else None,tie_count=len(anc['core_ties']),context_tie_resolved=anc.get('context_tie_resolved',False),core_context_disagreement=conflict,diagnostic_hold_reasons=anc.get('diagnostic_hold_reasons',[]),path_is_concatenated_independent_parts=True))
        relations=[]
        for left,right in zip(d['anchors'],d['anchors'][1:]):
            if any(x['state'] in ['no_candidate','observation_insufficient'] for x in [left,right]):state='insufficient'
            elif any(x['state']=='ambiguous_occurrence' for x in [left,right]):state='ambiguous'
            else:
                x=left['selected']['query_core'];y=right['selected']['query_core'];state='forward_candidate' if x[1]<y[0] else 'reverse_candidate' if y[1]<x[0] else 'overlap'
            relations.append(dict(first=left['segment_id'],second=right['segment_id'],state=state,context_disagreement=left.get('old_context_disjoint',False) or right.get('old_context_disjoint',False),educational_verdict=None))
        compare(name+'/relations',d['relations'],relations,'state');ck(name+'/outside_all_query_frames',d['outside_core_candidates'],sorted(set(range(n))-covered),'scope')
        if previous:
            compare(name+'/raw_numeric',d['numerical_raw'],expected_raw,'correspondence');compare(name+'/selected_numeric',d['numerical_selected'],expected_selected,'correspondence')
            controls[d['control']].append(dict(name=name,query=a,role=d['role'],stages=stage_evidence,relations=relations))
        else:pairs.append(dict(query=a,reference=b,role=d['role'],states=dict(Counter(derived_states)),conflicts=sum(x.get('old_context_disjoint',False) for x in d['anchors']),seconds_incremental=d['seconds_incremental_core_split'],old_seconds_saved=d['old_context_seconds_saved']))
        allstates.update(derived_states)
        expected_ui=dict(id=name,title=f"{rows[a]['label']} {a} / "+(d.get('control') or f'참고 {b}'),kind=d['kind'],query=a,reference=b,role=d['role'],control=d.get('control'),query_pts=(qraw['timestamp_us']/1e6).tolist(),original_query_pts=(raw[a]['timestamp_us']/1e6).tolist(),reference_pts=(raw[b]['timestamp_us']/1e6).tolist(),original_indices=indices,anchors=ui_anchors,relations=relations,outside_core_candidates=d['outside_core_candidates'])
        compare(name+'/report_case',reportcases[name],expected_ui,'report')
        case_outputs.append(dict(name=name,kind=d['kind'],anchors=len(d['anchors']),states=dict(Counter(derived_states))))
        if jobidx%20==0:print('verified',jobidx,'/162 failures',len(failures),flush=True)
    metrics={}
    for control,jobs in controls.items():
        ss=[s for job in jobs for s in job['stages']];nn=len(ss)
        metrics[control]=dict(cases=len(jobs),stages=nn,mean_hits=sum(s['mean_hit'] for s in ss),context_hits=sum(s['context_hit'] for s in ss),raw_core_hits=sum(s['raw_core']['center_hit'] for s in ss),selected_hits=sum(s['selected']['center_hit'] for s in ss),raw_to_selected_losses=sum(s['raw_core']['center_hit'] and not s['selected']['center_hit'] for s in ss),raw_to_selected_gains=sum(not s['raw_core']['center_hit'] and s['selected']['center_hit'] for s in ss),supported_hits=sum(s['selected']['center_hit'] and s['supported_unambiguous'] for s in ss),context_complete_hits=sum(s['selected']['center_hit'] and s['supported_unambiguous'] and s['context_complete'] for s in ss),conflicts=sum(s['conflict'] for s in ss),states=dict(Counter(s['state'] for s in ss)),mean_core_recall=math.fsum(s['selected']['recall'] for s in ss)/nn,mean_core_precision=math.fsum(s['selected']['precision'] for s in ss)/nn,tie_resolved=sum(s['tie_resolved'] for s in ss),relations=dict(Counter(r['state'] for job in jobs for r in job['relations'])))
    times=defaultdict(list);oldtimes=defaultdict(list);devstates=Counter()
    for p in pairs:
        if p['role']!='fit':times[p['query']].append(p['seconds_incremental']);oldtimes[p['query']].append(p['old_seconds_saved']);devstates.update(p['states'])
    gates=dict(identity_all45=metrics['identity']['selected_hits']==45,slow_all45=metrics['speed_075']['selected_hits']==45,swap_not_worse_than_mean40=metrics['swap_adjacent_core_blocks']['selected_hits']>=40,no_context_override_damage=all(v['raw_to_selected_losses']==0 for v in metrics.values()),right_missing_no_supported=metrics['right_hand_unobserved']['states'].get('supported_core_candidate',0)==0,semantic_correctness_not_tested=True)
    expected_summary=dict(status='COMPLETE_PENDING_VERIFICATION',controls=metrics,gates=gates,person_pairs=len(pairs),development_states=dict(devstates),incremental_seconds_by_dev_query={k:math.fsum(v) for k,v in times.items()},previous_context_seconds_by_dev_query={k:math.fsum(v) for k,v in oldtimes.items()},runtime_note='core-only+anchoredcontext measured now; previouscontextcached; not fullpipeline or combined measuredruntime',word_specific_branches=False,validated_expressions=len({r['word_id'] for r in rows.values()}),all_word_generalization_verified=False,neural_training=0,production_adopted=False)
    compare('SUMMARY',read(U/'SUMMARY.json'),expected_summary,'summary');compare('REPORT_summary',report['summary'],expected_summary,'report');compare('CONTROL_EVALUATION',read(U/'CONTROL_EVALUATION.json'),dict(controls=dict(controls)),'summary');compare('PAIR_EVALUATION',read(U/'PAIR_EVALUATION.json'),dict(pairs=pairs),'summary')
    for p,h in frozen['pins'].items():ck('final_pin/'+p,sha(p),h,'provenance')
    for n,h in initial_hashes.items():ck('unchanged_during_verification/'+n,sha(U/n),h,'provenance')
    out=dict(status='PASS' if not failures else 'FAIL',reviewer='Codex agent B; independent saved arithmetic, same model family, not human KSL ground truth',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=sum(counts.values()),checks_by_category=dict(counts),failures=len(failures),failure_details=failures,max_float_error=max_error,results_checked=len(case_outputs),counts=dict(stats),states=dict(allstates),hold_reason_counts=dict(holds),tie_set_sizes=dict(tiesizes),verified_control_metrics=metrics,verified_gates=gates,artifact_pins=initial_hashes,verifier_sha256=sha(__file__),independent_name_based_feature_summary_oracle_sha256=sha(ORACLE),no_search=True,no_DP=True,no_model_calls=True,no_evaluator_import=True,limitations=['Selected/raw core and all stored context-path objectives are independently recomputed with scalar frame-group arithmetic. Global search optimality is not rerun.','All saved tied windows obey EPS and policy; completeness of the saved tie list is not independently searched. Nonselected tied core windows have no saved core path, so their recorded core cost cannot be independently reconstructed without forbidden DP/search.','Every saved context path and fixed reference expected/missing/unaligned group summary is checked. No-path feasibility uses the closed dimension bound for max2 axial runs, not DP.','Summary and all gates are recalculated from fixed original-index correspondence; these numerical labels are not semantic correctness or mandatory-stage completion.','Test count is assertions over saved artifacts, not independent sign-language accuracy examples.','Incremental timing is prepared-feature computation only; previous context timings are cached separate runs, not measured full capture-to-screen latency.'],result_checks=case_outputs)
    with (U/'review/SAVED_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:out[k] for k in ['status','checks','failures','max_float_error','results_checked','counts']}),flush=True)
    return 0 if not failures else 1
if __name__=='__main__':raise SystemExit(main())
