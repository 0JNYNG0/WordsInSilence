"""Independent saved-value audit. No torch/model/evaluation-module import or forward.
Run once only after root confirms PROCESS_AUTO_EVALUATION.json is complete.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import ast, hashlib, json, math, traceback
import numpy as np

U=Path(__file__).resolve().parent
HIGH,LOW,MAX_CANDIDATES=.8,.2,8
QUERIES=('right_hand_toward_left_wrist_along_arm','hands_move_apart',
         'right_hand_rises_relative_left','right_hand_descends_face_to_torso',
         'right_hand_falls_relative_left')
COUNTS=Counter()
HASHES={}

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need(ok,why):
    COUNTS['assertions']+=1
    if not ok: raise AssertionError(why)
def digest(p):
    p=str(Path(p).resolve())
    if p not in HASHES:
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''): h.update(block)
        HASHES[p]=h.hexdigest()
    return HASHES[p]
def pin(p,h): need(digest(p)==h,'Hash mismatch: '+str(p))
def equal(a,b,where):
    if isinstance(b,dict):
        need(isinstance(a,dict),where+' dict')
        for k,v in b.items():
            need(k in a,where+' missing '+k)
            equal(a[k],v,where+'.'+k)
    elif isinstance(b,list):
        need(isinstance(a,list) and len(a)==len(b),where+' list length')
        for i,(x,y) in enumerate(zip(a,b)): equal(x,y,where+'['+str(i)+']')
    elif isinstance(b,float):
        need(isinstance(a,(int,float)) and math.isfinite(a) and abs(a-b)<=1e-12,where+' float')
    else: need(a==b and (not isinstance(b,bool) or type(a)==bool),where)
def prob(x,where):
    need(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0<=x<=1,where)
    return float(x)
def constants():
    tree=ast.parse((U/'process_qualification.py').read_text(encoding='utf-8'))
    values={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ['HIGH','LOW','MAX_INTERVAL_CANDIDATES']:
                    values[target.id]=ast.literal_eval(node.value)
    equal(values,dict(HIGH=HIGH,LOW=LOW,MAX_INTERVAL_CANDIDATES=MAX_CANDIDATES),'qualification constants')

def candidate_geometry(point,available):
    # Derive connected visible components, then pair setup-run entrances with
    # the earliest later point in each terminal run. No target/witness access.
    need(len(point)==len(available),'point/availability rows')
    for pair in point:
        need(len(pair)==2,'point columns')
        for x in pair: prob(x,'point probability')
    need(all(type(x)==bool for x in available),'boolean availability')
    components=[];start=None
    for i,ok in enumerate(available+[False]):
        if ok and start is None:start=i
        if not ok and start is not None:
            components.append(list(range(start,i)));start=None
    pairs=[]
    for g in components:
        starts=[i for i in g if point[i][0]>=HIGH and (i==g[0] or point[i-1][0]<HIGH)]
        ends=[];run=[]
        for i in g:
            if point[i][1]>=HIGH:run.append(i)
            elif run:ends.append(run);run=[]
        if run:ends.append(run)
        for s in starts:
            for terminal in ends:
                valid=[t for t in terminal if t>s]
                if valid:pairs.append((s,valid[0]))
    pairs.sort(key=lambda p:(p[1],p[0]))
    return [dict(start=s,end=t,output_indices=list(range(s,t+1)),setup_index=s,terminal_index=t,
                 setup_probability=float(point[s][0]),terminal_probability=float(point[t][1]))
            for s,t in pairs[:MAX_CANDIDATES]],len(pairs)

def decision(p,chain,available):
    if p>=HIGH and chain:return 'OBSERVED_PROCESS_EVIDENCE',1,chain[0]
    if not any(available):return 'INSUFFICIENT_OBSERVATION',None,None
    if p<=LOW and chain:return 'PROCESS_AND_LOCAL_EVIDENCE_DISAGREE',None,None
    if p<=LOW:return 'PROCESS_NOT_CONFIRMED',None,None
    return 'PROCESS_EVIDENCE_UNCONFIRMED',None,None

def qualification(saved,p,point,available,where,seams=None):
    p=prob(p,where+' process')
    geometry,total=candidate_geometry(point,available)
    observed=saved['interval_candidates']
    need(len(observed)==len(geometry)<=MAX_CANDIDATES,where+' candidate count')
    candidates=[]
    for i,(c,g) in enumerate(zip(observed,geometry)):
        equal(c,g,where+' candidate '+str(i))
        need(all(available[j] for j in g['output_indices']),where+' candidate crosses unavailable image')
        score=prob(c['motion_probability'],where+' interval score')
        candidates.append(dict(g,motion_probability=score))
    chain=[c for c in candidates if c['motion_probability']>=HIGH]
    status,display,witness=decision(p,chain,available)
    expected=dict(status=status,qualified_display=display,raw_probability=p,
                  raw_display=1 if p>=HIGH else 0 if p<=LOW else None,
                  witnessed_chain=witness,interval_candidates=candidates,candidate_count=len(candidates),
                  uncapped_candidates=total,candidate_cap_reached=total>MAX_CANDIDATES,
                  unavailable_indices=[i for i,ok in enumerate(available) if not ok],
                  fragment_provenance='availability_detector_only',source_seam_oracle_diagnostic=False,
                  observation_only=True,observed_process_absence_certified=False,
                  whole_word_correctness=None,learner_error=None,required_repetitions=None,
                  complete_action_verified=False,raw_low_is_not_word_error=True)
    equal(saved,expected,where)
    COUNTS['runtime_candidates']+=len(candidates)
    COUNTS['runtime_confirmed']+=display==1
    if seams is None:return expected
    # A privileged seam diagnostic reuses the same scores. It does NOT replace
    # runtime output or claim a detector could know source editing metadata.
    oracle_candidates=[dict(c,crosses_known_source_seam=any(c['start']<s<=c['end'] for s in seams)) for c in candidates]
    clean=[c for c in oracle_candidates if c['motion_probability']>=HIGH and not c['crosses_known_source_seam']]
    status,display,witness=decision(p,clean,available)
    oracle=dict(expected,status=status,qualified_display=display,witnessed_chain=witness,
                interval_candidates=oracle_candidates,fragment_provenance='known_source_edit_metadata',
                source_seam_oracle_diagnostic=True,source_seam_indices=sorted(set(seams)),
                same_runtime_candidates_and_scores_reused=True,independent_real_runtime_generalization=False)
    COUNTS['runtime_candidates_crossing_known_seam']+=sum(c['crosses_known_source_seam'] for c in oracle_candidates)
    COUNTS['runtime_confirmed_chain_crossing_known_seam']+=bool(expected['witnessed_chain']) and any(
        expected['witnessed_chain']['start']<s<=expected['witnessed_chain']['end'] for s in seams)
    return expected,oracle

def summary(rows,field='qualification'):
    pos=[r for r in rows if r['truth']==1];neg=[r for r in rows if r['truth']==0]
    yes=lambda r:r[field]['qualified_display']==1
    tp=sum(yes(r) for r in pos)
    return dict(total=len(rows),known_positive=len(pos),known_zero=len(neg),
                unknown=sum(r['truth'] is None for r in rows),
                confirmed_positive_correct=tp,confirmed_positive_wrong=sum(yes(r) for r in neg),
                unknown_confirmed_positive=sum(r['truth'] is None and yes(r) for r in rows),
                known_positive_recall=tp/len(pos) if pos else None,
                raw_low_on_known_positive=sum(r[field]['raw_display']==0 for r in pos),
                raw_low_on_known_zero=sum(r[field]['raw_display']==0 for r in neg),
                all_absence_claims=False,qualified_negative_is_not_defined=True)
def null_language(obj,where):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k in ['whole_word_correctness','learner_error','required_repetitions',
                     'required_repetition_count','learned_selected_chain_exact_time_truth']:
                need(v is None,where+' unsupported truth '+k)
            if k in ['exact_boundary_accuracy_claim','complete_action_verified']:
                need(v is False,where+' unsupported complete/boundary '+k)
            null_language(v,where+'.'+k)
    elif isinstance(obj,list):
        for v in obj:null_language(v,where)

def reconstruct_proposal(row,full,old_result):
    ids=full['frame_id'].tolist();ts=full['timestamp_us'].tolist()
    p=np.asarray(old_result['path'])
    anchor=row['proposal']['canonical_anchor'];m=anchor['prototype_length']
    need(p.ndim==2 and p.shape[1]==2 and np.issubdtype(p.dtype,np.integer),'saved DTW axes')
    need(np.array_equal(p[0],[0,0]) and np.array_equal(p[-1],[len(ids)-1,m-1]),'saved DTW endpoints')
    d=np.diff(p,axis=0)
    need(((d>=0)&(d<=1)).all() and (d.sum(1)>0).all(),'saved DTW allowed steps')
    need(np.array_equal(np.unique(p[:,0]),np.arange(len(ids))) and np.array_equal(np.unique(p[:,1]),np.arange(m)),'saved DTW coverage')
    lo,hi=anchor['prototype_index_extent']
    mapped=[int(x) for x in p[(p[:,1]>=lo)&(p[:,1]<=hi),0]]
    need(bool(mapped),'anchor mapped')
    left=max(0,min(mapped)-12);right=min(len(ids)-1,max(mapped)+12)
    expected=dict(status='PROPOSED_CONTEXT_ONLY',output_indices=list(range(left,right+1)),
                  full_native_index_extent=[left,right],mapped_anchor_index_extent=[min(mapped),max(mapped)],
                  source_native_frame_extent=[ids[left],ids[right]],source_time_us_extent=[ts[left],ts[right]],
                  context_frames=12,max_proposals_per_query=1,exact_event_boundary=False,
                  observed_process_truth=None,whole_word_correctness=None,additional_DTW=0,
                  individual_reference_video_comparisons=0)
    equal(row['proposal'],expected,'saved proposal '+row['query_id'])
    return list(range(left,right+1))

def native_support(target,frames):
    yes=[s for s in target['sequence_links'] if s['visible_chain']=='yes']
    supported=[]
    for link in yes:
        required=set(link['setup_frames'])|set(link['terminal_frames'])
        if link.get('motion_interval'):
            lo,hi=link['motion_interval'];required.update(range(lo,hi+1))
        if required and required<=set(frames):supported.append(link)
    return (1 if yes else None),(1 if supported else None),supported

def main():
    out=U/'PROCESS_AUTO_EVALUATION_VERIFIED.json'
    need(not out.exists(),'Do not repeat completed saved-result verification')
    required=['PROCESS_AUTO_EVALUATION.json','PROCESS_EVALUATION.json','PROCESS_COMPARISON_FROZEN.json',
              'PROCESS_TRAIN_RESULT.json','PROCESS_AUTO_NATIVE_MANIFEST.json']
    need(all((U/n).exists() for n in required),'Wait for completed learning/evaluation artifacts; never launch them here')
    constants()
    result=read(U/required[0]);raw=read(U/required[1]);contract=read(U/required[2])
    trained=read(U/required[3]);native_manifest=read(U/required[4])
    for p,h in contract['pins'].items():pin(p,h)
    for p,h in native_manifest['pins'].items():pin(p,h)
    for obj in [result,raw,trained]:pin(U/'PROCESS_COMPARISON_FROZEN.json',obj['contract_sha256'])
    pin(U/'PROCESS_EVALUATION.json',result['saved_raw_evaluation_sha256'])
    pin(trained['checkpoint'],trained['checkpoint_sha256']);pin(trained['normalization'],trained['normalization_sha256'])
    equal(result['checkpoint_sha256'],trained['checkpoint_sha256'],'checkpoint lineage')
    equal(result,dict(old_models_executed=0,new_DTW=0,baseline_full_process_score_available=False,whole_goal_complete=False,
                      candidate_forward_budget=dict(repeated_raw_process_point=0,automatic_process_point=60,
                      interval_batch_per_query_at_most=1,max_intervals_per_batch=8)),'evaluation scope')
    views={r['view_id']:r for name in ['PROCESS_VIEW_TARGETS.json','DEV_PROCESS_VIEW_TARGETS.json'] for r in read(U/name)['rows']}
    native_targets={r['id']:r for name in ['PROCESS_TARGETS.json','DEV_PROCESS_TARGETS.json'] for r in read(U/name)['rows']}
    raw_by_id={r['view_id']:r for r in raw['rows']}
    transformed=result['view_rows'];automatic=result['automatic_rows']
    need(len(views)==len(raw['rows'])==len(raw_by_id)==len(transformed)==297,'transformed denominator')
    need(len({r['view_id'] for r in transformed})==297 and set(raw_by_id)==set(views)=={r['view_id'] for r in transformed},'transformed identity set')
    for r in transformed:
        t=views[r['view_id']];saved=raw_by_id[r['view_id']]
        equal(r,{k:t[k] for k in ['view_id','query_id','clip_id','role','physical_query']},'view identity')
        equal(r,dict(person=t['recorded_person_id'],kind=t['construction_kind_audit_only'],truth=t['observed_process_evidence'],
                     raw_process_point_forward_repeated=False,annotated_interval_truth_used_to_select_candidates=False),'view provenance')
        equal(saved,dict(truth=t['observed_process_evidence'],source_frames=[x['source_native_frame'] for x in t['ordered_frame_map']],
                         source_timestamps_us=[x['source_native_timestamp_us'] for x in t['ordered_frame_map']],
                         view_timestamps_us=[x['view_timestamp_us'] for x in t['ordered_frame_map']],
                         image_available=[x['image_available'] for x in t['ordered_frame_map']]),'raw axis')
        rt,oracle=qualification(r['qualification'],saved['probability'],saved['point_probabilities'],saved['image_available'],
                                r['view_id'],t['seam_output_indices'])
        equal(r['oracle_known_seam_diagnostic'],oracle,r['view_id']+' oracle')
        null_language(t,'target');null_language(r,'view')
    nm={r['query_id']:r for r in native_manifest['rows']}
    need(len(nm)==len(native_targets)==len(automatic)==60,'automatic denominator')
    need(set(nm)==set(native_targets)=={r['query_id'] for r in automatic} and len({r['query_id'] for r in automatic})==60,'automatic identities')
    full_cache={};result_cache={}
    # Validate canonical anchors from existing FIT01 guides, not DEV boundaries.
    P=U.parent/'word100_learned_guide_feedback_20261102'
    guides=read(P/'GUIDE_SUPPORT.json')['guides']
    canonical={r['word_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows'] if r['role']=='fit' and r['person']==1}
    for word,anchor in native_manifest['anchors'].items():
        g=[x for x in guides if x['word_id']==word]
        need(g and all(x['boundary_provenance']=='canonical01_saved_reviewed_span' for x in g),'canonical guide scope')
        extent=[min(int(x['frames'][0]) for x in g),max(int(x['frames'][1]) for x in g)]
        cr=canonical[word];pin(cr['input'],cr['input_sha256'])
        with np.load(cr['input'],allow_pickle=False) as z:axis=z['frame_id']
        need(np.array_equal(axis,np.arange(len(axis))),'canonical identity-axis contract')
        equal(anchor,dict(source_native_extent=extent,prototype_index_extent=extent,prototype_length=len(axis),
                          guide_ids=[x['canonical_stage_id'] for x in g]),'canonical anchor')
    for r in automatic:
        m=nm[r['query_id']];target=native_targets[r['query_id']]
        equal(r,{k:m[k] for k in ['query_id','clip_id','person','role','physical_query']},'automatic identity')
        need(m['role'] in ['fit','development'] and m['person'] in (range(1,10) if m['role']=='fit' else [11,12,13]),'public role allowlist')
        equal(m['source_sha256'],target['source_sha256'],'native target source pin')
        pin(m['input'],m['input_sha256']);pin(m['result'],m['result_sha256'])
        if m['input'] not in full_cache:
            with np.load(m['input'],allow_pickle=False) as z:full_cache[m['input']]={k:z[k] for k in ['frame_id','timestamp_us','image_available']}
            result_cache[m['result']]=read(m['result'])
        a=full_cache[m['input']];old=result_cache[m['result']]
        equal(a['frame_id'].tolist(),m['frame_ids'],'full axis IDs')
        equal(a['timestamp_us'].tolist(),m['timestamp_us'],'full axis times')
        equal(old,dict(source_sha256=m['source_sha256'],word_id=m['word_id'],
                       path_axes=['query_native_index','learned_canonical_index'],individual_reference_comparisons=0,
                       learned_prototype_comparisons=1),'saved native source/path provenance')
        equal(old['overlay']['times'],(a['timestamp_us']/1e6).tolist(),'overlay clock')
        equal(m['proposal']['canonical_anchor'],native_manifest['anchors'][m['word_id']],'proposal anchor binding')
        ii=reconstruct_proposal(m,a,old);equal(r['proposal'],m['proposal'],'automatic saved proposal')
        frames=a['frame_id'][ii].tolist();times=a['timestamp_us'][ii].tolist();av=a['image_available'][ii].tolist()
        equal(r,dict(source_frame_ids=frames,source_timestamps_us=times,image_available=av,
                     learned_selected_chain_exact_time_truth=None,exact_boundary_accuracy_claim=False,additional_DTW=0),'auto clock and limits')
        qualification(r['qualification'],r['probability'],r['point_probabilities'],av,'auto '+r['query_id'])
        c=r['qualification']['witnessed_chain']
        if c:
            full=[ii[j] for j in c['output_indices']]
            equal(c,dict(full_native_indices=full,source_frame_ids=a['frame_id'][full].tolist(),
                         source_timestamps_us=a['timestamp_us'][full].tolist()),'witness source clock')
        truth,scope,supported=native_support(target,frames)
        equal(r,dict(truth=truth,proposal_scope_truth=scope,proposal_contains_reviewed_positive_chain=bool(supported),
                     recovered=truth==1 and scope==1 and r['qualification']['qualified_display']==1),'support bookkeeping')
        null_language(r,'automatic');null_language(target,'native target')
    rebuilt={};macro={}
    for role in ['fit','development']:
        vv=[r for r in transformed if r['role']==role];aa=[r for r in automatic if r['role']==role]
        need(len(vv)==(217 if role=='fit' else 80),'role view denominator')
        need(len(aa)==(45 if role=='fit' else 15),'role automatic denominator')
        perq={}
        for query in QUERIES:
            ar=[r for r in aa if r['physical_query']==query]
            perq[query]=dict(total=len(ar),known_positive=sum(r['truth']==1 for r in ar),
                            recovered=sum(r['recovered'] for r in ar),
                            proposal_without_reviewed_chain=sum(r['truth']==1 and r['proposal_scope_truth'] is None for r in ar))
        denominator=sum(r['truth']==1 for r in aa);rec=sum(r['recovered'] for r in aa)
        rebuilt[role]=dict(real_runtime_qualification=summary(vv),
          oracle_only_known_edit_seam_diagnostic=summary(vv,'oracle_known_seam_diagnostic'),
          per_query={q:summary([r for r in vv if r['physical_query']==q]) for q in QUERIES},
          per_kind={k:summary([r for r in vv if r['kind']==k]) for k in sorted({r['kind'] for r in vv})},
          per_person={str(p):summary([r for r in vv if r['person']==p]) for p in sorted({r['person'] for r in vv})},
          automatic=dict(total=len(aa),known_positive_denominator=denominator,recovered=rec,
            recall=rec/denominator if denominator else None,per_query=perq,
            unknown=sum(r['truth'] is None for r in aa),
            unknown_confirmed_positive=sum(r['truth'] is None and r['qualification']['qualified_display']==1 for r in aa),
            per_person={str(p):dict(known_positive=sum(r['truth']==1 for r in aa if r['person']==p),
                         recovered=sum(r['recovered'] for r in aa if r['person']==p)) for p in sorted({r['person'] for r in aa})},
            proposal_without_reviewed_positive_chain=sum(r['truth']==1 and r['proposal_scope_truth'] is None for r in aa),
            raw_low_on_known_positive=sum(r['truth']==1 and r['qualification']['raw_display']==0 for r in aa)))
        rates=[x['recovered']/x['known_positive'] for x in perq.values() if x['known_positive']]
        sq=rebuilt[role]['per_query']
        recalls=[x['known_positive_recall'] for x in sq.values() if x['known_positive']]
        false_rates=[x['confirmed_positive_wrong']/x['known_zero'] for x in sq.values() if x['known_zero']]
        macro[role]=dict(automatic_query_recall=sum(rates)/len(rates) if rates else None,
                        automatic_query_denominators=perq,automatic_macro_query_count=len(rates),
                        transformed_query_positive_recall=sum(recalls)/len(recalls) if recalls else None,
                        transformed_query_positive_denominator_count=len(recalls),
                        transformed_query_confirmed_on_zero_rate=sum(false_rates)/len(false_rates) if false_rates else None,
                        transformed_query_zero_denominator_count=len(false_rates),
                        native_known_zero_count=sum(r['truth']==0 for r in aa),
                        native_false_positive_rate=None,
                        interpretation='Automatic recovery only confirms a broad proposal containing a reviewed positive plus qualified output; selected exact interval correctness remains unknown. Edited-view0 is partial evidence, not learner error.')
    equal(result['summaries'],rebuilt,'saved summary independent reconstruction')
    report=dict(status='PASS_SAVED_AUTO_EVALUATION_ARITHMETIC_PROVENANCE_AND_QUALIFICATION',
                at=datetime.now(timezone.utc).isoformat(),reviewer='Codex egypt_source_binding_review, separate saved-value verifier author; not independent human/KSL expert',
                view_rows=297,automatic_rows=60,thresholds=dict(high=HIGH,low=LOW,max_interval_candidates=MAX_CANDIDATES),
                checks=dict(COUNTS),reconstructed_summaries=rebuilt,query_macro=macro,
                result_sha256=digest(U/'PROCESS_AUTO_EVALUATION.json'),raw_sha256=digest(U/'PROCESS_EVALUATION.json'),
                native_manifest_sha256=digest(U/'PROCESS_AUTO_NATIVE_MANIFEST.json'),
                contract_sha256=digest(U/'PROCESS_COMPARISON_FROZEN.json'),verifier_sha256=digest(__file__),
                hashed_file_count=len(HASHES),new_model_inferences=0,new_training=0,new_DTW=0,old_models_executed=0,
                selected_exact_interval_truth=None,whole_word_correctness=None,learner_error=None,required_repetitions=None,
                limitations=['Learned probabilities are range/lineage checked, not numerically reproduced by another forward.',
                             'Macro denominators use physical queries and recorded source groups; edited views are not independent people.',
                             'Known edit seams are a privileged diagnostic and never substituted for real runtime qualification.',
                             'Source witness containment is bookkeeping, not exact selected-event or language correctness.',
                             'Repeated DEV is not a fresh independent final test.'],
                whole_goal_complete=False)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=report['status'],checks=dict(COUNTS),query_macro=macro),ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        first=U/'PROCESS_AUTO_SAVED_VERIFICATION_FIRST_FAILURE.txt'
        if not first.exists():first.write_text(traceback.format_exc(),encoding='utf-8')
        raise

