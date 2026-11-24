"""Reconstruct saved downstream arithmetic without engine/feedback/torch imports."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import hashlib
import json
import math
import statistics
import numpy as np
U=Path(__file__).resolve().parent.parent
OLD=U.parent/'word100_stage_feedback_cycle_20261031'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
STATES=['SUPPORTED_STATE','OPPOSITE_STATE','UNOBSERVED','UNCERTAIN_MODEL','ALIGNMENT_UNAVAILABLE','REFERENCE_UNOBSERVED','REFERENCE_OBSERVER_CONFLICT']

def classify(p,valid):
    p=float(p)
    if not valid or not math.isfinite(p) or not 0<=p<=1:return 'UNOBSERVED'
    if p>=.8:return 'SUPPORTED_STATE'
    if 1-p>=.8:return 'OPPOSITE_STATE'
    return 'UNCERTAIN_MODEL'

def runs(data,column):
    active=np.zeros(len(data['times_us']),bool)
    for low,high in data['intervals']:active|=(data['times_us']>=low*1e6)&(data['times_us']<=high*1e6)
    groups=[];group=[];period=statistics.median(np.diff(data['times_us']).tolist())
    for i in range(len(active)):
        p=float(data['prob'][i,column]);known=active[i] and data['support'][i,column] and math.isfinite(p) and .8<=p<=1
        continuation=bool(group) and i==group[-1]+1 and data['frame_ids'][i]==data['frame_ids'][group[-1]]+1 and data['times_us'][i]-data['times_us'][group[-1]]<=2*period
        if not known or not continuation:
            if len(group)>=2:groups.append(group)
            group=[]
        if known:group.append(i)
    if len(group)>=2:groups.append(group)
    return [dict(start_index=g[0],end_index=g[-1],start_seconds=float(data['times_us'][g[0]]/1e6),end_seconds=float(data['times_us'][g[-1]]/1e6),frames=len(g)) for g in groups]

def main(output):
    checks=0;fail=[]
    def check(ok,name,detail=None):
        nonlocal checks
        checks+=1
        if not bool(ok):fail.append(dict(check=name,detail=detail))
    frozen=read(U/'FROZEN.json');result=read(U/'RESULT.json');contracts=read(U/'source_audit/SOURCE_CONTRACTS.json');old=read(OLD/'LEARNING_INPUTS.json')
    targets=frozen['targets'];selected=frozen['target_indices'];rows=frozen['rows'];rowmap={r['alias']:r for r in rows};index={r['alias']:i for i,r in enumerate(old['rows'])}
    check(len(rows)==48 and Counter(r['role'] for r in rows)=={'fit':36,'development':12},'fitdev_only48')
    check(sorted({r['person'] for r in rows if r['role']=='fit'})==[1,2,3] and sorted({r['person'] for r in rows if r['role']=='development'})==[4],'person_split')
    for category in ['pins','protected']:
        for p,h in frozen[category].items():check(sha(p)==h,category+'/'+Path(p).name)
    prior_rows=read(OLD/'DEVELOPMENT_COMPARISON.json')['rows'];prior={r['query']:r for r in prior_rows if r['condition']=='pose_capacity_matched'}
    probabilities=np.load(OLD/'runs/pose_capacity_matched_42/probabilities.npy',mmap_mode='r',allow_pickle=False)
    data={};input_pins={};targets_cache={}
    for row in rows:
        alias=row['alias']
        with np.load(row['input'],allow_pickle=False) as z:a={k:z[k] for k in ['mask','frame_id','timestamp_us']}
        with np.load(row['observation'],allow_pickle=False) as z:xy=z['xy_px'];usable=z['usable'];obframes=z['frame_id']
        box=row['box'];inside=(xy[:,:,0]>=box[0])&(xy[:,:,0]<box[2])&(xy[:,:,1]>=box[1])&(xy[:,:,1]<box[3]);usable=usable&inside
        check(np.array_equal(a['frame_id'],obframes),'input_observation_alignment/'+alias)
        support=np.ones((len(a['mask']),len(targets)),bool);names=row['feature_names']
        for j,feature in enumerate(targets):
            for side in ['R','L'] if feature=='hands_overlap' else [feature[0]]:
                support[:,j]&=a['mask'][:,[names.index(side+'_hand_wrist_x'),names.index(side+'_hand_wrist_y')]].all(1)
                if '_motion_' in feature:support[:,j]&=a['mask'][:,[names.index(side+'_wrist_velocity_x'),names.index(side+'_wrist_velocity_y')]].all(1)
                elif feature.endswith(('index_only','open_four')):
                    required=[names.index(f'{side}_{finger}_{joint}_angle') for finger in ['index','middle','ring','little'] for joint in ['mcp','pip','dip']]
                    support[:,j]&=a['mask'][:,required].all(1)
            if feature.endswith('face_overlap'):support[:,j]&=usable[:,[0,2,5]].all(1)&np.isfinite(xy[:,[0,2,5]]).all((1,2))
        handframes=np.flatnonzero(usable[:,33:75].any(1));low,high=int(handframes[0]),int(handframes[-1]);ts=a['timestamp_us']
        requested=np.linspace(ts[low],ts[high],96)
        # Independent nearest-time enumeration; tie chooses the later source sample.
        sample=np.array(sorted({min(range(low,high+1),key=lambda i:(abs(float(ts[i])-t),-i)) for t in requested}),dtype=np.int64)
        d=dict(prob=np.array(probabilities[index[alias],:len(ts)],copy=True),support=support,frame_ids=a['frame_id'],times_us=ts,sample_indices=sample,intervals=row['intervals'])
        h=hashlib.sha256()
        for key in ['prob','support','frame_ids','times_us','sample_indices']:
            value=np.asarray(d[key]);h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
        d['digest']=h.hexdigest()
        if row['role']=='fit':
            with np.load(row['targets_file'],allow_pickle=False) as z:y=z['y'][:,selected];mask=z['target_mask'][:,selected]
            active=np.zeros(len(ts),bool)
            for lo,hi in row['intervals']:active|=(ts/1e6>=lo)&(ts/1e6<=hi)
            d['anchors']={name:np.flatnonzero(active&mask[:,j]&(y[:,j]==1)).tolist() for j,name in enumerate(targets)}
        data[alias]=d
    contract_by_word={c['word_id']:c for c in contracts['expressions']};declared_inputs={x['alias']:x for x in result['inputs']}
    clause_statuses=Counter();order_statuses=Counter();all_anchor_cells=0;clauses_executable=0;raw_count=0;outside_count=0;diagnostics=[]
    expected_queries=[r['alias'] for r in rows if r['role']=='development']
    check([r['query'] for r in result['rows']]==expected_queries,'all12_query_order')
    for out in result['rows']:
        alias=out['query'];q=data[alias];row=rowmap[alias];contract=contract_by_word[row['word_id']];oldpairs={p['reference']:p for p in prior[alias]['pairs']}
        check(declared_inputs[alias]['observer_digest']==q['digest'],'independent_observer_digest/'+alias)
        check(declared_inputs[alias]['refs']==list(oldpairs),'fixed_reference_set/'+alias)
        check(all(rowmap[r]['role']=='fit' and rowmap[r]['word_id']==row['word_id'] for r in oldpairs),'fit_same_expression_references/'+alias)
        check(out['raw_physical_difference_candidates']==prior[alias]['events'],'unchanged_raw_comparison/'+alias)
        check(out['baseline_distance']==prior[alias]['distance'],'unchanged_prior_dtw_distance/'+alias)
        check(out['whole_sign_verdict'] is None and out['linguistically_acceptable_variation'] is None,'no_linguistic_verdict/'+alias)
        raw_count+=len(out['raw_physical_difference_candidates'])
        active_features={c.get('feature') for c in contract['clauses'] if c['link_status'] in ['partial_supported','proxy_only']}
        expected_outside=[e for e in prior[alias]['events'] if e['feature'] not in active_features]
        check(out['outside_source_contract_candidates']==expected_outside,'outside_contract_candidates_preserved/'+alias);outside_count+=len(expected_outside)
        check([c['id'] for c in out['clauses']]==[c['id'] for c in contract['clauses']],'all_source_clauses_preserved/'+alias)
        evaluated={}
        for clause,actual in zip(contract['clauses'],out['clauses']):
            label=alias+'/'+clause['id'];clause_statuses[actual['status']]+=1
            check(actual['source_span']==clause.get('source_span') and actual['source_url']==clause.get('source_url') and actual['link_status']==clause['link_status'],'source_provenance/'+label)
            check(actual['educational_verdict'] is None and actual['linguistic_tolerance'] is None,'clause_no_educational_claim/'+label)
            if clause['link_status'] not in ['partial_supported','proxy_only'] or clause.get('feature') not in targets:
                check(actual['status']=='SOURCE_LINK_WITHHELD' and actual['references']==[],'rejected_or_unsupported_withheld/'+label);continue
            clauses_executable+=1;j=targets.index(clause['feature']);votes=defaultdict(set);expected_count=0;observed_count=0;supportrefs=[];oppositerefs=[]
            check([r['reference'] for r in actual['references']]==list(oldpairs),'clause_three_separate_refs/'+label)
            for ar in actual['references']:
                refname=ar['reference'];ref=data[refname];anchors=ref['anchors'][clause['feature']];expected_count+=len(anchors)
                check(ar['expected_anchor_count']==len(anchors)==len(ar['cells']),'complete_fit_positive_anchors/'+label+'/'+refname)
                counts=Counter()
                for k,cell in zip(anchors,ar['cells']):
                    counts[cell['state']]+=1
                    check(cell['reference_frame']==int(ref['frame_ids'][k]) and cell['reference_seconds']==float(ref['times_us'][k]/1e6),'original_reference_anchor/'+label)
                    sample=ref['sample_indices'];period=statistics.median(np.diff(ref['times_us']).tolist())
                    if k<sample[0] or k>sample[-1]:
                        check(cell['state']=='ALIGNMENT_UNAVAILABLE' and cell['query_frame'] is None,'out_of_alignment_anchor_preserved/'+label);continue
                    nearest=min(range(len(sample)),key=lambda s:(abs(int(ref['times_us'][sample[s]])-int(ref['times_us'][k])),s))
                    error=abs(int(ref['times_us'][sample[nearest]])-int(ref['times_us'][k]))
                    if error>int(period)+1:
                        check(cell['state']=='ALIGNMENT_UNAVAILABLE' and cell['query_frame'] is None,'quantized_anchor_preserved/'+label);continue
                    qi=int(q['sample_indices'][oldpairs[refname]['reference_to_query'][nearest]])
                    state=classify(q['prob'][qi,j],q['support'][qi,j])
                    if not ref['support'][k,j]:state='REFERENCE_UNOBSERVED'
                    elif classify(ref['prob'][k,j],True)=='OPPOSITE_STATE':state='REFERENCE_OBSERVER_CONFLICT'
                    expected=dict(query_frame=int(q['frame_ids'][qi]),query_seconds=float(q['times_us'][qi]/1e6),query_index=qi,state=state,query_probability=float(q['prob'][qi,j]),reference_probability=float(ref['prob'][k,j]),alignment_quantization_us=error)
                    check(all(cell.get(key)==val for key,val in expected.items()),'independent_anchor_state_alignment/'+label,dict(reference=refname,frame=int(ref['frame_ids'][k])))
                    if state in ['SUPPORTED_STATE','OPPOSITE_STATE']:
                        votes[(cell['query_frame'],state)].add(refname);observed_count+=1
                        evaluated[(cell['query_frame'],clause['feature'])]=dict(frame=cell['query_frame'],feature=clause['feature'],state=state,index=qi,column=j)
                check(ar['counts']=={s:counts[s] for s in STATES},'anchor_counts/'+label+'/'+refname)
                if counts['SUPPORTED_STATE']:supportrefs.append(refname)
                if counts['OPPOSITE_STATE']:oppositerefs.append(refname)
            consensus=[dict(query_frame=key[0],state=key[1],references=sorted(value)) for key,value in votes.items() if len(value)>=2]
            check(actual['coherent_query_frame_consensus']==consensus,'same_query_frame_distinct_reference_consensus/'+label)
            status='GUIDE_RELEVANT_DIFFERENCE_CANDIDATE' if any(c['state']=='OPPOSITE_STATE' for c in consensus) else 'DESCRIBED_FEATURE_OBSERVED' if consensus else 'INSUFFICIENT_LINKED_EVIDENCE'
            check(actual['status']==status and actual['expected_anchor_count']==expected_count and actual['observed_anchor_count']==observed_count,'clause_state_and_denominator/'+label)
            check(actual['supported_references']==supportrefs and actual['opposite_references']==oppositerefs,'reference_lists/'+label);all_anchor_cells+=expected_count
        check([x['id'] for x in out['orders']]==[x['id'] for x in contract['orders']],'all_source_orders_preserved/'+alias)
        for order,actual in zip(contract['orders'],out['orders']):
            order_statuses[actual['status']]+=1;label=alias+'/'+order['id']
            check(actual['educational_verdict'] is None and actual['mandatory_omission_verdict'] is None,'order_no_omission_claim/'+label)
            if order['link_status'] not in ['partial_supported','proxy_only'] or order['before'] not in targets or order['after'] not in targets:
                check(actual['status']=='ORDER_NOT_OBSERVABLE' and actual['before_events']==actual['after_events']==[],'unsupported_order_withheld/'+label);continue
            left=runs(q,targets.index(order['before']));right=runs(q,targets.index(order['after']))
            check(actual['before_events']==left and actual['after_events']==right,'independent_original_time_events/'+label)
            forward=[(a,b) for a in left for b in right if a['end_index']<b['start_index']];reverse=[(b,a) for a in left for b in right if b['end_index']<a['start_index']]
            expected='DESCRIBED_ORDER_PATTERN_OBSERVED' if forward else 'REVERSE_ORDER_CANDIDATE' if reverse else 'ORDER_UNCONFIRMED'
            check(actual['status']==expected,'independent_order_status/'+label)
            check(actual['witness']==(dict(before=forward[0][0],after=forward[0][1]) if forward else None),'actual_order_witness/'+label)
        # Development draft labels used only after reconstructing all decisions for this query.
        with np.load(row['targets_file'],allow_pickle=False) as z:y=z['y'][:,selected];mask=z['target_mask'][:,selected]
        cells=[]
        for item in evaluated.values():
            i,j=item['index'],item['column'];cells.append({k:item[k] for k in ['frame','feature','state']}|dict(draft_label=int(y[i,j]) if mask[i,j] else None))
        known=[c for c in cells if c['draft_label'] is not None];expected=dict(alias=alias,evaluated_cells=len(cells),known_draft_cells=len(known),agree=sum((c['state']=='SUPPORTED_STATE')==(c['draft_label']==1) for c in known),cells=cells)
        actual=next(x for x in result['physical_draft_audit'] if x['alias']==alias)
        check(all(actual.get(k)==v for k,v in expected.items()),'independent_descriptive_draft_audit/'+alias)
        diagnostics.append({k:v for k,v in expected.items() if k!='cells'})
    s=result['summary']
    check(s['source_anchored_cells']==all_anchor_cells,'summary_anchor_cells')
    check(s['clause_statuses']=={k:clause_statuses[k] for k in s['clause_statuses']},'summary_clause_statuses')
    check(s['order_statuses']=={k:order_statuses[k] for k in s['order_statuses']},'summary_order_statuses')
    check(s['raw_difference_candidate_cells']==raw_count and s['outside_source_contract_cells']==outside_count,'summary_raw_preserved')
    check(math.isclose(s['comparison_median_seconds'],statistics.median(t['source_comparison_seconds'] for t in result['timing']),abs_tol=1e-12),'summary_timing_arithmetic')
    check(s['new_training']==s['new_model_inference']==s['whole_sign_verdict_count']==s['linguistic_tolerance_claim_count']==0 and s['educational_accuracy'] is None,'summary_no_new_inference_or_semantic_accuracy')
    report=dict(status='PASS' if not fail else 'FAIL',reviewer='Codex rgb_labels_b; separate from root implementation, not independent human truth',checks=checks,failures=fail,imports='Standard library and NumPy only; no engine/run/feedback/torch import',new_model_inference=0,training=0,video_decode=0,fit_clips=36,development_queries=12,held_or_fresh_numeric_slices_accessed=0,executable_clauses=clauses_executable,source_anchor_cells=all_anchor_cells,clause_statuses=dict(clause_statuses),order_statuses=dict(order_statuses),raw_candidates=raw_count,outside_source_contract_candidates=outside_count,descriptive_draft_audits=diagnostics,protected_hashes_verified=len(frozen['protected']),pins={str(U/p):sha(U/p) for p in ['FROZEN.json','RESULT.json','engine.py','run.py','source_audit/SOURCE_CONTRACTS.json']},verifier_sha256=sha(__file__),limitations=['Known-positive reference anchors are sparse physical support, not all semantic stages or complete DTW coverage.','The order predicate checks an existential original-time witness, not exclusive ordering or exact repetitions.','Saved RGB draft agreement is not pedagogical accuracy; all linguistic/whole-sign/tolerance verdicts remain withheld.'])
    with (U/'review'/output).open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=report['status'],checks=checks,failures=len(fail)),ensure_ascii=False))
    return int(bool(fail))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='SAVED_RESULT_VERIFICATION.json');args=parser.parse_args();raise SystemExit(main(args.output))
