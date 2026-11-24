"""Source-linked physical differences, not an unrestricted KSL correctness judge.

The observer is immutable and target blind. Source contracts select which
already observed features to compare; they cannot change observations.
"""
import copy, hashlib, json
import numpy as np

CONFIDENCE = .8  # existing frozen observer confidence, no outcome-based tuning
MIN_RUN_FRAMES = 2
MIN_REFERENCES = 2

def feature_support(array, names, targets, old_support):
    support=np.array(old_support,copy=True,dtype=bool)
    for j,target in enumerate(targets):
        if target.endswith(('index_only','open_four')):
            side=target[0]
            required=[names.index(f'{side}_{finger}_{joint}_angle') for finger in ['index','middle','ring','little'] for joint in ['mcp','pip','dip']]
            support[:,j]&=array['mask'][:,required].all(1)
    return support

def classify_probability(p, usable, expected=1):
    if not usable or not np.isfinite(p) or p<0 or p>1:return 'UNOBSERVED'
    if expected not in [0,1]:raise ValueError('Explicit binary expected state required')
    certainty=p if expected else 1-p
    if certainty>=CONFIDENCE:return 'SUPPORTED_STATE'
    if certainty<=1-CONFIDENCE:return 'OPPOSITE_STATE'
    return 'UNCERTAIN_MODEL'

def safe_probability(p):
    return float(p) if np.isfinite(p) and 0<=p<=1 else None

def select_anchor_indices(row, target_index, y, mask):
    ts=np.asarray(row['timestamps_us'])/1e6
    active=np.zeros(len(ts),bool)
    for start,end in row['intervals']:active|=(ts>=start)&(ts<=end)
    # Draft RGB positive labels define reference anchors, never model predictions.
    return np.flatnonzero(active & mask[:,target_index] & (y[:,target_index]==1))

def compare_clause(clause, refs, query, targets):
    """refs retain every source anchor, even unobserved/unmappable anchors."""
    base=dict(id=clause['id'],feature=clause.get('feature'),source_span=clause.get('source_span'),source_url=clause.get('source_url'),link_status=clause['link_status'],educational_verdict=None,linguistic_tolerance=None)
    if clause['link_status'] not in ['partial_supported','proxy_only'] or clause.get('feature') not in targets:
        return dict(base,status='SOURCE_LINK_WITHHELD',references=[],reason=clause.get('rationale','Source-feature relation not established'))
    j=targets.index(clause['feature']);expected=clause.get('expected',1)
    if expected!=1:raise ValueError('Current source anchors only establish positive attributes; 0 is not fist')
    out=[]
    for ref in refs:
        anchors=ref['anchors'].get(clause['feature'],[]);cells=[]
        for k in anchors:
            rs=ref['sample_indices'];ts=ref['times_us'];qts=query['times_us']
            cell=dict(reference_frame=int(ref['frame_ids'][k]),reference_seconds=float(ts[k]/1e6))
            if not len(rs) or k<min(rs) or k>max(rs):
                cells.append(dict(cell,state='ALIGNMENT_UNAVAILABLE',query_frame=None,query_seconds=None));continue
            sample=int(np.argmin(np.abs(ts[rs]-ts[k])))
            # Reject an anchor farther than one original frame from saved DTW support.
            error=abs(int(ts[rs[sample]])-int(ts[k]));period=int(np.median(np.diff(ts)))
            if error>period+1:
                cells.append(dict(cell,state='ALIGNMENT_UNAVAILABLE',query_frame=None,query_seconds=None));continue
            qi=int(query['sample_indices'][ref['reference_to_query'][sample]])
            state=classify_probability(query['prob'][qi,j],query['support'][qi,j],expected)
            if not ref['support'][k,j]:state='REFERENCE_UNOBSERVED'
            elif classify_probability(ref['prob'][k,j],True,expected)=='OPPOSITE_STATE':state='REFERENCE_OBSERVER_CONFLICT'
            # Uncertain reference model need not erase the visible draft anchor;
            # its RGB provisional label and observation status stay explicit.
            cells.append(dict(cell,query_frame=int(query['frame_ids'][qi]),query_seconds=float(qts[qi]/1e6),query_index=qi,state=state,query_probability=safe_probability(query['prob'][qi,j]),reference_probability=safe_probability(ref['prob'][k,j]),alignment_quantization_us=error))
        counts={state:sum(x['state']==state for x in cells) for state in ['SUPPORTED_STATE','OPPOSITE_STATE','UNOBSERVED','UNCERTAIN_MODEL','ALIGNMENT_UNAVAILABLE','REFERENCE_UNOBSERVED','REFERENCE_OBSERVER_CONFLICT']}
        out.append(dict(reference=ref['alias'],expected_anchor_count=len(cells),counts=counts,cells=cells))
    supported_refs=[r['reference'] for r in out if r['counts']['SUPPORTED_STATE']>0]
    opposite_refs=[r['reference'] for r in out if r['counts']['OPPOSITE_STATE']>0]
    votes={}
    for r in out:
        for cell in r['cells']:
            if cell['state'] in ['SUPPORTED_STATE','OPPOSITE_STATE']:
                votes.setdefault((cell['query_frame'],cell['state']),set()).add(r['reference'])
    consensus=[dict(query_frame=k[0],state=k[1],references=sorted(v)) for k,v in votes.items() if len(v)>=MIN_REFERENCES]
    # Separate times are not collapsed into a whole-sign pass/fail.
    if any(v['state']=='OPPOSITE_STATE' for v in consensus):status='GUIDE_RELEVANT_DIFFERENCE_CANDIDATE'
    elif any(v['state']=='SUPPORTED_STATE' for v in consensus):status='DESCRIBED_FEATURE_OBSERVED'
    else:status='INSUFFICIENT_LINKED_EVIDENCE'
    return dict(base,status=status,references=out,supported_references=supported_refs,opposite_references=opposite_refs,coherent_query_frame_consensus=consensus,
                expected_anchor_count=sum(r['expected_anchor_count'] for r in out),
                observed_anchor_count=sum(r['counts']['SUPPORTED_STATE']+r['counts']['OPPOSITE_STATE'] for r in out),
                note='Counts are source-linked draft-reference anchor cells. Neither a whole-sign score nor an omission verdict.')

def positive_runs(obs, target, targets):
    if target not in targets:return []
    j=targets.index(target);t=np.asarray(obs['times_us']);ids=np.asarray(obs['frame_ids']);p=obs['prob'][:,j]
    active=np.zeros(len(t),bool)
    for a,b in obs['intervals']:active|=(t>=a*1e6)&(t<=b*1e6)
    good=active & obs['support'][:,j] & np.isfinite(p) & (p>=CONFIDENCE) & (p<=1)
    period=np.median(np.diff(t));runs=[];start=None
    for i in range(len(t)+1):
        continuity=i<len(t) and (i==0 or (ids[i]-ids[i-1]==1 and t[i]-t[i-1]<=2*period))
        if i<len(t) and good[i] and (start is None or continuity):
            if start is None:start=i
        else:
            if start is not None and i-start>=MIN_RUN_FRAMES:runs.append(dict(start_index=start,end_index=i-1,start_seconds=float(t[start]/1e6),end_seconds=float(t[i-1]/1e6),frames=int(i-start)))
            start=i if i<len(t) and good[i] else None
    return runs

def check_order(order, obs, targets):
    base=dict(id=order['id'],before=order['before'],after=order['after'],source_span=order.get('source_span'),source_url=order.get('source_url'),link_status=order['link_status'],educational_verdict=None,mandatory_omission_verdict=None)
    if order['link_status'] not in ['partial_supported','proxy_only'] or order['before'] not in targets or order['after'] not in targets:
        return dict(base,status='ORDER_NOT_OBSERVABLE',before_events=[],after_events=[])
    before=positive_runs(obs,order['before'],targets);after=positive_runs(obs,order['after'],targets)
    forward=[(a,b) for a in before for b in after if a['end_index']<b['start_index']]
    reverse=[(b,a) for a in before for b in after if b['end_index']<a['start_index']]
    if forward:status='DESCRIBED_ORDER_PATTERN_OBSERVED'
    elif reverse:status='REVERSE_ORDER_CANDIDATE'
    else:status='ORDER_UNCONFIRMED'
    witness=None if not forward else dict(before=forward[0][0],after=forward[0][1])
    return dict(base,status=status,before_events=before,after_events=after,witness=witness,
                note='Uses original time order, never DTW-reordered time. Partial wrist-motion proxy, not full semantic phase or exact repetition count.')

def explanation(item):
    name=item.get('feature') or (item.get('before','')+' → '+item.get('after',''))
    return {
      'GUIDE_RELEVANT_DIFFERENCE_CANDIDATE':f'설명과 연결한 {name} 항목에서 여러 참고와 다른 관측이 나왔습니다. 해당 장면 확인이 필요합니다.',
      'DESCRIBED_FEATURE_OBSERVED':f'설명과 연결한 {name} 특징이 대응 장면에서 관측됐습니다. 전체 표현의 정답 판정은 아닙니다.',
      'INSUFFICIENT_LINKED_EVIDENCE':f'{name} 항목을 설명과 비교할 근거가 부족합니다.',
      'SOURCE_LINK_WITHHELD':f'{name} 항목은 설명과 실제 표현형 또는 관측 기능의 연결이 확인되지 않았습니다.',
      'DESCRIBED_ORDER_PATTERN_OBSERVED':f'{name}의 순서가 관측됐습니다. 설명에 나온 동작의 일부만 확인한 결과입니다.',
      'REVERSE_ORDER_CANDIDATE':f'{name}과 반대 순서의 관측이 있습니다. 동작 구간과 설명 연결을 확인해야 합니다.',
      'ORDER_UNCONFIRMED':f'{name}의 순서를 확인할 관측이 충분하지 않습니다. 동작 생략으로 단정하지 않습니다.',
      'ORDER_NOT_OBSERVABLE':f'{name}의 순서는 현재 관측 기능으로 판단하지 않습니다.'}[item['status']]

def run_expression(contract, refs, query, targets):
    clauses=[compare_clause(c,refs,query,targets) for c in contract['clauses']]
    orders=[check_order(o,query,targets) for o in contract.get('orders',[])]
    for item in clauses+orders:item['message_ko']=explanation(item)
    return dict(word_id=contract['word_id'],label=contract['label'],query=query['alias'],clauses=clauses,orders=orders,
      unsupported_claims=contract.get('unsupported_claims',[]),
      whole_sign_verdict=None,linguistically_acceptable_variation=None,
      scope='Frozen learned physical observer + declarative source-linked comparison. Guide affects comparison targets only. No newly trained semantic model.')
