"""Classify stored correspondence failures; no new model/scoring/selection."""
from pathlib import Path
import json,hashlib,collections
U=Path(__file__).resolve().parent
P=U.parent/'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/full_bag_rate_rank_learning/evaluation/COMPARISON.json'
j=json.loads(P.read_text('utf-8'));groups=collections.defaultdict(collections.Counter);details=[]
condition='full_bag_rate_rank' # The already completed latest rank condition, not score selection.
for r in j['rows']:
    q=r['conditions'][condition];g=groups[(r['seed'],r['kind'])];g['scenarios']+=1;cards=[]
    for c in q['cards']:
        t=c.get('truth') or {};w=t.get('witness') or [];other=t.get('wrong_witness') or []
        chosen=c.get('chosen');span=chosen.get('span') if chosen else None
        included=lambda k:bool(span and span[0]<=k<=span[1])
        hits=sum(included(k) for k in w);wrong=sum(included(k) for k in other)
        if w:
            assert t['all_witness']==(hits==len(w))
            assert t['wrong_points']==wrong
            state=('missing_and_mixed' if wrong else 'missing_direct_points') if hits<len(w) else 'all_direct_points_but_mixed' if wrong else 'direct_points_preserved'
            g['cards_with_direct_points']+=1;g[state]+=1
        else:state='no_direct_truth_in_saved_case';g[state]+=1
        cards.append(dict(card=c['card'],family=c['family'],span=span,state=state,own_points=w,other_points=other,hits=hits,wrong_points=wrong,semantic_completion=None))
    orders=r.get('pts_order',{}).get(condition,[])
    for order in orders:
        g['order_'+order['state']]+=1
        if order['state']=='ambiguous_overlap':
            relevant=[c for c in cards if c['card'] in order['cards']]
            if any(c['state']=='no_direct_truth_in_saved_case' for c in relevant):g['overlap_without_direct_truth']+=1
            else:g['overlap_with_clean_direct_points' if all(c['state']=='direct_points_preserved' for c in relevant) else 'overlap_with_binding_issue']+=1
    if any(c['state'] not in ['direct_points_preserved','no_direct_truth_in_saved_case'] for c in cards) or any(o['state']=='ambiguous_overlap' for o in orders):
        details.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],cards=cards,orders=orders))
summary=[dict(seed=k[0],kind=k[1],**v) for k,v in sorted(groups.items())]
out=dict(source=str(P),sha256=hashlib.sha256(P.read_bytes()).hexdigest(),condition=condition,
    scenarios=82,seeds=3,rows=246,summary=summary,diagnostics=details,
    interpretation='Direct source-role witnesses, extent overlap and semantic completion are separate. Source-role identity after synthetic reversal is not a whole-action correctness label.',
    new_scoring=False,new_inference=False,gate_waived=False,web_candidate_adopted=False)
out['correction']='First audit called absent stored truth no_surviving_direct_points and mixed it with binding issues. Empty truth proves neither absence nor model failure. Original audit preserved; this version separates no truth.'
with (U/'BINDING_CAUSE_AUDIT_V2.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(normal=[r for r in summary if r['kind'] in ['full_normal','speed075','speed125']],non_normal_diagnostics=len([r for r in details if r['kind'] not in ['full_normal','speed075','speed125']])),ensure_ascii=False))
