"""New provenance diagnostics on saved outputs, not a new accuracy benchmark."""
from pathlib import Path
import json,hashlib,collections
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
E=O/'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/full_bag_rate_rank_learning/evaluation'
P=O/'word100_action_linkage_repair_20261113'
comparison=read(E/'COMPARISON.json');mapping=read(P/'SPEED_TRUTH_TRANSPORT.json')
assert mapping['original_comparison_sha256']==sha(E/'COMPARISON.json')
condition=read(U/'PLAN.json')['fixed_candidate']
normals={(r['seed'],r['clip_id']):r for r in comparison['rows'] if r['kind']=='full_normal'}
mapped={(r['clip_id'],r['kind'],r['card']):r for r in mapping['mapped_cards']}
details=[];counts=collections.Counter()
for row in comparison['rows']:
    if row['kind'] not in ('full_normal','speed075','speed125'):continue
    base=normals[row['seed'],row['clip_id']]['conditions'][condition]['cards']
    bc={c['card']:c for c in base}
    cards=[]
    for card in row['conditions'][condition]['cards']:
        span=card.get('chosen',{}).get('span') if card.get('chosen') else None
        w=bc[card['card']].get('truth',{}).get('witness',[])
        own=[];foreign=[];dropped=[]
        for other in base:
            points=other.get('truth',{}).get('witness',[])
            if row['kind']=='full_normal':entries=[dict(source_index=k,query_indices=[k]) for k in points]
            else:entries=mapped.get((row['clip_id'],row['kind'],other['card']),{}).get('witnesses',[])
            for entry in entries:
                item=dict(card=other['card'],source_index=entry['source_index'],query_indices=entry['query_indices'])
                item['inside']=bool(span and any(span[0]<=i<=span[1] for i in entry['query_indices']))
                if not entry['query_indices']:dropped.append(item)
                elif other['card']==card['card']:own.append(item)
                else:foreign.append(item)
        if not w:state='no_direct_identity_truth'
        elif not own:state='all_own_witnesses_resampled_out'
        elif any(e['inside'] for e in foreign):state='other_action_evidence_in_selected_interval'
        elif all(e['inside'] for e in own):state='retained_identity_witnesses_preserved'
        else:state='some_retained_identity_witnesses_outside'
        counts[(row['kind'],state)]+=1
        cards.append(dict(card=card['card'],span=span,identity_state=state,own=own,other=foreign,dropped=dropped,
            source_extent_complete=None,linguistic_completion=None))
    orders=row.get('pts_order',{}).get(condition,[])
    overlaps=[o for o in orders if o['state']=='ambiguous_overlap']
    details.append(dict(seed=row['seed'],clip_id=row['clip_id'],kind=row['kind'],cards=cards,extent_order=orders,
        overlap_with_clean_retained_identity=bool(overlaps and all(c['identity_state']=='retained_identity_witnesses_preserved' for c in cards))))
out=dict(status='SAVED_SOURCE_IDENTITY_DIAGNOSTIC_COMPLETE',condition=condition,
    source_files={str(p):sha(p) for p in [U/'PLAN.json',E/'COMPARISON.json',P/'SPEED_TRUTH_TRANSPORT.json']},
    rows=details,summary=[dict(kind=k[0],state=k[1],cards=v) for k,v in sorted(counts.items())],
    overlap_cases=[r for r in details if any(o['state']=='ambiguous_overlap' for o in r['extent_order'])],
    does_not_change_saved_scores=True,does_not_waive_G2=True,new_model_inference=False,
    limitation='Sparse retained witnesses diagnose source identity only. They do not certify full boundaries or natural-user completion.')
with (U/'LINKAGE_DIAGNOSTIC.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'summary':out['summary'],'overlap_cases':out['overlap_cases']},ensure_ascii=False))
