"""Describe fixed saved results; no inference, tuning, or old rescoring."""
from verify_bridge import *

def main():
    verified=read(V/'VERIFIED.json');result={}
    for tag in ['role_only','content_connected']:
        r=read(OUT/'evaluations'/f'{tag}.json')
        normal=[s for s in r['sequences'] if s['kind'] in NORMAL]
        counts=Counter(s['status'] for s in normal)
        cases=[]
        for s in normal:
            points=[p for p in r['points'] if p['key']==s['key'] and p['y']<8]
            losses=[p for p in points if p['pred']==p['y'] and not p['eligible']]
            cases.append(dict(key=s['key'],word=s['word'],kind=s['kind'],status=s['status'],
                known=len(points),raw_wrong=[p['frame'] for p in points if p['pred']!=p['y']],
                correct_excluded=[p['frame'] for p in losses],
                missing_main_families=[w['group'] for w in s['role_main_witnesses'] if not w['observed']],
                unexpected_frames=s['unexpected_action_frames'],
                uncertain_old_core=s['previous_reviewed_core_unconfirmed'],
                role_rest_conflicts=s['rest_action_conflict_frames'],
                order_evidence={k:s[k] for k in ['states','conflicts','short_candidates','missing_groups'] if k in s}))
        reverse=[dict(key=s['key'],status=s['status'],missing=[w['group'] for w in s['role_main_witnesses'] if not w['observed']])
            for s in r['sequences'] if s['kind']=='actual_reverse_main' and s['precedences']]
        result[tag]=dict(normal_status_counts=dict(counts),normal_cases=cases,short_reverse_cases=reverse)
    lossrows=read(OUT/'evaluations/content_connected.json')['main_eligibility_losses']
    unique=sorted({(p['clip_id'],p['source_frame']) for p in lossrows})
    save(V/'CAUSE_AUDIT.json',dict(status=verified['status'],conditions=result,
        suppressed_known_main_source_points=unique,
        scope='Existing exposed DEV diagnostics, not independent final validation. No thresholds changed, no new inference or source labels.',
        next_hypothesis_not_proven=True))
    print({k:v['normal_status_counts'] for k,v in result.items()})
    print('Unique source main exclusions',unique)

if __name__=='__main__':main()
