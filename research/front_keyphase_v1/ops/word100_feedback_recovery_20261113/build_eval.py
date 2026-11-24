from common import *
s=(PREV/'evaluate.py').read_text('utf-8-sig')
a=s.index('    gates=[]\n');z=s.index("    write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFICATION'",a)
s=s[:a]+'''    gates=[]
    for condition in cfg['conditions']:
      for seed in cfg['seeds']:
        c=out[f'{condition}_{seed}'];a=read(PREV/'scores'/f'A_{seed}.json')
        cn=c['finger_by_variant']['normal'];an=a['finger_by_variant']['normal']
        improvement=cn['macro_f1']>=an['macro_f1']+.01
        hands=all(c['finger_by_hand'][s]['accuracy']>=a['finger_by_hand'][s]['accuracy'] for s in ['L','R'])
        states=all((cv or 0)>=(av or 0) for av,cv in zip(an['state_f1'],cn['state_f1']))
        # Fixed prior development denominators are kept, even when observation is absent.
        variants=[]
        for v in FINE_VARIANTS:
            cv=c['finger_by_variant'][v];av=a['finger_by_variant'][v]
            assert cv['total']==av['total']
            if cv['correct']<av['correct'] or any((x or 0)<(y or 0) for x,y in zip(cv['state_f1'],av['state_f1'])):
                variants.append(dict(variant=v,before=av,after=cv))
        ret=next(r for r in retention_rows if r['condition']==condition and r['seed']==seed)
        coverage=cn['withheld']<=an['withheld']
        passed=improvement and hands and states and coverage and ret['full_preservation'] and not variants
        gates.append(dict(condition=condition,seed=seed,consistent_fine_gain=improvement,hands_preserved=hands,states_preserved=states,normal_abstention_not_increased=coverage,variant_failures=variants,legacy_preserved=ret['full_preservation'],pass_candidate=bool(passed)))
    condition_pass={c:all(r['pass_candidate'] for r in gates if r['condition']==c) for c in cfg['conditions']}
    write(U/'EVALUATION_COMPLETE.json',dict(status='DEVELOPMENT_COMPARISON_COMPLETE',models=out,retention_all_conditions=retention_rows,gates=gates,condition_pass=condition_pass,any_pass=any(condition_pass.values()),no_production_promotion=True,old_baseline_from_stored_scores=True,development_historically_exposed=True))
''' +s[z:]
s=s.replace("'EVALUATED_PENDING_VERIFICATION'","'SHAPE_EVALUATED_PENDING_VERIFICATION'")
(U/'evaluate.py').write_text(s,'utf-8')
