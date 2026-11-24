"""Append-only context verification; reuse independent scalar audit helpers."""
from pathlib import Path
import datetime
import json
from verify_cost_components import U, OPS, Verification, verify_cells, verify_parent_rows


def main():
    v=Verification()
    initial=v.read(U/'review/COST_COMPONENT_VERIFICATION.json')
    v.check('initial_arithmetic_pass',initial['status']=='PASS' and not initial['failures'])
    for p,h in initial['source_pins'].items():v.check('initial_inputs_still_pinned',v.pin(p)==h,p)
    prior=v.read(U/'COST_COMPONENTS.json')
    scope=v.read(U/'CONTEXT_AUDIT_SCOPE.json')
    d=v.read(U/'COST_CONTEXT_SUPPLEMENT.json')
    v.check('context_scope_pin',d['scope_sha256']==v.pin(U/'CONTEXT_AUDIT_SCOPE.json'))
    v.check('context_parent_scope_pin',scope['primary_scope_sha256']==v.pin(U/'COST_AUDIT_SCOPE.json'))
    v.check('context_preserved_initial_pin',scope['component_sha256']==v.pin(U/'COST_COMPONENTS.json'))
    v.check('context_code_pin',scope['code_sha256']==v.pin(U/'audit_context.py'))
    inp=v.read(OPS/'word100_phase_correspondence_20261101/INPUTS.json')
    for alias,row in inp['inputs'].items():v.array(alias,row['original'])
    for key,old in prior['cells'].items():v.check('initial_cell_preserved_exactly',d['cells'].get(key)==old,key)
    extra={k:x for k,x in d['cells'].items() if k not in prior['cells']}
    verify_cells(v,{'cells':extra})
    verify_parent_rows(v,d,inp,whole_counterparts=True)
    v.check('all_existing_query_scope_preserved',
            [(p['parent_id'],[r['q'] for r in p['links']]) for p in d['parents']]==
            [(p['parent_id'],[r['q'] for r in p['links']]) for p in prior['parents']])
    v.check('motivating_outside_counterpart_retained','P066_68_P063_55' in d['cells'])
    v.check('motivating_inside_counterpart_retained','P066_68_P063_58' in d['cells'])
    examples=[]
    for key in ['P066_68_P063_58','P066_68_P063_55']:
        c=d['cells'][key]
        examples.append(dict(key=key,**{k:c[k] for k in ['geometry11','direction2','full13','query_motion','reference_motion']}))
    result=dict(status='PASS' if not v.failures else 'FAIL',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                checks=dict(v.checks),check_total=sum(v.checks.values()),failures=v.failures,
                initial_cells_preserved=len(prior['cells']),new_context_cells=len(extra),total_unique_cells=len(d['cells']),
                parents=len(d['parents']),local_ordering_flip_parent_rows=sum(x['direction_changes_local_ordering'] is True for p in d['parents'] for x in p['links']),
                maximum_float64_cost_error=v.maximum_cost_error,examples=examples,
                original_verification={'path':str(U/'review/COST_COMPONENT_VERIFICATION.json'),'sha256':v.pin(U/'review/COST_COMPONENT_VERIFICATION.json'),'checks':initial['check_total']},
                source_pins=v.pins,verifier_sha256=v.pin(Path(__file__)),
                method='Verify prior872 cells unchanged by exact JSON equality and pinned independent audit; independently derive95 new cells and recount all saved B/C counterpart links for the same parent query-frame scopes.',
                limits=['The same query frame may occur under more than one parent; local flip counts are parent-row occurrences, not unique physical errors.',
                        'This compares the existing C13 objective on local saved links; it is not a rederived B objective and not global-path causal attribution.',
                        'No correct occurrence, educational error, tiny-jitter label or direction suppression threshold is established.'],
                new_paths=0,new_inference=0,old_scores_changed=False,global_causation_proven=False)
    out=U/'review/COST_CONTEXT_VERIFICATION.json'
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['status','check_total','initial_cells_preserved','new_context_cells','total_unique_cells','parents','local_ordering_flip_parent_rows']}))


if __name__=='__main__':main()
