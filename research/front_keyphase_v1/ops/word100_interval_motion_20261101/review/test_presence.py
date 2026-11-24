"""Small semantic synthetic suite; never reads actual pilot labels."""
from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
W = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('presence_under_test', W / 'audit_presence.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    sources, rows = [], []
    for i in range(24):
        sid = f'SYN{i:02d}'
        sources.append(dict(sample_id=sid, alias=f'P{i+1:03d}', clip_id=f'clip{i}', person=i % 3 + 1,
                            word_id=f'word{i % 15}', role='fit', strata=['hash_ordinary'] if i >= 12 else ['challenge'],
                            frame_ids=list(range(100, 113)), timestamps_us=list(range(1000, 1013)), center_frame=106,
                            center_timestamp_us=1006, image='synthetic.jpg', image_sha256='synthetic'))
        values = {'down': int(i % 4 in (1, 3)), 'hold': int(i % 4 in (2, 3))}
        segments = {'down': [dict(first_frame=101, last_frame=102, in_roi=True, note='Direct synthetic downward step')] if values['down'] else [],
                    'hold': [dict(first_frame=106, last_frame=108, in_roi=True, note='Direct synthetic maintained positions')] if values['hold'] else []}
        rows.append(dict(sample_id=sid, alias=f'P{i+1:03d}', presence={h: deepcopy(values) for h in 'RL'},
                         segments={h: deepcopy(segments) for h in 'RL'}, full13_position_observable=dict(R=True, L=True),
                         full13_roi=dict(R=True, L=True), unknown_intervals=[], events=[], notes='Synthetic fixture',
                         actual_image_viewed=True, image_sha256='synthetic'))
    return dict(rows=sources), [dict(status='COMPLETE', rows=deepcopy(rows)) for _ in range(2)]


def unknown(first=109, last=111, component='translation', predicates=('down',), hand='R'):
    return dict(first_frame=first, last_frame=last, component=component, affected_predicates=list(predicates), hand=hand, reason='Synthetic uncertainty')


def set_value(row, hand, predicate, value):
    row['presence'][hand][predicate] = value
    row['segments'][hand][predicate] = ([] if value != 1 else
        [dict(first_frame=101 if predicate == 'down' else 106, last_frame=102 if predicate == 'down' else 108,
              in_roi=True, note='Synthetic direct evidence')])


def main():
    gates = audit.read(W / 'PILOT_GATES.json')
    outcomes = []
    def case(name, change=None, expectation=None, rejects=False):
        manifest, reviews = fixture()
        if change:
            change(reviews)
        before = deepcopy((manifest, reviews))
        try:
            result = audit.calculate(manifest, reviews, gates)
            passed = not rejects and (expectation(result) if expectation else all(result['checks'].values()))
            passed = passed and before == (manifest, reviews)
            outcomes.append(dict(name=name, passed=bool(passed), outcome=result['status']))
        except ValueError as error:
            outcomes.append(dict(name=name, passed=rejects, outcome='REJECTED_INVALID_SCHEMA', detail=str(error)))
    case('Valid fixtures include 11 coexistence, all raw denominators and both classes', expectation=lambda r: all(r['checks'].values()) and r['all_predicate_cells']==96 and r['rows'][3]['predicates']['R_down']['model_label']==r['rows'][3]['predicates']['R_hold']['model_label']==1)
    def nulls(reviews, hand=None):
        for review in reviews:
            for row in review['rows']:
                for h in ([hand] if hand else 'RL'):
                    for p in ('down','hold'): set_value(row,h,p,None)
    case('All unknown cannot become negative or pass coverage',lambda r:nulls(r),lambda r: all(m['both_known']==0 and m['known_conflict_rate'] is None for m in r['metrics'].values()) and not r['checks']['R_two_bit_coverage'] and all(c['model_label'] is None for row in r['rows'] for c in row['predicates'].values()))
    case('One known hand cannot hide the other unknown hand',lambda r:nulls(r,'L'),lambda r:not r['checks']['L_two_bit_coverage'] and not r['checks']['L_down_known_coverage'])
    def partial_negative(reviews):
        for review in reviews:review['rows'][0]['full13_position_observable']['R']=False
    case('Negative needs full13 visible position and identity',partial_negative,lambda r:not r['checks']['annotation_contract'] and r['rows'][0]['predicates']['R_down']['model_label'] is None)
    def predicate_unknown(reviews):
        for review in reviews:
            row=review['rows'][0];set_value(row,'R','hold',None);row['unknown_intervals']=[unknown(predicates=('hold',))]
    case('Hold uncertainty does not erase a separately justified down absence',predicate_unknown,lambda r:r['checks']['annotation_contract'] and r['rows'][0]['predicates']['R_down']['model_label']==0 and r['rows'][0]['predicates']['R_hold']['model_label'] is None)
    def elsewhere(reviews):
        for review in reviews:review['rows'][1]['unknown_intervals']=[unknown()]
    case('Unknown elsewhere does not erase a direct positive witness',elsewhere,lambda r:r['checks']['annotation_contract'] and r['rows'][1]['predicates']['R_down']['model_label']==1)
    def overlap_unknown(reviews):
        for review in reviews:review['rows'][1]['unknown_intervals']=[unknown(101,102)]
    case('Relevant unknown cannot overlap positive evidence',overlap_unknown,lambda r:not r['checks']['annotation_contract'] and r['rows'][1]['predicates']['R_down']['model_label'] is None)
    def identity(reviews):
        for review in reviews:
            row=review['rows'][0];row['full13_position_observable']['R']=False;row['unknown_intervals']=[unknown(component='identity',predicates=())]
    case('Identity uncertainty blocks both negatives even with empty predicate hint',identity,lambda r:sum(i['reason']=='negative_with_relevant_unknown' for i in r['contract_failures'])==4)
    def orientation(reviews):
        for review in reviews:review['rows'][1]['unknown_intervals']=[unknown(101,102,component='orientation',predicates=())]
    case('Orientation uncertainty alone does not erase direct translation',orientation,lambda r:r['checks']['annotation_contract'] and r['rows'][1]['predicates']['R_down']['model_label']==1)
    def short(reviews):
        for review in reviews:
            row=review['rows'][3];row['segments']['R']['down'][0]['last_frame']=101;row['segments']['R']['hold'][0]['last_frame']=107
    case('Single-frame down and two-frame hold lack required witnesses',short,lambda r:sum(i['reason']=='positive_segment_too_short' for i in r['contract_failures'])==4)
    def outside(reviews):
        for review in reviews:
            for row in review['rows']:
                for h in 'RL':
                    row['full13_roi'][h]=False
                    for p in ('down','hold'):
                        for s in row['segments'][h][p]:s['in_roi']=False
    case('Physical agreement outside ROI cannot pass usable-model gates',outside,lambda r:r['checks']['annotation_contract'] and all(m['both_known']==24 and m['eligible_agreed_known']==0 for m in r['metrics'].values()) and not r['checks']['R_down_eligible_coverage'])
    def endpoint(reviews, first):
        for review in reviews:review['rows'][3]['segments']['R']['hold'][0].update(first_frame=first,last_frame=first+2)
    case('Down and hold may share one observed endpoint',lambda r:endpoint(r,102),lambda r:r['checks']['annotation_contract'])
    case('Down and hold cannot claim the same two observed frames',lambda r:endpoint(r,101),lambda r:not r['checks']['annotation_contract'])
    def diluted(reviews):
        set_value(reviews[1]['rows'][0],'R','down',1);set_value(reviews[1]['rows'][1],'R','hold',1)
    case('Two-bit gate catches disagreements diluted across predicates',diluted,lambda r:r['checks']['R_down_known_conflict'] and r['checks']['R_hold_known_conflict'] and not r['checks']['R_two_bit_conflict'] and r['pattern_metrics']['R']['conflicts']==2)
    def five_unknown(reviews):
        for row in reviews[1]['rows'][:5]:set_value(row,'R','down',None)
    case('Known-unknown rate retains original24 denominator',five_unknown,lambda r:r['metrics']['R_down']['known_unknown_mismatch_rate']==5/24 and not r['checks']['R_down_unknown_mismatch'])
    case('Fractional native frames rejected',lambda r:r[0]['rows'][1]['segments']['R']['down'][0].update(first_frame=101.5),rejects=True)
    case('Boolean presence rejected rather than treated as positive',lambda r:r[0]['rows'][1]['presence']['R'].update(down=True),rejects=True)
    output=dict(status='PASS_SYNTHETIC_PRESENCE_SEMANTICS' if all(o['passed'] for o in outcomes) else 'FAIL_SYNTHETIC_PRESENCE_SEMANTICS',
                passed=sum(o['passed'] for o in outcomes),failed=sum(not o['passed'] for o in outcomes),checks=outcomes,
                source_pins={str(p):audit.sha(p) for p in (W/'audit_presence.py',W/'PILOT_GATES.json',Path(__file__))},
                actual_pilot_labels_read=False,training_runs=0,inference_runs=0,raw_inputs_mutated=False)
    audit.write_new(W/'review/PRESENCE_SELFTEST.json',output)
    print(json.dumps({k:output[k] for k in ('status','passed','failed')}))


if __name__=='__main__':main()
