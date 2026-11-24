"""Preserve ordered visible events without manufacturing missing frame truth.

This is an evidence adapter, not a detector, new label adjudicator or learning
readiness gate. Original labels, observations and model scores are untouched.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
sys.dont_write_bytecode = True

U = Path(__file__).resolve().parent
COMPONENTS = ('translation', 'orientation', 'shape')
VALUES = {
    'translation': {'hold', 'up', 'down', 'horizontal', 'turn', 'mixed', 'unresolved'},
    'orientation': {'rotation', 'change', 'stable', 'unresolved'},
    'shape': {'change', 'stable', 'unresolved'},
}

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    with Path(p).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)

def encode(row, side, events):
    assert side in ('R', 'L')
    frames, times = row['frame_ids'], row['timestamps_us']
    assert len(frames) == len(times) == 13 and len(set(frames)) == len(frames)
    assert all(type(f) is int for f in frames) and all(type(t) is int for t in times)
    assert all(b == a + 1 for a, b in zip(frames, frames[1:]))
    assert all(b > a for a, b in zip(times, times[1:]))
    if 'center_frame' in row:
        assert row['center_frame'] == frames[6]
    if 'center_timestamp_us' in row:
        assert row['center_timestamp_us'] == times[6]
    seen = set(frames)
    cells = [{c: {'direct_values': set(), 'uncertain': False, 'event_indices': []}
              for c in COMPONENTS} for _ in frames]
    for eidx, event in enumerate(events):
        c, v, evidence = event['component'], event['value'], event['evidence']
        assert c in COMPONENTS and v in VALUES[c]
        assert evidence in ('direct', 'context', 'uncertain')
        lo, hi = event['frame_start'], event['frame_end']
        assert isinstance(lo, int) and isinstance(hi, int)
        assert lo <= hi and set(range(lo, hi + 1)) <= seen
        boundary = event.get('boundary_range')
        if boundary is not None:
            assert len(boundary) == 2 and boundary[0] <= boundary[1]
            assert set(range(boundary[0], boundary[1]+1)) <= seen
        for i, frame in enumerate(frames):
            cell = cells[i][c]
            if lo <= frame <= hi:
                cell['event_indices'].append(eidx)
                if evidence == 'direct' and v != 'unresolved':
                    cell['direct_values'].add(v)
                if evidence == 'uncertain' or v == 'unresolved':
                    cell['uncertain'] = True
            if boundary is not None and boundary[0] <= frame <= boundary[1]:
                cell['uncertain'] = True
                if eidx not in cell['event_indices']:
                    cell['event_indices'].append(eidx)
    timeline = []
    for frame, timestamp, channels in zip(frames, times, cells):
        entry = {'frame': frame, 'timestamp_us': timestamp, 'components': {}}
        for c, cell in channels.items():
            values = sorted(cell['direct_values'])
            known = len(values) == 1 and not cell['uncertain']
            reason = ('direct_scoped_event' if known else
                      'uncertain_boundary_or_event' if cell['uncertain'] else
                      'conflicting_direct_events' if len(values) > 1 else
                      'no_direct_event')
            entry['components'][c] = dict(value=values[0] if known else None,
                direct=known, reason=reason, source_event_indices=cell['event_indices'])
        timeline.append(entry)
    runs = {c: [] for c in COMPONENTS}
    for c in COMPONENTS:
        for t in timeline:
            value = t['components'][c]['value']
            if runs[c] and runs[c][-1]['value'] == value:
                runs[c][-1]['frame_end'] = t['frame']
                runs[c][-1]['timestamp_end_us'] = t['timestamp_us']
            else:
                runs[c].append(dict(frame_start=t['frame'], frame_end=t['frame'],
                    timestamp_start_us=t['timestamp_us'], timestamp_end_us=t['timestamp_us'], value=value))
    return dict(sample_id=row['sample_id'], side=side, original_frames=frames[:],
        original_timestamps_us=times[:], events=copy.deepcopy(events), timeline=timeline,
        ordered_runs=runs, semantic_importance=None, semantic_phase=None,
        educational_correctness=None, training_eligible=False,
        estimated_values_are_direct=False, original_model_masks_modified=False)

def selftest():
    row = dict(sample_id='control', frame_ids=list(range(100, 113)),
               timestamps_us=[round(i*1e6/30) for i in range(100, 113)])
    original = copy.deepcopy(row)
    def event(lo, hi, value, component='translation', evidence='direct', **kw):
        return dict(frame_start=lo, frame_end=hi, component=component, value=value, evidence=evidence, **kw)
    def vals(out, c='translation'):
        return [t['components'][c]['value'] for t in out['timeline']]
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(name)
    a=encode(row,'R',[event(100,109,'hold'),event(110,112,'down')])
    check('brief_lowering_after_hold_preserved', vals(a)==['hold']*10+['down']*3)
    check('two_runs_no_majority_collapse', [r['value'] for r in a['ordered_runs']['translation']]==['hold','down'])
    a=encode(row,'R',[event(100,102,'hold'),event(110,112,'hold')])
    check('equal_endpoints_do_not_fill_gap', vals(a)==['hold']*3+[None]*7+['hold']*3)
    a=encode(row,'R',[event(100,112,'rotation','orientation')])
    check('rotation_does_not_imply_translation', vals(a)==[None]*13)
    check('rotation_remains_observed', vals(a,'orientation')==['rotation']*13)
    a=encode(row,'L',[event(100,112,'hold'),event(104,108,'down')])
    check('conflicting_direct_claims_masked', vals(a)==['hold']*4+[None]*5+['hold']*4)
    a=encode(row,'L',[event(100,112,'hold'),event(104,108,'unresolved')])
    check('occlusion_unknown_not_negative', vals(a)==['hold']*4+[None]*5+['hold']*4)
    a=encode(row,'R',[event(100,112,'down',evidence='context')])
    check('temporal_guess_not_direct_truth', vals(a)==[None]*13)
    a=encode(row,'R',[event(100,109,'hold',boundary_range=[108,110]),event(110,112,'down')])
    check('uncertain_onset_blocks_exact_boundary', vals(a)==['hold']*8+[None]*3+['down']*2)
    check('boundary_provenance_retained_outside_event_span', a['timeline'][10]['components']['translation']['source_event_indices']==[0,1])
    a=encode(row,'L',[event(100,112,'hold'),event(105,107,'change','shape')])
    check('finger_change_and_translation_separate', vals(a)==['hold']*13 and vals(a,'shape')[5:8]==['change']*3)
    a=encode(row,'R',[event(100,103,'up'),event(104,106,'hold'),event(107,112,'down')])
    check('reversal_order_preserved', [r['value'] for r in a['ordered_runs']['translation']]==['up','hold','down'])
    shifted=copy.deepcopy(row);shifted['timestamps_us']=[t*2+1000 for t in row['timestamps_us']]
    b=encode(shifted,'R',a['events'])
    check('timestamp_change_does_not_relabel_action', vals(a)==vals(b))
    check('actual_timestamps_retained', b['original_timestamps_us']==shifted['timestamps_us'])
    check('all_frames_remain_in_denominator', all(len(o['timeline'])==13 for o in (a,b)))
    check('original_input_not_mutated', row==original)
    check('semantic_phase_unknown', a['semantic_phase'] is None and a['semantic_importance'] is None)
    check('diagnosis_never_training_authorization', a['training_eligible'] is False)
    for bad in [event(99,100,'hold'), event(100,113,'down'),event(100,112,'rotation')]:
        try:encode(row,'R',[bad])
        except AssertionError:pass
        else:raise AssertionError('invalid event accepted')
    check('out_of_scope_or_wrong_component_rejected', True)
    for badrow in [dict(sample_id='bad',frame_ids=[],timestamps_us=[]),
                   dict(row,frame_ids=row['frame_ids'][:12],timestamps_us=row['timestamps_us'][:12]),
                   dict(row,center_frame=105), dict(row,center_timestamp_us=0),
                   dict(row,frame_ids=[float(x) for x in row['frame_ids']]),
                   dict(row,timestamps_us=[float(x) for x in row['timestamps_us']])]:
        try:encode(badrow,'R',[])
        except AssertionError:pass
        else:raise AssertionError('invalid original window accepted')
    check('empty_short_noninteger_and_wrong_center_window_rejected', True)
    exact=dict(row,center_frame=106,center_timestamp_us=row['timestamps_us'][6])
    check('exact_original_center_accepted', encode(exact,'R',[])['original_frames']==row['frame_ids'])
    return dict(status='PASS_SYNTHETIC_SEQUENCE_BOOKKEEPING_ONLY',count=len(checks),checks=checks,
        no_real_motion_accuracy_claim=True,code_sha256=sha(__file__))

if __name__ == '__main__':
    if len(sys.argv) in (2,3) and sys.argv[1] == 'selftest':
        result=selftest();target=(U/sys.argv[2]).resolve() if len(sys.argv)==3 else U/'review/MOTION_SEQUENCE_SELFTEST.json'
        assert target.parent==U/'review'
        write(target,result)
        print(json.dumps(result,ensure_ascii=False))
    else:
        raise SystemExit('Use selftest. Actual diagnostic export is built only after RGB review.')
