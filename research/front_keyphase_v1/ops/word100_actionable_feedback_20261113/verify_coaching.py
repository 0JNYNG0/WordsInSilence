"""Audit presentation contract on saved public outputs; no detector/model rerun."""
from pathlib import Path
import json,copy,hashlib,time,collections
import numpy as np
from coaching import compose,attach,GUIDES,DIRS
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_positive_direction_20261109'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
checks=[]
def check(name,ok):
    checks.append(dict(name=name,passed=bool(ok)))
    if not ok:raise AssertionError(name)
check('empty_result_not_success',compose({})['status']=='uncertain')
check('empty_cards_not_success',compose(dict(word='WORD1133',cards=[],timestamp_us=[0,33333]))['status']=='uncertain')
audits=[];times=[];fixtures={}
for p in sorted((D/'final_decisions').glob('*.json')):
    x=read(p)['after'];word=p.name.split('_')[2];z=np.load(D/'predictions'/(p.stem+'.npz'))
    x.update(word=word,timestamp_us=z['timestamp_us'].tolist(),source='public_saved_contract_check')
    before=json.dumps(x,sort_keys=True);t=time.perf_counter();result=attach(x);times.append((time.perf_counter()-t)*1000)
    c=result['coaching'];without=copy.deepcopy(result);without.pop('coaching')
    check(p.stem+':original_unchanged',json.dumps(without,sort_keys=True)==before and json.dumps(x,sort_keys=True)==before)
    normal=any(k in p.stem for k in ['_normal','_speed','_translation_scale'])
    if normal:check(p.stem+':no_new_correction',not c['cues'])
    for cue in c['cues']:
        a,b=cue['focus_indices'];check(p.stem+':native_time_valid',0<=a<=b<len(x['timestamp_us']) and cue['start_seconds']==x['timestamp_us'][a]/1e6)
        check(p.stem+':no_wrong_verdict',cue['educational_wrong'] is False and cue['whole_action_complete'] is None)
    if 'WORD1133' in p.stem and any(k in p.stem for k in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']):
        expected={'handshape','direction'} if 'closed_hand_and_reversed' in p.stem else {'handshape'} if 'right_closed_hand_main' in p.stem else {'direction'}
        check(p.stem+':expected_cue_types',{q['kind'] for q in c['cues']}==expected)
        if 'direction' in expected:check(p.stem+':direction_not_duplicated',sum(q['kind']=='direction' for q in c['cues'])==1)
        fixtures[p.stem]=x
    audits.append(dict(case=p.stem,normal=normal,status=c['status'],cues=c['cues'],notes=c['notes'],rejected=c['rejected']))

shape=copy.deepcopy(next(v for k,v in fixtures.items() if 'right_closed_hand_main' in k))
direction=copy.deepcopy(next(v for k,v in fixtures.items() if 'main_reversed' in k))
def changed(name,original,fn):
    x=copy.deepcopy(original);fn(x);check(name,not compose(x)['cues'])
changed('wrong_word_not_rebound',shape,lambda x:x.update(word='WORD1567'))
changed('bad_timestamps',shape,lambda x:x.update(timestamp_us=[0]*len(x['timestamp_us'])))
changed('unknown_is_not_error',shape,lambda x:[c.update(display_status='uncertain') for c in x['cards']])
changed('quarantine_is_not_error',shape,lambda x:[c.update(observation_quarantine_overlap=True) for c in x['cards']])
changed('outside_time_not_corrected',shape,lambda x:[e.update(contradicted_spans=[[9998,9999]]) for c in x['cards'] for e in c['shape_evidence']])
changed('string_false_not_valid',shape,lambda x:[e.update(conflicting_predictions='false') for c in x['cards'] for e in c['shape_evidence']])
changed('projection_disagreement',shape,lambda x:[e.update(projection_consistency_required=False) for c in x['cards'] for e in c['shape_evidence']])
changed('missing_span_not_filled_by_dtw',shape,lambda x:[c.update(query_indices=None) for c in x['cards']])
changed('wrong_hand_direction_not_corrected',direction,lambda x:[c['motion_evidence'].update(moving_hand='L') for c in x['cards'] if c.get('motion_evidence')])
changed('unbound_direction_sign',direction,lambda x:[c['motion_evidence'].update(expected_sign=-1) for c in x['cards'] if c.get('motion_evidence')])
changed('direction_nan',direction,lambda x:[c['motion_evidence'].update(signed_change=float('nan')) for c in x['cards'] if c.get('motion_evidence')])
changed('missing_all_samples',shape,lambda x:x.update(long_missing_runs=[[0,len(x['timestamp_us'])-1]]))
# Hand symmetry and new expression IDs: same generic logic with explicit guide requirements.
mirrored=copy.deepcopy(shape);mirrored['word']='TEST_GENERIC';gs=copy.deepcopy(GUIDES[shape['word']]);convert={1:2,2:1,3:4,4:3}
for card in [*mirrored['cards'],*gs]:
    card['features']={str(convert.get(int(k),int(k))):v for k,v in card['features'].items()}
    card['hands']=['L' if h=='R' else 'R' for h in card['hands']]
for card in mirrored['cards']:
    card['motion_evidence']=None
    for e in card['shape_evidence']:
        e['feature']=convert.get(e['feature'],e['feature']);e['opposite_feature']=convert.get(e['opposite_feature'],e['opposite_feature'])
mc=compose(mirrored,{'TEST_GENERIC':gs},{})
check('generic_left_hand_same_logic',bool(mc['cues']) and all(c['hand']=='L' and '왼손' in c['action'] for c in mc['cues']))

out=dict(passed=True,checks=checks,saved_public_conditions=len(audits),normal_conditions=sum(x['normal'] for x in audits),
    audited_cues=sum(len(x['cues']) for x in audits),synthetic_error_conditions=len(fixtures),
    timing_ms=dict(median=float(np.median(times)),max=float(max(times))),
    source='93 pre-existing exposed public conditions; mostly transformations of 12 videos; not 93 independent videos, not fresh inference or accuracy',
    limitation='Contract checks do not independently validate automatic clause membership or educational correctness.',cases=audits)
(U/'COACHING_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['checks','cases']},ensure_ascii=False))
