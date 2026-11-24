"""Independent saved-contract audit/recount; no implementation imports."""
import json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
U=Path(__file__).resolve().parent;B=U.parent/'word100_hand_region_uncertainty_20261017'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(U/'SCOPE.json');out=read(U/'OUTPUT.json');prior=read(B/'PREDICTIONS.json')
assert out['scope_sha256']==sha(U/'SCOPE.json') and out['source_predictions_sha256']==sha(B/'PREDICTIONS.json')
for p,h in {**s['dependencies'],**s['source_pins'],**s['protected_pins']}.items():assert sha(p)==h
lookup={r['clip_id']:r for r in out['records']};assert len(lookup)==60
states=normal=location=unobserved=0;counts={k:Counter() for k in ['position','shape','overlap','proximity_hint']}
for r in prior['records']:
    rows=lookup[r['clip_id']]['frames'];assert len(rows)==len(r['frames'])
    for old,new in zip(r['frames'],rows):
        assert all(old[k]==new[k] for k in ['frame','timestamp_us','core'])
        for side,(base,region,hand) in enumerate(zip(old['baseline'],old['hands'],new['hands'])):
            state=base['state'];p=hand['position'];shape=hand['shape'];overlap=hand['overlap']
            assert hand['side']==side
            if state in ('inside_missing','overlap_likely'):expected='wrist_interior'
            elif state=='boundary_outside_likely':expected='wrist_boundary_or_outside'
            elif state=='observed':expected='observed_points'
            elif state=='partial_observation':expected='partial_points'
            elif state=='assignment_uncertain':expected='assignment_unresolved'
            else:assert state=='unknown';expected='unknown'
            assert p['state']==expected and p['evidence']==base['evidence'];location+=1
            assert p['whole_hand_visibility_proven'] is False
            if state=='observed':assert shape['state']=='usable_observed_points';normal+=1
            elif state=='partial_observation':assert shape['state']=='partial_observation'
            else:assert shape['state']=='unobserved';unobserved+=1
            assert shape['estimated_region_is_observed_shape'] is False
            assert overlap['proximity_hint']==(state=='overlap_likely')
            assert overlap['occlusion_cause_confirmed'] is False
            if state not in ('inside_missing','overlap_likely','boundary_outside_likely'):support='not_assessed_for_missing_hand'
            elif region['region_status']!='estimated_envelope':support='region_unavailable'
            elif region['state'] in ('region_uncertain','boundary_uncertain'):support='region_uncertain'
            elif region['state']=='overlap_likely':support='area_supported_suspicion'
            else:support='not_supported_across_envelope'
            assert overlap['area_evidence']==support
            assert hand['region']['uncertainty_pixels']==region['evidence'].get('uncertainty_pixels')
            assert hand['region']['age_seconds']==region['evidence'].get('region_age_seconds')
            assert hand['region']['state']==region['state'] and hand['region']['status']==region['region_status']
            assert hand['coordinate_repair'] is False and hand['eligible_for_automatic_exclusion'] is False
            assert 'state' not in hand  # No destructive single combined status.
            for k,v in [('position',p['state']),('shape',shape['state']),('overlap',support),('proximity_hint',str(overlap['proximity_hint']))]:counts[k][v]+=1
            states+=1
assert counts==out['counts'] and states==15926
summary=defaultdict(lambda:{k:Counter() for k in ['baseline','combined','separated']})
preserved=restored=0;issues=[];manual=read(B/'VISUAL_LABELS.json')['labels']
assert len(manual)==len(out['visual_comparisons'])==141
for x,item in zip(manual,out['visual_comparisons']):
    assert all(item[k]==v for k,v in x.items())
    h=lookup[x['clip_id']]['frames'][x['frame']]['hands'][x['side']]
    assert h==item['separated']
    origin=x['origin'];label=x['visual_category']
    if label=='observed_control':
        assert h['position']['state']=='observed_points'
        for k in summary[origin]:summary[origin][k]['controls_preserved']+=1
        continue
    truth='out' if label=='boundary_or_outside' else 'in'
    def loc(state):
        if state in ('wrist_boundary_or_outside','boundary_outside_likely'):return 'out'
        if state in ('wrist_interior','inside_missing','overlap_likely'):return 'in'
        return '?'
    actual={k:loc(state) for k,state in [('baseline',item['baseline']['state']),('combined',item['combined_candidate']['state']),('separated',h['position']['state'])]}
    for k,v in actual.items():summary[origin][k]['abstain' if v=='?' else 'agree' if v==truth else 'disagree']+=1
    if actual['baseline']==truth:
        assert actual['separated']==truth;preserved+=1
        if actual['combined']!=truth:restored+=1
    if actual['separated'] not in (truth,'?'):
        issues.append({'review_id':x['review_id'],'clip_id':x['clip_id'],'frame':x['frame'],'side':x['side'],
                       'whole_hand_visual_category':label,'position':h['position']['state'],'region':h['region']['state']})
assert preserved==101 and restored==52 and len(issues)==2
tests=read(U/'CONTRACT_TESTS.json');assert tests['status']=='PASS'
result={'status':'PASS','at':datetime.now().astimezone().isoformat(timespec='seconds'),
 'operator':'Same Codex operator, independent saved-contract audit, not a second visual reviewer',
 'hand_states_checked':states,'position_and_evidence_preserved':location,'observed_states_preserved':normal,'unobserved_not_promoted':unobserved,
 'metamorphic_cases':tests['metamorphic_cases'],'public_stream_hand_states':tests['public_stream_hand_states'],
 'counts':counts,'visual_regression_tables':summary,'previously_agreeing_missing_samples_preserved':preserved,
 'restored_vs_rejected_combined_candidate':restored,'remaining_whole_hand_boundary_disagreements':issues,
 'source_pins_verified':len(s['source_pins']),'protected_pins_verified':len(s['protected_pins']),
 'limitation':'Location is specifically arm/pose-wrist evidence. Old whole-hand labels retained for comparison; do not interpret positional preservation as improved physical classification.141 labels are reused regression, not new blind data.',
 'new_training':0,'classifier_inference':0,'new_detection':0,'user_recording_reads':0,
 'evidence_sha256':{n:sha(U/n) for n in ['SCOPE.json','separate.py','OUTPUT.json','CONTRACT_TESTS.json']}}
with (U/'INDEPENDENT_CHECK.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:result[k] for k in ['status','hand_states_checked','observed_states_preserved','previously_agreeing_missing_samples_preserved','restored_vs_rejected_combined_candidate','visual_regression_tables','remaining_whole_hand_boundary_disagreements']},ensure_ascii=False,indent=2))
