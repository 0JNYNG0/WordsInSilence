"""Fixed saved-public comparison + analytical component contracts. No inference."""
from pathlib import Path
import json,sys,copy,hashlib,time
import numpy as np
from localized_direction import refine,along_forearm,original_along_forearm,legacy,localize_cues
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_positive_direction_20261109'
sys.path.insert(0,str(O/'word100_actionable_feedback_20261113'))
from coaching import compose
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[];rows=[];times=[]
def check(name,value):checks.append(dict(name=name,passed=bool(value)))
for path in sorted((D/'final_decisions').glob('*.json')):
    z=np.load(D/'predictions'/(path.stem+'.npz'));saved=read(path);before=saved['before'];old=copy.deepcopy(saved['after'])
    start=time.perf_counter();new=refine(before,z['attributes'],z['x'],z['timestamp_us'],z['frame_id'],z['phase']);times.append((time.perf_counter()-start)*1000)
    for a,b in zip(old['cards'],new['cards']):
        if a.get('observation_quarantine_overlap'):
            for k in ['status','display_status','observation_quarantine_overlap']:b[k]=a[k]
    word=path.name.split('_')[2]
    for r in (old,new):r.update(word=word,timestamp_us=z['timestamp_us'].tolist())
    previous=compose(old);current=localize_cues(new,compose)
    normal=any(k in path.stem for k in ['_normal','_speed','_translation_scale'])
    if normal:
        check(path.stem+':no_new_normal_correction',not current['cues'])
        check(path.stem+':normal_status_preserved',[c['display_status'] for c in old['cards']]==[c['display_status'] for c in new['cards']])
    check(path.stem+':shape_evidence_identical',[c['shape_evidence'] for c in old['cards']]==[c['shape_evidence'] for c in new['cards']])
    if previous['cues']:check(path.stem+':existing_correction_kinds_preserved',{c['kind'] for c in previous['cues']}=={c['kind'] for c in current['cues']})
    for c in new['cards']:
        m=c.get('motion_evidence')
        if not m:continue
        for e in m['events']:
            a,b=e['span'];check(path.stem+':no_phase_gap_inside_event',e['source_phase_indices']==list(range(a,b+1)))
            check(path.stem+':completion_unknown',e['whole_action_complete'] is None)
        if len(m['events'])==1:
            prior=next(previous_card['motion_evidence'] for previous_card in old['cards'] if previous_card['id']==c['id'])
            for k in ['state','signed_change','trend','deadband']:
                check(path.stem+':single_run_same_'+k,m.get(k)==prior.get(k))
    for q in current['cues']:
        a,b=q['focus_indices'];check(path.stem+':native_time_exact',q['start_seconds']==z['timestamp_us'][a]/1e6 and q['end_seconds']==z['timestamp_us'][b]/1e6)
    rows.append(dict(case=path.stem,normal=normal,previous_cues=previous['cues'],current_cues=current['cues'],
        previous_states=[c['display_status'] for c in old['cards']],current_states=[c['display_status'] for c in new['cards']],
        motion=[dict(card=c['id'],evidence=c['motion_evidence']) for c in new['cards'] if c.get('motion_evidence')],
        prediction_sha256=sha(D/'predictions'/(path.stem+'.npz'))))

# Analytical geometry fixtures use the same relative forearm measure. They test
# the observer after phase association, not recognizer accuracy or real errors.
contract=copy.deepcopy(legacy.CONTRACTS['brush_arm']);n=48;ts=np.arange(n,dtype=np.int64)*33333
def fixture(progress,active=None,left=False):
    x=np.zeros((len(progress),280),np.float32);pts=x[:,:102].reshape(-1,51,2);valid=x[:,186:237]
    c=copy.deepcopy(contract)
    if left:c.update(moving_hand='L',anchor_arm='R')
    e,w=(4,6) if left else (3,5);h=9 if left else 30
    pts[:,e]=[0,0];pts[:,w]=[1,0]
    for j in [h,h+5,h+9,h+13,h+17]:pts[:,j,0]=progress;pts[:,j,1]=.1
    valid[:,[e,w,h,h+5,h+9,h+13,h+17]]=1
    ph=np.zeros((len(progress),14),np.float32);ph[:,0]=1
    if active is None:active=np.ones(len(progress),bool)
    ph[active,0]=0;ph[active,c['phase_groups'][0]+1]=1
    return x,ph,c
controls=[]
for left in [False,True]:
    # The two opposite strokes have the same global start/end. A global net
    # displacement cannot tell the user about the second, reversed stroke.
    p=np.r_[np.linspace(0,1,18),np.ones(12),np.linspace(1,0,18)]
    active=np.r_[np.ones(18,bool),np.zeros(12,bool),np.ones(18,bool)]
    x,ph,c=fixture(p,active,left);old=original_along_forearm(x,ts,ph,c);new=along_forearm(x,ts,ph,c)
    check('opposite_strokes_do_not_cancel_'+str(left),old['state']=='unknown' and new['state']=='contradicted' and new['contradicted_spans']==[[30,47]])
    check('both_events_retained_'+str(left),[e['state'] for e in new['events']]==['supported','contradicted'])
    controls.append(dict(kind='two_opposite_strokes',side='L' if left else 'R',previous=old,candidate=new))
    p=np.r_[np.linspace(0,1,18),np.ones(12),np.linspace(1,0,18)];active=np.r_[np.ones(18,bool),np.zeros(30,bool)]
    x,ph,c=fixture(p,active,left);new=along_forearm(x,ts,ph,c)
    check('unassigned_return_not_a_direction_error_'+str(left),new['state']=='supported' and new['span']==[0,17])
    ph[31:33,0]=0;ph[31:33,c['phase_groups'][0]+1]=1;new=along_forearm(x,ts,ph,c)
    check('short_island_not_hidden_'+str(left),new['state']=='unknown' and new['unknown_spans']==[[31,32]] and len(new['events'])==2)
    x,ph,c=fixture(np.linspace(0,1,n),left=left);x[:,186:237]=0;new=along_forearm(x,ts,ph,c)
    check('missing_not_error_or_completion_'+str(left),new['state']=='unknown' and new['whole_action_complete'] is None)
    x,ph,c=fixture(np.linspace(1,0,n),left=left);new=along_forearm(x,ts,ph,c)
    check('one_reverse_stroke_detected_'+str(left),new['state']=='contradicted')
    xp=x.copy();xp[:,:102]=xp[:,:102]*2.1+.47;scaled=along_forearm(xp,ts,ph,c)
    check('body_relative_scale_translate_'+str(left),scaled['state']==new['state'] and np.isclose(scaled['signed_change'],new['signed_change'],atol=1e-6))
    slow=along_forearm(x,ts*2,ph,c)
    check('time_scale_not_direction_'+str(left),slow['state']==new['state'])
    gapts=ts.copy();gapts[24:]+=300000;gap=along_forearm(x,gapts,ph,c)
    check('timestamp_gap_split_'+str(left),[e['span'] for e in gap['events']]==[[0,23],[24,47]])
    check('no_labels_or_completion_in_runtime_'+str(left),gap['expected_order_used'] is False and gap['whole_action_complete'] is None)
out=dict(passed=all(c['passed'] for c in checks),checks=checks,public_conditions=len(rows),normal_conditions=sum(r['normal'] for r in rows),
    cases=rows,component_controls=controls,timing_ms=dict(median=float(np.median(times)),max=float(max(times))),
    frozen_plan_sha256=sha(U/'LOCAL_DIRECTION_PLAN.json'),candidate_sha256=sha(U/'localized_direction.py'),
    evidence='Existing exposed public outputs plus analytical component controls. No new network inference, real learner errors or whole-sign accuracy.',
    whole_G2_pass=False)
with (U/'LOCAL_DIRECTION_COMPARISON_V2.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(dict(passed=out['passed'],public=out['public_conditions'],normal=out['normal_conditions'],checks=len(checks),failed=[c for c in checks if not c['passed']],timing_ms=out['timing_ms']),ensure_ascii=False))
