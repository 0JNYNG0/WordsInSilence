"""Narrow revision checks; rules are not relabelled as learned repetitions.

The original 22-check report remains immutable. This script proves the revised
integration contract, not normal-motion accuracy or cross-person performance.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
V=Path(__file__).resolve().parent;U=V.parent
sys.path.insert(0,str(V))
from independent_checks import ShapeStub,ActionStub
from coach import Coach
from action_runtime import ActionObserver

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(2)
    paths=[U/'coach.py',U/'action_runtime.py']
    hashes={p.name:sha(p)for p in paths}
    frozen=[V/'independent_checks.py',V/'INDEPENDENT_CHECKS.json']
    previous={p.name:sha(p)for p in frozen}
    with np.load(U/'source/reference.npz')as z:r={k:z[k].copy()for k in z.files}
    xy,score,ts=r['xy_px'],r['score'],r['timestamp_us'];w,h=int(r['source_width']),int(r['source_height'])
    checks=[]
    def ck(name,passed,**details):checks.append(dict(name=name,passed=bool(passed),**details))
    for answer,name in [(2,'supported'),(1,'hold')]:
        a=ActionStub(answer);c=Coach(model=ShapeStub(0),action=a)
        for i in range(48):o=c.update(xy[i],score[i],int(ts[i]),w,h)
        ck('sweep_learned_gate_'+name,c.completed[0]==(answer==2),completed=c.completed,queries=len(a.queries))
    a=ActionStub(2);c=Coach(model=ShapeStub(1),action=a)
    for i in range(48):o=c.update(xy[i],score[i],int(ts[i]),w,h)
    ck('wrong_shape_keeps_action_observation',len(a.history)==48 and not any(c.completed))

    def repeat(mode):
        a=ActionStub(1);c=Coach(model=ShapeStub(1),action=a)
        c.stage_index=1;c.completed[0]=True
        for i in range(48,95):
            j=53 if mode=='held' else min(i,54)if mode=='one_then_hold' else i
            p=xy[j].copy()
            if mode=='one_hand_held':p[[7,9]+list(range(91,112))]=xy[51][[7,9]+list(range(91,112))]
            if mode=='missing_hand':p[91:112]=np.nan
            o=c.update(p,score[j],int(ts[i]),w,h)
        return a,c,o
    a,c,o=repeat('normal')
    ck('repetition_explicit_not_learned',c.completed[1] and not a.queries and len(c.events)==2 and all('temporal direction'not in e['evidence']for e in c.events),events=c.events)
    ck('scope_discloses_rule_repetition',o['learned_direction_axes']==[2]and o['diagnostic_only_direction_axes']==[0]and o['whole_word_correct']is None,role=o['model_role'])
    for name in ['held','one_then_hold','one_hand_held','missing_hand']:
        a,c,o=repeat(name)
        ck(name+'_cannot_complete_repetition',not c.completed[1],event_count=len(c.events),phase=o['phase'])
    a=ActionStub(2);c=Coach(model=ShapeStub(0),action=a)
    for i in range(40):c.update(xy[i],score[i],int(ts[i]),w,h)
    o=c.update(xy[40]*.5,score[40],int(ts[40]),w//2,h//2)
    ck('dimensions_reset_hand_window_and_pending',len(c.frames)==1 and c.start_q is None and c.last_assessed is None and o['phase']=='observing')

    model=ActionObserver()
    for i in range(47):o=model.update(xy[i],score[i],int(ts[i]),w,h)
    stamps=model.channels['timestamp_us'];end=int(o['assessed_timestamp_us']);b=int(np.flatnonzero(stamps==end)[0])
    old_g,old_ok,_=model.rc.prior.geometry(model.x,stamps,b-3,b,2,'R','L')
    new_g,new_ok,ev=model._geometry(stamps,b-3,b,2,'R','L',3)
    ck('four_plus_features_unchanged',new_ok==old_ok and np.array_equal(new_g,old_g),evidence=ev)
    three=model.movement(int(stamps[b-2]),end,2,'R','L',min_samples=3)
    strict=model.movement(int(stamps[b-2]),end,2,'R','L',min_samples=4)
    two=model.movement(int(stamps[b-1]),end,2,'R','L',min_samples=3)
    ck('three_real_observations_optional_policy',three['eligible']and not strict['eligible']and not two['eligible']and three['evidence']['actual_observed_samples']==3 and three['evidence']['synthesized_observations']==0,three=three,strict=strict,two=two)
    short=stamps.copy();short[b-2:b+1]=[100,101,102]
    _,ok,ev=model._geometry(short,b-2,b,2,'R','L',3)
    ck('too_short_three_observation_interval_rejected',not ok,evidence=ev)
    duplicate=stamps.copy();duplicate[b-1]=duplicate[b-2]
    _,ok,ev=model._geometry(duplicate,b-2,b,2,'R','L',3)
    ck('duplicate_three_observation_times_rejected',not ok,evidence=ev)
    physical=model.x.copy();model.x[b-1]=0
    _,ok,ev=model._geometry(stamps,b-2,b,2,'R','L',3)
    model.x=physical
    ck('three_observation_interval_missing_middle_rejected',not ok and ev['actual_observed_samples']<3,evidence=ev)
    ck('original_check_report_unchanged',all(sha(p)==previous[p.name]for p in frozen),sha256=previous)
    stable=all(sha(p)==hashes[p.name]for p in paths)
    report=dict(passed=all(c['passed']for c in checks),checks=checks,code_sha256=hashes,code_unchanged_during_test=stable,
        original_report_preserved=True,active_minimum_policy=model.min_direction_samples,
        scope='Revision contract only. Three observations are not evidence of restored normal accuracy. Repetition remains explicit geometry; learned axis2 admissions require root public regression review.',
        actual_camera=False,private_data=False,training=False,whole_word_accuracy=None)
    (V/'INDEPENDENT_CHECKS_V2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],unchanged=stable,checks=[dict(name=c['name'],passed=c['passed'])for c in checks]),ensure_ascii=False))

if __name__=='__main__':main()
