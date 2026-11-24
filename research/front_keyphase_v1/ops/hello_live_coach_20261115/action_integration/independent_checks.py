"""Independent integration tests; public coordinates and model stubs, no camera.

Stub tests prove routing/veto contracts, never recognition accuracy. Actual-model
tests cover prefix causality and observation eligibility, not new-person accuracy.
"""
from pathlib import Path
import sys, json, hashlib, copy
import numpy as np
import torch

V=Path(__file__).resolve().parent;U=V.parent
sys.path.insert(0,str(U))
from coach import Coach
from action_runtime import ActionObserver

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {k:clean(x)for k,x in v.items()if k not in ['compute_ms','observer_ms']}
    if isinstance(v,list):return [clean(x)for x in v]
    return v

class ShapeStub:
    def __init__(self,state=0):self.state=state
    def sequence(self,x,hand):
        return dict(states=np.full(len(x),self.state),fingers=np.full((len(x),4),1 if self.state==0 else 0 if self.state==1 else -1))

class ActionStub:
    def __init__(self,answer):self.answer=answer;self.reset_count=0;self.reset()
    def reset(self):self.history=[];self.queries=[];self.reset_count+=1
    def update(self,p,s,t,w,h):
        self.history.append(t)
        return dict(status='observed',calls=len(self.history),prefix_frames=len(self.history),whole_word_correct=None)
    def movement(self,a,b,axis,moving,anchor):
        observed=sum(a<=t<=b for t in self.history);eligible=observed>=4
        self.queries.append(dict(a=a,b=b,axis=axis,moving=moving,observed=observed))
        return dict(eligible=eligible,state=self.answer if eligible else -1,reason=None if eligible else 'insufficient_observed_geometry')

def main():
    torch.set_num_threads(2)
    initial={p.name:sha(p)for p in [U/'coach.py',U/'action_runtime.py']}
    raw_path=U/'source/reference.npz';raw_sha=sha(raw_path)
    with np.load(raw_path)as z:r={k:z[k].copy()for k in z.files}
    xy,score,ts=r['xy_px'],r['score'],r['timestamp_us'];w,h=int(r['source_width']),int(r['source_height'])
    checks=[]
    def record(name,passed,**detail):checks.append(dict(name=name,passed=bool(passed),**detail))

    # Same geometry and hand evidence, only learned movement answer changes.
    for answer,name in [(2,'supported'),(1,'hold'),(0,'opposite'),(-1,'uncertain')]:
        action=ActionStub(answer);shape=ShapeStub(0);c=Coach(model=shape,action=action);outs=[]
        for i in range(48):outs.append(c.update(xy[i],score[i],int(ts[i]),w,h))
        got=c.completed[0]
        record('learned_sweep_'+name,got==(answer==2),first_completed=got,model_updates=len(action.history),movement_queries=len(action.queries))
        record('hold_not_reported_as_opposite_'+name,answer!=1 or not any('방향을 바꿔' in o['message']for o in outs))
    for shape_state,name in [(1,'wrong_shape'),(-1,'unknown_shape')]:
        a=ActionStub(2);c=Coach(model=ShapeStub(shape_state),action=a)
        for i in range(48):out=c.update(xy[i],score[i],int(ts[i]),w,h)
        record(name+'_still_enters_action_model',len(a.history)==48 and not any(c.completed),model_updates=len(a.history),completed=c.completed)
    # Geometry contains both real downward strokes. A model that supports only
    # downward movement must NOT permit the return/rearm or count them as two.
    a=ActionStub(0);c=Coach(model=ShapeStub(1),action=a)
    c.stage_index=1;c.completed[0]=True
    for i in range(48,95):c.update(xy[i],score[i],int(ts[i]),w,h)
    down_events=[e for e in c.events if e['kind']=='downstroke_observed']
    record('repeat_requires_learned_upward_return',len(down_events)==1 and not c.completed[1],down_events=len(down_events),completed=c.completed)
    class OneHandHold(ActionStub):
        def movement(self,a,b,axis,moving,anchor):
            q=super().movement(a,b,axis,moving,anchor)
            if q['eligible'] and moving=='R':q['state']=1
            return q
    a=OneHandHold(0);c=Coach(model=ShapeStub(1),action=a)
    c.stage_index=1;c.completed[0]=True
    for i in range(48,95):c.update(xy[i],score[i],int(ts[i]),w,h)
    queried={q['moving']for q in a.queries}
    record('repeat_requires_both_learned_wrist_directions',queried=={'L','R'} and not c.events and not c.completed[1],queried_hands=sorted(queried),completed=c.completed)
    # Reset clears model context, stage state and the seven-frame hand window.
    a=ActionStub(2);c=Coach(model=ShapeStub(0),action=a)
    for i in range(48):c.update(xy[i],score[i],int(ts[i]),w,h)
    before=a.reset_count;c.reset()
    record('coach_reset_clears_both_models',a.reset_count==before+1 and not a.history and not c.frames and c.completed==[False,False])
    for i in range(39):c.update(xy[i],score[i],int(ts[i]),w,h)
    out=c.update(xy[43],score[43],int(ts[43])+1000000,w,h)
    record('gap_clears_pending_geometry',c.start_q is None and len(c.frames)==1 and not any(c.completed),phase=out['phase'])

    model=ActionObserver()
    prefix=47
    def stream(variant):
        model.reset();out=[]
        for i in range(57):
            p=xy[i].copy()
            if i>=prefix and variant=='altered_future':p[:,0]=w-p[:,0]
            q=model.update(p,score[i],int(ts[i]),w,h)
            json.dumps(q,allow_nan=False)
            if i<prefix:out.append(copy.deepcopy(clean(q)))
        return out
    a=stream('normal');b=stream('altered_future')
    record('actual_model_same_prefix_independent_of_future',a==b,prefix_frames=prefix)
    record('frozen_existing_checkpoint',model.provenance['parameters_frozen']and not model.provenance['target_word_input']and not model.provenance['trained_this_session'],checkpoint_sha256=model.checkpoint_sha256)
    # Query ends at the assessed frame, never at an unassessed future endpoint.
    model.reset()
    for i in range(47):out=model.update(xy[i],score[i],int(ts[i]),w,h)
    end=int(out['assessed_timestamp_us']);idx=int(np.flatnonzero(ts==end)[0])
    three=model.movement(int(ts[idx-2]),end,2,'R','L')
    four=model.movement(int(ts[idx-3]),end,2,'R','L')
    future=model.movement(int(ts[idx-3]),int(ts[idx+1]),2,'R','L')
    record('minimum_four_observed_samples',not three['eligible']and four['eligible'],three=three,four=four)
    record('unassessed_endpoint_rejected',not future['eligible'],result=future)
    gap=model.update(xy[47],score[47],int(ts[47])+1000000,w,h)
    old=model.movement(int(ts[39]),int(ts[43]),2,'R','L')
    record('actual_model_gap_resets_history',gap['history_reset']=='observation_gap'and gap['prefix_frames']==1 and not old['eligible'],gap=clean(gap),old_query=old)
    model.reset()
    for i in range(12):missing=model.update(np.full((133,2),np.nan),score[i],int(ts[i]),w,h)
    record('missing_body_is_uncertain_no_stale_cache',missing['status']=='uncertain'and model.h is None and not model.movement(0,100000,2)['eligible'],status=missing['status'])
    model.reset();record('actual_reset_clears_cached_evidence',not model.frames and model.h is None and model.channels is None and model.last_ts is None)
    record('public_source_unchanged',sha(raw_path)==raw_sha)
    current={p.name:sha(p)for p in [U/'coach.py',U/'action_runtime.py']}
    report=dict(passed=all(c['passed']for c in checks),checks=checks,
        code_sha256=initial,code_unchanged_during_test=initial==current,source_sha256=raw_sha,
        scope='Independent integration routing and causality tests. Stub outputs are not accuracy. One public recording; no new-person result.',
        actual_camera=False,private_data=False,training=False,whole_word_accuracy=None)
    (V/'INDEPENDENT_CHECKS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],unchanged=report['code_unchanged_during_test'],checks=[dict(name=c['name'],passed=c['passed'])for c in checks]),ensure_ascii=False))

if __name__=='__main__':main()
