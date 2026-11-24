"""Independent read-only audit of an isolated motion-head candidate.

No training, camera, private recording, old-report rewrite, or admission occurs.
"""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
import torch
V=Path(__file__).resolve().parent;U=V.parent;D=V/'finetune'
sys.path.insert(0,str(V))
from independent_checks import ShapeStub,ActionStub,clean
from coach import Coach
from action_runtime import ActionObserver,_load_model,CHECKPOINT,CHECKPOINT_SHA256

def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(2)
    watch=[U/'coach.py',U/'action_runtime.py',D/'experiment.py',D/'SOURCE_LABELS.json',D/'CANDIDATE.pt',CHECKPOINT]
    hashes={str(p.relative_to(U))if p.is_relative_to(U)else str(p):sha(p)for p in watch}
    preserved=[V/'INDEPENDENT_CHECKS.json',V/'INDEPENDENT_CHECKS_V2.json',V/'PUBLIC_COMPARISON.json']
    previous={str(p):sha(p)for p in preserved}
    index=read(D/'FEATURE_INDEX.json');plan=read(D/'FROZEN_PLAN.json');results=read(D/'RESULTS.json');labels=read(D/'SOURCE_LABELS.json')
    checks=[]
    def ck(name,passed,**details):checks.append(dict(name=name,passed=bool(passed),**details))
    fit=index['old_fit'];dev=index['old_dev'];supp=index['supplemental_fit'];diagnostic=index['controller_diagnostics']
    fit_clips={r['clip']for r in fit+supp};dev_clips={r['clip']for r in dev}
    fit_people={r['person']for r in fit+supp};dev_people={r['person']for r in dev}
    ck('public_fit_development_separation',not(fit_clips&dev_clips)and not(fit_people&dev_people)
       and all(r['role']=='fit'and r['clip'].startswith('NIA_SL_WORD')for r in fit)
       and all(r['role']=='development'for r in dev)
       and all(r['role']=='supplemental_fit'and r['clip']=='NIA_SL_SEN0354_REAL01_F'for r in supp),
       fit_people=sorted(fit_people),development_people=sorted(dev_people),supplemental_source_clips=sorted({r['clip']for r in supp}))
    expected={(36,42,2,'R',2),(51,56,0,'R',0),(51,56,0,'L',0),(56,60,0,'R',2),(56,60,0,'L',2),(60,65,0,'R',0),(60,65,0,'L',0),(69,75,0,'R',1),(69,75,0,'L',1)}
    actual={(x['start_frame'],x['end_frame'],x['axis'],x['moving'],x['target'])for x in labels['intervals']}
    ck('supplement_matches_independent_coarse_direction_review',actual==expected and len(supp)==36 and all(not r['independent']for r in supp),reviewed_ranges=len(expected),derived_feature_rows=len(supp))
    ck('fixed_plan_and_source_hashes',plan['steps']==400 and plan['seed']==42 and plan['threshold']==.65
       and plan['code_sha256']==sha(D/'experiment.py')and plan['source_labels_sha256']==sha(D/'SOURCE_LABELS.json')
       and plan['source_raw_sha256']==sha(Path(labels['source_raw'])),steps=plan['steps'],threshold=plan['threshold'])
    # Independently audited source: optimizer inputs are only these two sets;
    # development and unlabeled diagnostics are evaluated after the fixed loop.
    code=(D/'experiment.py').read_text('utf-8')
    ck('optimizer_uses_only_fit_features',"fitsets=[oldfit,supp]"in code and "copy.deepcopy(net.motion_head)"in code
       and "for step in range(400):"in code and code.index('for name,rows in datasets.items():')>code.index('for step in range(400):')
       and all(r['target']is None and r['role']=='diagnostic'for r in diagnostic))
    model=ActionObserver(candidate_path=D/'CANDIDATE.pt');_,original,_,_= _load_model()
    original_state=original.state_dict();candidate_state=model.net.state_dict()
    changed=[k for k in original_state if not torch.equal(original_state[k],candidate_state[k])]
    ck('only_motion_head_weights_changed',bool(changed)and all(k.startswith('motion_head.')for k in changed),changed_tensor_names=changed)
    ck('candidate_and_original_frozen_at_runtime',all(not p.requires_grad for p in model.net.parameters())and all(not p.requires_grad for p in original.parameters())and sha(CHECKPOINT)==CHECKPOINT_SHA256)
    ck('reported_historical_development_preserved',results['results']['old_dev']['native']['before']['correct']==32
       and results['results']['old_dev']['native']['after']['correct']==32
       and not results['results']['old_dev']['all']['regressions'],
       note='Historical development regression evidence from fixed experiment; not a new independent test.')
    ck('provenance_discloses_finetuned_head_only',model.provenance['direction_head_finetuned']and not model.provenance['base_trained_this_session']and not model.provenance['greeting_stage_training']and not model.provenance['target_word_input']and model.provenance['whole_word_correct']is None)

    with np.load(U/'source/reference.npz')as z:r={k:z[k].copy()for k in z.files}
    xy,score,ts=r['xy_px'],r['score'],r['timestamp_us'];w,h=int(r['source_width']),int(r['source_height'])
    for answer,name in [(2,'supported'),(1,'hold'),(0,'opposite'),(-1,'unknown')]:
        a=ActionStub(answer);c=Coach(model=ShapeStub(0),action=a)
        for i in range(48):o=c.update(xy[i],score[i],int(ts[i]),w,h)
        ck('sweep_model_veto_'+name,c.completed[0]==(answer==2),completed=c.completed,queries=len(a.queries))
    a=ActionStub(2);c=Coach(model=ShapeStub(1),action=a)
    for i in range(48):c.update(xy[i],score[i],int(ts[i]),w,h)
    ck('shape_error_does_not_stop_action_observation',len(a.history)==48 and not any(c.completed))
    a=ActionStub(0);c=Coach(model=ShapeStub(1),action=a);c.stage_index=1;c.completed[0]=True
    for i in range(48,95):c.update(xy[i],score[i],int(ts[i]),w,h)
    ck('repeat_needs_learned_upward_return',len(c.events)==1 and not c.completed[1],events=len(c.events))
    class RightHold(ActionStub):
        def movement(self,a,b,axis,moving,anchor):
            out=super().movement(a,b,axis,moving,anchor)
            if out['eligible']and moving=='R':out['state']=1
            return out
    a=RightHold(0);c=Coach(model=ShapeStub(1),action=a);c.stage_index=1;c.completed[0]=True
    for i in range(48,95):o=c.update(xy[i],score[i],int(ts[i]),w,h)
    ck('repeat_needs_both_learned_wrist_directions',not c.events and not c.completed[1]and {q['moving']for q in a.queries}=={'L','R'})
    ck('final_scope_reports_learned_direction_and_explicit_completion',set(o['learned_direction_axes'])=={0,2}and o['whole_word_correct']is None,role=o['model_role'])

    def prefix_run(alter):
        model.reset();outputs=[]
        for i in range(54):
            p=xy[i].copy()
            if alter and i>=47:p[:,0]=w-p[:,0]
            o=model.update(p,score[i],int(ts[i]),w,h)
            if i<47:outputs.append(copy.deepcopy(clean(o)))
        return outputs
    ck('candidate_same_received_prefix_ignores_future',prefix_run(False)==prefix_run(True))
    model.reset()
    for i in range(47):o=model.update(xy[i],score[i],int(ts[i]),w,h)
    end=o['assessed_timestamp_us'];idx=int(np.flatnonzero(ts==end)[0])
    three=model.movement(int(ts[idx-2]),end,2);two=model.movement(int(ts[idx-1]),end,2)
    future=model.movement(int(ts[idx-2]),int(ts[idx+1]),2)
    ck('three_genuine_samples_and_unassessed_endpoint_guard',three['eligible']and not two['eligible']and not future['eligible']and three['evidence']['actual_observed_samples']==3 and three['evidence']['synthesized_observations']==0)
    gap=model.update(xy[47],score[47],int(ts[47])+1000000,w,h)
    ck('candidate_gap_clears_evidence',gap['history_reset']=='observation_gap'and model.h is None and len(model.frames)==1)
    model.reset()
    for i in range(12):o=model.update(np.full((133,2),np.nan),score[i],int(ts[i]),w,h)
    ck('missing_body_cannot_reuse_candidate_evidence',o['status']=='uncertain'and model.h is None and not model.movement(0,100000,2)['eligible'])
    a=ActionStub(2);c=Coach(model=ShapeStub(0),action=a)
    for i in range(40):c.update(xy[i],score[i],int(ts[i]),w,h)
    c.update(xy[40]*.5,score[40],int(ts[40]),w//2,h//2)
    ck('dimensions_clear_delayed_hand_state',len(c.frames)==1 and c.last_assessed is None and c.start_q is None)
    c.reset();model.reset()
    ck('reset_clears_models_and_pending',not c.frames and not a.history and not model.frames and model.h is None)
    ck('earlier_failure_and_review_reports_preserved',all(sha(p)==previous[str(p)]for p in preserved),sha256=previous)
    current={str(p.relative_to(U))if p.is_relative_to(U)else str(p):sha(p)for p in watch}
    report=dict(passed=all(c['passed']for c in checks),checks=checks,code_and_artifact_sha256=hashes,files_unchanged_during_review=hashes==current,
        scope='Read-only training separation audit and candidate integration contracts. Root whole-controller public regression is a separate admission gate. Supplemental source is fitting/development, not independent new-person performance.',
        camera=False,private_data=False,training_executed_by_review=False,admission_written_by_review=False,independent_accuracy=None)
    (V/'INDEPENDENT_FT_REVIEW.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],unchanged=hashes==current,checks=[dict(name=c['name'],passed=c['passed'])for c in checks]),ensure_ascii=False))

if __name__=='__main__':main()
