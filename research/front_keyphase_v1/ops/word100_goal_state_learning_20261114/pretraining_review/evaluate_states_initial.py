"""One fixed evaluation on previously exposed public records, no tuning."""
from pathlib import Path
import json,hashlib,time,collections,copy
import numpy as np
import torch
from state_model import StateHeads,CLASSES,features,consensus,transition_evidence,attach_component
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 with (U/name).open('x',encoding='utf-8')as f:json.dump(value,f,ensure_ascii=False,indent=2)
cfg=read(U/'TRAINING_FROZEN.json');meta=read(U/'LEARNING_INPUTS.json');trained=read(U/'TRAINING_COMPLETE.json');torch.set_num_threads(2)
assert not(U/'EVALUATION.json').exists()
for p,h in cfg['pins'].items():assert sha(Path(p))==h
lookup={(r['clip_id'],r['frame'],r['domain'],r['hand']):r for r in meta['rows']}
predrows=[];metrics=[];models={};state_lookup={}
for kind in cfg['models']:
 models[kind]=[]
 for t in trained['rows']:
    if t['kind']!=kind:continue
    model=StateHeads(kind);model.load_state_dict(torch.load(t['model'],weights_only=True,map_location='cpu'));model.eval();models[kind].append(model)
 for d in CLASSES:
    rr=[r for r in meta['rows']if r['domain']==d]
    pp=[np.load(t['predictions'])[d]for t in trained['rows']if t['kind']==kind]
    decisions=consensus(pp,[r['available']for r in rr],d,cfg['confidence'])
    for r,decision in zip(rr,decisions):
        pr=dict(index=r['index'],kind=kind,split=r['split'],domain=d,clip_id=r['clip_id'],frame=r['frame'],
                hand=r['hand'],truth=r['label'],available=r['available'],decision=decision)
        predrows.append(pr);state_lookup[kind,r['index']]=decision
    for split in ['fit','exposed_probe']:
        ss=[p for p in predrows if(p['kind'],p['domain'],p['split'])==(kind,d,split)and p['truth']is not None]
        metrics.append(dict(kind=kind,domain=d,split=split,n=len(ss),correct=sum(p['decision']==p['truth']for p in ss),
            wrong=sum(p['decision']is not None and p['decision']!=p['truth']for p in ss),unknown=sum(p['decision']is None for p in ss),
            observation_missing=sum(not p['available']for p in ss)))
# Direct-state sequence controls. They do not measure automatic localization.
domain_map={2:'forearm',3:'forearm',4:'arm_region',5:'height',7:'separation'}
paths={2:['wrist_above_elbow','forearm_across'],3:['forearm_across','wrist_above_elbow'],
 4:['upper_arm_side','wrist_side'],5:['active_hand_higher','opposite_hand_higher'],7:['hands_near','hands_apart']}
groups=collections.defaultdict(list)
for r in read(U/'REVIEW_INPUTS.json')['rows']:groups[r['group_id']].append(r)
temporal=[];unscored=[]
for gid,rr in groups.items():
    rr=sorted(rr,key=lambda r:r['review_slot']);f=rr[0]['family'];d=domain_map[f]
    lab=[lookup[(r['clip_id'],r['frame'],d,'R')]for r in rr];truth=[r['label']for r in lab];path=paths[f].copy()
    if f==5 and truth[0]==path[1]:path=path[::-1]
    if not all(s in truth for s in path):
        unscored.append(dict(group_id=gid,reason='Three review frames do not contain both states; full-video transition cannot be marked absent.'));continue
    goal=next(i for i,s in enumerate(truth)if s==path[-1]);start=next(i for i,s in enumerate(truth)if s==path[0])
    if goal<=start:unscored.append(dict(group_id=gid,reason='No reviewed ordered pair'));continue
    first=lab[start];last=lab[goal]
    # Two direct witnesses are enough for this very narrow state-change claim;
    # never sufficient to claim shape, actual trajectory or whole-sign accuracy.
    variants=[('normal_reviewed',lab,None),('short_same_states',[first,last],None),
        ('slow_same_states',[first,first,last,last],None),('stop_before_goal',[first,first],None),
        ('goal_pose_only',[last,last],None),('reversed_states',[last,first],None),
        ('missing_goal_observation',[first,last],'last'),('all_missing',[first,last],'all')]
    for name,seq,mask in variants:
        gold=[r['label']for r in seq];avail=[True]*len(seq)
        if mask=='last':gold[-1]=None;avail[-1]=False
        if mask=='all':gold=[None]*len(seq);avail=[False]*len(seq)
        expected=transition_evidence(gold,path,avail)
        for kind in cfg['models']:
            states=[state_lookup[kind,r['index']]for r in seq];actual_av=[r['available']for r in seq]
            if mask=='last':states[-1]=None;actual_av[-1]=False
            if mask=='all':states=[None]*len(seq);actual_av=[False]*len(seq)
            actual=transition_evidence(states,path,actual_av)
            temporal.append(dict(group_id=gid,person=rr[0]['person'],split=rr[0]['split'],family=f,domain=d,kind=kind,variant=name,
                frames=[r['frame']for r in seq],label_indices=[r['index']for r in seq],path=path,states=states,available=actual_av,
                expected=expected['status'],actual=actual,correct=actual['status']==expected['status']))
summary=[]
for kind in cfg['models']:
 for split in ['fit','exposed_probe']:
  for variant in sorted({r['variant']for r in temporal}):
    rr=[r for r in temporal if(r['kind'],r['split'],r['variant'])==(kind,split,variant)]
    summary.append(dict(kind=kind,split=split,variant=variant,n=len(rr),correct=sum(r['correct']for r in rr)))
# Preserve all existing located comparison fields, even when the new diagnostic
# reports insufficient goal evidence. Uses saved cases, not another model run.
old=read(O/'word100_action_completion_binding_20261114/FULLPATH_COMPARISON.json');kept=0
for r in old['rows']:
    located=r['conditions']['fixed_rank'];original=copy.deepcopy(located)
    new=attach_component(located,[dict(status='observed_before_goal',semantic_completion=None),dict(status='observation_insufficient',semantic_completion=None)])
    assert located==original
    del new['goal_state_diagnostics'];del new['goal_state_diagnostics_display_enabled'];assert new==original;kept+=1
# Invariance + visibility and prefix-causality tests operate on raw coordinates.
data=np.load(meta['arrays']);checks=0;maxdiff=0.
for cid in sorted({r['clip_id']for r in meta['rows']}):
    x=data[cid+'_x'];sel=[r for r in meta['rows']if r['clip_id']==cid]
    for d,hand in sorted({(r['domain'],r['hand'])for r in sel}):
        a,ok=features(x,d,hand)
        trans=x.copy();pts=trans[:,:102].reshape(-1,51,2);pts[:]=pts*1.4+np.array([.9,-.5])
        b,v=features(trans,d,hand);diff=float(np.max(np.abs(a[ok]-b[ok])))if ok.any()else 0
        assert np.array_equal(ok,v)and diff<1e-4;checks+=1;maxdiff=max(maxdiff,diff)
        rot=x.copy();pts=rot[:,:102].reshape(-1,51,2);t=np.deg2rad(10);mat=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        pts[:]=pts@mat.T;b,v=features(rot,d,hand);assert np.array_equal(ok,v)and np.allclose(a,b,atol=1e-4);checks+=1
        swap=[0,2,1,4,3,6,5,8,7]+list(range(30,51))+list(range(9,30));mirror=x.copy()
        pts=mirror[:,:102].reshape(-1,51,2);pts[:,:,0]*=-1;pts[:]=pts[:,swap];mirror[:,186:237]=mirror[:,186:237][:,swap]
        b,v=features(mirror,d,'L'if hand=='R'else'R');assert np.array_equal(ok,v)and np.allclose(a,b,atol=1e-4);checks+=1
        blank=x.copy();blank[:,186:237]=0;b,v=features(blank,d,hand);assert not v.any()and not b.any();checks+=1
        cut=len(x)//2;b,v=features(x[:cut],d,hand);assert np.array_equal(ok[:cut],v)and np.array_equal(a[:cut],b);checks+=1
# Only the added component is timed. RTMW, TCN/GRU, camera and UI excluded.
times=[]
for cid in sorted({r['clip_id']for r in meta['rows']if r['split']=='exposed_probe'}):
    x=data[cid+'_x'];start=time.perf_counter()
    with torch.no_grad():
        for d in CLASSES:
            f,ok=features(x,d,'R');pp=[torch.softmax(m(torch.from_numpy(f),d),-1).numpy()for m in models['mlp']]
            consensus(pp,ok,d,cfg['confidence'])
    times.append(dict(clip_id=cid,frames=len(x),seconds=time.perf_counter()-start))
admissible={d:all(m['correct']==m['n']for m in metrics if(m['kind'],m['split'],m['domain'])==('mlp','exposed_probe',d))for d in CLASSES}
write('EVALUATION.json',dict(metrics=metrics,predictions=predrows,temporal=temporal,temporal_summary=summary,unscored=unscored,
    observable_state_probe_pass=admissible,invariance_checks=checks,max_feature_difference=maxdiff,preserved_existing_conditions=kept,
    timing=times,timing_scope='Added four state heads/three seeds on cached joints only. Not end-to-end latency.',
    compared_old_feedback_predictions=False,baseline='Matched newly trained linear state classifier. Old web has no equivalent endstate output; old cards preserved separately.',
    automatic_localization_tested=False,whole_sign_correctness_tested=False,real_user_errors_tested=False,web_changed=False,
    training_complete_sha256=sha(U/'TRAINING_COMPLETE.json'),private_used=False,final8_opened=False))
print(json.dumps(dict(probe=[m for m in metrics if m['split']=='exposed_probe'],preserved=kept,invariance=checks,admissible=admissible),ensure_ascii=False))
