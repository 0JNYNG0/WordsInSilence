import time,traceback
from model import *
import evaluate_episode as previous
NORMAL=previous.NORMAL;SUPPORTED=previous.SUPPORTED

def observe(pr,role,r,graph):
    actor=pr.copy();rp=role.argmax(1);rc=role.max(1);available=r['available'];interpreted=pr.copy()
    contextual=(rp!=1)&(rc>=.8)&available
    interpreted[contextual]=0;interpreted[contextual,8]=1
    seq=previous.observe(interpreted,r,graph)
    witnessed=[];top=pr.argmax(1);conf=pr.max(1);main=(rp==1)&(rc>=.8)&available
    for g in graph['groups']:
        yes=(top==g)&(conf>=.8)&main;stable=(yes&np.r_[False,yes[:-1]])|(yes&np.r_[yes[1:],False])
        witnessed.append(dict(group=g,frames=np.flatnonzero(stable).tolist(),observed=bool(stable.any())))
    if not all(w['observed'] for w in witnessed) and seq['status'] in [SUPPORTED,'CONFIGURATIONS_OBSERVED_ORDER_RULE_UNSUPPORTED']:seq['status']='ACTION_ROLE_OR_CONTENT_UNCONFIRMED'
    seq.update(role_main_witnesses=witnessed,context_frames=np.flatnonzero(contextual&(rp==0)).tolist(),
      rest_observed_frames=np.flatnonzero(contextual&(rp==2)).tolist(),role_uncertain_frames=np.flatnonzero((rc<.8)|~available).tolist(),
      raw_action_states=previous.observe(actor,r,graph)['states'],role_does_not_replace_observation=True,
      scope='Simultaneous raw action content and process-role channels. Only confident context changes event interpretation. Unknown roles retain short contrary candidates. Native timeline, shape, body relation and motion remain intact.')
    assert np.array_equal(actor,pr)
    return seq,interpreted

def behavior():
    graph=dict(groups=[0,1,2],precedences=[[0,1],[0,2],[1,2]])
    def call(ids,roles,gaps=(),ambiguous=()):
        n=len(ids);r=dict(current=np.zeros((n,10)),available=np.ones(n,bool),times=np.arange(n)/30,y=np.ones(n,int),episode=np.asarray(ids))
        r['available'][list(gaps)]=False;pr=np.eye(9)[ids];rr=np.eye(3)[roles];rr[list(ambiguous)]=1/3
        q,after=observe(pr,rr,r,graph);assert q['comparison_frames']==n and np.array_equal(pr,np.eye(9)[ids]);return q
    assert call([0,0,1,1,2,2,1,1],[1]*6+[0,0])['status']==SUPPORTED
    assert call([2,2,1,1,0,0],[1]*6)['status']=='PHYSICAL_ORDER_CONFLICT'
    assert call([0,0,1,1,0,2,2],[1]*7)['status']!=SUPPORTED
    assert call([0,0,1,1,0,2,2],[1]*7,ambiguous=[4])['status']!=SUPPORTED
    assert call([0,0,2,2],[1]*4)['status']!=SUPPORTED
    assert call([0,0,1,1,2,2],[1]*6,gaps=[3])['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    assert call([0,0,1,1,2,2],[0]*6)['status']!=SUPPORTED
    return dict(context_not_action_completion=True,missing_action_held=True,short_contrary_retained=True,uncertain_role_not_erased=True,reverse_detected=True,missing_image_held=True,native_time_and_raw_preserved=True)

def main():
    assert read(F/'TRAIN_PROGRESS.json')['status']=='COMPLETE' and not (F/'EVALUATION.json').exists();torch.set_num_threads(2)
    plan=read(F/'FROZEN.json');stats=read(D/'NORMALIZATION.json');targets={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(D/'DEV_ACTION_TARGETS.json')['rows']}
    rows=episode.attach_targets(process.add_rgb(process.old.inputs(False)),targets)
    for r in rows:r['role']=np.where(r['episode']<0,-1,np.where(r['episode']<8,1,2)).astype('int64')
    graphs=previous.graphs();pins={r['input']:r['input_sha256'] for r in rows}
    for r in rows:pins.update(dict(r['extra_pins']));pins[r['base_path']]=r['base_sha256']
    for p in [F/'FROZEN.json',F/'evaluate_split.py',D/'DEV_ACTION_TARGETS.json']:pins[str(p)]=sha(p)
    exclusive(F/'EVALUATION_PROTOCOL.json',dict(at=clock(),primary=plan['primary'],graphs=graphs,pins=pins,behavior=behavior(),cases=88,comparisons=176,old_standalone_inference_or_rescore=0,
      metrics='Both raw action and role-qualified interpretation. Gate on effective interpretation so excluding a true action is an error. DEV role labels only known action/rest; no claim of measured approach/return accuracy.',
      uncertainty='A clear context role is a learned interpretation, never deletion or linguistic irrelevance. Strong short contrary candidates with uncertain roles remain. Audit true action/core loss explicitly.'))
    reports=[]
    for condition in plan['conditions']:
        name=condition+'_42';path=F/'runs'/name/'model.pt';assert sha(path)==read(path.parent/'COMPLETE.json')['sha256'];m=SplitNet(condition=='frozen_encoder');m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));m.eval()
        points=[];raw_points=[];role_points=[];sequences=[];feedback=[];seconds=[]
        for r in rows:
            x,im,l,_,_=batch([r],stats);start=time.perf_counter()
            with torch.inference_mode():ac,ro=m(x,im,l);pr=ac.softmax(-1)[0].numpy();rp=ro.softmax(-1)[0].numpy()
            seconds.append(time.perf_counter()-start);q,interpreted=observe(pr,rp,r,graphs[r['word_id']]);sq=dict(key=r['key'],word=r['word_id'],kind=r['kind'],**q);sequences.append(sq)
            fb=previous.previous.compose(r,sq);fb.update(key=r['key'],kind=r['kind'],word=r['word_id']);feedback.append(fb);pins[fb['saved_movement_source']]=fb['saved_movement_sha256']
            for f in np.flatnonzero(r['episode']>=0):
                base=dict(key=r['key'],clip_id=r['clip_id'],person=r['person'],word=r['word_id'],kind=r['kind'],frame=int(f),source_frame=int(r['mapping'][f]))
                points.append(dict(base,y=int(r['episode'][f]),pred=int(interpreted[f].argmax()),probability=interpreted[f].tolist()))
                raw_points.append(dict(base,y=int(r['episode'][f]),pred=int(pr[f].argmax()),probability=pr[f].tolist()))
                role_points.append(dict(base,y=int(r['role'][f]),pred=int(rp[f].argmax()),probability=rp[f].tolist()))
            dest=F/'predictions'/name/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,probability=interpreted,raw_action_probability=pr,role_probability=rp,current_probability=r['current'],image_available=r['available'],frame_id=np.arange(len(pr)))
        normal=[p for p in points if p['kind'] in NORMAL];raw=[p for p in raw_points if p['kind'] in NORMAL and p['y']<8]
        report=dict(condition=name,points=points,raw_points=raw_points,role_points=role_points,sequences=sequences,feedback=feedback,
          normal_metrics=process.state.metrics(normal),normal_action_metrics=process.state.metrics([p for p in normal if p['y']<8]),raw_normal_action_metrics=process.state.metrics(raw),normal_role_metrics=process.state.metrics([p for p in role_points if p['kind'] in NORMAL]),head_median_seconds=float(np.median(seconds)))
        exclusive(F/'evaluations'/f'{name}.json',report);reports.append({k:v for k,v in report.items() if k not in ['points','raw_points','role_points','sequences','feedback']})
        print(name,'effective',report['normal_action_metrics']['correct'],'raw',report['raw_normal_action_metrics']['correct'],'order',sum(q['kind'] in NORMAL and q['status']==SUPPORTED for q in sequences),flush=True)
    for p,h in pins.items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(F/'EVALUATION.json',dict(at=clock(),reports=reports,pins=pins,cases=88,comparisons=176,whole_goal_complete=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (F/'EVAL_FIRST_FAILURE.txt').exists():(F/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
