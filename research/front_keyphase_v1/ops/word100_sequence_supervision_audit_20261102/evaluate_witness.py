import time,traceback
from sequence_model import *
import evaluate_split as prior
NORMAL=prior.NORMAL;SUPPORTED=prior.SUPPORTED

def observe(pr,activity,r,graph):
    role,conflict=hierarchy(activity,r['current'],r['available'])
    used=role.copy();used[conflict]=1/3
    sq,effective=prior.observe(pr,used,r,graph)
    if conflict.any() and sq['status']!='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED':
        sq['status']='REST_ACTION_OBSERVATION_CONFLICT'
    sq.update(rest_action_conflict_frames=np.flatnonzero(conflict).tolist(),retained_rest_probability=True,
        sequence_training_scope='CTC only labelled witness subsequence, no full transcript or constrained decoding.',
        rest_is_observation_not_truth=True)
    return sq,effective,role

def behavior():
    graph=dict(groups=[0,1,2],precedences=[[0,1],[0,2],[1,2]])
    def call(ids,activity,rest=None,gaps=()):
        n=len(ids);cur=np.zeros((n,10));cur[:,0]=1
        if rest is not None:
            cur[:]=0;cur[:,0]=1-np.asarray(rest);cur[:,9]=rest
        r=dict(current=cur,available=np.ones(n,bool),times=np.arange(n)/30,y=np.ones(n,int),episode=np.asarray(ids))
        r['available'][list(gaps)]=False
        return observe(np.eye(9)[ids],np.eye(2)[activity],r,graph)[0]
    assert call([0,0,1,1,2,2],[1]*6)['status']==SUPPORTED
    assert call([2,2,1,1,0,0],[1]*6)['status']=='PHYSICAL_ORDER_CONFLICT'
    assert call([0,0,1,1,0,2,2],[1]*7)['status']!=SUPPORTED
    assert call([0,0,2,2],[1]*4)['status']!=SUPPORTED
    assert call([0,0,1,1,2,2],[1]*6,gaps=[3])['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    q=call([0,0,1,1,2,2],[1]*6,rest=[0,0,1,1,0,0]);assert q['status']=='REST_ACTION_OBSERVATION_CONFLICT' and q['rest_action_conflict_frames']==[2,3]
    assert q['comparison_frames']==6
    return dict(rest_conflict_held=True,short_contrary_preserved=True,no_forced_order=True,missing_image_held=True,native_axis=True)

def main():
    assert read(J/'TRAIN_PROGRESS.json')['status']=='COMPLETE' and not (J/'EVALUATION.json').exists();torch.set_num_threads(2)
    plan=read(J/'FROZEN.json');stats=read(D/'NORMALIZATION.json')
    targets={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(D/'DEV_ACTION_TARGETS.json')['rows']}
    rows=episode.attach_targets(process.add_rgb(process.old.inputs(False)),targets)
    for r in rows:r['role']=np.where(r['episode']<0,-1,np.where(r['episode']<8,1,2)).astype('int64')
    graphs=prior.previous.graphs();pins={r['input']:r['input_sha256'] for r in rows}
    for r in rows:pins.update(dict(r['extra_pins']));pins[r['base_path']]=r['base_sha256']
    pins[str(D/'DEV_ACTION_TARGETS.json')]=sha(D/'DEV_ACTION_TARGETS.json')
    exclusive(J/'EVALUATION_PROTOCOL.json',dict(at=clock(),cases=88,comparisons=176,primary=plan['primary'],behavior=behavior(),graphs=graphs,pins=pins,
        old_inference_or_rescore=0,whole_goal_complete=False))
    reports=[]
    for condition in plan['conditions']:
        name=condition+'_42';path=J/'runs'/name/'model.pt';assert sha(path)==read(path.parent/'COMPLETE.json')['sha256']
        model=WitnessNet();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
        points=[];raw_points=[];role_points=[];sequences=[];feedback=[];seconds=[]
        for r in rows:
            x,im,l,_,_=batch_j([r],stats);start=time.perf_counter()
            with torch.inference_mode():ac,act,seq=model(x,im,l);pr=ac.softmax(-1)[0].numpy();activity=act.softmax(-1)[0].numpy()
            seconds.append(time.perf_counter()-start)
            q,effective,role=observe(pr,activity,r,graphs[r['word_id']]);sq=dict(key=r['key'],word=r['word_id'],kind=r['kind'],**q);sequences.append(sq)
            fb=prior.previous.previous.compose(r,sq);fb.update(key=r['key'],word=r['word_id'],kind=r['kind']);feedback.append(fb);pins[fb['saved_movement_source']]=fb['saved_movement_sha256']
            for f in np.flatnonzero(r['episode']>=0):
                base=dict(key=r['key'],clip_id=r['clip_id'],person=r['person'],word=r['word_id'],kind=r['kind'],frame=int(f),source_frame=int(r['mapping'][f]))
                points.append(dict(base,y=int(r['episode'][f]),pred=int(effective[f].argmax()),probability=effective[f].tolist()))
                raw_points.append(dict(base,y=int(r['episode'][f]),pred=int(pr[f].argmax()),probability=pr[f].tolist()))
                role_points.append(dict(base,y=int(r['role'][f]),pred=int(role[f].argmax()),probability=role[f].tolist()))
            dest=J/'predictions'/name/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,probability=effective,raw_action_probability=pr,role_probability=role,activity_probability=activity,
                current_probability=r['current'],image_available=r['available'],frame_id=np.arange(len(pr)))
        norms=[p for p in points if p['kind'] in NORMAL];raw=[p for p in raw_points if p['kind'] in NORMAL and p['y']<8]
        report=dict(condition=name,points=points,raw_points=raw_points,role_points=role_points,sequences=sequences,feedback=feedback,
            normal_metrics=process.state.metrics(norms),normal_action_metrics=process.state.metrics([p for p in norms if p['y']<8]),
            raw_normal_action_metrics=process.state.metrics(raw),normal_role_metrics=process.state.metrics([p for p in role_points if p['kind'] in NORMAL]),head_median_seconds=float(np.median(seconds)))
        exclusive(J/'evaluations'/f'{name}.json',report);reports.append({k:v for k,v in report.items() if k not in ['points','raw_points','role_points','sequences','feedback']})
        print(name,'action',report['normal_action_metrics']['correct'],'raw',report['raw_normal_action_metrics']['correct'],'order',sum(q['kind'] in NORMAL and q['status']==SUPPORTED for q in sequences),flush=True)
    for p,h in pins.items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(J/'EVALUATION.json',dict(at=clock(),reports=reports,pins=pins,cases=88,comparisons=176,whole_goal_complete=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (J/'EVAL_FIRST_FAILURE.txt').exists():(J/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
