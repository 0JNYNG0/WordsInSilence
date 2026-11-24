"""Once per frozen condition; no old inference or evaluation rerun."""
import time,traceback
from condition_model import *
import evaluate_episode as old_eval
NORMAL=old_eval.NORMAL;SUPPORTED=old_eval.SUPPORTED

def observe(pr,r,graph):
    # Non-action mass is only a compatibility adapter for the existing sequence
    # calculation. Class9 is still reported as movement, never resting/absent.
    p=np.concatenate([pr[:,:8],(pr[:,8]+pr[:,9])[:,None]],axis=1)
    seq=old_eval.observe(p,r,graph)
    top=pr.argmax(1);conf=pr.max(1);av=r['available']
    seq['rest_observed_frames']=np.flatnonzero((top==8)&(conf>=.8)&av).tolist()
    seq['approach_return_observed_frames']=np.flatnonzero((top==9)&(conf>=.8)&av).tolist()
    seq['uncertain_frames']=np.flatnonzero((conf<.8)|~av).tolist()
    seq['scope']='Target-guide action interpretation with explicit approach/return. Every native frame retained; physical/shape/directed motion outputs untouched. No linguistic irrelevance claim.'
    return seq

def behavior():
    graph={'groups':[0,1,2],'precedences':[[0,1],[1,2],[0,2]]}
    def call(ids,gaps=()):
        n=len(ids);r=dict(current=np.zeros((n,10)),available=np.ones(n,bool),times=np.arange(n)/30,y=np.ones(n,int),episode=np.array(ids))
        r['available'][list(gaps)]=False;return observe(np.eye(10)[ids],r,graph)
    assert call([9,9,0,0,1,1,2,2,9,9,8,8])['status']==SUPPORTED
    assert call([9,9,2,2,1,1,0,0,9,9])['status']=='PHYSICAL_ORDER_CONFLICT'
    assert call([0,0,1,1,0,2,2])['status']!=SUPPORTED
    assert call([0,0,9,9,2,2])['status']!=SUPPORTED
    assert call([0,0,1,1,2,2],[3])['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    q=call([9,9,0,0,1,1,2,2,9,9,8,8]);assert q['comparison_frames']==12 and q['rest_observed_frames']==[10,11] and q['approach_return_observed_frames']==[0,1,8,9]
    return dict(context_is_movement_not_rest=True,native_axis=True,reverse=True,short_contrary=True,missing_action=True,image_gap=True)

def main():
    assert read(E/'TRAIN_PROGRESS.json')['status']=='COMPLETE' and not (E/'EVALUATION.json').exists()
    torch.set_num_threads(2);plan=read(E/'FROZEN.json');stats=read(E/'NORMALIZATION.json')
    targets={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(D/'DEV_ACTION_TARGETS.json')['rows']}
    rows=episode.attach_targets(process.add_rgb(process.old.inputs(False)),targets);graphs=old_eval.graphs();assert len(rows)==88
    pins={r['input']:r['input_sha256'] for r in rows}
    for r in rows:pins.update(dict(r['extra_pins']));pins[r['base_path']]=r['base_sha256']
    for p in [D/'DEV_ACTION_TARGETS.json',E/'evaluate_condition.py',E/'FROZEN.json']:pins[str(p)]=sha(p)
    exclusive(E/'EVALUATION_PROTOCOL.json',dict(at=clock(),primary=plan['primary'],cases=88,comparisons=352,graphs=graphs,behavior=behavior(),pins=pins,old_reinference=0,
      labels='Reuse frozen D sparse evaluation action/rest targets. No new DEV context labels. Interpretation metrics only; not word recognition or all-sign correctness.',
      gain_audit='Same D: no gain accepted when old reviewed core is not confirmed action or new D action witness is misassigned. This audit never filters runtime frames.'))
    reports=[]
    for spec in plan['conditions']:
        name=spec['tag'];path=E/'runs'/name/'model.pt';assert sha(path)==read(path.parent/'COMPLETE.json')['sha256'];m=ConditionNet()
        m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));m.eval();points=[];sequences=[];feedback=[];seconds=[]
        for r in rows:
            x,im,l,y=batch([r],stats,spec['guide']);start=time.perf_counter()
            with torch.inference_mode():pr=m(x,im,l).softmax(-1)[0].numpy()
            seconds.append(time.perf_counter()-start)
            sq=dict(key=r['key'],word=r['word_id'],kind=r['kind'],**observe(pr,r,graphs[r['word_id']]));sequences.append(sq)
            fb=old_eval.previous.compose(r,sq);fb.update(key=r['key'],kind=r['kind'],word=r['word_id']);feedback.append(fb);pins[fb['saved_movement_source']]=fb['saved_movement_sha256']
            for f in np.flatnonzero(r['episode']>=0):points.append(dict(key=r['key'],clip_id=r['clip_id'],person=r['person'],word=r['word_id'],kind=r['kind'],frame=int(f),source_frame=int(r['mapping'][f]),y=int(r['episode'][f]),pred=int(pr[f].argmax()),probability=pr[f].tolist()))
            dest=E/'predictions'/name/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,probability=pr,current_probability=r['current'],image_available=r['available'],frame_id=np.arange(len(pr)))
        norms=[p for p in points if p['kind'] in NORMAL];actions=[p for p in norms if p['y']<8]
        report=dict(condition=name,points=points,sequences=sequences,feedback=feedback,normal_metrics=process.state.metrics(norms),normal_action_metrics=process.state.metrics(actions),head_median_seconds=float(np.median(seconds)))
        exclusive(E/'evaluations'/f'{name}.json',report);reports.append({k:v for k,v in report.items() if k not in ['points','sequences','feedback']})
        print(name,'action',report['normal_action_metrics']['correct'],'/',len(actions),'order',sum(q['kind'] in NORMAL and q['status']==SUPPORTED for q in sequences),flush=True)
    for p,h in pins.items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(E/'EVALUATION.json',dict(at=clock(),reports=reports,cases=88,comparisons=352,pins=pins,whole_goal_complete=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (E/'EVAL_FIRST_FAILURE.txt').exists():(E/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
