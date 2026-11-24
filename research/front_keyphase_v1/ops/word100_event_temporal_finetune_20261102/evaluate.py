import time,traceback
from experiment import *

def main():
    assert not (U/'EVALUATION.json').exists();assert read(U/'TRAIN_PROGRESS.json')['status']=='COMPLETE'
    plan=read(U/'FROZEN.json');torch.set_num_threads(2);rows=eval_rows();stats=read(D/'NORMALIZATION.json');graphs=previous.graphs();pins={}
    for r in rows:
        pins[r['input']]=r['input_sha256'];pins.update(dict(r['extra_pins']));pins[r['base_path']]=r['base_sha256']
    pins[str(D/'DEV_ACTION_TARGETS.json')]=sha(D/'DEV_ACTION_TARGETS.json')
    exclusive(U/'EVALUATION_PROTOCOL.json',dict(at=clock(),cases=88,comparisons=352,pins=pins,graphs=graphs,behavior=behavior(),old_inference=0,whole_goal_complete=False))
    reports=[]
    for tag in plan['conditions']:
        path=OUT/'runs'/tag/'model.pt';assert sha(path)==read(path.parent/'COMPLETE.json')['sha256']
        m=model_for(tag);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));m.eval()
        points=[];raw=[];roles=[];sequences=[];seconds=[]
        for r in rows:
            x,im,l,_,_=batch_j([r],stats);start=time.perf_counter()
            with torch.inference_mode():ac,act,_=m(x,im,l);pr=ac.softmax(-1)[0].numpy();ap=act.softmax(-1)[0].numpy()
            seconds.append(time.perf_counter()-start);sq,effective,ro=observe(pr,ap,r,graphs[r['word_id']])
            sq.update(key=r['key'],kind=r['kind'],word=r['word_id'],raw_evaluation_action_errors=[int(f) for f in np.flatnonzero((r['episode']>=0)&(r['episode']<8)) if pr[f].argmax()!=r['episode'][f]])
            sequences.append(sq)
            for f in np.flatnonzero(r['episode']>=0):
                meta=dict(key=r['key'],clip_id=r['clip_id'],person=r['person'],word=r['word_id'],kind=r['kind'],frame=int(f),source_frame=int(r['mapping'][f]))
                points.append(dict(meta,y=int(r['episode'][f]),pred=int(effective[f].argmax())))
                raw.append(dict(meta,y=int(r['episode'][f]),pred=int(pr[f].argmax())))
                roles.append(dict(meta,y=int(r['role'][f]),pred=int(ro[f].argmax())))
            dest=OUT/'predictions'/tag/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,probability=effective,raw_action_probability=pr,role_probability=ro,activity_probability=ap,
                current_probability=r['current'],image_available=r['available'],frame_id=np.arange(len(pr)),source_frame_id=r['mapping'],timestamp=r['times'])
        rep=dict(condition=tag,points=points,raw_points=raw,role_points=roles,sequences=sequences,head_median_seconds=float(np.median(seconds)),whole_feedback_certified=False)
        rep['summary']=summary(rep);exclusive(OUT/'evaluations'/f'{tag}.json',rep);reports.append(dict(condition=tag,**rep['summary']));print(tag,rep['summary'],flush=True)
    for p,h in pins.items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(U/'EVALUATION.json',dict(at=clock(),reports=reports,cases=88,comparisons=352,whole_goal_complete=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (U/'EVAL_FIRST_FAILURE.txt').exists():(U/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
