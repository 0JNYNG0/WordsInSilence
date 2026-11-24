import time,traceback
from flow_model import *
from evidence_contract import observe,OK
import evaluate_episode as previous
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}

def main():
    assert read(FLOW_UNIT/'TRAIN_PROGRESS.json')['status']=='COMPLETE' and not (FLOW_UNIT/'EVALUATION.json').exists()
    torch.set_num_threads(2);normal=read(FLOW_UNIT/'FLOW_INPUTS.json');plan=read(FLOW_UNIT/'FROZEN.json');rr=get_rows('dev');stats=read(D/'NORMALIZATION.json')
    ck=FLOW_OUT/'run/model.pt';assert sha(ck)==read(FLOW_UNIT/'TRAINING_RESULT.json')['sha256']
    m=FlowRoleNet();m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True));m.eval();graphs=previous.graphs()
    exclusive(FLOW_UNIT/'EVALUATION_PROTOCOL.json',dict(at=clock(),cases=88,checkpoint_sha256=sha(ck),old_scores_recomputed=0,graphs=graphs,whole_goal_complete=False))
    points=[];roles=[];seqs=[];seconds=[]
    for r in rr:
        x,im,l,_,_=batch_j([r],stats);flow=flow_batch([r],normal);start=time.perf_counter()
        with torch.inference_mode():ac,ro=m(x,im,l,flow);pr=ac.softmax(-1)[0].numpy();activity=ro.softmax(-1)[0].numpy()
        seconds.append(time.perf_counter()-start)
        old=np.load(DEST/'predictions/seated_variation'/f"{r['key']}.npz")
        assert np.array_equal(pr,old['raw_action_probability']) and np.array_equal(r['current'],old['current_probability'])
        z=observe(pr,activity,r,graphs[r['word_id']]);s=z['sequence'];s.update(key=r['key'],kind=r['kind'],word=r['word_id']);seqs.append(s)
        for f in np.flatnonzero(r['episode']>=0):
            b=dict(key=r['key'],clip_id=r['clip_id'],person=r['person'],kind=r['kind'],word=r['word_id'],frame=int(f),source_frame=int(r['mapping'][f]))
            points.append(dict(b,y=int(r['episode'][f]),pred=int(pr[f].argmax()),eligible=bool(z['eligible'][f])))
            roles.append(dict(b,y=int(r['role'][f]),pred=int(z['role_probability'][f].argmax())))
        dest=FLOW_OUT/'predictions'/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
        np.savez_compressed(dest,content_probability=pr,role_probability=z['role_probability'],activity_probability=activity,eligible=z['eligible'],
            current_probability=r['current'],image_available=r['available'],frame_id=np.arange(len(pr)),source_frame_id=r['mapping'],timestamp=r['times'])
    major=[p for p in points if p['kind'] in NORMAL and p['y']<8]
    report=dict(points=points,role_points=roles,sequences=seqs,raw_major=process.state.metrics(major),
        main_eligibility_losses=[p for p in major if p['pred']==p['y'] and not p['eligible']],
        normal_order=sum(s['kind'] in NORMAL and s['status']==OK for s in seqs),head_median_seconds=float(np.median(seconds)))
    exclusive(FLOW_OUT/'evaluation.json',report)
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(FLOW_UNIT/'EVALUATION.json',dict(at=clock(),cases=88,raw_major=report['raw_major'],main_eligibility_losses=len(report['main_eligibility_losses']),
        normal_order=report['normal_order'],head_median_seconds=report['head_median_seconds'],whole_goal_complete=False))
    print('raw',report['raw_major']['correct'],'excluded',len(report['main_eligibility_losses']),'order',report['normal_order'])

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (FLOW_UNIT/'EVAL_FIRST_FAILURE.txt').exists():(FLOW_UNIT/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
