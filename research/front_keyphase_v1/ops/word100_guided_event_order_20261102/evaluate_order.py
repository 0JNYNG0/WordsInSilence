import time,traceback
from order_model import *
from order_contract import observe,OK
import evaluate_episode as previous
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}

def main():
    assert read(ORDER_UNIT/'TRAIN_PROGRESS.json')['status']=='COMPLETE' and not (ORDER_UNIT/'EVALUATION.json').exists();torch.set_num_threads(2)
    plan=read(ORDER_UNIT/'FROZEN.json');rr=get_rows('dev');stats=read(D/'NORMALIZATION.json');flow=read(FLOW_UNIT/'FLOW_INPUTS.json');graphs=previous.graphs()
    exclusive(ORDER_UNIT/'EVALUATION_PROTOCOL.json',dict(at=clock(),new_cases=88,new_comparisons=176,graphs=graphs,old_model_reexecution=0,whole_goal_complete=False))
    reports=[]
    for tag in plan['conditions']:
        path=ORDER_OUT/'runs'/tag/'model.pt';assert sha(path)==read(path.parent/'COMPLETE.json')['sha256'];m=OccurrenceOrderNet();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));m.eval()
        points=[];roles=[];sequences=[];seconds=[]
        for r in rr:
            x,im,l,_,_=batch_j([r],stats);fl=flow_batch([r],flow);start=time.perf_counter()
            with torch.inference_mode():ac,ro,oc=m(x,im,l,fl);pr=ac.softmax(-1)[0].numpy();activity=ro.softmax(-1)[0].numpy();occ=oc.sigmoid()[0].numpy()
            seconds.append(time.perf_counter()-start);old=np.load(DEST/'predictions/seated_variation'/f"{r['key']}.npz")
            assert np.array_equal(pr,old['raw_action_probability']) and np.array_equal(r['current'],old['current_probability'])
            z=observe(pr,activity,occ,r,graphs[r['word_id']]);s=z['sequence'];s.update(key=r['key'],kind=r['kind'],word=r['word_id']);sequences.append(s)
            for f in np.flatnonzero(r['episode']>=0):
                b=dict(key=r['key'],clip_id=r['clip_id'],person=r['person'],kind=r['kind'],word=r['word_id'],frame=int(f),source_frame=int(r['mapping'][f]))
                points.append(dict(b,y=int(r['episode'][f]),pred=int(pr[f].argmax()),eligible=bool(z['eligible'][f])))
                roles.append(dict(b,y=int(r['role'][f]),pred=int(z['role_probability'][f].argmax())))
            dest=ORDER_OUT/'predictions'/tag/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,content_probability=pr,activity_probability=activity,occurrence_probability=occ,role_probability=z['role_probability'],eligible=z['eligible'],
                current_probability=r['current'],image_available=r['available'],frame_id=np.arange(len(pr)),source_frame_id=r['mapping'],timestamp=r['times'])
        major=[p for p in points if p['kind'] in NORMAL and p['y']<8]
        report=dict(condition=tag,points=points,role_points=roles,sequences=sequences,raw_major=process.state.metrics(major),main_eligibility_losses=[p for p in major if p['y']==p['pred'] and not p['eligible']],
            normal_order=sum(s['kind'] in NORMAL and s['status']==OK for s in sequences),head_median_seconds=float(np.median(seconds)))
        exclusive(ORDER_OUT/'evaluations'/f'{tag}.json',report);reports.append({k:v for k,v in report.items() if k not in ['points','role_points','sequences']})
        print(tag,'raw',report['raw_major']['correct'],'excluded',len(report['main_eligibility_losses']),'order',report['normal_order'],flush=True)
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(ORDER_UNIT/'EVALUATION.json',dict(at=clock(),comparisons=176,reports=reports,whole_goal_complete=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (ORDER_UNIT/'EVAL_FIRST_FAILURE.txt').exists():(ORDER_UNIT/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
