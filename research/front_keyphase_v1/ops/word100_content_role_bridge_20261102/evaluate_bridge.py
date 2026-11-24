import time,traceback
from bridge_model import *
from evidence_contract import observe,behavior,OK
from evaluate_transfer import eval_rows,NORMAL
import evaluate_episode as previous

def main():
    assert not (V/'EVALUATION.json').exists() and read(V/'TRAIN_PROGRESS.json')['status']=='COMPLETE';torch.set_num_threads(2)
    plan=read(V/'FROZEN.json');rr=eval_rows();stats=read(D/'NORMALIZATION.json');graphs=previous.graphs();pins={}
    for r in rr:
        pins[r['input']]=r['input_sha256'];pins.update(dict(r['extra_pins']))
        p=DEST/'predictions/seated_variation'/f"{r['key']}.npz";pins[str(p)]=sha(p)
    exclusive(V/'EVALUATION_PROTOCOL.json',dict(at=clock(),cases=88,comparisons=176,pins=pins,graphs=graphs,behavior=behavior(),whole_goal_complete=False))
    reports=[]
    for tag in plan['conditions']:
        m=BridgeNet(tag=='content_connected');path=OUT/'runs'/tag/'model.pt';assert sha(path)==read(path.parent/'COMPLETE.json')['sha256'];m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));m.eval()
        points=[];roles=[];events=[];seconds=[]
        for r in rr:
            x,im,ll,_,_=batch_j([r],stats);start=time.perf_counter()
            with torch.inference_mode():ac,ro=m(x,im,ll);pr=ac.softmax(-1)[0].numpy();activity=ro.softmax(-1)[0].numpy()
            seconds.append(time.perf_counter()-start);old=np.load(DEST/'predictions/seated_variation'/f"{r['key']}.npz")
            assert np.array_equal(pr,old['raw_action_probability']) and np.array_equal(r['current'],old['current_probability'])
            z=observe(pr,activity,r,graphs[r['word_id']]);sq=z['sequence'];sq.update(key=r['key'],kind=r['kind'],word=r['word_id']);events.append(sq)
            for f in np.flatnonzero(r['episode']>=0):
                base=dict(key=r['key'],clip_id=r['clip_id'],person=r['person'],kind=r['kind'],word=r['word_id'],frame=int(f),source_frame=int(r['mapping'][f]))
                points.append(dict(base,y=int(r['episode'][f]),pred=int(pr[f].argmax()),eligible=bool(z['eligible'][f])))
                roles.append(dict(base,y=int(r['role'][f]),pred=int(z['role_probability'][f].argmax())))
            dest=OUT/'predictions'/tag/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,content_probability=z['content_probability'],role_probability=z['role_probability'],activity_probability=activity,eligible=z['eligible'],
                current_probability=r['current'],image_available=r['available'],frame_id=np.arange(len(pr)),source_frame_id=r['mapping'],timestamp=r['times'])
        major=[p for p in points if p['kind'] in NORMAL and p['y']<8];ns=[s for s in events if s['kind'] in NORMAL]
        report=dict(condition=tag,points=points,role_points=roles,sequences=events,raw_major=process.state.metrics(major),
            main_eligibility_losses=[p for p in major if p['pred']==p['y'] and not p['eligible']],normal_order=sum(s['status']==OK for s in ns),
            normal_clean_order=sum(s['status']==OK and not(s['evaluation_content_errors'] or s['evaluation_qualified_errors'] or s['previous_reviewed_core_unconfirmed']) for s in ns),head_median_seconds=float(np.median(seconds)))
        exclusive(OUT/'evaluations'/f'{tag}.json',report);reports.append({k:v for k,v in report.items() if k not in ['points','role_points','sequences']})
        print(tag,'raw',report['raw_major']['correct'],'excluded_correct',len(report['main_eligibility_losses']),'order',report['normal_order'],'clean',report['normal_clean_order'],flush=True)
    for p,h in pins.items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(V/'EVALUATION.json',dict(at=clock(),reports=reports,pins=pins,cases=88,comparisons=176,whole_goal_complete=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (V/'EVAL_FIRST_FAILURE.txt').exists():(V/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
