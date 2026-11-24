from retention_model import *
import time,traceback
sys.path.insert(0,str(O/'word100_nonmain_probability_20261102'))
import contract

def main():
    assert not (U/'EVALUATION.json').exists() and read(U/'TRAIN_PROGRESS.json')['status']=='COMPLETE'
    torch.set_num_threads(2);plan=read(U/'TRAINING_FREEZE.json');rows=ex.eval_rows();normal=read(fl.FLOW_UNIT/'FLOW_INPUTS.json');stats=read(ex.D/'NORMALIZATION.json');graphs=ex.previous.graphs()
    for i,r in enumerate(rows):r['flow_key']=f'dev_{i:03d}'
    assert len(rows)==88 and {r['person'] for r in rows}=={11,12,13}
    pins={}
    for r in rows:pins[r['input']]=r['input_sha256'];pins.update(dict(r['extra_pins']))
    exclusive(U/'EVALUATION_PROTOCOL.json',dict(at=clock(),cases=88,comparisons=176,graphs=graphs,pins=pins,new_models_only=True,behavior=contract.behavior(),whole_goal_complete=False))
    reports=[]
    for tag in plan['conditions']:
        ck=OUT/'runs'/tag/'model.pt';assert sha(ck)==read(ck.parent/'COMPLETE.json')['sha256'];m=RetainedRoleNet();m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True));m.eval()
        points=[];cases=[];seconds=[]
        for r in rows:
            x,rgb,l,_,_=ex.batch_j([r],stats);flow=fl.flow_batch([r],normal);start=time.perf_counter()
            with torch.inference_mode():a,b=m(x,rgb,l,flow);pr=a.softmax(-1)[0].numpy();act=b.softmax(-1)[0].numpy()
            seconds.append(time.perf_counter()-start)
            fixed=np.load(ex.OUT/'predictions/new_adapt'/f"{r['key']}.npz");assert np.array_equal(pr,fixed['raw_action_probability'])
            z=contract.observe(pr,act,r,graphs[r['word_id']]);q=z['sequence'];q.update(key=r['key'],kind=r['kind'],word=r['word_id']);cases.append(q)
            for f in np.flatnonzero(r['episode']>=0):points.append(dict(key=r['key'],kind=r['kind'],word=r['word_id'],person=r['person'],frame=int(f),source_frame=int(r['mapping'][f]),y=int(r['episode'][f]),raw_pred=int(pr[f].argmax()),eligible=bool(z['eligible'][f])))
            dest=OUT/'predictions'/tag/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
            np.savez_compressed(dest,content_probability=pr,activity_probability=act,role_probability=z['role_probability'],eligible=z['eligible'],
                current_probability=r['current'],image_available=r['available'],source_frame_id=r['mapping'],timestamp=r['times'],frame_id=np.arange(len(pr)))
        main=[p for p in points if p['kind'] in contract.NORMAL and p['y']<8];ns=[q for q in cases if q['kind'] in contract.NORMAL];rev=[q for q in cases if q['kind']=='actual_reverse_main' and q['precedences']]
        metrics=dict(raw_correct=sum(p['y']==p['raw_pred'] for p in main),used_correct=sum(p['y']==p['raw_pred'] and p['eligible'] for p in main),
            family_used={str(g):sum(p['y']==g and p['raw_pred']==g and p['eligible'] for p in main) for g in range(8)},
            normal_order=sum(q['status']==contract.OK and bool(q['precedences']) for q in ns),
            clean_order=sum(q['status']==contract.OK and bool(q['precedences']) and not(q['evaluation_action_errors'] or q['evaluation_content_errors'] or q['previous_reviewed_core_unconfirmed']) for q in ns),
            short_reverse=sum(q['status']=='PHYSICAL_ORDER_CONFLICT' for q in rev),reverse_false_pass=sum(q['status']==contract.OK for q in rev),false_order=sum(q['status']=='PHYSICAL_ORDER_CONFLICT' for q in ns),
            head_median_seconds=float(np.median(seconds)))
        exclusive(OUT/'evaluations'/f'{tag}.json',dict(condition=tag,points=points,sequences=cases,metrics=metrics,whole_goal_complete=False));reports.append(dict(condition=tag,**metrics));print(tag,metrics,flush=True)
    for p,h in pins.items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(U/'EVALUATION.json',dict(at=clock(),reports=reports,comparisons=176,whole_goal_complete=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (U/'EVAL_FIRST_FAILURE.txt').exists():(U/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
