import time,traceback
from collections import Counter
from role_model import *
from role_sequence import observe,behavior
NORMAL=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']
def main():
    assert not (A/'EVALUATION.json').exists();assert read(A/'TRAIN_PROGRESS.json')['status']=='COMPLETE';torch.set_num_threads(2)
    plan=read(A/'FROZEN_TRAINING.json');s=read(A/'NORMALIZATION.json');rows=inputs(False);graphs=read(S/'LEARNED_STATE_GRAPH.json')['graphs']
    exclusive(A/'EVALUATION_PROTOCOL.json',dict(at=clock(),conditions=plan['conditions'],seeds=plan['seeds'],primary=plan['primary'],cases=len(rows),gates=plan['development_gate'],sequence_code_sha256=sha(A/'role_sequence.py'),evaluation_code_sha256=sha(A/'evaluate_roles.py'),behavior_tests=behavior(),
      boundary='Exposed development, not new independent final evaluation. No thresholds/seed selection from these results. Old predictions reused and preserved.'))
    pins={r['input']:r['input_sha256'] for r in rows}
    for r in rows:pins.update(dict(r['extra_pins']))
    exclusive(A/'EVALUATION_INPUTS.json',dict(pins=pins,rows=[{k:v for k,v in r.items() if k not in ['x','y','z','m','current','available','times']} for r in rows]))
    reports=[]
    for kind in plan['conditions']:
        for seed in plan['seeds']:
            name=f'{kind}_{seed}';m=RoleNet(kind=='temporal');path=A/'runs'/name/'model.pt';assert sha(path)==read(path.parent/'COMPLETE.json')['sha256'];m.load_state_dict(torch.load(path,weights_only=True));m.eval()
            points=[];sequences=[];secs=[]
            for r in rows:
                x,l,y=batch([r],s);start=time.perf_counter()
                with torch.inference_mode():pr=m(x,l).softmax(-1)[0].numpy()
                elapsed=time.perf_counter()-start;secs.append(elapsed);v=observe(r['current'],pr,r['available'],graphs[r['word_id']],r['times'])
                sequences.append(dict(key=r['key'],word=r['word_id'],kind=r['kind'],**v))
                for i in np.flatnonzero(r['y']>=0):points.append(dict(key=r['key'],word=r['word_id'],person=r['person'],kind=r['kind'],frame=int(i),source_frame=int(r['mapping'][i]),y=int(r['y'][i]),pred=int(pr[i].argmax()),probability=pr[i].tolist()))
                dest=A/'predictions'/name/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();np.savez_compressed(dest,role_probability=pr,current_probability=r['current'],frame_id=np.arange(len(pr)),image_available=r['available'])
            met={k:state.metrics([p for p in points if p['kind']==k]) for k in sorted({p['kind'] for p in points})}
            full=dict(condition=name,metrics=met,points=points,sequences=sequences,head_median_seconds=float(np.median(secs)))
            exclusive(A/'evaluations'/f'{name}.json',full);reports.append({k:v for k,v in full.items() if k not in ['points','sequences']})
            print(name,'native',met['native'],'normal sequence',dict(Counter(v['status'] for v in sequences if v['kind'] in NORMAL)),flush=True)
    assert all(sha(p)==h for p,h in pins.items()) and all(sha(p)==h for p,h in plan['pins'].items())
    exclusive(A/'EVALUATION.json',dict(reports=reports,new_role_comparisons=528,cases=88,old_state_predictions_reused=True,whole_goal_complete=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (A/'EVAL_FIRST_FAILURE.txt').exists():(A/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
