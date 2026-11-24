"""New frozen neural heads once; old predictions reused without re-inference."""
import os,time,traceback
from collections import Counter
from state_model import *
from sequence_full import observe
import evaluate_phase as old_eval

def main():
    assert not (T/'EVALUATION.json').exists();assert read(T/'TRAIN_PROGRESS.json')['status']=='COMPLETE';torch.set_num_threads(2)
    rows=old_eval.cases();graph=read(S/'LEARNED_STATE_GRAPH.json')['graphs'];plan=read(T/'FROZEN_TRAINING.json')
    for r in rows:
        r['y']=labels_for(r['clip_id'],r['mapping']);r['y'][~r['available']]=-1
    exclusive(T/'EVALUATION_INPUTS.json',dict(rows=[{k:v for k,v in r.items() if k not in ['x','y','available','times']} for r in rows],no_new_sensor_extraction=True))
    reports=[]
    configs=[('old_phase',None),('previous_temporal_42',None)]+[(f'{c}_{seed}',T/'runs'/f'{c}_{seed}'/'model.pt') for seed in [42,43,44] for c in plan['conditions']]
    for name,path in configs:
        if path is not None:
            model=StateNet(name.startswith('anchored'));model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
        points=[];sequences=[];seconds=[];anchor_checked=anchor_changed=0
        for r in rows:
            if path is None:
                oldname='old_phase' if name=='old_phase' else 'temporal_readout_42';src=S/'predictions'/oldname/f"{r['key']}.npz"
                with np.load(src) as saved:pr=saved['probability'].copy()
                base=pr;allow=np.zeros(len(pr),bool);sec=None
            else:
                x,l,w,y=batch([r]);start=time.perf_counter()
                with torch.inference_mode():out,b,gate=model(x,l,w,details=True);pr=out.softmax(-1)[0].numpy();base=b.softmax(-1)[0].numpy();allow=gate[0].numpy()
                sec=time.perf_counter()-start;seconds.append(sec)
                if name.startswith('anchored'):
                    anchor_checked+=int((~allow).sum());anchor_changed+=int(np.any(pr[~allow]!=base[~allow],axis=1).sum())
                dest=T/'predictions'/name/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
                np.savez_compressed(dest,probability=pr,current_probability=base,context_allowed=allow,frame_id=np.arange(len(pr)),image_available=r['available'])
            ev=observe(pr,r['available'],graph[r['word_id']],r['times']);sequences.append(dict(key=r['key'],word=r['word_id'],kind=r['kind'],head_seconds=sec,**ev))
            for i in np.flatnonzero(r['y']>=0):points.append(dict(key=r['key'],clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],kind=r['kind'],frame=int(i),source_frame=int(r['mapping'][i]),y=int(r['y'][i]),pred=int(pr[i].argmax()),probability=pr[i].tolist()))
        bykind={k:dict(action=metrics([p for p in points if p['kind']==k and p['y']<9]),rest=metrics([p for p in points if p['kind']==k and p['y']==9])) for k in sorted({r['kind'] for r in rows})}
        full=dict(condition=name,metrics=bykind,points=points,sequences=sequences,head_median_seconds=float(np.median(seconds)) if seconds else None,anchor_checked=anchor_checked,anchor_changed=anchor_changed)
        exclusive(T/'evaluations'/f'{name}.json',full);reports.append({k:v for k,v in full.items() if k not in ['points','sequences']})
        print(name,{k:(v['action']['correct'],v['action']['known'],v['rest']['correct'],v['rest']['known']) for k,v in bykind.items() if k in ['native','actual_reverse','actual_reverse_main','speed_0.8','speed_1.25']},flush=True)
    assert all(sha(p)==h for p,h in plan['pins'].items())
    assert all(sha(r['input'])==r['input_sha256'] and sha(r['base_path'])==r['base_sha256'] and sha(r['axis_path'])==r['axis_sha256'] for r in rows)
    exclusive(T/'EVALUATION.json',dict(reports=reports,cases=88,new_head_comparisons=528,old_saved_comparisons=176,total=704,source_scope='Repeatedly exposed development; no new final generalization claim',old_preserved=True))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (T/'EVAL_FIRST_FAILURE.txt').exists():(T/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
