"""One saved-input head comparison; no completed sensor/backbone re-extraction."""
import os,time,traceback
from collections import Counter
import numpy as np,torch
from phase_model import *
from sequence_evidence import learn_graph,observe

def cases():
    targets=read(S/'PHYSICAL_STATE_TARGETS.json');meta=read(P/'INPUT_MANIFEST.json')['rows'];result=[]
    def add(cid,word,kind,path,mapping,base,available=None,axis=None):
        a=dict(np.load(path));n=len(a['latent']);assert len(mapping)==n
        aa=a if axis is None else dict(np.load(axis))
        assert len(aa['frame_id'])==n and np.array_equal(aa['frame_id'],np.arange(n))
        b=read(base);ix=sorted(set(i for s in b['stages'] for i in s['frame_ids']));assert len(ix)>0
        full=list(range(min(ix),max(ix)+1));assert max(full)<n
        av=a.get('image_available',np.ones(n,bool)) if available is None else available
        y=labels_for(cid,mapping);y[~np.asarray(av,bool)]=-1
        result.append(dict(key=f'{cid}_{kind}',clip_id=cid,word_id=word,kind=kind,person=int(cid.split('REAL')[1][:2]),input=str(path),input_sha256=sha(path),base_path=str(base),base_sha256=sha(base),
          axis_path=str(axis) if axis is not None else str(path),axis_sha256=sha(axis if axis is not None else path),
          x=a['latent'].copy(),y=y,mapping=mapping,scope=full,available=np.asarray(av,bool),times=aa['timestamp_us']/1e6))
    for r in meta:
        if r['role']!='development':continue
        cid=r['clip_id'];path=P/'runs/projection_finetune_42/predictions'/f'{cid}.npz'
        add(cid,r['word_id'],'native',path,list(range(r['frames'])),V/'development_feedback'/f'rgb_geometry_42_{cid}.json')
    for r in read(M/'REVERSE_MANIFEST.json')['rows']:
        d=M/'reverse_extracted'/r['clip_id'];add(r['clip_id'],r['word_id'],'actual_reverse',d/'analysis.npz',r['source_frame_indices'],d/'RESULT.json')
    for r in read(U/'VIDEO_CONTROLS.json')['rows']:
        d=U/'actual_controls'/f"{r['word_id']}_{r['kind']}";cid=f"NIA_SL_{r['word_id']}_REAL12_F"
        add(cid,r['word_id'],r['kind'],d/'analysis.npz',r['source_frame_indices'],d/'RESULT.json')
    for r in read(G/'MAIN_ORDER_RESULTS.json')['rows']:
        add(f"NIA_SL_{r['word']}_REAL12_F",r['word'],'actual_reverse_main',r['input'],r['source_frame_indices'],r['result'])
    for r in read(G/'SENSOR_GAP_RESULTS.json')['rows']:
        cid=f"NIA_SL_{r['word']}_REAL12_F";path=G/'gap_predictions'/f"{r['case']}.npz";n=len(np.load(path)['latent']);available=np.ones(n,bool)
        if r['kind']=='pose_and_image_7':available[r['frames']]=False
        axis=U/'actual_controls'/f"{r['word']}_reencoded_normal"/'analysis.npz'
        add(cid,r['word'],f"{r['kind']}_f{r['center']}",path,list(range(n)),G/'gap_feedback'/f"{r['case']}.json",available,axis)
    assert len(result)==88 and len({r['key'] for r in result})==88
    return result

def main():
    assert not (S/'EVALUATION.json').exists(),'Do not repeat completed comparison';assert read(S/'TRAIN_PROGRESS.json')['status']=='COMPLETE'
    torch.set_num_threads(2);rows=cases();graph=learn_graph();exclusive(S/'LEARNED_STATE_GRAPH.json',graph)
    exclusive(S/'EVALUATION_INPUTS.json',dict(rows=[{k:v for k,v in r.items() if k not in ['x','y','available','times']} for r in rows],new_sensor_or_backbone_inference=0))
    configs=[('old_phase',None)]+[(f'{c}_{seed}',S/'runs'/f'{c}_{seed}'/'model.pt') for c in ['phase_readout','temporal_readout'] for seed in [42,43,44]];reports=[]
    for name,path in configs:
        condition='temporal_readout' if name.startswith('temporal') else 'phase_readout';model=PhaseNet(condition)
        if path is not None:model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        model.eval();points=[];seq=[];seconds=[]
        for r in rows:
            x,l,w,y=batch([r]);t=time.perf_counter()
            with torch.inference_mode():pr=model(x,l,w).softmax(-1)[0].numpy()
            elapsed=time.perf_counter()-t;seconds.append(elapsed)
            evidence=observe(pr,r['available'],r['scope'],graph['graphs'][r['word_id']],r['times'])
            seq.append(dict(key=r['key'],word=r['word_id'],person=r['person'],kind=r['kind'],head_seconds=elapsed,**evidence))
            for i in np.flatnonzero(r['y']>=0):points.append(dict(key=r['key'],clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],kind=r['kind'],frame=int(i),source_frame=int(r['mapping'][i]),y=int(r['y'][i]),pred=int(pr[i].argmax()),probability=pr[i].tolist()))
            dest=S/'predictions'/name/f"{r['key']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();np.savez_compressed(dest,probability=pr,frame_id=np.arange(len(pr)),image_available=r['available'])
        bykind={k:metrics([p for p in points if p['kind']==k]) for k in sorted({p['kind'] for p in points})}
        report=dict(condition=name,metrics=bykind,points=points,sequences=seq,head_median_seconds=float(np.median(seconds)))
        exclusive(S/'evaluations'/f'{name}.json',report);reports.append(dict(condition=name,metrics=bykind,head_median_seconds=report['head_median_seconds'],sequence_counts=dict(Counter(r['status'] for r in seq))))
        print(name,{k:(v['correct'],v['known'],round(v['macro_recall'],4)) for k,v in bykind.items() if k in ['native','actual_reverse','actual_reverse_main','speed_0.8','speed_1.25']},flush=True)
    assert all(sha(r['input'])==r['input_sha256'] and sha(r['base_path'])==r['base_sha256'] and sha(r['axis_path'])==r['axis_sha256'] for r in rows)
    exclusive(S/'EVALUATION.json',dict(reports=reports,cases_per_condition=88,comparisons=616,old_arrays_and_feedback_preserved=True,
      scope='Already-exposed development physical-state accuracy and coarse-window order diagnostic. Not independent final/educational evaluation.'))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (S/'EVAL_FIRST_FAILURE.txt').exists():(S/'EVAL_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
