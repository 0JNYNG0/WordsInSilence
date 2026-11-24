"""One diagnostic pass over frozen public candidates; no new pose/encoder run."""
from pathlib import Path
import sys,json,hashlib,time,traceback
import numpy as np,torch
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
from prepare_linker import load
from quality_model import LocalMotionQuality,raw_sequence

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def summary(z,a,b,k):
    h=torch.tensor(z['z'][a:b+1,228:292]);q=torch.tensor(z['z'][a:b+1,292:392].reshape(-1,10,10)[:,k]);f=torch.tensor(z['fh'][a:b+1])
    return torch.cat([h.mean(0),h[0],h[-1],q.mean(0),q[0],q[-1],q[-1]-q[0],f.mean(0),torch.nn.functional.one_hot(torch.tensor(k//2),5),torch.nn.functional.one_hot(torch.tensor(k%2),2),torch.tensor([(b-a+1)/100.])]).numpy()

def prepare():
    assert not (W/'EVAL_INPUTS_COMPLETE.json').exists()
    role=load('_local_motion_eval_role',O/'word100_candidate_role_learning_20261111/role_core.py')
    sources=read(O/'word100_candidate_role_learning_20261111/EVALUATION_INPUTS.json')['rows']+[dict(r,kind='dynamic_complete_blocks') for r in read(U/'dynamic_blocks/INPUTS_COMPLETE.json')['rows']]
    bp={(r['clip_id'],r['kind']):r for r in read(U/'boundary_candidate_probe/RESULTS.json')}
    pp={(r['clip_id'],r['kind']):r for r in read(U/'candidate_quality_probe/RESULTS.json')}
    assert len(sources)==82 and len(bp)==82 and len(pp)==82
    rows=[]
    for s in sources:
        assert sha(s['path'])==s['sha256'];z=np.load(s['path'],allow_pickle=False);key=s['clip_id'],s['kind']
        pool=[dict(c,origin='boundary',span=[c['a'],c['b']]) for c in bp[key]['candidates']]+[dict(c,origin='phase') for c in pp[key]['candidates']]
        series={}
        for c in pool:
            a,b=c['span'];axis=c['axis'];moving=c['moving'];k=axis*2+(moving=='L')
            if k not in series:series[k]=role.prior.wrist_series(z['x'],role.prior.motion.KINDS[axis],moving,'L' if moving=='R' else 'R')
            values,good=series[k];seq=raw_sequence(values,good,z['ts'],a,b)
            p=W/'eval_inputs'/f'{len(rows):05d}.npz';p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,summary=summary(z,a,b,k),sequence=seq)
            rows.append(dict(id=len(rows),path=str(p),sha256=sha(p),clip_id=key[0],kind=key[1],origin=c['origin'],span=[a,b],axis=axis,moving=moving,card=c.get('card'),was_selected=c.get('was_selected'),role_peak=c.get('role_peak'),old_quality=c['quality'],old_quality_class=c['quality_class'],source_input=s['path'],source_sha256=s['sha256'],observed_frames=int(seq[:,8].sum()),frames=len(seq)))
    write(W/'EVAL_INPUTS_COMPLETE.json',dict(at=stamp(),rows=rows,input_clips=82,scope='Previously exposed public development diagnostics, not fresh test. All candidates retained; unknowns not removed.',pose_or_encoder_inference=False,private=False))
    print('Prepared diagnostic candidates',len(rows),flush=True)

def evaluate():
    assert not (W/'EVALUATION_COMPLETE.json').exists()
    cfg=read(W/'EVALUATION_FROZEN.json')
    for p,h in cfg['pins'].items():assert sha(p)==h,p
    runs=read(W/'TRAINING_COMPLETE.json')['runs'];rows=read(W/'EVAL_INPUTS_COMPLETE.json')['rows']
    assert len(runs)==6
    data=[]
    for r in rows:
        assert sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:data.append({k:z[k].copy() for k in ['summary','sequence']})
    torch.set_num_threads(4);torch.set_num_interop_threads(1);out=[]
    for run in runs:
        condition,seed=run['condition'],run['seed'];p=W/'runs'/condition/str(seed)/'DIAGNOSTIC.json'
        assert not p.exists(),'completed diagnostic not rerun'
        assert sha(run['path'])==run['sha256'];net=LocalMotionQuality(condition,seed,O)
        net.load_state_dict(torch.load(run['path'],weights_only=True,map_location='cpu'));net.eval();prob=[];start=time.perf_counter()
        with torch.inference_mode():
            for a in range(0,len(data),64):
                part=data[a:a+64];lengths=[len(v['sequence']) for v in part];seq=np.zeros((len(part),max(lengths),11),np.float32)
                for j,v in enumerate(part):seq[j,:lengths[j]]=v['sequence']
                logits,_=net(torch.tensor(np.stack([v['summary'] for v in part])),torch.tensor(seq),torch.tensor(lengths));prob.extend(logits.softmax(-1).numpy().tolist())
        write(p,dict(condition=condition,seed=seed,probabilities=prob,seconds=time.perf_counter()-start,auxiliary_used=False))
        out.append(dict(condition=condition,seed=seed,path=str(p),sha256=sha(p),seconds=read(p)['seconds']))
        print(condition,seed,'diagnostic',round(read(p)['seconds'],3),'s',flush=True)
    write(W/'EVALUATION_COMPLETE.json',dict(at=stamp(),runs=out,rows_per_run=len(rows),web_changed=False,source_encoder_replay=False,old_quality_A_reused=True,scope='Exposed development candidate quality diagnostic, not end-to-end feedback validation.'))

if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='evaluate':evaluate()
        else:raise ValueError('choose prepare/evaluate')
    except BaseException:
        (W/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
