"""One fixed scoped fine-tune; no common100 or deployed model modification."""
from pathlib import Path
import json,hashlib,collections,time,traceback
import numpy as np
import torch
from extent_model import ExtentChoice
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111';N=G/'typed_motion_recovery_20261113/null_candidate_learning'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not(U/'TRAINING_FROZEN.json').exists() and not(U/'TRAINING_COMPLETE.json').exists()
    verification=read(U/'TARGETS_VERIFIED.json');assert verification['passed'] and verification['train_justified']
    for p,h in verification['pins'].items():assert sha(Path(p))==h
    meta=read(U/'ABSENCE_BAGS.json');assert sha(Path(meta['arrays']))==meta['arrays_sha256']
    raw=np.load(meta['arrays']);rows=meta['rows'];data=[(raw[f'x_{i}'],raw[f'y_{i}'])for i in range(len(rows))]
    pools={}
    for i in meta['train_indices']:
        r=rows[i];assert r['person'] in range(2,8)
        pools.setdefault(r['family'],{}).setdefault(r['kind'],{}).setdefault(r['clip_id'],[]).append(i)
    assert sorted(pools)==[3,4,5,7]
    parents=read(N/'TRAINING_COMPLETE.json')['rows'];protections=read(G/'PROTECTIONS.json')
    for p,h in protections.items():assert sha(Path(p))==h
    pins={str(p):sha(p) for p in [Path(__file__),U/'extent_model.py',U/'ABSENCE_BAGS.json',U/'TARGETS_VERIFIED.json',N/'TRAINING_COMPLETE.json']}
    cfg=dict(status='FROZEN_BEFORE_NEW_EXTENT_CHOICE_TRAINING',seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,gradient_clip=2,
        hypothesis='Hold-only null supervision does not teach other-moving-action or partial-source-extent bags. Add both explicit bag types, while retaining full and held examples.',
        architecture='Same local1160->96 GELU->5 rank plus5 learned null logits. Corresponding old null seed weights and normalization initialize each run.',
        inputs='Existing candidate-local common100 Conv+GRU + body/hand relations. No word ID, expected order, frame ID, absolute time, candidate duration or outside-window encoder context added.',
        selection='Fixed final1200step checkpoint; no hyperparameter/seed/epoch selection.',
        sampling='Each step each family equally weighted1/4; each available bag type equally weighted within family; one source clip uniformly, then one derived bag uniformly. All positives and <=56 sampled negatives, same candidate cap as parent null training.',
        loss='Listwise full-positive probability mass or learned null target when no full reviewed source extent exists.',
        labels='Physical source-extent coverage ONLY. Partial-source null is not linguistic error; progression remains separate. Missing observation is not trained as absent action.',
        gates=['Each seed preserve parent known full-source selection and reviewed hold rejection on exposed record08/09.',
               'Improve partial and other-action rejection without accepting fewer known full sources.',
               'Same saved82 public scenarios: preserve normal/speed/cross/shape/direction witnesses and original G2 gates.',
               'No web adoption if any protected gate fails; no final8 before G2/G3 pass.'],
        test_split='Records02..07 FIT;08/09 exposed probe only;11..13 stored full-path development only.',
        no_common100_update=True,no_private=True,no_final8=True,no_web_update=True,
        pins=pins,parents=parents,protected=protections)
    write(U/'TRAINING_FROZEN.json',cfg);torch.set_num_threads(4);torch.set_num_interop_threads(1);runs=[]
    for seed in cfg['seeds']:
        parent=next(p for p in parents if p['seed']==seed);assert sha(Path(parent['model']))==parent['model_sha256']
        torch.manual_seed(seed);rng=np.random.default_rng(seed);st=torch.load(parent['model'],map_location='cpu',weights_only=True)
        net=ExtentChoice(st);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);logs=[];t=time.perf_counter();draws=collections.Counter()
        for step in range(cfg['steps']):
            if any((q/n).exists() for q in [U,G]for n in ['STOP','STOP.txt','STOP_REQUESTED.json']):raise RuntimeError('Stop requested')
            xs=[];pieces=[];off=0
            for family,types in pools.items():
                for kind,clips in types.items():
                    clip=list(clips)[int(rng.integers(len(clips)))];bi=int(rng.choice(clips[clip]));x,y=data[bi]
                    pos=np.flatnonzero(y);neg=np.flatnonzero(~y)
                    neg=rng.choice(neg,min(56,len(neg)),replace=False);ix=np.r_[pos,neg]
                    xs.append(x[ix]);pieces.append((off,off+len(ix),len(pos),[2,3,4,5,7].index(family),1/(4*len(types))));off+=len(ix);draws[bi]+=1
            scores=net(torch.tensor(np.concatenate(xs)));loss=torch.tensor(0.)
            for a,b,np_,j,weight in pieces:
                den=torch.logsumexp(torch.cat([scores[a:b,j],net.null_logits[j:j+1]]),0)
                num=torch.logsumexp(scores[a:a+np_,j],0)if np_ else net.null_logits[j]
                loss=loss+weight*(den-num)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);opt.step()
            if step%300==0 or step==cfg['steps']-1:
                logs.append(dict(step=step+1,loss=float(loss.detach())));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING_FIXED_RECIPE',seed=seed,step=step+1,completed_seeds=[r['seed']for r in runs],no_private=True));print(seed,step+1,float(loss.detach()),flush=True)
        net.eval();dest=U/'runs'/str(seed);dest.mkdir(parents=True);torch.save(net.state_dict(),dest/'model.pt')
        # Infer each original feature bag once, then reuse candidate subsets.
        oldrows=read(N/'TRAINING_INPUT_INDEX.json')['rows'];pred={}
        with torch.inference_mode():
            for i,r in enumerate(oldrows):
                assert sha(Path(r['path']))==r['sha256'];z=np.load(r['path']);pred[f'logits_{i}']=net(torch.tensor(z['local_full'])).numpy()
        pred['null_logits']=net.null_logits.detach().numpy();np.savez_compressed(dest/'predictions.npz',**pred)
        run=dict(seed=seed,model=str(dest/'model.pt'),model_sha256=sha(dest/'model.pt'),predictions=str(dest/'predictions.npz'),predictions_sha256=sha(dest/'predictions.npz'),
            initial_model=parent['model'],initial_sha256=parent['model_sha256'],seconds=time.perf_counter()-t,steps=1200,log=logs,bag_draw_counts=dict(draws))
        runs.append(run);write(U/'TRAINING_PROGRESS.json',dict(rows=runs))
    for p,h in protections.items():assert sha(Path(p))==h
    write(U/'TRAINING_COMPLETE.json',dict(status='THREE_FIXED_SEEDS_COMPLETE_NOT_ADOPTED',rows=runs,training_frozen_sha256=sha(U/'TRAINING_FROZEN.json'),common100_changed=False,web_changed=False))
    print('Training complete; evaluation and verification remain.',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        write(U/f'TRAIN_ERROR_{time.time_ns()}.json',dict(traceback=traceback.format_exc()));raise
