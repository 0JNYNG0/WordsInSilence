"""Fixed source-configuration supervision, no DTW transferred labels in loss."""
import time,os,sys,traceback,argparse
import numpy as np,torch
from torch.nn import functional as F
from phase_model import *

def freeze():
    plan=dict(at=clock(),conditions=['phase_readout','temporal_readout'],seeds=[42,43,44],primary='temporal_readout_42',epochs=100,batch_size=12,learning_rate=.001,weight_decay=.01,
      supervision='Sparse directly reviewed physical configurations. Unknown ignored. Each clip/variant receives equal loss, class balancing from FIT labels only.',
      original_observers_frozen=True,new_backbone_finetuning=False,old_signrep_projection_finetune_retained=True,
      inputs='144 actual native/speed/reversed FIT pixel extractions plus36sensor-only3-frame gap observations with identical RGB. Same original frame state labels follow source maps, no reversed cached-RGB substitution.',
      revision='V2 before any new phase training or development predictions: include the already planned short sensor gaps. Original freeze file preserved; no result-driven change.',
      validation='Pilot head trainingIDs01-06 and head validationIDs07-09. Original frozen backbone has prior exposure to allFIT01-09; this pilot is not a fully unseen-person estimate. FinalDEV11-13 excluded from all training; already repeatedly exposed development.',
      pilot_gate=dict(minimum_macro_recall=.8,no_worse_than_initial_macro=True),
      development_gate=dict(normal_macro_no_regression=True,reversed_macro_gain_minimum=.03,each_group_normal_drop_maximum=.05,old_outputs_preserved=True),
      decoding='Observe neural configurations in native order before any monotonic DTW. Derive strict precedences only if >=3FIT IDs support one direction and none oppose. Cyclic states do not acquire first/last/count rules. Missing observations cannot satisfy the graph.',
      scope='No complete educational correctness or independent expert labels. No H14/15, ID16, private user/mentor input. No old demo/default replacement.',
      pins={str(p):sha(p) for p in [S/'PHYSICAL_STATE_TARGETS.json',S/'SOURCE_AUDIT.json',P/'runs/projection_finetune_42/model.pt',P/'GUIDE_SUPPORT.json',N/'INPUT_MANIFEST.json',V/'runs/rgb_geometry_42/model.pt']})
    if (S/'FROZEN_TRAINING_V2.json').exists():
        old=read(S/'FROZEN_TRAINING_V2.json');plan['at']=old['at'];assert plan==old
    else:exclusive(S/'FROZEN_TRAINING_V2.json',plan)
    return plan

def train(condition,seed,rows,tag,plan):
    dest=S/'runs'/tag;assert not dest.exists(),'No rerun of partial or completed training';dest.mkdir(parents=True)
    torch.manual_seed(seed);rng=np.random.default_rng(seed);torch.set_num_threads(2)
    model=PhaseNet(condition);initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    counts=np.bincount(np.concatenate([r['y'][r['y']>=0] for r in rows]),minlength=9)
    weights=torch.tensor(np.clip(np.mean(counts[counts>0])/np.maximum(counts,1),.25,4),dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay']);start=time.perf_counter();log=[]
    for epoch in range(plan['epochs']):
        model.train();order=rng.permutation(len(rows));losses=[]
        for j in range(0,len(order),plan['batch_size']):
            rr=[rows[k] for k in order[j:j+plan['batch_size']]];x,l,w,y=batch(rr);scores=model(x,l,w)
            ce=F.cross_entropy(scores.transpose(1,2),y,ignore_index=-1,weight=weights,reduction='none');known=y>=0
            loss=(ce.sum(1)/known.sum(1).clamp_min(1)).mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();losses.append(float(loss.detach()))
        if epoch%20==0 or epoch==plan['epochs']-1:
            v=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.perf_counter()-start);log.append(v);print(tag,v,flush=True)
    changes={k:float((v-initial[k]).abs().max()) for k,v in model.state_dict().items() if v.is_floating_point() and k not in ['guide_vectors']}
    assert changes['query.weight']>0 and changes['embedding']>0
    torch.save(model.state_dict(),dest/'model.pt');exclusive(dest/'COMPLETE.json',dict(condition=condition,seed=seed,training_people=sorted({r['person'] for r in rows}),clips=len(rows),seconds=time.perf_counter()-start,log=log,weight_changes=changes,sha256=sha(dest/'model.pt')))
    return model

def preflight(rows,plan):
    torch.set_num_threads(2);base=PhaseNet('phase_readout').eval();temporal=PhaseNet('temporal_readout').eval()
    for r in rows[:4]:
        x,l,w,y=batch([r]);a=base(x,l,w);b=temporal(x,l,w)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        with np.load(r['input']) as old:
            grouped=np.stack([old['stage_probability'][:,ids].sum(1) for ids in base.aliases],-1)
            np.testing.assert_allclose(a.softmax(-1).detach().numpy()[0],grouped,rtol=1e-5,atol=1e-6)
    x,l,w,y=batch(rows[:2]);xx=x.clone().requires_grad_();score=base(xx,l,w);loss=F.cross_entropy(score.transpose(1,2),y,ignore_index=-1,reduction='sum');loss.backward()
    assert torch.all(xx.grad[y<0]==0),'Frame-only ignored positions must contribute no supervision'
    assert all(sha(p)==h for p,h in plan['pins'].items())
    exclusive(S/'PREFLIGHT.json',dict(initial_group_probabilities_match_old=True,temporal_zero_initialization_exact=True,unknown_frame_loss_ignored=True,actual_fit_inputs_verified=len(rows),old_pins_preserved=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args();plan=freeze()
    if args.freeze_only:return
    assert not (S/'TRAINING_RESULT.json').exists();rows=fit_rows()
    if not (S/'PREFLIGHT.json').exists():preflight(rows,plan)
    pilot=[r for r in rows if r['person']<=6];validation=[r for r in rows if r['person']>=7]
    if not (S/'PILOT_VALIDATION.json').exists():
        baseline=measure(PhaseNet('phase_readout'),validation);model=train('temporal_readout',42,pilot,'pilot_temporal_42',plan);pred=measure(model,validation)
        old=metrics(baseline);new=metrics(pred);gate=new['macro_recall']>=.8 and new['macro_recall']>=old['macro_recall']
        exclusive(S/'PILOT_VALIDATION.json',dict(baseline=old,candidate=new,baseline_points=baseline,candidate_points=pred,passed=gate,scope=plan['validation']))
    assert read(S/'PILOT_VALIDATION.json')['passed'],'Preserve failed pilot and diagnose before final runs'
    done=[]
    for condition in plan['conditions']:
        for seed in plan['seeds']:
            tag=f'{condition}_{seed}';complete=S/'runs'/tag/'COMPLETE.json'
            if complete.exists():assert sha(complete.parent/'model.pt')==read(complete)['sha256']
            else:train(condition,seed,rows,tag,plan)
            done.append(read(complete));write(S/'TRAIN_PROGRESS.json',dict(status='RUNNING',pid=os.getpid(),at=clock(),done=len(done),total=6,last=tag))
    assert all(sha(p)==h for p,h in plan['pins'].items())
    exclusive(S/'TRAINING_RESULT.json',dict(runs=done,old_models_preserved=True,new_backbone_updates=0,supervised_phase_head_finetuning=True))
    write(S/'TRAIN_PROGRESS.json',dict(status='COMPLETE',pid=os.getpid(),at=clock(),done=6,total=6))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (S/'TRAIN_FIRST_FAILURE.txt').exists():(S/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
