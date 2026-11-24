"""Two-stage supervised fine-tuning with frozen observer and current anchor."""
import time,os,traceback,argparse
from torch.nn import functional as F
from state_model import *

def freeze():
    plan=dict(at=clock(),primary='anchored_context_42',seeds=[42,43,44],conditions=['current','anchored_context'],
      epochs_current=100,epochs_context=100,batch_size=12,learning_rate=.001,weight_decay=.01,
      inputs='216FIT variants36originalclips:108normal/speed,36fullreverse,36pose3gap,36localreorder. Direct sparse state+rest labels follow source map; no dense DTW targets.',
      architecture='Pretrained phase query/guide embedding fine-tuned for9states plus learned rest. Its weights frozen before training an extraBiGRU residual that can change only current maxprob<0.8. Latent64 already has earlier temporal context.',
      current_anchor='Exact confident-current logits preserved. A confidently wrong current prediction remains a limitation; not independent RGB confirmation.',
      loss='Class-balanced maskedCE on reviewed points, equal clip/variant weights; uncertain points ignored. Context loss only where current confidence<0.8. No forced known gesture through gaps.',
      validation='Head fitIDs01-06, head validation07-09; backbone already exposed01-09, so not full unseen-person validation. DEV11-13 repeatedly exposed development only.',
      pilot_gate=dict(current_state_macro_minimum=.8,rest_recall_minimum=.9,context_current_macro_drop_maximum=.02),
      development_gate=dict(normal_macro_no_worse_than_old=True,each_state_normal_drop_maximum=.05,reversed_macro_gain_minimum=.03,native_rest_recall_minimum=.9,
        action_as_rest_rate_maximum=.05,confident_anchor_changes=0,normal_order_conflict=0,main_reverse_supported_with_strict_edges=0,old_arrays_preserved=True),
      sequence='Inspect full original axis before DTW; rest is an explicit class and cannot erase the axis. No per-word time windows. Require2adjacent confidentobservations. Each strict state precedence checks all observed spans; interleaving remains uncertain, not auto-error. Unsupported cycles/counts remain unknown.',
      latency='Head timing is not full runtime. If gates pass run one actual public upload and preserve olddemo.',
      scope='Physical states/order only, not full educational correctness. No private/mentor/H14-15/ID16 access or tuning. No backbone replacement/download.',
      pins={str(p):sha(p) for p in [T/'BACKGROUND_TARGETS.json',T/'SOURCE_PLAN.json',S/'PHYSICAL_STATE_TARGETS.json',old.P/'runs/projection_finetune_42/model.pt',old.P/'GUIDE_SUPPORT.json',S/'FROZEN_TRAINING_V2.json',T/'state_model.py']})
    p=T/'FROZEN_TRAINING.json'
    if p.exists():oldplan=read(p);plan['at']=oldplan['at'];assert oldplan==plan
    else:exclusive(p,plan)
    return plan

def train(kind,seed,rows,tag,plan,initial=None):
    dest=T/'runs'/tag;assert not dest.exists();dest.mkdir(parents=True)
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=StateNet(kind=='anchored_context')
    if initial is not None:
        missing,extra=model.load_state_dict(initial.state_dict(),strict=False);assert not extra and all(k.startswith(('context.','delta.')) for k in missing)
        model.freeze_current()
    start_state={k:v.detach().clone() for k,v in model.state_dict().items()};start=time.perf_counter();log=[]
    counts=np.bincount(np.concatenate([r['y'][r['y']>=0] for r in rows]),minlength=10)
    weights=torch.tensor(np.clip(np.mean(counts[counts>0])/np.maximum(counts,1),.25,4),dtype=torch.float32)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['learning_rate'],weight_decay=plan['weight_decay'])
    epochs=plan['epochs_current' if kind=='current' else 'epochs_context']
    for ep in range(epochs):
        model.train();order=rng.permutation(len(rows));losses=[]
        for j in range(0,len(rows),plan['batch_size']):
            rr=[rows[k] for k in order[j:j+plan['batch_size']]];x,l,w,y=batch(rr);logits,base,allow=model(x,l,w,details=True)
            target=y.clone()
            if kind=='anchored_context':target[~allow]=-1
            known=target>=0
            if not known.any():continue
            ce=F.cross_entropy(logits.transpose(1,2),target,ignore_index=-1,weight=weights,reduction='none')
            loss=(ce.sum(1)/known.sum(1).clamp_min(1)).mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();losses.append(float(loss.detach()))
        if ep%25==0 or ep==epochs-1:
            item=dict(epoch=ep+1,loss=float(np.mean(losses)) if losses else 0.,seconds=time.perf_counter()-start);log.append(item);print(tag,item,flush=True)
    changes={k:float((v-start_state[k]).abs().max()) for k,v in model.state_dict().items() if v.is_floating_point()}
    if kind=='anchored_context':assert all(v==0 for k,v in changes.items() if k.startswith(('current.','rest.')))
    else:assert changes['current.query.weight']>0 and changes['rest.weight']>0
    torch.save(model.state_dict(),dest/'model.pt');exclusive(dest/'COMPLETE.json',dict(condition=kind,seed=seed,people=sorted({r['person'] for r in rows}),inputs=len(rows),seconds=time.perf_counter()-start,log=log,changes=changes,sha256=sha(dest/'model.pt')))
    return model

def preflight(rows,plan):
    c=StateNet(False).eval();a=StateNet(True).eval();a.load_state_dict(c.state_dict(),strict=False);x,l,w,y=batch(rows[:2]);aa,base,allow=a(x,l,w,True)
    torch.testing.assert_close(c(x,l,w),aa,rtol=0,atol=0)
    a.delta.bias.data.fill_(10);out,base,allow=a(x,l,w,True);assert torch.equal(out[~allow],base[~allow])
    a.freeze_current();assert all(not p.requires_grad for p in a.current.parameters()) and all(not p.requires_grad for p in a.rest.parameters())
    single=c(x[:1,:l[0]],l[:1],w[:1]);torch.testing.assert_close(single,c(x,l,w)[:1,:l[0]],rtol=1e-5,atol=1e-5)
    assert all(sha(p)==h for p,h in plan['pins'].items())
    exclusive(T/'PREFLIGHT.json',dict(inputs=len(rows),old_pins_preserved=True,zero_residual_initial_match=True,anchor_exact_even_large_residual=True,current_freeze=True,padding_invariant=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');args=ap.parse_args();plan=freeze()
    if args.freeze_only:return
    assert not (T/'TRAINING_RESULT.json').exists();assert read(T/'INPUT_VERIFICATION.json')['status']=='PASS';torch.set_num_threads(2);rows=fit_rows()
    if not (T/'PREFLIGHT.json').exists():preflight(rows,plan)
    pilot=[r for r in rows if r['person']<=6];val=[r for r in rows if r['person']>=7]
    def load_or_train(kind,seed,rr,tag,initial=None):
        p=T/'runs'/tag/'COMPLETE.json'
        if not p.exists():return train(kind,seed,rr,tag,plan,initial)
        assert sha(p.parent/'model.pt')==read(p)['sha256'];m=StateNet(kind=='anchored_context');m.load_state_dict(torch.load(p.parent/'model.pt',weights_only=True));return m
    if not (T/'PILOT_VALIDATION.json').exists():
        cm=load_or_train('current',42,pilot,'pilot_current_42');am=load_or_train('anchored_context',42,pilot,'pilot_anchored_context_42',cm)
        cp=measure(cm,val);apoints=measure(am,val);cs=metrics([p for p in cp if p['y']<9]);ac=metrics([p for p in apoints if p['y']<9]);rest=metrics([p for p in cp if p['y']==9])
        passed=cs['macro_recall']>=.8 and rest['macro_recall']>=.9 and ac['macro_recall']>=cs['macro_recall']-.02
        exclusive(T/'PILOT_VALIDATION.json',dict(current=cs,context=ac,rest=rest,current_points=cp,context_points=apoints,passed=passed,scope=plan['validation']))
    assert read(T/'PILOT_VALIDATION.json')['passed'],'Retain failed pilot before further final training'
    done=[]
    for seed in plan['seeds']:
        c=load_or_train('current',seed,rows,f'current_{seed}');a=load_or_train('anchored_context',seed,rows,f'anchored_context_{seed}',c)
        for kind in plan['conditions']:done.append(read(T/'runs'/f'{kind}_{seed}'/'COMPLETE.json'))
        write(T/'TRAIN_PROGRESS.json',dict(at=clock(),status='RUNNING',pid=os.getpid(),done=len(done),total=6))
    assert all(sha(p)==h for p,h in plan['pins'].items())
    exclusive(T/'TRAINING_RESULT.json',dict(runs=done,old_models_preserved=True,supervised_phase_finetuning=True,new_backbone_updates=0))
    write(T/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',pid=os.getpid(),done=6,total=6))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (T/'TRAIN_FIRST_FAILURE.txt').exists():(T/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
