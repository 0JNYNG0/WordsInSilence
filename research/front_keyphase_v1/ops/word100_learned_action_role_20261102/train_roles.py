import os,time,traceback
from torch.nn import functional as F
from role_model import *

def protocol():
    p=A/'FROZEN_TRAINING.json'
    if p.exists():return read(p)
    obj=dict(at=clock(),primary='temporal_42',seeds=[42,43,44],conditions=['local','temporal'],epochs=100,batch_size=12,lr=.001,weight_decay=.01,
      targets='Sparse direct visual role points plus reviewed T rest. No dense labels or false unknown=transition. Source-frame mapped actual reversals, local reorders, speeds, pose gaps.',
      inputs='Protected T.current42 probabilities + existing64D RGB/pose temporal latent +14body-relative coordinates and validity/displacement; no absolute time or user data.',
      observation_preservation='Existing state/shape/relation/movement neural weights and outputs never changed. New separate3-class supervised role head.',
      split='FIT01-09. Pilot headfit01-06/headval07-09; observer previously exposed allFIT, not independent person validation. DEV11-13 exposed development, H14/15 locked.',
      pilot_gate=dict(macro_min=.8,presented_action_recall_min=.9,rest_recall_min=.9),
      sequence_rule='Role confidence>=0.8 can identify transit/rest for interpreting config order, but raw state, movement, native axis and uncertain candidates remain. No image gap may certify sequence. No forced ordered alignment.',
      development_gate=dict(role_macro_min=.85,normal_presented_action_recall_min=.95,normal_usable_order_min=21,normal_order_conflicts=0,strict_localreverse_detected_min=2,strict_localreverse_false_supported=0,blackout_certified=0,observations_unchanged=True),
      limitations='Physical reference process; not linguistic expert truth, contact/depth or whole-word correctness. Data labels are sparse Codex visual review. Current42 choice is prior-development-informed.',
      pins={str(p):sha(p) for p in [A/'ROLE_TARGETS.json',A/'role_model.py',A/'train_roles.py',T/'runs/current_42/model.pt',P/'runs/projection_finetune_42/model.pt',P/'GUIDE_SUPPORT.json',S/'LEARNED_STATE_GRAPH.json']})
    exclusive(p,obj);return obj

def run(kind,seed,rows,s,tag,plan):
    dest=A/'runs'/tag
    if (dest/'COMPLETE.json').exists():
        assert sha(dest/'model.pt')==read(dest/'COMPLETE.json')['sha256'];m=RoleNet(kind=='temporal');m.load_state_dict(torch.load(dest/'model.pt',weights_only=True));return m
    assert not dest.exists();dest.mkdir(parents=True);torch.manual_seed(seed);rng=np.random.default_rng(seed);m=RoleNet(kind=='temporal')
    initial={k:v.clone() for k,v in m.state_dict().items()};opt=torch.optim.AdamW(m.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay'])
    counts=np.bincount(np.concatenate([r['y'][r['y']>=0] for r in rows]),minlength=3);weights=torch.tensor(np.clip(counts.mean()/counts,.25,4),dtype=torch.float32)
    start=time.perf_counter();log=[]
    for ep in range(plan['epochs']):
        m.train();order=rng.permutation(len(rows));losses=[]
        for j in range(0,len(rows),plan['batch_size']):
            rr=[rows[k] for k in order[j:j+plan['batch_size']]];x,l,y=batch(rr,s);out=m(x,l)
            ce=F.cross_entropy(out.transpose(1,2),y,ignore_index=-1,weight=weights,reduction='none');loss=(ce.sum(1)/(y>=0).sum(1).clamp_min(1)).mean()
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();losses.append(float(loss.detach()))
        if ep%25==0 or ep==plan['epochs']-1:
            v=dict(epoch=ep+1,loss=float(np.mean(losses)),seconds=time.perf_counter()-start);log.append(v);print(tag,v,flush=True)
    changed={k:float((v-initial[k]).abs().max()) for k,v in m.state_dict().items()};assert changed['embed.0.weight']>0
    torch.save(m.state_dict(),dest/'model.pt');exclusive(dest/'COMPLETE.json',dict(condition=kind,seed=seed,people=sorted({r['person'] for r in rows}),inputs=len(rows),seconds=time.perf_counter()-start,changes=changed,log=log,sha256=sha(dest/'model.pt')))
    return m

def main():
    assert not (A/'TRAINING_RESULT.json').exists();plan=protocol();torch.set_num_threads(2);rows=inputs(True)
    assert len(rows)==216 and {r['person'] for r in rows}==set(range(1,10))
    pinned={r['input']:r['input_sha256'] for r in rows}
    for r in rows:pinned.update(dict(r['extra_pins']))
    assert all(sha(p)==h for p,h in pinned.items());s=stats(rows)
    exclusive(A/'NORMALIZATION.json',s);exclusive(A/'INPUT_VERIFICATION.json',dict(status='PASS',clips=36,inputs=216,split='FIT01-09',input_pins=pinned,known=int(sum((r['y']>=0).sum() for r in rows))))
    # Padding and masks are checked without reading DEV or modifying observers.
    test=RoleNet(True).eval();x,l,y=batch(rows[:2],s)
    with torch.inference_mode():torch.testing.assert_close(test(x,l)[0,:l[0]],test(x[:1,:l[0]],l[:1])[0],rtol=1e-5,atol=1e-5)
    assert all(not p.requires_grad for p in observer().parameters())
    exclusive(A/'PREFLIGHT.json',dict(status='PASS',padding_invariant=True,observer_frozen=True,no_parameter_sharing=True,unknowns_masked=True,dimension=130))
    pilot=[r for r in rows if r['person']<=6];val=[r for r in rows if r['person']>=7];ps=stats(pilot);results={}
    for kind in plan['conditions']:
        m=run(kind,42,pilot,ps,'pilot_'+kind+'_42',plan);pts=measure(m,val,ps);results[kind]=dict(metrics=state.metrics(pts),points=pts)
    mm=results['temporal']['metrics'];passed=mm['macro_recall']>=.8 and mm['groups']['1']['correct']/mm['groups']['1']['known']>=.9 and mm['groups']['2']['correct']/mm['groups']['2']['known']>=.9
    exclusive(A/'PILOT_VALIDATION.json',dict(results=results,passed=passed,scope=plan['split']))
    if not passed:
        write(A/'TRAIN_PROGRESS.json',dict(at=clock(),status='PILOT_GATE_FAILED',done=0,total=6));print('Pilot gate failed; preserve diagnosis before final training.',flush=True);return
    done=[]
    for kind in plan['conditions']:
        for seed in plan['seeds']:
            tag=f'{kind}_{seed}';run(kind,seed,rows,s,tag,plan);done.append(read(A/'runs'/tag/'COMPLETE.json'));write(A/'TRAIN_PROGRESS.json',dict(at=clock(),status='RUNNING',pid=os.getpid(),done=len(done),total=6))
    assert all(sha(p)==h for p,h in plan['pins'].items()) and all(sha(p)==h for p,h in pinned.items())
    exclusive(A/'TRAINING_RESULT.json',dict(runs=done,old_models_preserved=True,new_head_supervised=True,new_backbone_updates=0))
    write(A/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',pid=os.getpid(),done=6,total=6))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (A/'TRAIN_FIRST_FAILURE.txt').exists():(A/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
