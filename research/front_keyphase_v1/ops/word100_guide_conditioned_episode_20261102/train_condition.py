import time,os,traceback
from condition_model import *

def main():
    assert not (E/'TRAINING_RESULT.json').exists();torch.set_num_threads(2)
    assert read(E/'TARGETS_VERIFIED.json')['status']=='PASS'
    rows0=fit_rows(False);rows1=fit_rows(True);stats=process.stats(rows0)
    assert stats==process.stats(rows1)
    pins={str(p):sha(p) for p in [E/'PLAN.json',E/'CONTEXT_TARGETS.json',E/'TARGETS_VERIFIED.json',E/'condition_model.py',E/'train_condition.py',D/'episode_model.py',D/'ACTION_TARGETS.json',B/'PREPARED.json',B/'runs/joint_42/model.pt',T/'runs/current_42/model.pt',P/'GUIDE_SUPPORT.json']}
    plan=dict(at=clock(),primary=tag(True,True),conditions=[dict(guide=g,context=c,tag=tag(g,c)) for g,c in CONDITIONS],seed=42,epochs=80,batch_size=12,lr=.0005,weight_decay=.01,
      comparison='Matched2x2 input of retained goal-conditioned current posterior on/off x additional directly observed approach/return/rest supervision on/off. Same10-way architecture, initialization, data variants, minibatch order,80epochs. No target mask on output.',
      task='Source-guide action-family interpretation. Existing physical, handshape, relation and motion observation arrays remain intact; interpretation may depend on target expression. No claim target-conditioned posterior is independent visual truth.',
      data='36publicFIT recordedIDs01-09,216cachedvariants. DEV11-13exposed development only. H14/15locked. No private/mentor.',
      uncertainty='All original frames remain. Extra class9 is explicit observed approach/return, not rest or meaningless. Unknowns receive no supervision. Edit seams use unchanged D mask. No required repetition count.',
      gate=read(D/'FROZEN.json')['gate'],previous_complete_runs_not_repeated=True,new_backbone_training=False,new_backbone_inference=0,
      supervised_counts=[sum(int((r['episode']>=0).sum()) for r in rr) for rr in [rows0,rows1]],pins=pins,whole_goal_complete=False)
    pins[str(E/'evaluate_condition.py')]=sha(E/'evaluate_condition.py')
    exclusive(E/'FROZEN.json',plan);exclusive(E/'NORMALIZATION.json',stats)
    torch.manual_seed(42);m=ConditionNet().eval();x,im,l,y=batch(rows1[:2],stats,True)
    with torch.inference_mode():
        a=m(x,im,l);torch.testing.assert_close(a[0,:l[0]],m(x[:1,:l[0]],im[:1,:l[0]],l[:1])[0],rtol=1e-5,atol=1e-5)
        missing=im.clone();missing[:,:,-1]=0;out=m(x,missing,l);missing[:,:,:768]=37;torch.testing.assert_close(out,m(x,missing,l))
    snapshot=rows1[0]['current'].copy();alter=dict(rows1[0],current=np.ones_like(snapshot)*.1)
    off=batch(rows1[:1],stats,False)[0];torch.testing.assert_close(off,batch([alter],stats,False)[0])
    on=batch(rows1[:1],stats,True)[0];assert not torch.equal(on,batch([alter],stats,True)[0])
    assert np.array_equal(snapshot,rows1[0]['current'])
    assert all(np.array_equal(a['x'],b['x']) and np.array_equal(a['rgb'],b['rgb']) and np.array_equal(a['current'],b['current']) for a,b in zip(rows0,rows1))
    assert all(np.array_equal(a['episode'][a['episode']>=0],b['episode'][a['episode']>=0]) for a,b in zip(rows0,rows1))
    loss_fn(m(x,im,l),y,torch.ones(10)).backward();assert m.query.weight.grad is not None and m.approach_return.weight.grad is not None
    exclusive(E/'PREFLIGHT.json',dict(at=clock(),status='PASS',padding=True,image_absence=True,condition_input_factor=True,old_labels_and_inputs_unchanged=True,guide_and_context_gradients=True))
    reports=[]
    for g,c in CONDITIONS:
        name=tag(g,c);dest=E/'runs'/name;assert not dest.exists();dest.mkdir(parents=True)
        rows=rows1 if c else rows0;counts=np.bincount(np.concatenate([r['episode'][r['episode']>=0] for r in rows]),minlength=10)
        weights=torch.tensor(np.clip(counts[counts>0].mean()/np.maximum(counts,1),.25,4),dtype=torch.float32)
        torch.manual_seed(42);rng=np.random.default_rng(42);m=ConditionNet();initial={k:v.clone() for k,v in m.state_dict().items()}
        opt=torch.optim.AdamW(m.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay']);start=time.perf_counter();history=[]
        for epoch in range(plan['epochs']):
            m.train();ids=rng.permutation(len(rows));losses=[]
            for j in range(0,len(rows),plan['batch_size']):
                rr=[rows[k] for k in ids[j:j+plan['batch_size']]];x,im,l,y=batch(rr,stats,g);loss=loss_fn(m(x,im,l),y,weights)
                assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();losses.append(float(loss.detach()))
            if epoch%20==0 or epoch==plan['epochs']-1:
                rec=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.perf_counter()-start);history.append(rec);print(name,rec,flush=True)
                write(E/'TRAIN_PROGRESS.json',dict(at=clock(),status='TRAINING',pid=os.getpid(),tag=name,epoch=epoch+1))
        changes={k:float((v-initial[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point}
        assert changes['query.weight']>0 and changes['delta.weight']>0
        torch.save(m.state_dict(),dest/'model.pt');result=dict(tag=name,guide=g,context=c,sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-start,changes=changes,history=history,class_counts=counts.tolist())
        exclusive(dest/'COMPLETE.json',result);reports.append(result)
    for p,h in pins.items():assert sha(p)==h
    exclusive(E/'TRAINING_RESULT.json',dict(at=clock(),runs=reports,protected_files_unchanged=True,new_backbone_inference=0,whole_goal_complete=False))
    write(E/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None,runs=4))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (E/'TRAIN_FIRST_FAILURE.txt').exists():(E/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
