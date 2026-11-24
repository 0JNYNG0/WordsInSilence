import time,os,traceback
from model import *

def weights(rows,key,count):
    c=np.bincount(np.concatenate([r[key][r[key]>=0] for r in rows]),minlength=count)
    return torch.tensor(np.clip(c.mean()/np.maximum(c,1),.25,4),dtype=torch.float32)

def main():
    assert not (F/'FROZEN.json').exists();torch.set_num_threads(2)
    rows=fit_rows();stats=read(D/'NORMALIZATION.json');plan=read(F/'PLAN.json')
    pins={str(p):sha(p) for p in [F/'PLAN.json',F/'INPUT_CONTRACT.json',F/'model.py',F/'train.py',F/'evaluate_split.py',D/'runs/episode_42/model.pt',D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',D/'NORMALIZATION.json',B/'PREPARED.json',T/'runs/current_42/model.pt']}
    frozen=dict(at=clock(),primary='frozen_encoder_42',conditions=['frozen_encoder','joint_encoder'],seed=42,epochs=80,batch_size=12,lr=.0005,weight_decay=.01,
      architecture='Identical copied D episode42 feature encoder and trainable9-way action interpretation readout plus separate BiGRU3-way process role. Primary freezes embed/RGB/context/delta, but learns action readout and role. Control learns both encoder and readouts. Original T/shape/relation/motion arrays are immutable in both.',
      losses='CE sparse D action content + CE sparse E process-role, equal coefficients. Additional E context labels train role only, never another mutually exclusive action/pose class.',
      preparation_contract=plan,gate=read(D/'FROZEN.json')['gate'],pins=pins,action_labels=sum(int((r['episode']>=0).sum()) for r in rows),role_labels=sum(int((r['role']>=0).sum()) for r in rows),whole_goal_complete=False)
    exclusive(F/'FROZEN.json',frozen)
    aw=weights(rows,'episode',9);rw=weights(rows,'role',3)
    torch.manual_seed(42);m=SplitNet(True).eval();x,im,l,a,r=batch(rows[:2],stats)
    with torch.inference_mode():
        out,role=m(x,im,l);oo,rr=m(x[:1,:l[0]],im[:1,:l[0]],l[:1]);torch.testing.assert_close(out[0,:l[0]],oo[0],rtol=1e-5,atol=1e-5);torch.testing.assert_close(role[0,:l[0]],rr[0],rtol=1e-5,atol=1e-5)
        blank=im.clone();blank[:,:,-1]=0;ab,rb=m(x,blank,l);blank[:,:,:768]=27;aa,ra=m(x,blank,l);torch.testing.assert_close(ab,aa);torch.testing.assert_close(rb,ra)
    out,role=m(x,im,l);(loss_fn(out,a,aw)+loss_fn(role,r,rw)).backward()
    assert m.actor.query.weight.grad is not None and m.role_output.weight.grad is not None and m.actor.delta.weight.grad is None
    from evaluate_split import behavior
    checks=behavior()
    exclusive(F/'PREFLIGHT.json',dict(at=clock(),status='PASS',padding=True,image_availability=True,frozen_encoder_gradient=True,trainable_action_and_role=True,behavior=checks))
    reports=[]
    for condition in frozen['conditions']:
        name=condition+'_42';dest=F/'runs'/name;assert not dest.exists();dest.mkdir(parents=True)
        torch.manual_seed(42);rng=np.random.default_rng(42);m=SplitNet(condition=='frozen_encoder');before={k:v.clone() for k,v in m.state_dict().items()};opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=frozen['lr'],weight_decay=frozen['weight_decay']);start=time.perf_counter();history=[]
        for epoch in range(frozen['epochs']):
            m.train();order=rng.permutation(len(rows));ls=[];la=[];lr=[]
            for j in range(0,len(rows),frozen['batch_size']):
                rr=[rows[k] for k in order[j:j+frozen['batch_size']]];x,im,l,ay,ry=batch(rr,stats);ac,ro=m(x,im,l)
                al=loss_fn(ac,ay,aw);rl=loss_fn(ro,ry,rw);loss=al+rl;assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();ls.append(float(loss.detach()));la.append(float(al.detach()));lr.append(float(rl.detach()))
            if epoch%20==0 or epoch==frozen['epochs']-1:
                rec=dict(epoch=epoch+1,loss=float(np.mean(ls)),action_loss=float(np.mean(la)),role_loss=float(np.mean(lr)),seconds=time.perf_counter()-start);history.append(rec);print(name,rec,flush=True)
                write(F/'TRAIN_PROGRESS.json',dict(at=clock(),status='TRAINING',pid=os.getpid(),tag=name,epoch=epoch+1))
        delta={k:float((v-before[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point}
        fixed=[k for k in delta if any(k.startswith('actor.'+n+'.') for n in ['embed','rgb','context','delta'])]
        if condition=='frozen_encoder':assert all(delta[k]==0 for k in fixed)
        else:assert any(delta[k]>0 for k in fixed)
        assert delta['actor.query.weight']>0 and delta['role_output.weight']>0
        torch.save(m.state_dict(),dest/'model.pt');report=dict(tag=name,seconds=time.perf_counter()-start,history=history,changes=delta,sha256=sha(dest/'model.pt'),encoder_frozen=condition=='frozen_encoder')
        exclusive(dest/'COMPLETE.json',report);reports.append(report)
    for p,h in pins.items():assert sha(p)==h
    exclusive(F/'TRAINING_RESULT.json',dict(at=clock(),runs=reports,protected_inputs_unchanged=True,new_backbone_inference=0,whole_goal_complete=False))
    write(F/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None,runs=2))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (F/'TRAIN_FIRST_FAILURE.txt').exists():(F/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
