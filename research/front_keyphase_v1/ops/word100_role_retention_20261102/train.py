from retention_model import *
import os,time,traceback

def soft_batch(rows):
    n=max(len(r['x']) for r in rows);p=np.full((len(rows),n,2),.5,np.float32);mask=np.zeros((len(rows),n),bool)
    for i,r in enumerate(rows):p[i,:len(r['x'])]=r['_teacher'];mask[i,:len(r['x'])]=r['_retention_mask']
    return torch.from_numpy(p),torch.from_numpy(mask)

def main():
    assert not (U/'TRAINING_RESULT.json').exists() and not (U/'TRAINING_FREEZE.json').exists()
    assert read(U/'INPUTS_VERIFIED.json')['status']=='FIT_SOFT_RETENTION_INPUTS_VERIFIED'
    torch.set_num_threads(2);plan=read(U/'PREPARATION_FREEZE.json');rows,normal=fit_rows();stats=read(ex.D/'NORMALIZATION.json')
    teacher=read(U/'TEACHER_INPUTS.json')['rows'];assert len(rows)==len(teacher)==252
    pins=dict(plan['pins'])
    for p in [U/'train.py',U/'evaluate.py',U/'verify.py',U/'INPUTS_VERIFIED.json',U/'TEACHER_INPUTS.json',U/'verify_inputs.py',O/'word100_nonmain_probability_20261102/contract.py']:
        pins[str(p)]=sha(p)
    for r,t in zip(rows,teacher):
        assert r['clip_id']==t['clip_id'] and r['kind']==t['kind'];assert sha(t['teacher_path'])==t['teacher_sha256']
        a=np.load(t['teacher_path']);r['_teacher']=a['probability'];r['_retention_mask']=a['mask'];assert np.array_equal(a['role_label'],r['role'])
        r['activity']=np.where(r['role']<0,-1,(r['role']==1).astype('int64'))
        pins[t['teacher_path']]=t['teacher_sha256'];pins[r['cache']]=r['cache_sha256']
    for p,h in pins.items():assert sha(p)==h
    plan['pins']=pins;plan['family_baseline']=read(O/'word100_guided_event_order_20261102/FROZEN.json')['baseline_family_eligible']
    exclusive(U/'TRAINING_FREEZE.json',plan);rw=ex.weights(rows,'activity',2);reports=[]
    for tag in plan['conditions']:
        dest=OUT/'runs'/tag;dest.mkdir(parents=True,exist_ok=False);torch.manual_seed(42);rng=np.random.default_rng(42)
        m=RetainedRoleNet();before={k:v.clone() for k,v in m.state_dict().items()};opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=plan['weight_decay'])
        weight=plan['retention_weight'] if tag.endswith('with_retention') else 0.;history=[];start=time.perf_counter()
        for epoch in range(plan['epochs']):
            order=rng.permutation(len(rows));m.train();losses=[]
            for i in range(0,len(rows),plan['batch_size']):
                rr=[rows[j] for j in order[i:i+plan['batch_size']]];x,rgb,l,_,a=ex.batch_j(rr,stats);flow=fl.flow_batch(rr,normal)
                _,logits=m(x,rgb,l,flow);p,mask=soft_batch(rr)
                supervised=ex.loss_fn(logits,a,rw);kd=retention_loss(logits,p,mask);loss=supervised+weight*kd;assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();losses.append([float(supervised.detach()),float(kd.detach())])
            if epoch%10==0 or epoch==plan['epochs']-1:
                item=dict(epoch=epoch+1,supervised=float(np.mean(losses,0)[0]),soft_retention=float(np.mean(losses,0)[1]),seconds=time.perf_counter()-start)
                history.append(item);print(tag,item,flush=True);ex.write(U/'TRAIN_PROGRESS.json',dict(at=clock(),status='TRAINING',condition=tag,epoch=epoch+1,pid=os.getpid()))
        delta={k:float((v-before[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point}
        assert all(v==0 for k,v in delta.items() if k.startswith('actor.')) and delta['flow_bridge.weight']>0 and delta['role_output.weight']>0
        torch.save(m.state_dict(),dest/'model.pt');r=dict(condition=tag,sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-start,history=history,changes=delta)
        exclusive(dest/'COMPLETE.json',r);reports.append(r)
    for p,h in pins.items():assert sha(p)==h
    exclusive(U/'TRAINING_RESULT.json',dict(at=clock(),runs=reports,raw_actor_fixed=True,whole_goal_complete=False))
    ex.write(U/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (U/'TRAIN_FIRST_FAILURE.txt').exists():(U/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
