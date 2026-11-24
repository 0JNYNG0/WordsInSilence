import os,time,traceback,shutil
from experiment import *

def main():
    assert not (U/'TRAINING_RESULT.json').exists();plan=read(U/'FROZEN.json');torch.set_num_threads(2)
    for p,h in plan['pins'].items():assert sha(p)==h
    OUT.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(OUT).free>1024**3
    old,new,_=data();stats=read(D/'NORMALIZATION.json');reports=[]
    for tag in plan['conditions']:
        dest=OUT/'runs'/tag;dest.mkdir(parents=True,exist_ok=False)
        rows=old if tag.startswith('old') else new
        for r in rows:r['activity']=np.where(r['role']<0,-1,(r['role']==1).astype('int64'))
        aw=weights(rows,'episode',9);rw=weights(rows,'activity',2)
        torch.manual_seed(42);rng=np.random.default_rng(42);m=model_for(tag);before={k:v.clone() for k,v in m.state_dict().items()}
        slow=[];fast=[]
        for name,p in m.named_parameters():
            if p.requires_grad:(slow if name.startswith(('actor.context.','actor.delta.')) else fast).append(p)
        groups=[dict(params=fast,lr=plan['lr'])]
        if slow:groups.append(dict(params=slow,lr=plan['encoder_lr']))
        opt=torch.optim.AdamW(groups,weight_decay=plan['weight_decay']);start=time.perf_counter();history=[]
        for epoch in range(plan['epochs']):
            m.train();order=rng.permutation(len(rows));losses=[]
            for i in range(0,len(rows),plan['batch_size']):
                rr=[rows[k] for k in order[i:i+plan['batch_size']]];x,rgb,l,y,a=batch_j(rr,stats)
                pred,act,_=m(x,rgb,l);loss=loss_fn(pred,y,aw)+loss_fn(act,a,rw);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();losses.append(float(loss.detach()))
            if epoch%10==0 or epoch==plan['epochs']-1:
                row=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.perf_counter()-start);history.append(row);print(tag,row,flush=True)
                write(U/'TRAIN_PROGRESS.json',dict(at=clock(),status='TRAINING',pid=os.getpid(),condition=tag,epoch=epoch+1))
        delta={k:float((v-before[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point}
        fixed=[k for k in delta if k.startswith(('actor.embed.','actor.rgb.')) or k=='actor.guide_vectors'];assert all(delta[k]==0 for k in fixed)
        for k in ['actor.context.weight_ih_l0','actor.delta.weight']:assert (delta[k]>0)==tag.endswith('adapt')
        assert delta['actor.query.weight']>0 and delta['role_output.weight']>0
        torch.save(m.state_dict(),dest/'model.pt');rep=dict(condition=tag,sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-start,history=history,changes=delta)
        exclusive(dest/'COMPLETE.json',rep);reports.append(rep)
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(U/'TRAINING_RESULT.json',dict(at=clock(),runs=reports,protected_inputs_unchanged=True,whole_goal_complete=False))
    write(U/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None,runs=4))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (U/'TRAIN_FIRST_FAILURE.txt').exists():(U/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
