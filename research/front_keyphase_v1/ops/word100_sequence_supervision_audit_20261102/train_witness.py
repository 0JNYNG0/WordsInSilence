import time,os,traceback,shutil
from sequence_model import *

def weights(rows,key,count):
    c=np.bincount(np.concatenate([r[key][r[key]>=0] for r in rows]),minlength=count)
    return torch.tensor(np.clip(c.mean()/np.maximum(c,1),.25,4),dtype=torch.float32)

def preflight():
    from evaluate_witness import behavior
    checks=behavior();torch.manual_seed(42);m=WitnessNet().eval()
    rows=fit_rows();stats=read(D/'NORMALIZATION.json');x,im,l,y,act=batch_j(rows[:2],stats)
    with torch.inference_mode():
        aa,bb,ss=m(x,im,l);a,b,s=m(x[:1,:l[0]],im[:1,:l[0]],l[:1]);torch.testing.assert_close(aa[0,:l[0]],a[0],rtol=1e-5,atol=1e-5);torch.testing.assert_close(bb[0,:l[0]],b[0],rtol=1e-5,atol=1e-5)
        blank=im.clone();blank[:,:,-1]=0;ab,rb,sb=m(x,blank,l);blank[:,:,:768]=27;ac,rc,sc=m(x,blank,l);torch.testing.assert_close(ab,ac);torch.testing.assert_close(rb,rc)
    z=torch.randn(1,7,9,requires_grad=True);yy=torch.tensor([[0,-1,0,1,-1,1,2]])
    loss=sequence_loss(z,yy);loss.backward();assert (z.grad[0,[1,4]]==0).all() and (z.grad[0,[0,2,3,5,6]].abs().sum(-1)>0).all()
    changed=z.detach().clone();changed[0,[1,4]]=999;torch.testing.assert_close(sequence_loss(z,yy),sequence_loss(changed,yy))
    aa,bb,ss=m(x,im,l);(aa.sum()+bb.sum()+sequence_loss(ss,y)).backward()
    assert m.actor.delta.weight.grad is None and m.actor.query.weight.grad is not None and m.blank.grad is not None
    return dict(status='PASS',behavior=checks,padding=True,image_availability=True,no_sequence_supervision_of_unlabelled_frames=True,
        frozen_encoder=True,trainable_action_activity_and_blank=True),rows,stats

def main():
    assert not (J/'FROZEN.json').exists();assert shutil.disk_usage(J).free>1024**3;torch.set_num_threads(2)
    check,rows,stats=preflight();exclusive(J/'PREFLIGHT.json',dict(at=clock(),**check))
    plan=dict(at=clock(),primary='witness_ctc_42',conditions=['point_ce','witness_ctc'],seed=42,epochs=80,batch_size=12,lr=.0005,weight_decay=.01,sequence_coefficient=.1,
       scope='Same hierarchical observer-preserving architecture, FIT rows, init, optimizer and per-point losses; only sampled-witness CTC auxiliary differs. No full-video CTC admission, no constrained runtime decoding.',
       activity='D/E directly reviewed main1 versus approach/return/rest0; unknown-1. T rest observation kept explicit, not relearnt as competing three-way class.',
       architecture='Copied D episode42 encoder frozen, action readout + two-way activity temporal head trainable. Separate CTC blank parameter, not rest output.',
       exclusions='No C/T3F/old demo edits, no user/mentor, no H14/15, no new downloads, no heldout tuning.',
       gate=read(D/'FROZEN.json')['gate'],extra_gates=dict(retained_rest_probability_exact=True,normal_role_rest_correct_min=96,no_correct_action_loss_by_context=True),
       whole_goal_complete=False)
    files=[J/'REST_INPUT_AUDIT.json',J/'SEQUENCE_INPUT_AUDIT.json',J/'sequence_model.py',J/'train_witness.py',J/'evaluate_witness.py',J/'PREFLIGHT.json',D/'runs/episode_42/model.pt',D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',D/'NORMALIZATION.json',B/'PREPARED.json',F/'model.py',F/'evaluate_split.py']
    plan['pins']={str(p):sha(p) for p in files}
    plan['fit_cache_pins']={r['cache']:r['cache_sha256'] for r in rows}
    plan['direct_action_points']=sum(int((r['episode']>=0).sum()) for r in rows)
    plan['witness_ctc_points']=sum(int(((r['episode']>=0)&(r['episode']<8)).sum()) for r in rows)
    exclusive(J/'FROZEN.json',plan)
    for r in rows:r['activity']=np.where(r['role']<0,-1,(r['role']==1).astype('int64'))
    aw=weights(rows,'episode',9);rw=weights(rows,'activity',2);reports=[]
    for condition in plan['conditions']:
        name=condition+'_42';dest=J/'runs'/name;assert not dest.exists();dest.mkdir(parents=True)
        torch.manual_seed(42);rng=np.random.default_rng(42);m=WitnessNet();before={k:v.clone() for k,v in m.state_dict().items()};opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=plan['weight_decay'])
        start=time.perf_counter();history=[]
        for epoch in range(plan['epochs']):
            m.train();order=rng.permutation(len(rows));losses=[]
            for j in range(0,len(rows),plan['batch_size']):
                rr=[rows[k] for k in order[j:j+plan['batch_size']]];x,im,l,y,a=batch_j(rr,stats);ac,act,sq=m(x,im,l)
                la=loss_fn(ac,y,aw);lr=loss_fn(act,a,rw);ls=sequence_loss(sq,y) if condition=='witness_ctc' else ac.sum()*0
                total=la+lr+plan['sequence_coefficient']*ls;assert torch.isfinite(total)
                opt.zero_grad();total.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();losses.append([float(v.detach()) for v in [total,la,lr,ls]])
            if epoch%20==0 or epoch==plan['epochs']-1:
                rec=dict(epoch=epoch+1,losses=np.mean(losses,0).tolist(),seconds=time.perf_counter()-start);history.append(rec);print(name,rec,flush=True)
                write(J/'TRAIN_PROGRESS.json',dict(at=clock(),status='TRAINING',pid=os.getpid(),tag=name,epoch=epoch+1))
        delta={k:float((v-before[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point}
        fixed=[k for k in delta if any(k.startswith('actor.'+n+'.') for n in ['embed','rgb','context','delta'])];assert fixed and all(delta[k]==0 for k in fixed)
        assert delta['actor.query.weight']>0 and delta['role_output.weight']>0
        torch.save(m.state_dict(),dest/'model.pt');report=dict(tag=name,seconds=time.perf_counter()-start,history=history,changes=delta,sha256=sha(dest/'model.pt'),encoder_frozen=True)
        exclusive(dest/'COMPLETE.json',report);reports.append(report)
    for p,h in plan['pins'].items():assert sha(p)==h
    for p,h in plan['fit_cache_pins'].items():assert sha(p)==h
    exclusive(J/'TRAINING_RESULT.json',dict(at=clock(),runs=reports,protected_inputs_unchanged=True,new_backbone_inference=0,whole_goal_complete=False))
    write(J/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None,runs=2))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (J/'TRAIN_FIRST_FAILURE.txt').exists():(J/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
