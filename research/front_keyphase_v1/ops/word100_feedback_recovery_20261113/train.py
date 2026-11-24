from model_core import *
import traceback

def main():
    cfg=read(U/'FROZEN.json');assert read(U/'PRETRAIN_CHECK.json')['status']=='PASS'
    for p,h in {**cfg['pins'],**cfg['protected']}.items():assert sha(p)==h,p
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    rr=rows_load('fit');common=[r for r in rr if r['origin']=='common100']
    fine=[r for r in common if (r['fy']>=0).any()];coarse=[r for r in common if r['ym'].sum()]
    ef=[r for r in rr if r['origin']=='selected_extra' and (r['fy']>=0).any()]
    bends=[r for r in fine if (r['fy']==1).any()]
    pos=sum((r['y']*r['ym']).sum(0) for r in coarse);neg=sum(((1-r['y'])*r['ym']).sum(0) for r in coarse)
    pw=torch.tensor(np.clip(neg/np.maximum(pos,1),.25,4),dtype=torch.float32)
    for condition in cfg['conditions']:
      for seed in cfg['seeds']:
        d=U/'runs'/f'{condition}_{seed}';d.mkdir(parents=True,exist_ok=True)
        if (d/'COMPLETE.json').exists():
            done=read(d/'COMPLETE.json');assert sha(d/'model.pt')==done['checkpoint_sha256'];continue
        torch.manual_seed(seed);net=FineNet(len(cfg['extra_words']))
        base=torch.load(B/f'runs/shared100_{seed}/model.pt',map_location='cpu',weights_only=True)
        missing,unexpected=net.load_state_dict(base,strict=False);assert not unexpected and all(k.startswith(('finger','extra')) for k in missing)
        teacher=old.SharedNet();teacher.load_state_dict(base);teacher.eval()
        for p in teacher.parameters():p.requires_grad=False
        for name,p in net.named_parameters():
            p.requires_grad=name.startswith('finger') or (condition=='finetune' and name.startswith(('base.input','base.temporal','base.context')))
        heads=[p for n,p in net.named_parameters() if n.startswith('finger')]
        body=[p for n,p in net.named_parameters() if p.requires_grad and n.startswith('base.')]
        opt=torch.optim.AdamW([dict(params=body,lr=0.),dict(params=heads,lr=cfg['lr_head'])],weight_decay=.01)
        history=[];seen=set();start_epoch=1;elapsed=0.;started=time.perf_counter()
        if (d/'last.pt').exists():
            ck=torch.load(d/'last.pt',map_location='cpu',weights_only=False);assert ck['frozen_sha256']==sha(U/'FROZEN.json')
            net.load_state_dict(ck['model']);opt.load_state_dict(ck['opt']);history=ck['history'];seen=set(ck['seen']);start_epoch=ck['epoch']+1;elapsed=ck['seconds'];torch.set_rng_state(ck['rng'])
        for epoch in range(start_epoch,cfg['epochs']+1):
            opt.param_groups[0]['lr']=cfg['lr_base'] if condition=='finetune' and epoch>cfg['warmup_epochs'] else 0.
            net.train();rng=np.random.default_rng(seed*1000+epoch);order=rng.permutation(len(common));vals=[]
            write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING_SHAPE',pid=os.getpid(),condition=condition,seed=seed,epoch=epoch,other_branches='see motion_recovery and avatar_motion_pilot_20261113'))
            for k in range(cfg['steps_per_epoch']):
                common_rows=[common[int(i)] for i in order[k*24:(k+1)*24]]
                selected=[fine[int(i)] for i in rng.choice(len(fine),6,replace=False)]
                selected += [bends[int(i)] for i in rng.choice(len(bends),3,replace=False)]
                selected=list({r['clip_id']:r for r in selected}.values())
                supervised=[coarse[int(i)] for i in rng.choice(len(coarse),4,replace=False)]
                extra=[ef[int(i)] for i in rng.choice(len(ef),4,replace=False)]
                view='upper' if (epoch+k)%2 else 'full'
                selected=list({r['clip_id']:r for r in selected+supervised}.values())
                b=batch(selected,view);ab=batch(common_rows,view);eb=batch(extra,'full')
                o=net(b['x'],b['lengths'],b['word'],b['hand']);ao=net(ab['x'],ab['lengths'],ab['word']);eo=net(eb['x'],eb['lengths'],eb['word'],eb['hand'])
                with torch.no_grad():to=teacher(ab['x'],ab['lengths'],ab['word'])
                lf=balanced_fine_loss(o,b);ls=supervision(o,b,pw);lr=retention(ao,to,ab);lw=F.cross_entropy(ao['word_logits'],ab['class']);le=balanced_fine_loss(eo,eb)
                loss=lf+.5*ls+cfg['retention_weight']*lr+.2*lw+.5*le
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
                seen.update(r['clip_id'] for r in common_rows);vals.append([float(v.detach()) for v in [loss,lf,ls,lr,lw,le]])
            seconds=elapsed+time.perf_counter()-started
            history.append(dict(epoch=epoch,seconds=seconds,**dict(zip(['loss','finger','old_supervision','retention','word100','extra_finger'],np.mean(vals,axis=0).tolist()))))
            tmp=d/'last.tmp.pt';torch.save(dict(model=net.state_dict(),opt=opt.state_dict(),epoch=epoch,history=history,seen=sorted(seen),seconds=seconds,rng=torch.get_rng_state(),frozen_sha256=sha(U/'FROZEN.json')),tmp);tmp.replace(d/'last.pt')
            event('shape_epoch',condition=condition,seed=seed,**history[-1])
        assert len(seen)==899
        delta={prefix:float(sum((net.state_dict()[k]-v).float().square().sum() for k,v in base.items() if k.startswith(prefix)).sqrt()) for prefix in ['base.input','base.temporal','base.context','base.attributes','base.stage_query','base.prototype','word_head']}
        assert all(delta[k]==0 for k in ['base.attributes','base.stage_query','base.prototype','word_head'])
        if condition=='frozen':assert all(torch.equal(net.state_dict()[k],v) for k,v in base.items())
        else:assert delta['base.temporal']>0 and delta['base.context']>0
        torch.save(net.state_dict(),d/'model.pt')
        write(d/'COMPLETE.json',dict(condition=condition,seed=seed,history=history,seconds=history[-1]['seconds'],shared100_seen=len(seen),checkpoint_sha256=sha(d/'model.pt'),deltas=delta,development_training=False,frozen_sha256=sha(U/'FROZEN.json')))
        event('shape_run_complete',condition=condition,seed=seed,seconds=history[-1]['seconds'])
    write(U/'TRAINING_COMPLETE.json',dict(status='SIX_FIXED_RUNS_COMPLETE',runs=[str(p) for p in sorted((U/'runs').glob('*/COMPLETE.json'))]))
    write(U/'CURRENT_EXECUTION.json',dict(status='SHAPE_TRAINED_PENDING_EVALUATION',pid=None))

if __name__=='__main__':
    try:main()
    except BaseException:
        (U/f'FAILURE_TRAIN_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
