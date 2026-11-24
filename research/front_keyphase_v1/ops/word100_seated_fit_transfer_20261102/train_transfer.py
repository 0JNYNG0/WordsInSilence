import time,os,traceback,shutil
from transfer_data import *
from evaluate_witness import behavior

def weights(rows,key,count):
    c=np.bincount(np.concatenate([r[key][r[key]>=0] for r in rows]),minlength=count)
    return torch.tensor(np.clip(c.mean()/np.maximum(c,1),.25,4),dtype=torch.float32)

def main():
    assert not (L/'TRAINING_FREEZE.json').exists();assert shutil.disk_usage(DEST).free>1024**3
    torch.set_num_threads(2);base,extras=load_pairs();stats=read(D/'NORMALIZATION.json');start_path=J/'runs/point_ce_42/model.pt'
    assert read(L/'INPUT_TARGET_AUDIT.json')['status']=='PASS_FOR_PAIRED_PUBLIC_FIT_EXPERIMENT'
    torch.manual_seed(42);model=WitnessNet();model.load_state_dict(torch.load(start_path,map_location='cpu',weights_only=True));model.blank.requires_grad_(False)
    xx,im,ll,yy,aa=batch_j(extras['seated_variation'][:2],stats)
    model.eval()
    with torch.inference_mode():
        pa,pb,_=model(xx,im,ll);sa,sb,_=model(xx[:1,:ll[0]],im[:1,:ll[0]],ll[:1]);torch.testing.assert_close(pa[0,:ll[0]],sa[0],rtol=1e-5,atol=1e-5);torch.testing.assert_close(pb[0,:ll[0]],sb[0],rtol=1e-5,atol=1e-5)
        altered=im.clone();altered[:,:,-1]=0;pa,pb,_=model(xx,altered,ll);altered[:,:,:768]=999;qa,qb,_=model(xx,altered,ll);torch.testing.assert_close(pa,qa);torch.testing.assert_close(pb,qb)
    pa,pb,_=model(xx,im,ll);(pa.sum()+pb.sum()).backward()
    assert model.actor.rgb[0].weight.grad is None and model.actor.query.weight.grad is not None
    files=[L/'PAIR_MANIFEST.json',L/'INPUT_TARGET_AUDIT.json',L/'transfer_data.py',L/'train_transfer.py',L/'evaluate_transfer.py',J/'sequence_model.py',F/'model.py',D/'NORMALIZATION.json',D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',start_path,T/'runs/current_42/model.pt']
    pins={str(p):sha(p) for p in files}
    for r in base+extras['original_replay']+extras['seated_variation']:pins[r['cache']]=r['cache_sha256']
    plan=dict(at=clock(),primary='seated_variation',conditions=['original_replay','seated_variation'],seed=42,epochs=80,batch_size=12,lr=.0005,weight_decay=.01,
        starting_checkpoint=str(start_path),rows_per_condition=252,original_replay_rows=216,extra_views=36,same_labels_init_updates=True,
        trainable='Action guide/query/rest readout and two-way recurrent activity head only. Same frozen SignRep/latent/RGB/process encoder and protected T observer in both.',
        hypothesis='Camera-view augmentation, not more updates or extra labels. Cropped extra rest/hidden shape withheld in BOTH matched extra sets. Original216labels kept.',
        split='FITrecorded01-09 only. Exposed DEV11-13 development, never training. H14/15locked; no user/mentor.',
        evaluation='88cases with12native replaced by already preparedK coherent sensor inputs;76unchanged live controls. Fixed2new models once each. No whole feedback claim using old native ROI guide arrays.',
        gate=read(D/'FROZEN.json')['gate'],additional_gate='No raw correct action lost by activity interpretation. Improve seated action/order without regressing coherent native/short reverse/imagegap protections.',
        pins=pins,preflight=dict(padding=True,missing_image_mask=True,encoder_frozen=True,behavior=behavior()),whole_goal_complete=False)
    exclusive(L/'TRAINING_FREEZE.json',plan);reports=[]
    for tag in plan['conditions']:
        rows=base+extras[tag]
        for r in rows:r['activity']=np.where(r['role']<0,-1,(r['role']==1).astype('int64'))
        aw=weights(rows,'episode',9);rw=weights(rows,'activity',2);dest=DEST/'runs'/tag;dest.mkdir(parents=True,exist_ok=False)
        torch.manual_seed(42);rng=np.random.default_rng(42);m=WitnessNet();m.load_state_dict(torch.load(start_path,map_location='cpu',weights_only=True));m.blank.requires_grad_(False)
        before={k:v.clone() for k,v in m.state_dict().items()};opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=plan['weight_decay'])
        start=time.perf_counter();history=[]
        for epoch in range(plan['epochs']):
            m.train();order=rng.permutation(len(rows));losses=[]
            for i in range(0,len(rows),12):
                rr=[rows[k] for k in order[i:i+12]];x,rgb,ll,yy,aa=batch_j(rr,stats);ac,act,_=m(x,rgb,ll);loss=loss_fn(ac,yy,aw)+loss_fn(act,aa,rw);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();losses.append(float(loss.detach()))
            if epoch%20==0 or epoch==plan['epochs']-1:
                r=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.perf_counter()-start);history.append(r);print(tag,r,flush=True)
                write(L/'TRAIN_PROGRESS.json',dict(at=clock(),status='TRAINING',pid=os.getpid(),tag=tag,epoch=epoch+1))
        delta={k:float((v-before[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point}
        fixed=[k for k in delta if any(k.startswith('actor.'+n+'.') for n in ['embed','rgb','context','delta'])];assert all(delta[k]==0 for k in fixed)
        assert delta['actor.query.weight']>0 and delta['role_output.weight']>0
        torch.save(m.state_dict(),dest/'model.pt');result=dict(condition=tag,sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-start,history=history,changes=delta,encoder_frozen=True)
        exclusive(dest/'COMPLETE.json',result);reports.append(result)
    for p,h in pins.items():assert sha(p)==h
    exclusive(L/'TRAINING_RESULT.json',dict(at=clock(),runs=reports,protected_inputs_unchanged=True,whole_goal_complete=False))
    write(L/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None,runs=2))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (L/'TRAIN_FIRST_FAILURE.txt').exists():(L/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
