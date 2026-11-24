import os,time,traceback
from episode_model import *

def freeze(rows):
    pins={str(p):sha(p) for p in [D/'PLAN.json',D/'ACTION_TARGETS.json',D/'TARGETS_VERIFIED.json',D/'episode_model.py',D/'train_episode.py',B/'PREPARED.json',B/'runs/joint_42/model.pt',T/'runs/current_42/model.pt',P/'GUIDE_SUPPORT.json']}
    plan=dict(at=clock(),primary='episode_42',conditions=['readout','episode'],seeds=[42,43,44],epochs=80,batch_size=12,lr=.0005,weight_decay=.01,
      task='9 sparse event-content families including directly observed rest; source event content independent of playback direction. Direction, completion and language correctness stay separate.',
      architecture='Copied SignRep guide/query neural readout and RGB/geometry adapter; episode additionally fine-tunes the copied recurrent process encoder. No new backbone training. Existing physical/shape/relation/motion outputs remain immutable.',
      blindness='No target word, normalized clock or expected order in the observational forward pass. Old target-masked current state probabilities are zeroed for input. Guide embeddings are learned class descriptions, not a forced target sequence.',
      split='Public FIT01-09 only,36 originals216 saved transformed inputs. DEV11-13 is exposed development. H14/15 locked. No private/mentor input.',
      evaluation='Freeze new evaluation-only sparse episode labels from public DEV chronological sheets without candidate predictions. They are not replacement physical/role labels and never enter training. Evaluate88existing DEV inputs once per model; old evaluations reused read-only.',
      gate=dict(normal_action_macro_min=.90,normal_family_min_recall=.85,normal_previous_order_loss=0,normal_false_order=0,normal_supported_order_min=21,local_reverse_detected_min=2,local_reverse_false_supported=0,cut_false_completion=0,image_gap_certified=0,old_observations_preserved=True),
      decision='Primary fixed before training; no best seed or DEV threshold selection. Improved family classification alone is not deployment or whole goal completion. Accept order gain only if every previously reviewed core frame remains an action observation and original shape/movement checks also pass.',
      uncertainty='No dense interval truth; edit seams masked for supervision only; original native axis and missing frames retained.0.8 confidence and2adjacent-frame evidence inherited; single contrary evidence remains uncertain.',
      pins=pins,inputs=len(rows),known=sum(int((r['episode']>=0).sum()) for r in rows))
    exclusive(D/'FROZEN.json',plan);return plan

def main():
    assert not (D/'TRAINING_RESULT.json').exists();torch.set_num_threads(2);rows=fit_rows();plan=freeze(rows);s=process.stats(rows)
    exclusive(D/'NORMALIZATION.json',s)
    torch.manual_seed(42);m=EpisodeNet(True).eval();x,rgb,l,y=batch(rows[:2],s)
    with torch.inference_mode():
        out=m(x,rgb,l);torch.testing.assert_close(out[0,:l[0]],m(x[:1,:l[0]],rgb[:1,:l[0]],l[:1])[0],rtol=1e-5,atol=1e-5)
        empty=rgb.clone();empty[:,:,-1]=0;a=m(x,empty,l);empty[:,:,:768]=31;torch.testing.assert_close(a,m(x,empty,l))
    assert all(np.isfinite(r['rgb']).all() and len(r['episode'])==len(r['x']) for r in rows)
    before=rows[0]['current'].copy();trial=dict(rows[0],current=np.ones_like(before)*.1);x2,im2,l2,_=batch([trial],s);x1,im1,l1,_=batch([rows[0]],s)
    torch.testing.assert_close(x1,x2);assert np.array_equal(before,rows[0]['current'])
    weights=torch.tensor(np.bincount(np.concatenate([r['episode'][r['episode']>=0] for r in rows]),minlength=9),dtype=torch.float32)
    weights=(weights.mean()/weights).clamp(.25,4)
    loss_fn(m(x,rgb,l),y,weights).backward();assert m.query.weight.grad is not None and m.guide[0].weight.grad is not None
    exclusive(D/'PREFLIGHT.json',dict(at=clock(),status='PASS',padding=True,image_absence_gate=True,target_masked_state_not_used=True,old_arrays_preserved=True,guide_gradient=True,input_count=len(rows),sparse_known=plan['known']))
    reports=[]
    for kind in plan['conditions']:
        for seed in plan['seeds']:
            tag=f'{kind}_{seed}';d=D/'runs'/tag;assert not d.exists();d.mkdir(parents=True)
            torch.manual_seed(seed);rng=np.random.default_rng(seed);m=EpisodeNet(kind=='episode');init={k:v.clone() for k,v in m.state_dict().items()}
            opt=torch.optim.AdamW(m.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay']);start=time.perf_counter();history=[]
            for ep in range(plan['epochs']):
                m.train();indices=rng.permutation(len(rows));losses=[]
                for j in range(0,len(rows),plan['batch_size']):
                    rr=[rows[k] for k in indices[j:j+plan['batch_size']]];xx,im,ll,yy=batch(rr,s);loss=loss_fn(m(xx,im,ll),yy,weights)
                    assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();losses.append(float(loss.detach()))
                if ep%20==0 or ep==plan['epochs']-1:
                    rec=dict(epoch=ep+1,loss=float(np.mean(losses)),seconds=time.perf_counter()-start);history.append(rec);print(tag,rec,flush=True)
                    write(D/'TRAIN_PROGRESS.json',dict(at=clock(),status='TRAINING',pid=os.getpid(),tag=tag,epoch=ep+1))
            changes={k:float((v-init[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point};assert changes['query.weight']>0 and changes['delta.weight']>0
            torch.save(m.state_dict(),d/'model.pt');report=dict(tag=tag,condition=kind,seed=seed,seconds=time.perf_counter()-start,sha256=sha(d/'model.pt'),changes=changes,history=history)
            exclusive(d/'COMPLETE.json',report);reports.append(report)
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(D/'TRAINING_RESULT.json',dict(at=clock(),runs=reports,protected_files_unchanged=True,new_backbone_inference=0,whole_goal_complete=False))
    write(D/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None,runs=len(reports)))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (D/'TRAIN_FIRST_FAILURE.txt').exists():(D/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
