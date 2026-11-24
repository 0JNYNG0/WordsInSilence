"""Matched source-frame control vs reviewed block-context augmentation.

Only local role labels are supervised. Neither condition receives a
whole-sign validity/completion target. Original models are read-only.
"""
from graph import *
from prepare_linker import load
import numpy as np, torch, time, traceback, sys
O=U.parent; R=O/'word100_candidate_role_learning_20261111'
G=O/'word100_guide_requirement_learning_20261111'
OLD=U/'occurrence_supervision'; W=U/'context_role_training'


def prepare():
    assert not (W/'INPUTS_COMPLETE.json').exists()
    admission=read(W/'ADMISSION.json'); assert admission['approved']
    block_file=OLD/'FIT_BLOCKS.json'
    assert admission['fit_blocks_sha256']==hashlib.sha256(block_file.read_bytes()).hexdigest()
    rc=load('_ctx_req',G/'requirement_core.py'); role=load('_ctx_role',R/'role_core.py')
    blocks={r['clip_id']:r for r in read(block_file)['rows'] if r['status']=='READY_LOCAL_ROLE_CONTEXT_AUGMENTATION'}
    for b in blocks.values():assert role.sha(b['source'])==b['source_sha256']
    # Filter metadata before touching arrays: no development frames in preparation.
    public={}
    for r in read(rc.S/'READY.json')['rows']:
        if r['role']!='fit' or r['clip_id'] not in blocks:continue
        assert role.sha(r['path'])==r['sha256']
        with np.load(r['path'],allow_pickle=False) as zz:
            public[r['clip_id']]=dict(r,**{k:zz[k].copy() for k in ['x_full','x_upper','timestamp_us']})
    sources=[s for s in read(OLD/'INPUTS_COMPLETE.json')['rows'] if s['condition']=='reviewed_roles']
    assert len(sources)==72 and len(set(s['clip_id'] for s in sources))==36
    assert blocks and set(blocks)==set(public)
    assert set(blocks)<=set(s['clip_id'] for s in sources)
    admitted={(r['clip_id'],r['sequence_index']):r['proposed_family'] for r in read(OLD/'DIRECT_ROLE_AUDIT.json')['rows'] if r['category']!='uncertain'}
    base=rc.frozen_observer(); temporal=role.frozen_model(); records=[]
    for s in sources:
        assert role.sha(s['path'])==s['sha256']; z=np.load(s['path']); cid=s['clip_id']
        rec={k:s[k] for k in ['clip_id','person','view','source_sha256','path','sha256']}
        rec['paired']=None
        if cid in blocks:
            b=blocks[cid]; r=public[cid]; assert r['source_sha256']==b['source_sha256']==s['source_sha256']
            ii=np.array(b['query_to_source_sequence_indices']); ts=np.array(b['query_timestamp_us'])
            assert np.array_equal(np.sort(ii),np.arange(len(z['y'])))
            assert len(ts)==len(ii) and np.all(np.diff(ts)>0)
            x=rc.prior.physical(r['x_'+s['view']])[ii].copy()
            fh,_=role.lc.be.ev.frozen_forward(base,x,r['word'])
            ob=role.features(temporal,x,ts,fh)
            qidx=np.array([a['query_index'] for a in b['eligible_sampled_labels']],int)
            sidx=np.array([a['source_sequence_index'] for a in b['eligible_sampled_labels']],int)
            yy=np.array([a['family'] for a in b['eligible_sampled_labels']],np.int64)
            assert len(yy)>0 and np.all(ii[qidx]==sidx)
            assert len(set(sidx))==len(sidx) and np.all((yy>=1)&(yy<=7))
            for a in b['eligible_sampled_labels']:
                assert admitted[cid,a['source_sequence_index']]==a['family']
                assert a['source_frame_id'] not in b['boundary_excluded_source_frame_ids']
                for seam in b['splices']:
                    if seam['synthetic_discontinuity']:
                        cut=(int(ts[seam['query_left']])+int(ts[seam['query_right']]))/2
                        assert abs(int(ts[a['query_index']])-cut)>133333
            assert np.array_equal(z['y'][sidx],yy) and np.all(z['w'][sidx]==1)
            # Equal samples, labels, update count and loss weights in both arms.
            # Only the temporal context of the paired input differs.
            p=W/'paired_inputs'/f'{cid}_{s["view"]}.npz';p.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(p,normal=z['z'][sidx],permuted=ob['z'][qidx],y=yy,
                source_index=sidx,query_index=qidx)
            rec['paired']=dict(path=str(p),sha256=role.sha(p),frames=len(yy),families={str(v):int((yy==v).sum()) for v in np.unique(yy)})
        records.append(rec)
    assert sum(r['paired'] is not None for r in records)>0
    write(W/'INPUTS_COMPLETE.json',dict(rows=records,source_videos=len(set(r['clip_id'] for r in records)),paired_views=sum(r['paired'] is not None for r in records),only_new_permuted_feature_inference=True,private_access=False))
    print('Prepared',len(records),'normal views;',sum(r['paired'] is not None for r in records),'paired views',flush=True)


def train():
    assert not (W/'TRAINING_COMPLETE.json').exists()
    role=load('_ctx_train_role',R/'role_core.py'); cfg=read(W/'FROZEN.json'); inputs=[]
    assert cfg['conditions']==['source_context_control','permuted_context'] and cfg['seeds']==[42,43,44]
    for s in read(W/'INPUTS_COMPLETE.json')['rows']:
        assert role.sha(s['path'])==s['sha256']; z=np.load(s['path'])
        v=dict(z=torch.tensor(z['z']),y=torch.tensor(z['y']).long(),w=torch.tensor(z['w'])>0,paired=None)
        if s['paired']:
            p=s['paired'];assert role.sha(p['path'])==p['sha256']; q=np.load(p['path'])
            v['paired']={k:torch.tensor(q[k]) for k in ['normal','permuted','y']}
        inputs.append(v)
    results=[]
    for condition in cfg['conditions']:
        for seed in cfg['seeds']:
            dst=W/'runs'/condition/str(seed); done=dst/'TRAINING.json'
            if done.exists():
                v=read(done);assert role.sha(v['path'])==v['sha256'];results.append(v);continue
            net=role.RoleAdapter(seed); initial=torch.load(OLD/'runs/reviewed_roles'/str(seed)/'model.pt',weights_only=True,map_location='cpu');net.load_state_dict(initial);net.eval()
            with torch.no_grad(): teachers=[net(v['z']).softmax(-1).detach() for v in inputs]
            torch.manual_seed(seed);net.train();opt=torch.optim.AdamW(net.role.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            log=[]; start=time.perf_counter(); steps=0
            for epoch in range(cfg['epochs']):
                losses=[]
                for i in torch.randperm(len(inputs)).tolist():
                    v=inputs[i];mask=v['w'];p=net(v['z'])
                    assert mask.any()
                    loss=torch.nn.functional.cross_entropy(p[mask],v['y'][mask])
                    if v['paired'] is not None:
                        pair=v['paired']; key='normal' if condition=='source_context_control' else 'permuted'
                        loss=.5*loss+.5*torch.nn.functional.cross_entropy(net(pair[key]),pair['y'].long())
                    if (~mask).any():
                        loss+=cfg['teacher_kl_weight']*torch.nn.functional.kl_div(p[~mask].log_softmax(-1),teachers[i][~mask],reduction='batchmean')
                    opt.zero_grad();loss.backward();opt.step();steps+=1;losses.append(float(loss.detach()))
                log.append(float(np.mean(losses)))
            final=net.state_dict();unchanged=[k for k in initial if not k.startswith('role.')]
            assert all(torch.equal(initial[k],final[k]) for k in unchanged)
            assert steps==1800
            dst.mkdir(parents=True,exist_ok=True);torch.save(final,dst/'model.pt')
            v=dict(condition=condition,seed=seed,seconds=time.perf_counter()-start,steps=steps,loss=log,path=str(dst/'model.pt'),sha256=role.sha(dst/'model.pt'),only_role_changed=True)
            write(done,v);results.append(v);print(condition,seed,'seconds',round(v['seconds'],2),flush=True)
    write(W/'TRAINING_COMPLETE.json',dict(runs=results,private_used=False,web_changed=False,common100_backbone_changed=False))


if __name__=='__main__':
    try:
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        {'prepare':prepare,'train':train}[sys.argv[1]]()
    except BaseException:
        W.mkdir(exist_ok=True);(W/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
