"""Freeze/prepare isolated public FIT inputs. Never starts training.

--freeze reads manifests only, filtering FIT30 before any array load.
--prepare reuses unchanged source latents, evaluates the existing boundary head
for new candidate windows, and recomputes ALL frozen features for admitted edits.
"""
from pathlib import Path
import argparse,json,hashlib,importlib.util,sys,datetime,collections,traceback
import numpy as np
import torch
from labels import direct_candidates,assign_extent,validate_aux_admission
from model import make_sequence

sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent
R=O/'word100_candidate_role_learning_20261111'
L=O/'word100_autonomous_localization_20261111'
S=O/'word100_shared_motion_learning_20261109'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def selected_metadata():
    annotations=read(L/'FIT_LABELS.json')['rows']
    assert len(annotations)==33 and all(r['role'] in ('fit','fit_additional') for r in annotations)
    wanted={r['clip_id'] for r in annotations}
    # Only manifest metadata is read before the public/FIT/clip filters below.
    selected=[r for r in read(S/'READY.json')['rows'] if r['role']=='fit' and r['clip_id'] in wanted]
    assert len(selected)==len(wanted)==30
    assert len({r['source_sha256'] for r in selected})==30
    cache={(r['clip_id'],r['view']):r for r in read(R/'TRAIN_INPUTS_COMPLETE.json')['rows'] if r['clip_id'] in wanted and not r['reverse']}
    assert len(cache)==60
    for row in selected:
        aa=[a for a in annotations if a['clip_id']==row['clip_id']]
        assert all(a['source_sha256']==row['source_sha256'] for a in aa)
        assert row['person'] in (1,3,5,7)
        for view in ('full','upper'):
            assert cache[row['clip_id'],view]['source_sha256']==row['source_sha256']
    return annotations,selected,cache


def freeze():
    assert not (W/'FROZEN.json').exists(),'frozen run already exists'
    annotations,rows,cache=selected_metadata()
    admissions=read(W/'AUX_ADMISSIONS.json').get('rows',[]) if (W/'AUX_ADMISSIONS.json').exists() else []
    paths=[W/'model.py',W/'labels.py',W/'prepare.py',W/'CONTRACT.json',L/'FIT_LABELS.json',L/'NORMALIZATION.json',S/'READY.json',R/'TRAIN_INPUTS_COMPLETE.json',R/'role_core.py',O/'word100_episode_joint_linkage_20261111/joint_core.py',O/'word100_broad_factorized_learning_20261110/core.py']
    paths += [R/'runs'/str(seed)/'model.pt' for seed in (42,43,44)]
    paths += [L/'runs'/str(seed)/'model.pt' for seed in (42,43,44)]
    paths.append(O/'word100_episode_joint_linkage_20261111/runs/42/model.pt')
    if (W/'train.py').exists():paths.append(W/'train.py')
    if (W/'AUX_ADMISSIONS.json').exists():paths.append(W/'AUX_ADMISSIONS.json')
    pins={str(p):sha(p) for p in paths}
    split={r['clip_id']:('head_probe' if r['person']==7 else 'train') for r in rows}
    cfg=dict(created=datetime.datetime.now().isoformat(),status='FROZEN_FOR_INPUT_PREPARATION_NOT_TRAINING_APPROVAL',
        conditions=['B','C'],seeds=[42,43,44],epochs=40,lr=.001,batch=64,optimizer='AdamW',weight_decay=.01,gradient_clip=2.,aux_loss_weight=1.,
        train_clips=26,head_probe_clips=4,head_probe_prior_encoder_exposure=True,
        head_probe_prior_quality_exposure=True,head_probe_meaning='Excluded from this B/C continuation only; previously used by encoder, localizer and copied R quality. Not fresh-person or unseen-clip performance.',split=split,views=['full','upper'],
        extent_classes=['hold_or_background','partial_reviewed_event','reviewed_extent','multiple_reviewed_events'],
        changed_class3_scope='Movement+terminal hold is excluded/unknown, not automatically class3; preserve distinction from old expanded proposal_adapter target.',
        edit_aux='Deliberate source-edit provenance only; not true sensor error or sign correctness.',synthetic_extent_loss=0,
        aux_admissions=len(admissions),aux_enabled=bool(admissions),aux_without_admission='Skip and record; never infer admissions from prediction or event midpoint.',
        no_word_or_desired_order_input=True,no_existing_A_replay=True,no_private=True,no_web_change=True,
        sources=[dict(clip_id=r['clip_id'],person=r['person'],source_sha256=r['source_sha256'],path=r['path'],sha256=r['sha256'],split=split[r['clip_id']]) for r in rows],
        source_context_caches=[dict(clip_id=k[0],view=k[1],path=r['path'],sha256=r['sha256']) for k,r in cache.items()],pins=pins)
    write(W/'FROZEN.json',cfg)
    return cfg


def reconstruct_source_ob(role,net,x,z):
    q=torch.tensor(role.old.relation_features(x)[None])
    h=torch.tensor(z[:,228:292][None]);fh=torch.tensor(z[:,392:456][None])
    with torch.inference_mode():
        norm=(q-net.qm)/net.qs
        norm=torch.cat([norm[...,:4]*q[...,8,None],norm[...,4:8]*q[...,9,None],q[...,8:]],-1)
    assert np.allclose(norm[0].numpy().reshape(len(x),100),z[:,292:392],atol=1e-6)
    return dict(h=h,q=q,fh=fh)


def boundary_candidates(net,role,ob):
    h,q,fh=ob['h'],ob['q'],ob['fh'];T=h.shape[1]
    with torch.inference_mode():
        norm=(q-net.qm)/net.qs
        norm=torch.cat([norm[...,:4]*q[...,8,None],norm[...,4:8]*q[...,9,None],q[...,8:]],-1)
        ids=torch.arange(10)
        emb=torch.cat([torch.nn.functional.one_hot(ids//2,5),torch.nn.functional.one_hot(ids%2,2)],-1).float()[None,None].expand(1,T,-1,-1)
        bp=net.boundary(torch.cat([h[:,:,None].expand(-1,-1,10,-1),norm,fh[:,:,None].expand(-1,-1,10,-1),emb],-1)).sigmoid()[0].numpy()
    return role.jc.candidate_spans(bp,q[0].numpy(),[])


def prepare():
    assert (W/'FROZEN.json').exists(),'run --freeze after code/provenance review'
    assert not (W/'INPUTS_COMPLETE.json').exists(),'completed preparation is not replayed'
    cfg=read(W/'FROZEN.json')
    for path,h in cfg['pins'].items():assert sha(path)==h,path
    annotations,selected,cache=selected_metadata()
    assert [r['clip_id'] for r in selected]==[r['clip_id'] for r in cfg['sources']]
    admissions=read(W/'AUX_ADMISSIONS.json').get('rows',[]) if cfg['aux_enabled'] else []
    assert len(admissions)==cfg['aux_admissions']
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    role=load('_local_quality_role',R/'role_core.py');net=role.frozen_model()
    base=None  # Loaded only if admitted NEW edited inputs need frozen encoders.
    result=[];unknown=[];input_audit=[];aux_audit=[]
    def save_sample(row,view,kind,x,ts,ob,spec,y,ids,reason,aux_y,aux_mask,origin):
        a,b,k=spec;axis=k//2;moving='L' if k%2 else 'R';anchor='R' if moving=='L' else 'L'
        values,observed=role.prior.wrist_series(x,role.prior.motion.KINDS[axis],moving,anchor)
        if y>=0 and not bool(observed[a:b+1].all()):
            unknown.append(dict(clip_id=row['clip_id'],view=view,span=[a,b],axis=axis,moving=moving,reason='extent_observation_incomplete',kind=kind,
                intended_extent_target=y,episode_ids=ids,origins=origin,interval_label_reason=reason,
                observed_frames=int(observed[a:b+1].sum()),total_frames=b-a+1));return
        with torch.inference_mode():summary=net.segment(ob['h'],ob['q'],ob['fh'],[(0,a,b,k)])[0].numpy()
        sequence=make_sequence(values[a:b+1],observed[a:b+1],ts[a:b+1])
        extent_mask=int(y>=0 and kind=='source_original')
        sample_id=f"sample_{len(result):06d}";path=W/'inputs'/f'{sample_id}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,summary=summary,sequence=sequence,timestamps_us=ts[a:b+1],extent_target=np.int64(y),extent_mask=np.float32(extent_mask),aux_target=np.int64(aux_y),aux_mask=np.float32(aux_mask))
        result.append(dict(id=sample_id,path=str(path),sha256=sha(path),clip_id=row['clip_id'],source_sha256=row['source_sha256'],person=row['person'],view=view,kind=kind,split=cfg['split'][row['clip_id']],a=a,b=b,axis=axis,moving=moving,episode_ids=ids,extent_target=y,extent_mask=extent_mask,aux_target=aux_y,aux_mask=aux_mask,label_reason=reason,origins=origin,frames=b-a+1))
    for row in selected:
        assert sha(row['path'])==row['sha256']
        raw=np.load(row['path'],allow_pickle=False)
        assert np.array_equal(raw['frame_id'],np.arange(len(raw['frame_id']))),'annotation index mapping must be explicitly checked'
        aa=[a for a in annotations if a['clip_id']==row['clip_id']]
        for view in cfg['views']:
            cm=cache[row['clip_id'],view];assert sha(cm['path'])==cm['sha256']
            cc=np.load(cm['path'],allow_pickle=False);x=role.prior.physical(raw['x_'+view]);ts=raw['timestamp_us'].copy();ob=reconstruct_source_ob(role,net,x,cc['z'])
            # Original context is reused unchanged, never shuffled for a new edit.
            pool=direct_candidates(aa)+boundary_candidates(net,role,ob);unique={}
            for c in pool:
                key=(int(c['a']),int(c['b']),int(c['axis']),c['moving'])
                unique.setdefault(key,[]).append(c.get('source','unknown'))
            for (a,b,axis,moving),origin in unique.items():
                y,ids,reason=assign_extent(a,b,axis,moving,aa)
                if y<0:
                    unknown.append(dict(clip_id=row['clip_id'],view=view,span=[a,b],axis=axis,moving=moving,reason=reason,origins=origin));continue
                save_sample(row,view,'source_original',x,ts,ob,(a,b,axis*2+(moving=='L')),y,ids,reason,0,int(cfg['aux_enabled']),origin)
            input_audit.append(dict(clip_id=row['clip_id'],view=view,source_cached_context_reused=True,source_frames=len(x),raw_candidates=len(pool),unique_candidates=len(unique)))
            for number,ad in enumerate(a for a in admissions if a['clip_id']==row['clip_id']):
                ar,e,left,right=validate_aux_admission(ad,aa,len(x))
                for proof in [ad['rgb_evidence']]:assert sha(proof['path'])==proof['sha256']
                values,good=role.prior.wrist_series(x,role.prior.motion.KINDS[ar['axis']],ar['moving'],ar.get('anchor','L'))
                proof_frames=[left,right]+ad['observed_removed_frames']
                if not bool(good[proof_frames].all()):
                    aux_audit.append(dict(clip_id=row['clip_id'],view=view,admission=number,status='held_observation_missing'));continue
                indices=np.r_[np.arange(left+1),np.arange(right,len(x))]
                nx=x[indices].copy();nt=ts[:len(indices)].copy()
                if base is None:
                    base=role.prior.Network('coupled',role.read(role.old.P/'NORMALIZATION.json'))
                    base.load_state_dict(torch.load(role.old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval();base.requires_grad_(False)
                # The encoder latent does not depend on word; 0 only fills the
                # unused legacy stage-selector argument. Stage scores discarded.
                fh,_=role.lc.be.ev.frozen_forward(base,nx,0)
                edited=role.features(net,nx,nt,fh)
                a=int(np.flatnonzero(indices==e['onset'][0])[0]);b=int(np.flatnonzero(indices==e['offset'][1])[0])
                save_sample(row,view,'edited_observation',nx,nt,edited,(a,b,ar['axis']*2+(ar['moving']=='L')),-1,[],
                    'admitted_source_edit_only_extent_loss_masked',1,1,['direct_RGB_aux_admission'])
                aux_audit.append(dict(clip_id=row['clip_id'],view=view,admission=number,status='prepared_aux_only',left=left,right=right,source_indices_audit_only=indices.tolist(),full_frozen_features_recomputed=True,extent_mask=0))
    # Exact contradictory observational inputs cannot become learnable targets.
    fingerprints={};conflicts=[]
    for r in result:
        z=np.load(r['path']);fingerprint=hashlib.sha256(z['summary'].tobytes()+z['sequence'].tobytes()).hexdigest()
        key=fingerprint
        if r['aux_mask'] and key in fingerprints and fingerprints[key]['aux_target']!=r['aux_target']:conflicts.append([fingerprints[key]['id'],r['id']])
        fingerprints.setdefault(key,r)
    assert not conflicts,('observationally identical opposite auxiliary labels',conflicts)
    counts=collections.Counter((r['split'],r['extent_target']) for r in result if r['extent_mask'])
    actual_aux_positive=sum(r['aux_mask'] and r['aux_target']==1 for r in result)
    write(W/'UNKNOWN_CANDIDATES.json',dict(rows=unknown))
    write(W/'SOURCE_INPUT_AUDIT.json',dict(rows=input_audit,aux=aux_audit,private_access=False,source_context_replayed=False))
    write(W/'INPUTS_COMPLETE.json',dict(rows=result,extent_counts={f'{s}/{c}':n for (s,c),n in counts.items()},aux_enabled=bool(actual_aux_positive),aux_positive=actual_aux_positive,
        aux_training_requires_positive_per_train_split=True,source_clips=30,train_clips=26,head_probe_clips=4,head_probe_prior_encoder_exposure=True,head_probe_prior_quality_exposure=True,
        unknown_candidates=len(unknown),synthetic_extent_mask_always_zero=True,training_started=False,web_changed=False))
    print('Prepared',len(result),'samples; unknown',len(unknown),'aux positives',actual_aux_positive)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.freeze == args.prepare:parser.error('choose exactly one of --freeze or --prepare; neither starts training')
    try:
        if args.freeze:freeze()
        else:prepare()
    except BaseException:
        write(W/f'FAILURE_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.json',dict(traceback=traceback.format_exc(),training_started=False));raise
