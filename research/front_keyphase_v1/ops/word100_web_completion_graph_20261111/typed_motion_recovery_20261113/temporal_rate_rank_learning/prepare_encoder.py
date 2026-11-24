"""Frozen canonical time-resampling then NEW candidate-local frozen encoding.

`freeze` writes a reviewable contract only. `run-reviewed` must be executed by
root after review. No optimizer/training/old-model prediction rerun exists here.
"""
import importlib.util,sys,time,traceback
import numpy as np
from audit_inputs import U,H,G,O,J,W,read,write,sha,check
sys.dont_write_bytecode=True
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def freeze():
    assert not(U/'PREPARATION_FROZEN.json').exists()
    cfg=read(U/'FROZEN.json');audit=read(U/'INPUT_VALIDITY_AUDIT.json')
    assert read(U/'SAVED_VERIFICATION.json')['status']=='SAVED_RATE_AUDIT_PASS_NOT_ENCODER_READY'
    check(cfg['pins'])
    pins={str(p):sha(p)for p in [U/'prepare_encoder.py',U/'FROZEN.json',U/'INPUT_VALIDITY_AUDIT.json',U/'SAVED_VERIFICATION.json',
        J/'prepare.py',J/'splice_core.py',W/'transfer_local_timefix_core.py',W/'local_rank_core.py',J/'FROZEN.json']}
    # Existing J encoder/normalization/dependency pins; no new network selected.
    pins.update(read(J/'FROZEN.json')['pins'])
    for row in audit['sequences']:pins[row['path']]=row['sha256']
    for row in audit['cases']:pins[row['label_path']]=row['label_sha256']
    for row in cfg['parent_models']:pins[row['model']]=row['model_sha256']
    contract=dict(status='PREPARATION_CODE_READY_ENCODER_NOT_EXECUTED',at=time.time(),
        supersedes_only='Original FROZEN.future_encoder raw guard/adapter assumption. Original contract and raw_audit preserved as provenance evidence.',
        reason='Root fixed this as rank time-resolution augmentation. Resampling already canonical J observations exactly preserves prior geometry, view masks and scale. Running guard/adapter again would add an observation/normalization change and confound this temporal-only test.',
        canonical_base='Splice: exact saved J sequence x. Normal/reverse: argsort(J.baseline_index) restores original baseline-orientation x0:279. Rebuild original/reversed PTS from pinned source timestamps and reset dt279.',
        rate_input='Apply the exact frozen audit base_query_index to canonical x; require x0:279 byte-equal to mapped base. Set dt279=[0,diff(queryPTS)/1e6]. Normalize coordinates once using original fixed stats.',
        guard_policy='No guard/adapter call on canonical x. Existing crop quarantine and full/upper masks remain exactly inherited. Raw_audit is provenance only; no detector-pipeline equivalence claim to raw re-conversion speed controls.',
        body_contract='Recomputed per-frame relation body-valid masks for family2/3 must equal inherited source valid mapped by baseline_index. If not, stop and report; do not silently relabel or force full positives.',
        encoder='Same frozen common100 recomputed inside each new selected candidate. No copied/concatenated/resampled hidden1160. Existing rate1 bags reused without encoding.',
        sampling=dict(seed=19,normal='All full positives + up to512 uniformly sampled inherited negatives.',
            splice='All full positives + up to512 same J duration-stratified known-join negatives; retained contained audit partners.',batch=128),
        output_schema=dict(file='PREPARATION_INPUTS_COMPLETE.json',bags='All288 new .8/1.2 case rows, including skipped78 with no feature path; viable rows carry local_full,y,spans,rank_mask arrays.',
            row_fields=['clip_id','person','view','reverse','family','base_kind','rate','path','sha256','positive','negative','skipped'],
            replay_original='178 immutable old bags, base_kind=normal,rate=1',replay_splice='72 immutable J bags,base_kind=splice,rate=1',parent_models='3 matching-seed J checkpoints'),
        proposed_training=dict(executed=False,copy_parent='J matching42/43/44 including normalization',steps=1200,lr=.001,weight_decay=.001,
            weight='Per source/view/reverse/family2/3: normal .5,splice .5; each half uniformly split among available rates1/.8/1.2. Held rates excluded from loss but kept in denominator. Other families original replay only. Same2 sources per family per step.',
            interpretation='Equal source/family mass despite unequal retained-rate counts; duplicates and speed variants are not independent source videos.',
            selection='All3 seeds and final1200-step checkpoint only; no rate,epoch,threshold or seed tuning.'),
        proposed_evaluation=dict(executed=False,new_conditions=['rate_head','old_local_full_plus_rate_half'],
            blend='Exactly0.5*original local_full logits +0.5*new rate-head logits; same seed, same candidate list. No alpha search.',
            references='Saved baseline/local_full/J/previous half-J blend; root pins final reference files. No old-head inference.',
            scope='All3 seeds and same82 scenarios, raw chosen/shape/direction policies unchanged, other family scores exact original, actualPTS relation arithmetic applied symmetrically. Absence contract reviewed separately; no automatic G2 approval.'),
        no_dev_preparation=True,no_encoder_execution_yet=True,no_training=True,pins=pins)
    check(pins);write(U/'PREPARATION_FROZEN.json',contract)
    print('Canonical encoder preparation contract frozen; encoder NOT executed')

def prepare():
    assert not(U/'PREPARATION_INPUTS_COMPLETE.json').exists()and not(U/'canonical').exists()
    cfg=read(U/'FROZEN.json');plan=read(U/'PREPARATION_FROZEN.json');audit=read(U/'INPUT_VALIDITY_AUDIT.json')
    check(cfg['pins']);check(plan['pins'])
    oldprep=load('_rate_J_prepare',J/'prepare.py');core,role,_,_=oldprep.libraries()
    sample=load('_rate_J_sampling',J/'splice_core.py')
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    relation_model=role.frozen_model();base=core.encoder()
    assert not any(p.requires_grad for p in base.parameters())
    rng=np.random.default_rng(plan['sampling']['seed']);sm={(r['clip_id'],r['view'],r['reverse']):r for r in cfg['splice_sequences']}
    cases_by_path={r['path']:[c for c in audit['cases']if c['raw_path']==r['path']]for r in audit['sequences']}
    cache={};out=[];seqout=[];assertions=0
    for directory in ['canonical','bags']:(U/directory).mkdir()
    for si,s in enumerate(audit['sequences']):
        sid=(s['clip_id'],s['view'],s['reverse']);q=np.load(s['path']);raw_meta=cfg['raw'][s['clip_id']]
        if sid not in cache:
            j=np.load(sm[sid]['path']);pts=np.load(raw_meta['full_path'])['timestamp_us']
            order=np.argsort(j['baseline_index'],kind='stable')
            assert np.array_equal(j['baseline_index'][order],np.arange(len(order)))
            normal=j['x'][order].copy()
            normal_pts=pts[-1]-pts[::-1]if s['reverse']else pts-pts[0]
            normal[:,279]=np.r_[0,np.diff(normal_pts).astype(np.float32)/1e6]
            cache[sid]=dict(normal=(normal,normal_pts),splice=(j['x'].copy(),j['timestamp_us'].copy()))
        base_x,base_pts=cache[sid][s['base_kind']];ids=q['base_query_index'];x=base_x[ids].copy()
        x[:,279]=np.r_[0,np.diff(q['timestamp_us']).astype(np.float32)/1e6]
        assert np.array_equal(x[:,:279],base_x[ids,:279])
        assert np.array_equal(base_pts[ids],q['base_query_pts'])
        assert np.all(np.diff(q['timestamp_us'])>0)and np.isfinite(x).all();assertions+=3
        path=U/'canonical'/__import__('pathlib').Path(s['path']).name
        np.savez_compressed(path,x=x,timestamp_us=q['timestamp_us'],baseline_index=q['baseline_index'],
            base_query_index=ids,source_index=q['source_index'],source_block_id=q['source_block_id'])
        source=np.load(s['source_sequence']);relations=role.old.relation_features(x)
        for f in cfg['family_scope']:
            old_valid=source['valid'][:,cfg['family_order'].index(f)].any(1)[q['baseline_index']]
            assert np.array_equal(relations[:,core.AX[f]*2,8]>.5,old_valid),'Canonical observation contract differs from audit'
            assertions+=1
        selected=[]
        for r in cases_by_path[s['path']]:
            t=np.load(r['label_path']);record=dict(r,split='train',skipped=r['held'],source_sha256=raw_meta['source_sha256'],
                canonical_path=str(path),canonical_sha256=sha(path),positive=r['positive'],negative=r['negative'])
            if r['held']:
                record['skip_reason']=';'.join(r['hold_reasons']);out.append(record);continue
            if s['base_kind']=='splice':
                choose=sample.select_bag(t,rng);ix=choose['selected'];partners=choose['negative_partners']
                assert not choose['skipped']
            else:
                pi=np.flatnonzero(t['positive']);ni=np.flatnonzero(t['negative'])
                if len(ni)>512:ni=np.sort(rng.choice(ni,512,replace=False))
                ix=np.r_[pi,ni];partners=np.empty(0,np.int64)
            y=t['positive'][ix];assert y.any()and(~y).any()and(t['positive'][ix]|t['negative'][ix]).all()
            record.update(positive=int(y.sum()),negative=int((~y).sum()),negative_available=r['negative'])
            selected.append((record,t['spans'][ix],y,ix,partners))
        union=sorted({tuple(v)for _,sp,_,_,_ in selected for v in sp});features=np.empty((0,1160),np.float32)
        if union:
            qn=(relations-relation_model.qm.numpy())/relation_model.qs.numpy()
            qn=np.concatenate([qn[...,:4]*relations[...,8,None],qn[...,4:8]*relations[...,9,None],relations[...,8:]],-1).reshape(-1,100)
            pose=core.normalized_pose(x);sp=np.asarray(union,np.int32)
            features=np.concatenate([core.encode_windows(base,pose,qn,sp[a:a+128])for a in range(0,len(sp),128)])
            assert features.shape==(len(sp),1160)and np.isfinite(features).all()
        lu={v:i for i,v in enumerate(union)}
        for r,sp,y,ix,partners in selected:
            dest=U/'bags'/f'{path.stem}_{r["family"]}.npz';ff=features[[lu[tuple(v)]for v in sp]]
            np.savez_compressed(dest,local_full=ff,y=y,spans=sp,rank_mask=np.ones(len(y),bool),target_grid_index=ix,negative_partners=partners)
            r.update(path=str(dest),sha256=sha(dest));out.append(r)
        seqout.append(dict(s,canonical_path=str(path),canonical_sha256=sha(path),encoded_candidates=len(union)))
        write(U/'PREPARATION_PROGRESS.json',dict(sequences=si+1,total=144,bags=len(out)))
        print('rate encode',si+1,144,'new candidates',len(union),flush=True)
    assert len(out)==288 and sum(r['skipped']for r in out)==78
    # Preserve audit order and all held denominators for training integration.
    index={(r['clip_id'],r['view'],r['reverse'],r['family'],r['base_kind'],r['rate']):i for i,r in enumerate(audit['cases'])}
    out.sort(key=lambda r:index[r['clip_id'],r['view'],r['reverse'],r['family'],r['base_kind'],r['rate']])
    check(cfg['pins']);check(plan['pins'])
    write(U/'PREPARATION_INPUTS_COMPLETE.json',dict(status='CANONICAL_RATE_FEATURES_READY_NOT_TRAINED',at=time.time(),bags=out,sequences=seqout,
        replay_original=[dict(r,base_kind='normal',rate=1.)for r in cfg['replay_original']],
        replay_splice=[dict(r,base_kind='splice',rate=1.)for r in cfg['replay_splice']],parent_models=cfg['parent_models'],
        new_cases=288,new_feature_bags=210,held_cases=78,source_clips=6,canonical_contract_checks=assertions,
        source_contract_sha256=sha(U/'FROZEN.json'),preparation_contract_sha256=sha(U/'PREPARATION_FROZEN.json'),
        no_guard_adapter_rerun=True,all_source_x0_279_exact=True,dt_rebuilt=True,candidate_local_frozen_encoder=True,
        no_dev_inputs=True,no_training=True))
    print('Canonical rate features ready;210 new bags;78 held;NO TRAINING')

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1]in ['freeze','run-reviewed']
        freeze()if sys.argv[1]=='freeze'else prepare()
    except BaseException:
        (U/f'PREPARATION_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
