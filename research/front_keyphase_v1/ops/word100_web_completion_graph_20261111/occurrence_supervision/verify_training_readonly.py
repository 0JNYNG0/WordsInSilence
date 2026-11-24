"""Separate artifact arithmetic audit. No model forward/training or DEV features."""
from pathlib import Path
import collections, hashlib, json, datetime
import numpy as np
import torch

W = Path(__file__).resolve().parent
G = W.parent
O = G.parent
R = O / 'word100_candidate_role_learning_20261111'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def require(test, message):
    if not test:
        raise AssertionError(message)

def main():
    audit = read(W/'DIRECT_ROLE_AUDIT.json')
    admission = read(W/'ROOT_ADMISSION.json')
    frozen = read(W/'FROZEN.json')
    manifest = read(W/'INPUTS_COMPLETE.json')
    trained = read(W/'TRAINING_COMPLETE.json')
    oldlabels = {r['clip_id']:r for r in read(R/'ROLE_LABELS.json')['rows']}
    sources = [r for r in read(R/'TRAIN_INPUTS_COMPLETE.json')['rows'] if not r['reverse'] and r['clip_id'] in oldlabels]
    src = {(r['clip_id'],r['view']):r for r in sources}
    accepted = {(r['clip_id'],r['sequence_index']):r for r in audit['rows'] if r['category'] != 'uncertain'}
    uncertain = {(r['clip_id'],r['sequence_index']):r for r in audit['rows'] if r['category'] == 'uncertain'}
    require(len(accepted)==76 and len(uncertain)==132, 'Unexpected review counts')
    require(admission['approved_for_isolated_fit'] and admission['audit_sha256']==sha(W/'DIRECT_ROLE_AUDIT.json'), 'Admission mismatch')
    require(frozen['admission_sha256']==sha(W/'ROOT_ADMISSION.json'), 'Frozen admission mismatch')
    require({f"{c}:{i}" for (c,i),r in accepted.items() if r['category']=='conflict'}==set(admission['accepted_conflicts']), 'Conflicts not exactly admitted')
    require(len(src)==72 and len(manifest['rows'])==144, 'Wrong input count')
    require(frozen['conditions']==['continued_old_direct','reviewed_roles'] and frozen['seeds']==[42,43,44], 'Condition/seed change')
    require(frozen['epochs']==25 and frozen['lr']==.0003 and frozen['weight_decay']==.01 and frozen['teacher_kl_weight']==.2, 'Schedule change')
    by = {(r['condition'],r['clip_id'],r['view']):r for r in manifest['rows']}
    require(len(by)==144, 'Duplicate inputs')
    pairchecks=[]
    condition_counts=collections.Counter()
    unique_changed=set(); unique_direct_added=set(); unique_y_changed=set()
    for key,s in src.items():
        cid,view=key
        require(sha(s['path'])==s['sha256'], 'Original cached feature pin changed')
        old=np.load(s['path'])
        require(np.array_equal(old['y'],np.asarray(oldlabels[cid]['y'])), 'Source label vector mismatch')
        require(np.array_equal(old['w'],np.asarray(oldlabels[cid]['w'],dtype=old['w'].dtype)), 'Source weight vector mismatch')
        require(np.isfinite(old['z']).all() and old['z'].shape[1]==684, 'Feature shape/nonfinite')
        expected_y=old['y'].copy();expected_w=(old['w']==1).astype(np.float32)
        expected_candidate_y=expected_y.copy();expected_candidate_w=expected_w.copy()
        for (c,i),r in accepted.items():
            if c != cid: continue
            require(r['source_sha256']==s['source_sha256'], 'Admitted source mismatch')
            require(int(expected_y[i])==r['old_family'], 'Admitted prior label mismatch')
            require(r['source_images'] and not r['boundary_bands'], 'Admitted row lacks RGB or has uncertain boundary')
            expected_candidate_y[i]=r['proposed_family'];expected_candidate_w[i]=1
        actual={}
        for condition,ey,ew in [('continued_old_direct',expected_y,expected_w),('reviewed_roles',expected_candidate_y,expected_candidate_w)]:
            m=by[(condition,cid,view)]
            require(m['source_sha256']==s['source_sha256'] and 1<=m['person']<=9, 'FIT source identity mismatch')
            require(sha(m['path'])==m['sha256'], 'New input pin changed')
            z=np.load(m['path']);require(set(z.files)=={'z','y','w'}, 'Unexpected model input fields')
            require(np.array_equal(z['z'],old['z']), 'Features changed')
            require(np.array_equal(z['y'],ey) and np.array_equal(z['w'],ew), 'Labels differ from exact admitted edition')
            require(set(np.unique(z['w']))<={0.,1.}, 'Weak labels used')
            require((z['y'][z['w']>0]>=0).all(), 'Unlabeled supervised target')
            require(int(z['w'].sum())==m['direct_frames'], 'Manifest direct count mismatch')
            condition_counts[condition]+=int(z['w'].sum())
            actual[condition]=z
        a=actual['continued_old_direct'];b=actual['reviewed_roles']
        changed=np.where((a['y']!=b['y'])|(a['w']!=b['w']))[0]
        for i in changed:
            require((cid,int(i)) in accepted, 'Change outside admitted sampled frames')
            unique_changed.add((cid,int(i)))
        for i in np.where(a['y']!=b['y'])[0]:unique_y_changed.add((cid,int(i)))
        for i in np.where(a['w']<b['w'])[0]:unique_direct_added.add((cid,int(i)))
        for (c,i),r in uncertain.items():
            if c==cid:require(a['y'][i]==b['y'][i] and a['w'][i]==b['w'][i], 'Uncertain frame changed')
        pairchecks.append(dict(clip_id=cid,view=view,source_sha256=s['source_sha256'],feature_shape=list(old['z'].shape),features_bit_equal=True,changed_indices=changed.tolist(),new_direct_indices=np.where(a['w']<b['w'])[0].tolist()))

    # Metadata only: no DEV arrays, videos, labels, predictions, or scores loaded.
    ready_path=O/'word100_shared_motion_learning_20261109/READY.json'
    ready=read(ready_path)['rows']
    fits=[r for r in ready if r['role']=='fit']
    dev=[r for r in ready if r['role']!='fit']
    fit_ids={r['clip_id']:r for r in fits}
    train_sha={s['source_sha256'] for s in sources}
    dev_sha={r['source_sha256'] for r in dev}
    require(len(fits)==899 and len(dev)==300, 'Unexpected common100 split')
    require(not train_sha & dev_sha, 'Training/DEV source SHA overlap')
    require(not {r['source_sha256'] for r in fits} & dev_sha, 'Common100 FIT/DEV source SHA overlap')
    require(all(s['clip_id'] in fit_ids and s['source_sha256']==fit_ids[s['clip_id']]['source_sha256'] for s in sources), 'Training clip not in FIT')
    require({r['person'] for r in dev}=={11,12,13}, 'Unexpected DEV record IDs')

    changes=[]
    expected_keys={'role.0.weight','role.0.bias','role.2.weight','role.2.bias'}
    for r in trained['runs']:
        require(r['condition'] in frozen['conditions'] and r['seed'] in frozen['seeds'], 'Unexpected run')
        require(sha(r['path'])==r['sha256'] and len(r['loss'])==25, 'Checkpoint pin/epoch mismatch')
        startpath=R/'runs'/str(r['seed'])/'model.pt'
        original=torch.load(startpath,weights_only=True,map_location='cpu')
        final=torch.load(r['path'],weights_only=True,map_location='cpu')
        require(set(original)==set(final), 'Checkpoint architecture changed')
        changed=[k for k in original if not torch.equal(original[k],final[k])]
        require(set(changed)==expected_keys, 'Non-role changed or role not updated')
        frozenkeys=[k for k in original if k not in expected_keys]
        require(frozenkeys and all(k.startswith(('quality.','pair.')) for k in frozenkeys), 'Unexpected frozen checkpoint keys')
        require(all(torch.isfinite(v).all() for v in final.values()), 'Nonfinite checkpoint')
        changes.append(dict(condition=r['condition'],seed=r['seed'],initial_path=str(startpath),initial_sha256=sha(startpath),final_sha256=r['sha256'],changed_tensors=changed,frozen_bit_equal_tensors=frozenkeys,role_max_abs_change={k:float((original[k]-final[k]).abs().max()) for k in changed}))
    require(len(changes)==6 and len({(r['condition'],r['seed']) for r in changes})==6, 'Missing/duplicate model')

    # No source/score parsing: hash selected pre-existing protected model/code/label files.
    protects=read(G/'PROTECTIONS.json')
    selected={k:v for k,v in protects.items() if Path(k).suffix=='.pt' or 'candidate_role_learning_20261111' in k or 'action_feedback_demo_20261108' in k or Path(k).name=='ROLE_LABELS.json'}
    protected=[]
    for p,h in selected.items():
        require(sha(p)==h,'Protected file changed: '+p)
        protected.append(dict(path=p,sha256=h))
    for pin in audit['source_label_pins']:
        require(sha(pin['path'])==pin['sha256'], 'Original direct-role source label changed')
    for pin in audit['image_pins']:
        require(sha(pin['path'])==pin['sha256'], 'RGB evidence changed')

    train_code=(G/'train_role_correction.py').read_text(encoding='utf-8')
    required_snippets=['opt=torch.optim.AdamW(n.role.parameters()',"valid=it['w']>0","p=n(it['z'])",'cross_entropy(p[valid],it[\'y\'][valid])','p[~valid].log_softmax(-1),teachers[i][~valid]','n.load_state_dict(initial)','for epoch in range(cfg[\'epochs\'])']
    require(all(s in train_code for s in required_snippets), 'Reviewed training contract differs')
    pins=[W/'FROZEN.json',W/'ROOT_ADMISSION.json',W/'DIRECT_ROLE_AUDIT.json',W/'INPUTS_COMPLETE.json',W/'TRAINING_COMPLETE.json',W/'FIRST_PREP_ERROR.json',G/'prepare_role_correction.py',G/'train_role_correction.py',R/'role_core.py',ready_path,Path(__file__)]
    out=dict(status='PASS_ARTIFACT_CODE_AND_SPLIT_AUDIT_NO_PERFORMANCE_CLAIM',at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),reviewer='Separate Codex sub-agent artifact/source audit; not independent human/KSL ground truth',counts=dict(source_clips=len(train_sha),source_views=len(src),condition_view_inputs=len(by),admitted_sampled_frames=len(accepted),uncertain_frames=len(uncertain),admission_categories=dict(collections.Counter(r['category'] for r in accepted.values())),actually_changed_unique_frames=len(unique_changed),label_value_changed_unique_frames=len(unique_y_changed),newly_direct_supervised_unique_frames=len(unique_direct_added),direct_supervision_frame_views=dict(condition_counts),model_runs=len(changes),protected_pins=len(protected)),input_checks=pairchecks,checkpoint_checks=changes,split_check=dict(manifest=str(ready_path),fit_clips=len(fits),dev_clips=len(dev),fit_record_ids=sorted({r['person'] for r in fits}),dev_record_ids=sorted({r['person'] for r in dev}),trained_source_sha_overlap_dev=0,common100_fit_source_sha_overlap_dev=0,training_sources_all_fit=True,dev_features_or_scores_loaded=False,scope='Existing exposed development split; no claim of fresh independent evaluation.'),code_review=dict(word_id_model_input=False,word_id_only_provenance_and_label_lookup=True,trainable_only_role_mlp=True,same_initial_matching_seed=True,same_schedule_order_and_features=True,only_label_and_direct_mask_edition_differs=True,teacher_kl_only_outside_current_direct_mask=True,teacher_probabilities_are_not_ground_truth=True,final_epoch_no_dev_selection=True,no_new_interpolation=True,weak_pseudo_label_targets_used=False),limitations=['Old w=1 reviewed intervals are inherited, not all independently sampled single-frame RGB truth. Uncertain new rows do not create new labels but inherited labels there remain unchanged in both conditions.','76 admitted frames are 6 FIT recordings of 2 expressions; all source frames/views are correlated, not 152 independent new facts.','Common100 backbone frozen does not mean every 100-word feedback role was trained or independently validated. This continuation uses 36 labeled public recordings.','This audit verifies artifacts and code, not improved predictive accuracy or runtime behavior. No forward pass, retraining, rescore, or DEV feature read occurred.','Single-record/word-dependent role labels refine a generic family taxonomy; no word ID or word-specific threshold enters the MLP. Broad generalization still requires separate comparison.'],protected_files=protected,pins=[dict(path=str(p),sha256=sha(p)) for p in pins],original_labels_modified=False,models_modified_by_verifier=False,training_or_inference_by_verifier=False,private_used=False,first_preparation_failure_preserved=True)
    target=W/'INDEPENDENT_TRAIN_VERIFICATION.json'
    require(not target.exists(), 'Do not overwrite completed audit')
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=out['status'],counts=out['counts'],split=out['split_check'],output=str(target)),ensure_ascii=False))

if __name__=='__main__':main()
