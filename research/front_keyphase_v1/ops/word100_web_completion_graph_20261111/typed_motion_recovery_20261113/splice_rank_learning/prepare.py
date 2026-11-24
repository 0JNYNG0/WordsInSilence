"""Freeze then prepare FIT-only splice inputs. No optimizer or train entrypoint."""
from pathlib import Path
import collections, hashlib, importlib.util, json, sys, time, traceback
import numpy as np
sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
H = U.parent; G = H.parent; O = G.parent; W = G / 'candidate_description_supervision'
L = W / 'local_rank_learning'; K = W / 'source_rank_learning'
sys.path.insert(0, str(W))
from splice_core import splice_timeline, extent_targets, select_bag

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)
def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def check_pins(pins):
    for p, h in pins.items(): assert sha(p) == h, p

def libraries():
    import local_rank_core as core
    role = load('_splice_role', O / 'word100_candidate_role_learning_20261111/role_core.py')
    sys.path.insert(0, str(O / 'word100_observation_guard_20261108'))
    guard = load('_splice_guard', O / 'word100_observation_guard_20261108/guard.py')
    adapter = load('_splice_adapter', O / 'word100_rtmw_shared_input_20261108/adapter.py')
    return core, role, guard, adapter

def freeze():
    assert not (U / 'FROZEN.json').exists(), 'Frozen contract is immutable'
    core, role, guard, adapter = libraries()
    design = read(H / 'body_action_binding_review/SPLICE_REPAIR_DESIGN.json')['proposed_sources']
    ids = {r['clip_id'] for r in design}
    assert len(ids) == 6 and {r['person'] for r in design} == set(range(2, 8))
    local = read(L / 'INPUTS_COMPLETE.json'); old = read(K / 'INPUTS_COMPLETE.json')
    replay = [r for r in local['bags'] if r['split'] == 'train']
    sequences = [r for r in old['sequences'] if r['clip_id'] in ids and r['split'] == 'train']
    ready = {r['clip_id']: r for r in read(O / 'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role'] == 'fit'}
    crops = {r['clip_id']: r for p in [core.R / 'CROP_INPUTS_COMPLETE.json', core.D / 'CROP_INPUTS_COMPLETE.json'] for r in read(p)['rows']}
    inputs = {cid: dict(source=ready[cid], crop=crops[cid]) for cid in sorted(ids)}
    assert len(sequences) == 36 and len(replay) == 178
    pins = {}
    def pin(p, expected=None):
        p = str(Path(p).resolve()); h = sha(p)
        assert expected is None or h == expected, p
        pins[p] = h
    for p in [Path(__file__), U / 'splice_core.py', U / 'verify.py',
              H / 'body_action_binding_review/SPLICE_REPAIR_DESIGN.json',
              L / 'INPUTS_COMPLETE.json', L / 'FROZEN.json', L / 'TRAINING_COMPLETE.json',
              L / 'evaluation/FROZEN.json', W / 'evaluate_local_rank.py', W / 'train_local_rank.py',
              K / 'INPUTS_COMPLETE.json', W / 'prepare_rank_learning.py',
              O / 'word100_rtmw_joint_learning_20261108/feedback_viewfix/FEEDBACK_NORMALIZATION.json',
              O / 'word100_broad_factorized_learning_20261110/NORMALIZATION.json',
              O / 'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt',
              role.J / 'runs/42/model.pt']:
        pin(p)
    for r in design: pin(r['annotation_path'])
    for r in replay + sequences: pin(r['path'], r['sha256'])
    for r in replay: pin(r['original_path'], r['original_sha256'])
    for r in inputs.values():
        s = r['source']; pin(s['path'], s['sha256']); pin(s['channel'], s['channel_sha256'])
        pin(r['crop']['path'], r['crop']['sha256'])
    priors = [r for r in read(L / 'TRAINING_COMPLETE.json')['rows'] if r['condition'] == 'local_full']
    assert {r['seed'] for r in priors} == {42, 43, 44}
    for r in priors: pin(r['model'], r['model_sha256'])
    # Imported repository Python dependencies are also frozen, without invoking models.
    for module in list(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p and Path(p).suffix == '.py' and Path(p).is_file() and Path(p).resolve().is_relative_to(O.parents[2]): pin(p)
    counts = collections.Counter(r['family'] for r in replay)
    cfg = dict(status='INPUT_PREPARATION_AUTHORIZED_TRAINING_NOT_AUTHORIZED', at=time.time(),
        designs=design, sequences=sequences, inputs=inputs, replay=replay, prior_models=priors,
        original_fit_bag_counts=dict(counts), original_fit_positive_counts={f: sum(r['positive'] for r in replay if r['family'] == f) for f in counts},
        source_count=6, old_replay_sources=24, new_family_scope=[2, 3], family_order=[2, 3, 4, 5, 7],
        views=['full', 'upper', 'actual_crop'], reverse=[False, True],
        enumerator='All inclusive [a,b] with b>=a+3; exactly old enumerator. Eligibility endpoints and >=80% inherited valid body.',
        timing='Preserve each source PTS in provenance. Preserve internal block intervals; each join uses median positive source interval. Query starts at0. Whole reverse reverses actual intervals. Rebuild channel279 seconds.',
        positive='Entire mapped source span contiguous in baseline orientation, satisfies all unchanged old extent bands/witnesses/exclusion/eligibility.',
        negative='Eligible candidate crossing explicit nonadjacent source join; ONLY contiguous-single-occurrence extent rank target. No action absence, handshape, direction or KSL correctness target.',
        unknown='Other candidates excluded from new rank loss. Partial retained as audit/progression outside rank. No full positive => hold entire bag; never fabricate it.',
        sampling=dict(seed=19, max_negatives=512, method='Round-robin equal duration strata; only lengths with eligible contained audit partner. All full positives. Partner remains unlabelled unless inherited positive.'),
        encoder='Unchanged frozen common100 base recomputed separately inside every candidate from permuted raw canonical x, relation observations and reconstructed dt. No old hidden concatenation.',
        unchanged='All 178 original FIT bags replay by pinned reference. No probe arrays or DEV arrays used in preparation. No label rewrite.',
        fixed_graph=dict(runtime='Raw score under same frozen body_window presence>=0 and body visibility; no shape gate or guide order in correspondence.',
                         comparison='Saved local_full baseline and frozen82 scenarios AFTER parent authorization. Prior exposed DEV only regression, never train/tune/select.',
                         qualifiers='Family 2/3 may route new rank only after review; families4/5/7 retain old outputs byte-identically. semantic_completion remains null. G2 must preserve normal/speed/partial/shape/direction/cross/order/omission/static; no final8/web switch.',
                         unsupported='Seven existing hand requirements do not cover thumb extension, depth/contact or all four-word semantics.'),
        proposed_training=dict(authorized=False, initialization='Copy each matching original local_full seed; keep original normalization unchanged',
            seeds=[42,43,44], steps=1200, lr=.001, weight_decay=.001, optimizer='AdamW new optimizer state',
            loss='Old listwise logsumexp(all)-logsumexp(full positives). Same all positives+56 sampled negatives. Gradient clip2.',
            sampling='Each step2 original bags per family. Families2/3 pair each original source/view/reverse with its new bag when eligible; old/new loss half each within that source/family, then average five family losses. Missing full-positive new bag => original loss only, no replacement or forced target.',
            protections='All five original family bags replay. Only family2/3 new output eligible at runtime; other family outputs always immutable original. No extra loss on partial or progress.',
            selection='Final1200-step checkpoint only, no sweep, no seed selection, no DEV tuning. Review FIT input contract before any training.'),
        pins=pins)
    assert read(L / 'FROZEN.json')['steps'] == cfg['proposed_training']['steps']
    write(U / 'FROZEN.json', cfg)
    print('FROZEN', len(pins), 'pins; replay', dict(counts), 'sequences', len(sequences), flush=True)

def prepare():
    assert not (U / 'INPUTS_COMPLETE.json').exists()
    assert not (U / 'sequences').exists(), 'No overwrite/resume of partially generated preparation'
    cfg = read(U / 'FROZEN.json'); check_pins(cfg['pins'])
    core, role, guard, adapter = libraries()
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    jn = role.frozen_model(); base = core.encoder()
    assert not any(p.requires_grad for p in base.parameters())
    rng = np.random.default_rng(cfg['sampling']['seed'])
    designs = {r['clip_id']: r for r in cfg['designs']}
    cache = {}; output = []; seqout = []; debug_summary = []; assertions = []
    for directory in ['sequences', 'targets', 'bags']: (U / directory).mkdir()
    for si, row in enumerate(cfg['sequences']):
        cid, view, reverse = row['clip_id'], row['view'], row['reverse']
        if (cid, view) not in cache:
            rr = cfg['inputs'][cid]['source']; z = np.load(rr['path']); ch = np.load(rr['channel']); ts = z['timestamp_us']
            if view == 'actual_crop':
                cr = cfg['inputs'][cid]['crop']; a = np.load(cr['path'])
                assert np.array_equal(ts, a['timestamp_us'])
                clean = guard.quarantine(a['xy_px'], a['score'], ts, 640, 480)
                x = adapter.convert(clean['xy_px'], a['score'], ts, a['frame_id'], 640, 480)['tcn280']
            else: x = ch['full'] if view == 'full' else ch['feedback'] if 'feedback' in ch.files else ch['upper_mask']
            cache[cid, view] = x.copy(), ts.copy()
        x, ts = cache[cid, view]
        query = splice_timeline(x, ts, designs[cid]['blocks_in_query_order'], reverse)
        qs = query.pop('join_interval_us')
        name = f'{cid}_{view}_{int(reverse)}'
        qp = U / 'sequences' / f'{name}.npz'
        np.savez_compressed(qp, **query)
        source = np.load(row['path']); m = query['baseline_index']
        bags = []
        for f in cfg['new_family_scope']:
            j = cfg['family_order'].index(f)
            targets = extent_targets(source, m, j)
            selection = select_bag(targets, rng)
            # Recomputing original target contract is algebra only, no old model rerun.
            old = extent_targets(source, np.arange(len(m)), j)
            prior = next(r for r in cfg['replay'] if (r['clip_id'],r['view'],r['reverse'],r['family']) == (cid,view,reverse,f))
            old_bag = np.load(prior['path'])
            old_set = {tuple(s) for s in old['spans'][old['positive']]}
            assert old_set == {tuple(s) for s in old_bag['spans'][old_bag['y']]}
            selected = selection['selected']; sp = targets['spans'][selected]
            debug = U / 'targets' / f'{name}_{f}.npz'
            np.savez_compressed(debug, **targets, selected=selected, negative_partners=selection['negative_partners'])
            counts = {k: int(targets[k].sum()) for k in ['eligible','crossing','positive','negative','rank_mask','partial_audit']}
            r = dict(clip_id=cid, person=row['person'], split='train', view=view, reverse=reverse, family=f,
                source_sha256=row['source_sha256'], source_sequence=row['path'], source_sequence_sha256=row['sha256'],
                query=str(qp), query_sha256=sha(qp), target_debug=str(debug), target_debug_sha256=sha(debug),
                original_replay_bag=prior['path'], original_positive_count=prior['positive'],
                enumerated_candidates=len(targets['spans']), counts=counts,
                unknown_rank_candidates=int((~targets['rank_mask']).sum()),
                positive=selection['positive_count'], negative=selection['negative_count'], skipped=selection['skipped'],
                unmatched_duration_negative_count=selection['unmatched_duration_negative_count'])
            bags.append((r, targets, selection)); debug_summary.append(r)
        union = sorted({tuple(s) for r,t,se in bags if not r['skipped'] for s in t['spans'][se['selected']]})
        features = np.empty((0,1160),np.float32)
        if union:
            q = role.old.relation_features(query['x'])
            qn = (q - jn.qm.numpy()) / jn.qs.numpy()
            qn = np.concatenate([qn[...,:4]*q[...,8,None], qn[...,4:8]*q[...,9,None], q[...,8:]], -1).reshape(-1,100)
            pose = core.normalized_pose(query['x']); spans = np.asarray(union,np.int32)
            features = np.concatenate([core.encode_windows(base,pose,qn,spans[a:a+128]) for a in range(0,len(spans),128)])
            assert features.shape == (len(union),1160) and np.isfinite(features).all()
        lookup = {v:i for i,v in enumerate(union)}
        for r,t,se in bags:
            if not r['skipped']:
                ix = se['selected']; sp = t['spans'][ix]
                indices = [lookup[tuple(v)] for v in sp]
                path = U / 'bags' / f'{name}_{r["family"]}.npz'
                np.savez_compressed(path, local_full=features[indices], y=t['positive'][ix], rank_mask=t['rank_mask'][ix],
                                    spans=sp, crossing=t['crossing'][ix], target_grid_index=ix)
                r.update(path=str(path),sha256=sha(path))
            output.append(r)
        seqout.append(dict(row, path=str(qp), sha256=sha(qp), source_sequence=row['path'],
                           source_sequence_sha256=row['sha256'], join_interval_us=qs,
                           joins=query['join_right'].tolist(), encoded_candidates=len(union)))
        write(U / 'PREPARATION_PROGRESS.json',dict(completed=si+1,total=len(cfg['sequences']),new_bags=len(output)))
        print('splice inputs',si+1,len(cfg['sequences']),'candidates',len(union),'bags',[(r['family'],r['positive'],r['negative'],r['skipped'])for r,_,_ in bags],flush=True)
    check_pins(cfg['pins'])
    counts = collections.Counter(r['family'] for r in output if not r['skipped'])
    result = dict(status='INPUTS_READY_NOT_TRAINED',at=time.time(),frozen_sha256=sha(U/'FROZEN.json'),
        sequences=seqout,bags=output,replay=cfg['replay'],fit_only=True,
        new_bag_counts=dict(counts),held_bags=[r for r in output if r['skipped']],
        source_count=6,original_fit_replay_bags=len(cfg['replay']),
        no_probe_or_dev_arrays=True,encoder_frozen=True,no_concatenated_hidden=True,
        no_optimizer_or_training=True,no_new_label_claim=True,source_pins_unchanged=True)
    write(U/'INPUTS_COMPLETE.json',result)
    print('INPUTS READY; NO TRAINING',dict(counts),flush=True)

if __name__ == '__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1] in ['freeze','prepare']
        globals()[sys.argv[1]]()
    except BaseException:
        (U/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
