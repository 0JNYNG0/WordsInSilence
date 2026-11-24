"""Prepare deterministic articulation descriptors; no model inference or training."""
from hand_core import *
from articulation_core import articulation24, FEATURE_NAMES, EPSILON
import copy


def main():
    assert not (U/'FROZEN.json').exists() and not (U/'INPUTS_COMPLETE.json').exists()
    meta = read(PARENT/'INPUTS_COMPLETE.json')
    cfg = read(PARENT/'FROZEN.json')
    tests = read(U/'PURE_TESTS.json'); assert tests['pass_'] and tests['tests'] == 10
    for p, h in tests['pins'].items(): assert sha(p) == h, p
    for p, h in cfg['pins'].items(): assert sha(p) == h, p
    completed = read(PARENT/'TRAINING_COMPLETE.json'); assert len(completed['runs']) == 18
    parents = {}; parent_pins = {}
    for held in cfg['folds']:
        parents[str(held)] = {}
        for seed in cfg['seeds']:
            cp = PARENT/'folds'/str(held)/str(seed)/'COMPLETE.json'
            rec = read(cp); mp = Path(rec['model'])
            assert sha(mp) == rec['model_sha256']
            assert rec['heldperson'] == held and rec['seed'] == seed
            assert held not in rec['split']['trainpeople']
            parents[str(held)][str(seed)] = rec['model_sha256']
            parent_pins[str(cp)] = sha(cp); parent_pins[str(mp)] = sha(mp)
    paths = [Path(__file__), U/'hand_core.py', U/'articulation_core.py', U/'train.py', U/'test_articulation.py', U/'PURE_TESTS.json',
             PARENT/'INPUTS_COMPLETE.json', PARENT/'FROZEN.json', PARENT/'TRAINING_COMPLETE.json',
             PARENT/'INDEPENDENT_SAVED_REVIEW.json', PARENT/'PARTIAL_CAUSE_REVIEW.json', ORIGINAL/'hand_core.py']
    frozen = copy.deepcopy(cfg)
    frozen.update(at=stamp(), scope='Isolated geometry residual fine-tuning diagnostic, not allFIT deployment or existing-model replacement.',
        model='Original R hand63 spatial Linear63to48 + zero Linear24to48 projected-articulation residual -> original GELU/Conv/GRU and original heads.',
        steps=300, lr=.0003,
        initialization='Matching R heldperson/seed weights and coordinate-normalization buffers copied byte-for-byte as tensors. New residual weight is zero and has no bias. All parameters then fine-tuned.',
        input='First63 values exactly R canonical input. Last24: index/middle/ring/little x PIP cosine,DIP cosine,tip-MCP/three-bone-chain ratio, then12availability. No thumb descriptor added.',
        feature_geometry='Projected2D scalars, invariant to in-plane rotation, scale and reflection. Numerical denominators >1e-6 only. These are not measured3D joint angles or new semantic labels.',
        normalization='R63 mean/std unchanged from parent checkpoint. New12scalar mean/std only available values in training people, stdfloor.05. Masks mean0/std1; unavailable standardized values zero.',
        targets='All original masked attribute labels, full/partial/stable windows, reverse and constant controls unchanged. Intermediate is not a catch-all other class, partial closing is not semantic completion.',
        remedy_rationale='REAL06 start74 source-level short closing was missed by2seeds, with retained measured finger changes. Provide explicit invariant articulation to the existing learner, not a source-specific rule or label change.',
        comparison='Fixed18LOPO300step final checkpoints. Same saved item order and thresholds as R. Compare every attribute/person/seed, full/partial/stable/closed-arm/static window and canonical source case. Source06 is development-exposed; repeated overlapping windows are not independent samples.',
        release_gate='No allFIT model or integration now. Root must assess short-partial and perattribute regressions together with full/stable/closed-arm controls; do not select a seed or threshold from fold results.',
        previous_results='Read saved R predictions for paired comparison. Do not rerun parent original predictions.',
        parent_models=parents,
        pins=dict(cfg['pins'], **parent_pins, **{str(p):sha(p) for p in paths}))
    write(U/'FROZEN.json', frozen)
    rows = []; audit = []
    for r in meta['sequences']:
        assert sha(r['path']) == r['sha256']
        z = np.load(r['path'], allow_pickle=False); extra = articulation24(z['x'])
        x = np.c_[z['x'], extra].astype(np.float32)
        assert np.array_equal(x[:, :63], z['x']) and np.isfinite(x).all()
        assert ((extra[:,12:] == 0) | (extra[:,12:] == 1)).all()
        assert not extra[:,:12][extra[:,12:]==0].any()
        payload = {k:z[k].copy() for k in z.files}; payload['x'] = x
        dest = U/'sources'/Path(r['path']).name; dest.parent.mkdir(exist_ok=True)
        assert not dest.exists(); np.savez_compressed(dest, **payload)
        rows.append(dict(r, path=str(dest), sha256=sha(dest), parent_path=r['path'], parent_sha256=r['sha256']))
        audit.append(dict(clip_id=r['clip_id'],side=r['side'],person=r['person'],frames=len(x),
            available_by_feature=extra[:,12:].sum(0).astype(int).tolist(),
            unavailable_by_feature=(extra[:,12:]==0).sum(0).astype(int).tolist(),
            labels_unchanged=bool(np.array_equal(payload['mask'],z['mask']) and np.array_equal(payload['y'],z['y'])),
            original63_unchanged=True))
    new = copy.deepcopy(meta); new.update(at=stamp(),sequences=rows,parent_manifest=str(PARENT/'INPUTS_COMPLETE.json'),
        parent_manifest_sha256=sha(PARENT/'INPUTS_COMPLETE.json'),input_basis='Rcanonical63 + projected_articulation24',
        articulation_feature_names=FEATURE_NAMES,articulation_audit=audit)
    assert all(new[k] == meta[k] for k in ['labels','windows','controls','counts'])
    write(U/'INPUTS_COMPLETE.json', new)
    write(U/'PREPARATION_VERIFICATION.json', dict(at=stamp(),pass_=True,source_hand_sequences=len(rows),
        source_frames=sum(r['frames'] for r in audit),input_dimensions=87,original63_unchanged=True,
        labels_unchanged=True,windows_unchanged=True,controls_unchanged=True,folds_unchanged=True,
        feature_names=FEATURE_NAMES,numerical_epsilon=EPSILON,
        unavailable_feature_values=sum(sum(r['unavailable_by_feature']) for r in audit),
        parent_checkpoints_verified=18,model_inference=False,training_executed=False,old_models_changed=False,
        root_review_required_before_training=True,
        pins={str(p):sha(p) for p in [U/'FROZEN.json', U/'INPUTS_COMPLETE.json', U/'train.py', U/'hand_core.py', U/'articulation_core.py']}))
    print('Prepared articulation residual branch:',new['counts'],flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException: preserve_failure('PREP'); raise
