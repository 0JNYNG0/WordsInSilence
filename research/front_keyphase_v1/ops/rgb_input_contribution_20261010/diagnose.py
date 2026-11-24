"""Two frozen-model interventions. Original RGB-on scores are reused, never recomputed."""
from pathlib import Path
import hashlib
import json
import sys
import traceback
import numpy as np
import torch

U = Path(__file__).resolve().parent
B = U.parents[1]
R = B / 'runs/rgb_input_contribution_v1'
OLD = B / 'runs/rgb_tcn_development_pair_v1'
sys.path[:0] = [str(B.parent/'sign_model_pilot_v1/src'),
               str(B/'ops/rgb_pose_bridge_20261010'),
               str(B/'ops/rgb_features_and_browser_20261010')]
from pilot.model_masked_v1 import MaskedSignTCN
from rgb_bridge import PoseRGBBridge
from feature_cache import load as load_features, model_inputs


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()


def main():
    plan = read(U/'FROZEN_DIAGNOSTIC.json')
    assert all(sha(p) == h for p, h in plan['pins'].items())
    R.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    ck = torch.load(B/'runs/partial_encoder_pair_v1/checkpoint_control_final.pt',
                    map_location='cpu', weights_only=True)
    parent = MaskedSignTCN(width=ck['width'], dilations=tuple(ck['dilations']),
                          kernel_size=ck['kernel_size'], dropout=ck['dropout'],
                          n_classes=len(ck['classes']))
    parent.load_state_dict(ck['state_dict'], strict=True)
    model = PoseRGBBridge(parent)
    model.load_state_dict(torch.load(OLD/'checkpoint_B.pt', map_location='cpu',
                                     weights_only=True)['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    original = read(OLD/'EVALUATION_ROWS.json')
    assert original['classes'] == ck['classes']
    manifest = {r['sample_id']: r for r in read(B/'runs/ksl77_rgb_paired_v1/MANIFEST.json')}
    indices = [i for i, r in enumerate(original['rows']) if r['group'].startswith('ksl_')]
    assert len(indices) == len(manifest) == 36
    with np.load(OLD/'EVALUATION_SCORES.npz', allow_pickle=False) as z:
        scores = {'baseline832': z['baseline'][indices].copy(),
                  'A_pose_adapted': z['A'][indices].copy(),
                  'B_rgb_on': z['B'][indices].copy(), 'y': z['y'][indices].copy()}
    outputs = {'B_rgb_off': [], 'B_flags_only': []}
    rows = []
    with torch.inference_mode():
        for idx in indices:
            meta = original['rows'][idx]
            source = manifest[meta['id']]
            assert source['label_metadata_only'] == meta['label']
            data = load_features(source['feature_path'], source['feature_sha256'])
            v = model_inputs(data, 'B')
            assert not v['estimated'].any()
            x, pad = v['pose_x'][None], v['pad'][None]
            zero_features = torch.zeros_like(v['features'])[None]
            no_hand = torch.zeros_like(v['available'])[None]
            outputs['B_rgb_off'].append(model(x, pad, zero_features, no_hand, no_hand).numpy()[0])
            outputs['B_flags_only'].append(model(x, pad, zero_features,
                                                  v['available'][None], no_hand).numpy()[0])
            # Interventions never touch the source arrays or original pose/time axis.
            assert np.array_equal(v['pose_x'].numpy(), data['pose_x'])
            assert np.array_equal(v['timestamp_us'].numpy(), data['timestamp_us'])
            assert np.array_equal(v['frame_id'].numpy(), data['frame_id'])
            row = dict(meta, feature_path=source['feature_path'],
                       feature_sha256=source['feature_sha256'], frames=len(data['pose_x']),
                       observed_images=int(data['observed_rgb'].sum()),
                       preserved_array_sha256={k: array_sha(data[k]) for k in
                                               ('pose_x', 'timestamp_us', 'frame_id', 'observed_rgb')})
            rows.append(row)
    scores.update({k: np.stack(v) for k, v in outputs.items()})
    assert all(scores[k].shape == (36, 93) and np.isfinite(scores[k]).all()
               for k in scores if k != 'y')
    write(R/'ROWS.json', {'classes': ck['classes'], 'rows': rows})
    with (R/'SCORES.npz').open('xb') as f:
        np.savez_compressed(f, **scores)
    assert all(sha(p) == h for p, h in plan['pins'].items())
    write(R/'COMPLETION.json', {'status': 'COMPUTED_PENDING_INDEPENDENT_VERIFICATION',
          'rows': 36, 'new_conditions': 2, 'training_steps': 0,
          'rgb_on_scores_reused': True, 'source_hashes_unchanged': True,
          'plan_sha256': sha(U/'FROZEN_DIAGNOSTIC.json'),
          'implementation_sha256': sha(__file__), 'scores_sha256': sha(R/'SCORES.npz'),
          'rows_sha256': sha(R/'ROWS.json')})
    print('Saved 36 clips x 2 frozen interventions; no training.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if not (U/'FIRST_FAILURE.json').exists():
            write(U/'FIRST_FAILURE.json', {'traceback': traceback.format_exc()})
        raise
