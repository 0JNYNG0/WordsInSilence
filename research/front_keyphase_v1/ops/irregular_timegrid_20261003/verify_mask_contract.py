"""Independent contract check using synthetic coordinates; no model or camera."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

H = Path(__file__).resolve().parent
cfg = json.loads((H / 'CONFIG_LOCAL_ONLY.json').read_text(encoding='utf-8-sig'))
sys.path[:0] = [cfg['helper_dir'], cfg['pilot_src']]
import timegrid
from pilot.data import build_channels, collate

for module in (Path(timegrid.__file__).resolve(), Path(sys.modules['pilot.data'].__file__).resolve()):
    pin = next(v for k, v in cfg['pins'].items() if Path(k).resolve() == module)
    assert hashlib.sha256(module.read_bytes()).hexdigest() == pin

# Two fully observed instants, 100 ms apart. Deliberately outside frozen gap limit.
joint = np.arange(67, dtype=np.float32)
pose = np.column_stack((100 + joint * 3, 200 + joint * 2))
xy = np.stack((pose, pose + np.array([6, 3], dtype=np.float32)))
present = np.ones((2, 67), dtype=bool)
ts = np.array([0, 100000], dtype=np.int64)
restored = timegrid.restore(xy, present, ts, 'linear')
assert restored['ts'].tolist() == [0, 33333, 66667, 100000]
assert restored['present'].any(axis=1).tolist() == [True, False, False, True]
assert not restored['estimated'].any()
features = build_channels(restored['xy'], restored['present'], restored['ts'])['x']
batch, pad, _ = collate([{'x': features, 'frames': len(features), 'label': 0}])
assert np.all(features[1:3, :-1] == 0)
assert np.all(features[1:3, -1] > 0)
assert pad.tolist() == [[1.0, 1.0, 1.0, 1.0]]
out = {
    'status': 'PASS_CONTRACT_OBSERVATION_ONLY',
    'synthetic_only': True,
    'restored_timestamp_us': restored['ts'].tolist(),
    'any_observed_or_estimated_joint': restored['present'].any(axis=1).tolist(),
    'padding_mask': pad.tolist(),
    'interior_coordinates_and_joint_masks_zero': True,
    'interior_dt_positive': True,
    'finding': 'Fully unsupported interior grid frames still have padding_mask=1; joint masks and padding mask have different meanings.',
    'causal_limit': 'No inference or intervention. Does not prove how much of the recognition regression is caused by these frames, nor that changing the padding mask fixes it.',
    'training': 0, 'model_inference': 0, 'camera': 0, 'user_data': 0,
}
with (H / 'MASK_CONTRACT_CHECK.json').open('x', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False))
