"""Independent saved-array verification: no fitting, inference or candidate changes."""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation

U = Path(__file__).resolve().parent
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for x in iter(lambda: f.read(1024 * 1024), b''): h.update(x)
    return h.hexdigest()
def rotations(q):
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]].reshape(-1, 4))
def degrees(a, b):
    shape = np.asarray(a).shape[:-1]
    return np.rad2deg((rotations(a).inv() * rotations(b)).magnitude()).reshape(shape)
def world_points(a, key):
    return a[key] @ a['rig_world'][:3, :3].T + a['rig_world'][:3, 3]

def main():
    checks = []
    def check(name, ok, measured=None):
        checks.append(dict(name=name, passed=bool(ok), measured=measured))
    frozen = json.loads((U / 'FROZEN.json').read_text('utf-8'))
    auth = json.loads((U / 'AUTHORIZATION.json').read_text('utf-8'))
    for p, h in {**auth['protected'], **frozen['pins']}.items():
        check('protected_sha:' + str(p), sha(p) == h)
    a = np.load(U / 'baseline_poses.npz', allow_pickle=False)
    b = np.load(U / 'candidate_poses.npz', allow_pickle=False)
    names = a['names'].tolist()
    ni = a['native_indices']
    wrists = [names.index('mixamorig1:' + side + 'Hand') for side in ['Left', 'Right']]
    others = [i for i in range(len(names)) if i not in wrists]
    fingers = [i for i, n in enumerate(names) if 'Hand' in n and i not in wrists]
    check('same_65_bones', len(names) == 65 and np.array_equal(a['names'], b['names']))
    check('same_505_quarter_samples', np.array_equal(a['source_frames'], b['source_frames']) and np.array_equal(a['source_frames'], np.arange(505) / 4))
    check('127_native_times', np.array_equal(ni, b['native_indices']) and np.array_equal(a['source_frames'][ni], np.arange(127)))
    check('rig_transform_unchanged', np.array_equal(a['rig_world'], b['rig_world']))
    for label, x in [('before', a), ('after', b)]:
        for key in ['head', 'tail', 'local_quaternion', 'global_quaternion', 'matrix', 'rig_world']:
            check(label + '_finite_' + key, np.isfinite(x[key]).all())
        m = x['matrix'][..., :3, :3]
        orth = np.max(np.abs(m @ np.swapaxes(m, -1, -2) - np.eye(3)))
        det = np.max(np.abs(np.linalg.det(m) - 1))
        check(label + '_proper_rotations', orth < 1e-4 and det < 1e-4, dict(max_orth_error=float(orth), max_det_error=float(det)))
    local_delta = degrees(a['local_quaternion'][ni], b['local_quaternion'][ni])
    global_delta = degrees(a['global_quaternion'][ni], b['global_quaternion'][ni])
    non_wrist = float(local_delta[:, others].max())
    wrist = float(local_delta[:, wrists].max())
    check('native_nonwrist_local_rotation_unchanged', non_wrist < 1e-4, non_wrist)
    check('native_wrist_max_change_within_2deg', wrist <= 2.0001, wrist)
    check('native_endpoint_rotations_unchanged', float(local_delta[[0, -1]].max()) < 1e-4, float(local_delta[[0, -1]].max()))
    head_a, head_b = world_points(a, 'head'), world_points(b, 'head')
    delta_wrist = np.linalg.norm(head_a[ni][:, wrists] - head_b[ni][:, wrists], axis=-1)
    check('native_wrist_positions_unchanged', float(delta_wrist.max()) < 1e-5, float(delta_wrist.max()))
    lengths_a = np.linalg.norm(a['tail'] - a['head'], axis=-1)
    lengths_b = np.linalg.norm(b['tail'] - b['head'], axis=-1)
    check('all_sample_bone_lengths_unchanged', float(np.abs(lengths_a - lengths_b).max()) < 1e-5, float(np.abs(lengths_a - lengths_b).max()))
    # An identical wrist path preserves its extrema and left/right ordering;
    # this is a transport check, not a human judgement of KSL motion.
    za = head_a[ni, wrists[0], 2] - head_a[ni, wrists[1], 2]
    zb = head_b[ni, wrists[0], 2] - head_b[ni, wrists[1], 2]
    mask = np.abs(za) > 1e-5
    check('native_two_wrist_vertical_order', np.array_equal(np.sign(za[mask]), np.sign(zb[mask])))
    report = {'checks': checks, 'native_max_finger_local_delta_deg': float(local_delta[:, fingers].max()),
              'native_max_finger_world_delta_deg': float(global_delta[:, fingers].max()),
              'wrist_windows': {}, 'fractional_rotation_steps': {}}
    for side, idx in zip(['left', 'right'], wrists):
        windows = {}
        for window, (start, end) in frozen['metrics']['windows'].items():
            vals = []
            for x in [a, b]:
                q = x['local_quaternion'][ni[start:end + 1], idx]
                r = rotations(q)
                v = (r[0].inv() * r).as_rotvec()
                acceleration = np.rad2deg(np.linalg.norm(np.diff(v, n=2, axis=0), axis=-1))
                vals.append(dict(mean_step_deg=float(degrees(q[:-1], q[1:]).mean()),
                                 local_angular_roughness_rms=float(np.sqrt(np.mean(acceleration ** 2)))))
            windows[window] = dict(before=vals[0], after=vals[1],
                                  roughness_reduction_pct=100 * (1 - vals[1]['local_angular_roughness_rms'] / vals[0]['local_angular_roughness_rms']))
        report['wrist_windows'][side] = windows
    # Quaternion sign changes themselves are not pose changes. Compare rotations.
    for name, idxs in [('all_bones', list(range(65))), ('wrists', wrists), ('finger_bones', fingers)]:
        vals = []
        for x in [a, b]:
            steps = degrees(x['global_quaternion'][:-1, idxs], x['global_quaternion'][1:, idxs])
            vals.append(dict(max_deg=float(steps.max()), p99_deg=float(np.percentile(steps, 99)), over_15deg=int((steps > 15).sum()), over_45deg=int((steps > 45).sum())))
        report['fractional_rotation_steps'][name] = dict(before=vals[0], after=vals[1])
    # Inter-hand tip proximity is descriptive: nearby tips are NOT ground-truth contact.
    tips = [[names.index('mixamorig1:' + side + 'Hand' + finger + '4') for finger in ['Thumb','Index','Middle','Ring','Pinky']] for side in ['Left','Right']]
    prox = []
    for h in [head_a[ni], head_b[ni]]:
        d = np.linalg.norm(h[:, tips[0], None] - h[:, None, tips[1]], axis=-1).min(axis=(1, 2))
        prox.append(dict(min_distance_world=float(d.min()), argmin_source_frame=int(d.argmin()), per_frame=d.tolist()))
    report['tip_proximity_not_contact'] = dict(before=prox[0], after=prox[1], max_change_world=float(np.max(np.abs(np.array(prox[0]['per_frame']) - prox[1]['per_frame']))))
    report.update(status='PASS_SAVED_ARRAY_PROTECTION_AND_METRICS' if all(c['passed'] for c in checks) else 'FAIL',
                  method='root SciPy rotation composition independent of implementation angle calculation; saved data only',
                  anatomical_or_ksl_accuracy_validated=False,
                  pins={p.name: sha(p) for p in [U/'FROZEN.json', U/'baseline_poses.npz', U/'candidate_poses.npz', U/'build_candidate.py']})
    (U / 'ROOT_ARRAY_VERIFIED.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', 'utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['tip_proximity_not_contact', 'checks', 'pins']}, ensure_ascii=False, indent=2))
    print('checks', len(checks), 'failed', [c for c in checks if not c['passed']])
    assert all(c['passed'] for c in checks)

if __name__ == '__main__': main()
