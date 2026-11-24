"""Synthetic adapter contract tests. No video/model/detector from a dataset."""
import hashlib
import json
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import adapter as a


def run():
    checks = []

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)

    def rejected(name, fn):
        try:
            fn()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    stats = dict(pose_mean=[2.]*69, pose_std=[2.]*69,
                 rgb_mean=[3.]*768, rgb_std=[4.]*768)
    mask = np.ones((18, 69), bool); x = np.full((18, 69), 6., np.float32)
    mask[3, 5] = False; x[3, 5] = np.nan
    data = dict(x=x, mask=mask, rgb=np.full((18, 768), 11., np.float32))
    xx, zz = a.normalize(data, stats)
    check('normalization shapes and float32', xx.shape == (18, 138) and zz.shape == (18, 768)
          and xx.dtype == zz.dtype == np.float32)
    check('saved means/stds used', xx[0, 0] == 2 and zz[0, 0] == 2)
    check('masked nan zero only under false mask', xx[3, 5] == xx[3, 69+5] == 0)
    check('support retained elsewhere', xx[0, 69:].all())
    other = dict(stats, pose_mean=[4.]*69, rgb_mean=[7.]*768)
    xx2, zz2 = a.normalize(data, other)
    check('different old/new normalization not conflated', xx2[0, 0] == 1 and zz2[0, 0] == 1)
    rejected('negative saved standard deviation rejected',
             lambda: a.normalize(data, dict(stats, pose_std=[-1.]*69)))
    rejected('nonfinite supported source rejected',
             lambda: a.normalize(dict(data, mask=np.ones((18, 69), bool)), stats))
    rejected('integer masks rejected', lambda: a.normalize(dict(data, mask=mask.astype(int)), stats))
    rejected('nonfinite RGB rejected', lambda: a.normalize(dict(data, rgb=np.full((18, 768), np.nan)), stats))
    check('normalization does not edit raw data', np.isnan(data['x'][3, 5]) and not mask[3, 5])
    check('13frame interior only', a.window_centers(np.arange(18)).tolist() == list(range(6, 12)))
    check('short video produces no padded centers', len(a.window_centers(np.arange(12))) == 0)
    check('one complete13frame window', a.window_centers(np.arange(13)).tolist() == [6])
    ids = np.arange(30); ids[15:] += 1
    check('windows cannot bridge dropped source frame', a.window_centers(ids).tolist() == [6, 7, 8, 21, 22, 23])
    rejected('nonmonotonic source rejected', lambda: a.window_centers(np.array([0, 2, 1])))

    observer = a._module('_live_selftest_feature_schema', a.OPS/'word100_guide_video_feedback_20261028/motion_observer.py')
    xy = np.full((18, 75, 2), np.nan, np.float32)
    ob = dict(xy_px=xy, observed=np.zeros((18, 75), bool), usable=np.zeros((18, 75), bool),
              frame_id=np.arange(18), timestamp_us=np.arange(18)*33333)
    features = observer.extract_features(ob)
    names = list(features['feature_names'])
    check('legacy geometry schema69', len(names) == 69 and features['mask'].shape == (18, 69))
    check('no joints produces no geometry support', not features['mask'].any())
    mm = np.ones((2, 69), bool)
    fingers = [i for i, n in enumerate(names) if n.startswith('R_') and '_angle' in n and 'elbow' not in n]
    mm[0, fingers[:4]] = False; mm[1, fingers[:5]] = False
    sup = a.input_support(mm, names)
    check('10of14 legacy support threshold', sup[0, 0] and not sup[1, 0])
    check('left shape independent of right missing', sup[:, 2].all())
    mm[0, names.index('L_palm_center_x')] = False
    sup = a.input_support(mm, names)
    check('both palm coordinates gate both relation proxies', not sup[0, 5] and not sup[0, 6])
    check('palm relation missing does not gate shape proxy', sup[0, 2])
    rejected('nonboolean support rejected', lambda: a.input_support(mm.astype(int), names))
    ob['xy_px'][:] = [5, 5]; ob['observed'][:] = True; ob['usable'][:] = True
    ob['xy_px'][0, 0] = [10, 10]
    roi = a.roi_arrays(ob, [0, 0, 10, 10])
    check('strict ROI upper edge excluded', not roi['usable'][0, 0])
    check('source observations remain intact', ob['usable'].all() and roi['observed'].all())
    check('ROI interior retained', roi['usable'][1:].all())
    check('ROI preserves source frames/PTS', np.array_equal(roi['frame_id'], ob['frame_id'])
          and np.array_equal(roi['timestamp_us'], ob['timestamp_us']))

    # Only generated pixels enter a fake OpenVINO callable. The production
    # encoder code is exercised, but no detector/backbone/checkpoint is run.
    class Frame:
        def __init__(self, p):
            self.pts = p
        def to_ndarray(self, format):
            return np.full((12, 16, 3), 200, np.uint8)
    pts = np.arange(18)*33333
    frames = [Frame(int(p+900000)) for p in pts]
    class Container:
        def __init__(self):
            self.streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 1000000))])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def demux(self, stream): return [SimpleNamespace(decode=lambda: iter(frames))]
    fake_av = SimpleNamespace(open=lambda path: Container())
    seen = []
    def compiled(inputs):
        seen.append(inputs[0].copy())
        return [np.full((1, 768), len(seen), np.float32)]
    with patch.dict('sys.modules', {'av': fake_av}):
        rgb, wf, starts, mids = a._encode('synthetic_not_a_file', [0, 0, 16, 6], np.arange(18), pts, compiled, None)
        check('RGB whole timeline and16frame windows retained', rgb.shape == (18, 768)
              and wf.shape == (2, 768) and starts.tolist() == [0, 2])
        check('exact PTS window centers', np.array_equal(mids, [(pts[0]+pts[15])/2, (pts[2]+pts[17])/2]))
        check('feature interpolation endpoints explicit', rgb[0, 0] == 1 and rgb[-1, 0] == 2)
        check('backbone layout matches official route', all(z.shape == (1, 3, 16, 224, 224) for z in seen))
        check('lower RGB replaced before encoder', float(seen[0][0, :, :, -1, :].max()) < .1)
        rejected('RGB body timestamp mismatch rejected',
                 lambda: a._encode('synthetic_not_a_file', [0, 0, 16, 6], np.arange(18), pts+1, compiled, None))
        limits = a._bounded_video_preflight('synthetic_not_a_file')
        check('actual limit uses first PTS even when nonzero', limits['maximum_elapsed_us'] == pts[-1]
              and limits['first_source_pts'] == 900000 and limits['decoded_frames'] == 18)
        frames = [Frame(900000), Frame(20900001)]
        rejected('actual elapsed over20s rejected before detectors',
                 lambda: a._bounded_video_preflight('synthetic_not_a_file'))
        frames = [Frame(900000+i) for i in range(1201)]
        rejected('1201 actual decoded frames rejected', lambda: a._bounded_video_preflight('synthetic_not_a_file'))
        frames = [Frame(900000+i) for i in range(1200)]
        check('1200 decoded frames allowed', a._bounded_video_preflight('synthetic_not_a_file')['decoded_frames'] == 1200)
    events = []; a._notify(events.append, 'test', frames=18)
    check('progress structured', events == [dict(stage='test', frames=18)])
    # Cache/warmup exercises use parameter-free synthetic heads and a fake RGB
    # engine. No model asset, checkpoint, actual detector or video is opened.
    import torch
    class Old(torch.nn.Module):
        def forward(self, x, z, valid):
            return torch.zeros(len(x), x.shape[1], 10)
    class New(torch.nn.Module):
        def forward(self, x, z):
            return torch.zeros(len(x), 4)
    old, new = Old().eval(), New().eval()
    calls = dict(compile=0, load=0, rgb=0)
    runtime = dict(old=old, new=new, old_stats=stats, new_stats=stats, old_columns=[0, 1, 2])
    def fake_compiled(inputs):
        calls['rgb'] += 1
        check('warmup only fixed synthetic16frame shape', inputs[0].shape == (1, 3, 16, 224, 224)
              and not inputs[0].any())
        return [np.zeros((1, 768), np.float32)]
    def compile_once():
        calls['compile'] += 1
        return fake_compiled, dict(actual=['SYNTHETIC'])
    def load_once():
        calls['load'] += 1
        return runtime
    a.warmup.cache_clear(); a._compile_signrep.cache_clear(); a._head_runtime.cache_clear()
    with patch.object(a, '_compile_signrep_uncached', compile_once), patch.object(a, '_load_head_models', load_once):
        first = a.warmup(); again = a.warmup()
        check('one startup compile/load/warm per process', calls == dict(compile=1, load=1, rgb=1)
              and first is again)
        model_data = dict(data, frame_id=np.arange(len(data['x'])))
        p1, c1 = a._heads(model_data); p2, c2 = a._heads(model_data)
        check('head resources reused but new inference result arrays', calls['load'] == 1
              and p1 is not p2 and np.array_equal(p1, p2, equal_nan=True) and np.array_equal(c1, c2))
        check('cached heads stay eval and untrainable', not old.training and not new.training
              and not any(p.requires_grad for p in list(old.parameters())+list(new.parameters())))
        new.train()
        try:
            a._heads(model_data)
        except RuntimeError:
            checks.append('mutated shared model rejected')
        else:
            raise AssertionError('mutated shared model accepted')
        new.eval()
    a.warmup.cache_clear(); a._compile_signrep.cache_clear(); a._head_runtime.cache_clear()
    attempted = []
    def fail_first():
        attempted.append(1)
        if len(attempted) == 1:
            raise RuntimeError('synthetic compile failure')
        return fake_compiled, {}
    with patch.object(a, '_compile_signrep_uncached', fail_first):
        try:
            a._compile_signrep()
        except RuntimeError:
            pass
        a._compile_signrep()
        check('failed initialization is not cached', len(attempted) == 2)
    a._compile_signrep.cache_clear()
    return dict(status='PASS', checks=len(checks), check_names=checks,
                adapter_sha256=a.sha(Path(a.__file__)), selftest_sha256=a.sha(Path(__file__)),
                public_or_user_video_read=False, model_or_detector_inference=0,
                old_predictions_recomputed=0, training=0,
                scope='Pure synthetic interface/normalization/mask/time/encoder layout; actual raw-video roundtrip still required')


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
