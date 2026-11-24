"""Provided-coordinate vs MediaPipe on two UNSEEN recordings of KNOWN signers.

Members are extracted from held archives; nothing is downloaded. The model is frozen.
"""
import hashlib, json, math, pathlib, re, sys, time, zipfile
import numpy as np
import torch

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
PILOT = L.parent / 'sign_model_pilot_v1'
sys.path[:0] = [str(L), str(L / 'src'), str(PILOT / 'src')]
import tools_prep_union93_candidate as u                    # noqa: E402
from pilot.data import build_channels                       # noqa: E402
from pilot.web_input import to_native_like                  # noqa: E402

AIHUB = pathlib.Path(r'C:\Users\local-user\Downloads\수어 영상\1.Training')
RAW = H / 'raw'
FPS = 30.0
FIELDS = [('pose_keypoints_2d', 25), ('hand_left_keypoints_2d', 21), ('hand_right_keypoints_2d', 21)]
FRAME_RE = re.compile(r'_F_(\d+)_keypoints\.json$')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def decode(raw):
    p = json.loads(raw)['people']
    vals = []
    for name, n in FIELDS:
        a = np.asarray(p[name], dtype=np.float32)
        assert a.shape == (n * 3,), (name, a.shape)
        vals.append(a.reshape(n, 3))
    xyc = np.concatenate(vals, axis=0)
    assert np.isfinite(xyc).all()
    assert not np.any((xyc[:, 2] < 0) | (xyc[:, 2] > 1))
    return xyc


def provided(clip_id, kp_archive, start_s, end_s):
    """Existing rule: ceil(start*FPS) .. ceil(end*FPS)-1, every frame must be present."""
    start = math.ceil(start_s * FPS - 1e-9)
    stop = math.ceil(end_s * FPS - 1e-9)
    wanted = list(range(start, stop))
    with zipfile.ZipFile(AIHUB / kp_archive) as z:
        frames = {}
        for n in z.namelist():
            if clip_id in n:
                m = FRAME_RE.search(n)
                if m:
                    frames[int(m.group(1))] = n
        missing = [i for i in wanted if i not in frames]
        assert not missing, f'{clip_id}: {len(missing)} frames missing'
        arrs = [decode(z.read(frames[i])) for i in wanted]
    xyc = np.stack(arrs)                                   # [T, 67, 3]
    xy = xyc[:, :, :2].astype(np.float32)
    present = ((xyc[:, :, 0] != 0) | (xyc[:, :, 1] != 0)) & np.isfinite(xyc[:, :, :2]).all(-1)
    ts = (np.asarray(wanted, dtype=np.int64) * (1_000_000 // int(FPS)))
    return build_channels(xy, present, ts), {'frames': len(wanted), 'first_frame': start,
                                             'last_frame': stop - 1,
                                             'joint_present_rate': float(present.mean())}


def mediapipe(video, clip_id, start_s, end_s):
    from wis_lab.capture.landmarker import Landmarker
    from wis_lab.capture.video_source import extract
    contract = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))
    with Landmarker(contract) as lm:
        extractor_id = lm.extractor_id()
        obs = extract(str(video), lm, clip_id)
    t0 = int(obs.timestamp_us[0])
    rel = (obs.timestamp_us - t0) / 1e6
    sel = np.flatnonzero((rel >= start_s) & (rel < end_s))
    assert sel.size >= 4, f'{clip_id}: only {sel.size} frames in window'
    nat, present = to_native_like(obs.xy_px[sel], obs.usable[sel])
    return build_channels(nat, present, obs.timestamp_us[sel]), {
        'frames': int(sel.size), 'row_first': int(sel[0]), 'row_last': int(sel[-1]),
        'joint_present_rate': float(present.mean()),
        'left_hand_rate': float(obs.usable[sel, 33:54].any(axis=1).mean()),
        'right_hand_rate': float(obs.usable[sel, 54:75].any(axis=1).mean()),
        'video_frames_total': int(obs.n_frames), 'extractor_id': extractor_id}


def score(model, classes, clip, target, target_idx):
    from pilot.data import collate
    xx, pad, _ = collate([dict(clip, label=0)])
    with torch.inference_mode():
        logit = model(torch.from_numpy(xx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
    yi = classes.index(target)
    out = {}
    for scope, inds in [('full93', list(range(93))), ('project18', target_idx)]:
        order = sorted(inds, key=lambda j: -float(logit[j]))
        rival = next(j for j in order if j != yi)
        out[scope] = {'pred': classes[order[0]], 'correct': bool(order[0] == yi),
                      'true_rank': order.index(yi) + 1,
                      'margin': round(float(logit[yi] - logit[rival]), 5)}
    return out, logit


def main():
    plan = json.loads((H / 'FROZEN_PLAN.json').read_text(encoding='utf-8'))
    assert plan['status'] == 'FROZEN_BEFORE_EXECUTION'
    RAW.mkdir(exist_ok=True)
    started = time.time()

    ckpt = L / 'runs/partial_encoder_pair_v1/checkpoint_control_final.pt'
    assert sha(ckpt) == plan['model']['sha256']
    ck = torch.load(ckpt, map_location='cpu', weights_only=True)
    model, classes = u.rebuild_from_payload(ck)
    model.eval()
    torch.set_num_threads(2)
    from wis_lab.service.neural import PROJECT_18
    target_idx = [i for i, c in enumerate(classes) if c in set(PROJECT_18.values())]

    results = []
    for t in plan['targets']:
        cid = t['clip_id']
        vpath = RAW / f'{cid}.mp4'
        if not vpath.exists():
            with zipfile.ZipFile(AIHUB / t['video_archive']) as z:
                data = z.read(t['video_member'])
            vpath.write_bytes(data)
        print(f'{cid}: video {vpath.stat().st_size:,} B', flush=True)

        pa, pstat = provided(cid, t['keypoint_archive'], t['annotation_start_s'], t['annotation_end_s'])
        mb, mstat = mediapipe(vpath, cid, t['annotation_start_s'], t['annotation_end_s'])
        sa, la = score(model, classes, pa, t['label'], target_idx)
        sb, lb = score(model, classes, mb, t['label'], target_idx)
        results.append({
            'clip_id': cid, 'signer_id': t['signer_id'], 'label': t['label'],
            'window_s': [t['annotation_start_s'], t['annotation_end_s']],
            'video_sha256': sha(vpath), 'video_bytes': vpath.stat().st_size,
            'provided': {'scores': sa, 'observation': pstat,
                         'body_scale': round(float(pa['body_scale']), 4),
                         'hand_scales': {k: (None if not np.isfinite(v) else round(float(v), 4))
                                         for k, v in pa['hand_scales'].items()},
                         'hand_valid_rate': round(float(pa['hand_valid'].mean()), 4),
                         'body_valid_rate': round(float(pa['body_valid'].mean()), 4)},
            'mediapipe': {'scores': sb, 'observation': mstat,
                          'body_scale': round(float(mb['body_scale']), 4),
                          'hand_scales': {k: (None if not np.isfinite(v) else round(float(v), 4))
                                          for k, v in mb['hand_scales'].items()},
                          'hand_valid_rate': round(float(mb['hand_valid'].mean()), 4),
                          'body_valid_rate': round(float(mb['body_valid'].mean()), 4)},
            'agreement': {'full93': sa['full93']['pred'] == sb['full93']['pred'],
                          'project18': sa['project18']['pred'] == sb['project18']['pred']},
            'max_logit_difference': round(float(np.max(np.abs(la - lb))), 5)})
        r = results[-1]
        print(f"   provided : {sa['full93']['pred']:<10} rank {sa['full93']['true_rank']:>2} "
              f"margin {sa['full93']['margin']:+8.3f} | frames {pstat['frames']}")
        print(f"   mediapipe: {sb['full93']['pred']:<10} rank {sb['full93']['true_rank']:>2} "
              f"margin {sb['full93']['margin']:+8.3f} | frames {mstat['frames']}")

    out = {'schema': 'wis_provided_vs_mediapipe_known_signer_result_v1',
           'at_kst': '2026-09-25', 'framing': plan['framing'],
           'model': plan['model'], 'plan_sha256': sha(H / 'FROZEN_PLAN.json'),
           'results': results,
           'training_runs': 0, 'downloads': 0, 'model_changed': 0, 'user_data': 0,
           'reserved_signers': 0,
           'limits': plan['limits'], 'seconds': round(time.time() - started, 1)}
    with (H / 'RESULT.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('\nwritten RESULT.json', sha(H / 'RESULT.json')[:16], '|', out['seconds'], 's')


if __name__ == '__main__':
    main()
