"""Minimal checks of the NEW web connection only, on stored data. No camera is opened.

1 equivalence : ShadowLandmarker primary == operating Landmarker, shadow == frozen StartLandmarker
2 worker path : the real CameraWorker recording code, fed decoded video frames for two takes with a
                session-continuous clock and a 0.2 s gap (< MAX_GAP_S, so a leak would be live):
                default path unchanged, MediaPipe clock continuous, candidate tracks reset per take
3 orientation : no flip anywhere in the path; sidecar mirrored=False; label mapping identical
4 webcam label : raw MediaPipe handedness vs pose anatomy on the stored webcam JPEG samples
"""
import io, json, pathlib, sys, zipfile
import cv2, numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[2]
sys.path[:0] = [str(L), str(L / 'src')]
from wis_lab.capture.landmarker import Landmarker                      # noqa: E402
from wis_lab.capture.video_source import extract                       # noqa: E402
from wis_lab.core.observation import empty_frames                      # noqa: E402
from wis_lab.service import camera as cam_mod                          # noqa: E402
from wis_lab.service.shadow_assignment import ShadowLandmarker, candidate_sha256  # noqa: E402
from start_landmarker import StartLandmarker                           # noqa: E402  (path set by shadow module)
from temporal_landmarker import MAX_GAP_S                              # noqa: E402

CONTRACT = json.loads((L / 'config' / 'contract.json').read_text(encoding='utf-8'))
CACHE = L / 'cache' / '2683eadbc5f1c588'
CLIPS = ['NIKL_MAEKJU_01', 'NIKL_GAM_01', 'NIKL_JOESONGHADA_01']
SEN = L / 'runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4'
KEYS = ('xy_px', 'observed', 'usable', 'conf', 'hand_score')


def src(clip):
    return json.loads((CACHE / f'{clip}.sidecar.json').read_text(encoding='utf-8'))['source_path']


def frames_of(path):
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return [f.to_ndarray(format='rgb24') for p in c.demux(s) for f in p.decode() if f.pts is not None]


def same(a, b):
    return all(np.array_equal(np.nan_to_num(a[k]), np.nan_to_num(b[k])) for k in KEYS)


def check_equivalence():
    rows = []
    for name, path in [(c, src(c)) for c in CLIPS] + [('SEN0354', str(SEN))]:
        with Landmarker(CONTRACT) as lm:
            op = extract(path, lm, name)
        with StartLandmarker(CONTRACT) as lm:
            ca = extract(path, lm, name)
        imgs = frames_of(path)
        assert len(imgs) == op.n_frames, (name, len(imgs), op.n_frames)
        prim, shad = empty_frames(len(imgs)), empty_frames(len(imgs))
        with ShadowLandmarker(CONTRACT) as lm:
            for i, (rgb, ts) in enumerate(zip(imgs, op.timestamp_us)):
                lm.process(rgb, int(ts), prim, i, shadow_out=shad, shadow_row=i)
        o = {k: getattr(op, k) for k in KEYS}; c = {k: getattr(ca, k) for k in KEYS}
        rows.append({'clip': name, 'frames': len(imgs), 'primary_equals_operating': same(prim, o),
                     'shadow_equals_candidate': same(shad, c),
                     'shadow_differs_from_primary': not same(prim, shad)})
        print('EQ', rows[-1], flush=True)
    return rows


class LoggingShadow(ShadowLandmarker):
    """records what MediaPipe was given and the candidate track state at each frame (diagnostic)"""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.ms_log, self.tracks_before, self.resets = [], [], 0
    def next_timestamp_ms(self, timestamp_us):
        ms = super().next_timestamp_ms(timestamp_us); self.ms_log.append((int(timestamp_us // 1000), ms)); return ms
    def reset_assignment_state(self):
        self.resets += 1; super().reset_assignment_state()
    def process(self, rgb, timestamp_us, out, row, shadow_out=None, shadow_row=0):
        self.tracks_before.append(sorted(self.tracks)); return super().process(rgb, timestamp_us, out, row, shadow_out, shadow_row)


class LoggingPlain(Landmarker):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.ms_log = []
    def next_timestamp_ms(self, timestamp_us):
        ms = super().next_timestamp_ms(timestamp_us); self.ms_log.append((int(timestamp_us // 1000), ms)); return ms


def drive(worker, lm, takes, gap_s):
    """the real begin_recording / _record_frame / take code, with a session-continuous clock"""
    now_ns, fid, out = 10_000_000_000, 0, []
    for imgs in takes:
        worker.begin_recording(store_video=False)
        for rgb in imgs:
            now_ns += 33_333_333; fid += 1
            worker._record_frame(rgb, now_ns, fid, lm)
        obs, meta = worker.take()
        out.append((obs, meta))
        now_ns += int(gap_s * 1e9)
    return out


def check_worker():
    takes = [frames_of(src('NIKL_MAEKJU_01'))[:70], frames_of(src('NIKL_GAM_01'))[:70]]
    gap = 0.2
    assert gap < MAX_GAP_S
    ws = cam_mod.CameraWorker(CONTRACT, shadow_assignment=True)
    wp = cam_mod.CameraWorker(CONTRACT)
    assert wp.shadow_assignment is False
    with LoggingShadow(CONTRACT) as ls, LoggingPlain(CONTRACT) as lp:
        rs = drive(ws, ls, takes, gap)
        rp = drive(wp, lp, takes, gap)
        tracks_end_take1 = None
        n1 = len(takes[0])
        res = {
            'primary_equals_plain_worker': [same({k: getattr(a[0], k) for k in KEYS}, {k: getattr(b[0], k) for k in KEYS})
                                             and np.array_equal(a[0].timestamp_us, b[0].timestamp_us) for a, b in zip(rs, rp)],
            'plain_worker_has_no_shadow': all('_shadow_obs' not in m for _, m in rp),
            'shadow_saved_each_take': all('_shadow_obs' in m for _, m in rs),
            'stored_timestamp_restarts_at_0_each_take': [int(o.timestamp_us[0]) for o, _ in rs],
            'mediapipe_ms_strictly_increasing_session_clock': all(b[1] > a[1] for a, b in zip(ls.ms_log, ls.ms_log[1:])),
            'mediapipe_ms_never_clamped': all(req == got for req, got in ls.ms_log),
            'mediapipe_ms_identical_shadow_vs_plain': ls.ms_log == lp.ms_log,
            'mediapipe_interval_ms_across_boundary': ls.ms_log[n1][1] - ls.ms_log[n1 - 1][1],
            'candidate_resets': ls.resets,
            'tracks_live_at_end_of_take1': ls.tracks_before[n1 - 1] if n1 else None,
            'tracks_before_first_frame_of_take2_after_reset': ls.tracks_before[n1],
            'sidecar_mirrored': [o.sidecar['mirrored'] for o, _ in rs],
            'shadow_sidecar_policy': rs[0][1]['_shadow_obs'].sidecar['hand_assignment_policy'],
            'shadow_same_frames_as_primary': all(np.array_equal(o.frame_id, m['_shadow_obs'].frame_id) and
                                                 np.array_equal(o.timestamp_us, m['_shadow_obs'].timestamp_us) for o, m in rs),
        }
    # counterfactual: the same two takes WITHOUT the take-boundary reset - does state actually carry?
    class NoReset(LoggingShadow):
        def reset_assignment_state(self):
            self.resets += 1
    with NoReset(CONTRACT) as nr:
        w2 = cam_mod.CameraWorker(CONTRACT, shadow_assignment=True)
        r2 = drive(w2, nr, takes, gap)
        res['without_reset_tracks_before_take2'] = nr.tracks_before[len(takes[0])]
        res['without_reset_take2_shadow_differs'] = not same({k: getattr(rs[1][1]['_shadow_obs'], k) for k in KEYS},
                                                             {k: getattr(r2[1][1]['_shadow_obs'], k) for k in KEYS})
    print('WORKER', res, flush=True)
    return res


def check_webcam_labels():
    """raw handedness label vs pose anatomy on stored webcam JPEGs (unambiguous frames only)"""
    from mediapipe import Image, ImageFormat
    zips = sorted((L / 'runs').glob('*/trials/*.capture.zip'))
    agree = disagree = half_agree = half_disagree = frames = 0
    per = []
    for z in zips:
        with zipfile.ZipFile(z) as a, Landmarker(CONTRACT) as lm:
            man = json.loads(a.read('manifest.json'))
            for e in man['frames']:
                bgr = cv2.imdecode(np.frombuffer(a.read(e['image']), np.uint8), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
                img = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                ms = lm.next_timestamp_ms(e['timestamp_us'])
                pose = lm.pose.detect_for_video(img, ms); hands = lm.hand.detect_for_video(img, ms)
                if not pose.pose_landmarks or not hands.hand_landmarks:
                    continue
                P = pose.pose_landmarks[0]
                if min(P[15].visibility, P[16].visibility, P[11].visibility, P[12].visibility) < lm.pose_vis_thr:
                    continue
                pw = {0: np.array([P[15].x * w, P[15].y * h]), 1: np.array([P[16].x * w, P[16].y * h])}
                sh = {0: np.array([P[11].x * w, P[11].y * h]), 1: np.array([P[12].x * w, P[12].y * h])}
                sw = float(np.linalg.norm(sh[0] - sh[1])); mid = (sh[0] + sh[1]) / 2
                for i, hl in enumerate(hands.hand_landmarks):
                    wr = np.array([hl[0].x * w, hl[0].y * h])
                    d = {s: float(np.linalg.norm(wr - pw[s]) / sw) for s in (0, 1)}
                    near = min(d, key=d.get)
                    if not (d[near] <= 0.35 and d[1 - near] > 0.35):
                        continue                      # unambiguous arm link only
                    frames += 1
                    lab = 0 if hands.handedness[i][0].category_name == 'Left' else 1
                    agree += lab == near; disagree += lab != near
                    half = 0 if float((wr - mid) @ (sh[0] - mid)) > 0 else 1
                    half_agree += half == near; half_disagree += half != near
        per.append({'capture': z.relative_to(L).as_posix(), 'frames_in_zip': len(man['frames'])})
    res = {'captures': per, 'unambiguous_hand_observations': frames,
           'raw_label_agrees_with_pose_anatomy': agree, 'raw_label_disagrees': disagree,
           'body_half_agrees_with_pose_anatomy': half_agree, 'body_half_disagrees': half_disagree,
           'note': 'stored JPEG samples (<=10 fps, downscaled) of ONE earlier development session; '
                   'checks the label direction only, used for no selection'}
    print('WEBCAM', res, flush=True)
    return res


def main():
    out = {'schema': 'wis_web_path_check_v1', 'at_kst': '2026-09-26', 'camera_opened': False,
           'candidate_sha256': candidate_sha256(),
           'equivalence': check_equivalence(), 'worker': check_worker(),
           'orientation': {'service_flips_frames': False, 'evidence': 'camera.py: cv2 frame -> cvtColor(BGR2RGB) -> process; '
                           'no cv2.flip in the recording path; preview mirror is CSS only (index.html #mirror, app.js toggles a class)',
                           'label_mapping': 'candidate uses the same raw mapping as the operating fallback: Left -> 0, Right -> 1'},
           'webcam_labels': check_webcam_labels()}
    (H / 'WEB_PATH_CHECK.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
