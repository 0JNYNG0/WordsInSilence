"""Independent saved-raw replay and full-source first-second contact sheets."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import importlib.util
import io
import json
import math
import time
from pathlib import Path
import numpy as np
import av
from PIL import Image, ImageDraw, ImageFont
import root_audit_legacy_preflight as audit

U = audit.U
J = audit.LAB / 'ops/word100_hand_assignment_joint_20261016'
spec = importlib.util.spec_from_file_location('root_probe_frozen_assign', J / 'assignment.py')
assign = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = assign
spec.loader.exec_module(assign)


def output(path, data):
    assert path.parent == U and path.name.startswith('root_probe_audit')
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def decode_frames(video, wanted, obs):
    found = {}
    row = 0
    with av.open(str(video)) as c:
        st = c.streams.video[0]
        first = None
        last = -1
        for frame in c.decode(st):
            if frame.pts is None: continue
            if first is None: first = frame.pts
            ts = int(round(float((frame.pts - first) * st.time_base) * 1_000_000))
            if ts <= last: continue
            last = ts
            if row in wanted:
                assert ts == int(obs.timestamp_us[row])
                arr = frame.to_ndarray(format='rgb24')
                assert (arr.shape[1], arr.shape[0]) == (obs.sidecar['width'], obs.sidecar['height'])
                found[row] = Image.fromarray(arr)
            if row >= max(wanted): break
            row += 1
    assert set(found) == set(wanted)
    return found


def contact(row, obs, diag):
    target = U / f"root_probe_audit_{row['clip_id']}.png"
    frames = (0, 9, 18, 27)
    pics = decode_frames(row['source_path'], frames, obs)
    font_path = 'C:/Windows/Fonts/malgun.ttf'
    title = ImageFont.truetype(font_path, 32)
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 19)
    sheet = Image.new('RGB', (1960, 1380), '#111827')
    d = ImageDraw.Draw(sheet)
    d.text((22, 14), f"{row['label']} · {row['clip_id']} · 원본 전체 화면 / 새 joint_v3", font=title, fill='white')
    d.text((22, 60), '청록: 촬영자 왼쪽 어깨–팔꿈치–손 손목   주황: 오른쪽 연결   관측된 점만 표시 · 화면 자르기 없음', font=font, fill='#d1d5db')
    for k, f in enumerate(frames):
        ox = 20 + (k % 2) * 970
        oy = 108 + (k // 2) * 612
        image = pics[f].resize((960, 540), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        scale = 960 / obs.sidecar['width']
        for name, s, e, w in assign_side_indices():
            color = '#22d3ee' if name == 'L' else '#fb923c'
            point = {}
            for idx in (s, e, w):
                if obs.observed[f, idx] and obs.usable[f, idx] and np.isfinite(obs.xy_px[f, idx]).all():
                    point[idx] = tuple(float(v) * scale for v in obs.xy_px[f, idx])
            for a, b in ((s, e), (e, w)):
                if a in point and b in point: draw.line((point[a], point[b]), fill=color, width=4)
            for idx, xy in point.items():
                radius = 7 if idx == w else 5
                draw.ellipse((xy[0]-radius, xy[1]-radius, xy[0]+radius, xy[1]+radius), fill=color, outline='white', width=1)
                if idx == w: draw.text((xy[0]+10, xy[1]-20), name+' wrist', font=small, fill=color, stroke_width=1, stroke_fill='#111827')
        sheet.paste(image, (ox, oy))
        d.text((ox + 8, oy + 545), f"프레임 {f} · {obs.timestamp_us[f]/1e6:.2f}초 · 원본 1920×1080 전체", font=font, fill='white')
        # Explicit image-derived arm values are local geometric evidence, not a correctness score.
        sw = float(np.linalg.norm(obs.xy_px[f, 11] - obs.xy_px[f, 12]))
        w = obs.xy_px[f, 54]; e = obs.xy_px[f, 14]
        if obs.usable[f, [11,12,14,54]].all() and sw > 0:
            drop = (w[1]-e[1])/sw
            d.text((ox + 8, oy + 576), f"오른 손목−팔꿈치 높이차: 어깨너비의 {drop:.3f}배 (팔내림 요구 ≥0.20)", font=small, fill='#fbbf24')
    d.text((22, 1340), '이 화면은 초기 준비자세 보정 조건 점검이다. 수어 정답/오답 판정이나 인식률 결과가 아니다.', font=small, fill='#d1d5db')
    buf = io.BytesIO()
    sheet.save(buf, format='PNG')
    if target.exists():
        assert target.read_bytes() == buf.getvalue(), 'Existing contact sheet differs; preserve it'
    else:
        with target.open('xb') as stream: stream.write(buf.getvalue())
    return str(target)


def assign_side_indices():
    return [('L', 11, 13, 33), ('R', 12, 14, 54)]


def main():
    started = time.perf_counter()
    result_path = U / 'CALIBRATION_PROBE_RESULT.json'
    scope_path = U / 'CALIBRATION_PROBE_SCOPE.json'
    scope = audit.read(scope_path)
    result = audit.read(result_path)
    old = {r['clip_id']: r for r in audit.read(U / 'root_audit_legacy_preflight_all_RESULT.json')['results']}
    assert [r['clip_id'] for r in result['rows']] == scope['clip_ids']
    contract = audit.read(audit.LAB / 'config/contract.json')
    pins = {str(p): audit.sha(p) for p in (result_path, scope_path, U / 'PREP_CONTEXT.json', audit.SRC,
        J / 'assignment.py', J / 'SCOPE.json', J / 'POINT_CONTRACT.json', Path(__file__),
        U / 'root_audit_legacy_preflight.py', U / 'root_audit_legacy_preflight_all_RESULT.json')}
    assert pins[str(U / 'PREP_CONTEXT.json')] == scope['context_sha256']
    assert audit.sha(U / 'probe_calibration.py') == scope['code_sha256']
    rows = []
    for row in result['rows']:
        cid = row['clip_id']
        assert row['role'] == 'train_candidate'
        pins.update(row['capture']['pins'])
        assert all(audit.sha(p) == h for p, h in row['capture']['pins'].items())
        video = Path(row['source_path'])
        assert audit.sha(video) == row['source_sha256']
        pins[str(video)] = row['source_sha256']
        rec = audit.read(audit.DATA / 'records' / f'{cid}.json')
        assert rec['source_sha256'] == row['source_sha256']
        obs = audit.Observation.load(row['capture']['candidate'])
        base = audit.Observation.load(row['capture']['baseline'])
        legacy = audit.Observation.load(old[cid]['observation'])
        assert obs.sidecar['assignment_sha256'] == assign.HandAssigner.identity()
        assert obs.sidecar['hand_assignment_policy'] == 'frozen_joint_v3'
        assert np.array_equal(obs.timestamp_us, legacy.timestamp_us) and np.array_equal(obs.frame_id, legacy.frame_id)
        with np.load(row['capture']['raw'], allow_pickle=False) as z: raw = {k:z[k] for k in z.files}
        with np.load(Path(row['capture']['candidate']).parent / 'assignment.npz', allow_pickle=False) as z: sides = z['sides']
        replay = assign.HandAssigner()
        replay_sides = np.full(sides.shape, -1, np.int8)
        chosen_hands = 0
        for f in range(obs.n_frames):
            assert int(raw['timestamp_us'][f]) == int(obs.timestamp_us[f])
            n = int(raw['raw_count'][f]); width=int(raw['width'][f]); height=int(raw['height'][f])
            chosen, _ = replay.assign(raw['hand_xy_px'][f,:n], raw['raw_label'][f,:n], raw['raw_score'][f,:n],
                raw['pose_xy_px'][f], raw['pose_usable'][f], int(obs.timestamp_us[f]), width, height)
            assigned = set()
            for i, side in chosen.items():
                if side is None: continue
                replay_sides[f,i] = side
                assigned.add(side); chosen_hands += 1
                sl=slice(33+side*21,54+side*21); pts=raw['hand_xy_px'][f,i].astype(np.float32)
                np.testing.assert_array_equal(obs.xy_px[f,sl], pts)
                margin=contract['usable_rule']['bounds_margin_ratio']
                usable=np.isfinite(pts).all(1)&(pts[:,0]>=-margin*width)&(pts[:,0]<=width*(1+margin))&(pts[:,1]>=-margin*height)&(pts[:,1]<=height*(1+margin))&(raw['raw_score'][f,i]>=contract['usable_rule']['hand_score_threshold'])
                np.testing.assert_array_equal(obs.usable[f,sl], usable)
                assert obs.observed[f,sl].all()
            for side in set((0,1))-assigned:
                assert not obs.observed[f,33+side*21:54+side*21].any()
        np.testing.assert_array_equal(replay_sides, sides)
        for key in ('xy_px','observed','usable','conf'):
            np.testing.assert_array_equal(getattr(obs,key)[:,:33], getattr(base,key)[:,:33])
        cal = audit.stable.calibrate_rest(obs,int(obs.sidecar['width']),int(obs.sidecar['height']))
        assert cal == row['calibration']
        diag = audit.diagnose(obs)
        assert diag['window_funnel'].get('qualifying_windows',0) == 0
        image = contact(row, obs, diag)
        old_equal = {key: bool(np.array_equal(getattr(obs,key), getattr(legacy,key), equal_nan=True))
                     for key in ('xy_px','observed','usable','timestamp_us','frame_id')}
        rows.append(dict(clip_id=cid,label=row['label'],calibration=cal,independent_window_diagnostics=diag,
            prior_legacy_diagnostics=old[cid]['diagnostics'],newjoint_vs_legacy_equal=old_equal,
            assignment_identity=assign.HandAssigner.identity(),assignment_raw_replay_exact=True,
            frames=obs.n_frames,selected_raw_hands=chosen_hands,source_and_pts_verified=True,
            full_frame_contact_sheet=image,contact_sheet_sha256=audit.sha(image)))
    assert all(audit.sha(p)==h for p,h in pins.items())
    out=dict(status='PASS_TWO_SAVED_PROBES_CONFIRMED_CALIBRATION_FAILURE',rows=rows,source_pins=pins,
        source_pins_unchanged=True,elapsed_seconds=time.perf_counter()-started,new_detector_runs=0,
        classifier_inference=0,training=0,user_or_heldout_access=0,
        visual_status='CONTACT_SHEETS_CREATED_PENDING_EXPLICIT_VIEW_IMAGE_REVIEW')
    output(U/'root_probe_audit_RESULT.json',out)
    print(json.dumps({'status':out['status'],'seconds':out['elapsed_seconds'],'images':[r['full_frame_contact_sheet'] for r in rows]},ensure_ascii=False))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
