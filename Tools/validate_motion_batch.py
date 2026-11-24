"""
Tools/validate_motion_batch.py
────────────────────────────────────────────────────────────────
blend_animate.py 파이프라인 범용성 검증 -배치 진단 스크립트

생산 코드(blend_animate.py)를 수정하지 않고 독립 실행.
Blender 없이 순수 Python 표준 라이브러리만 사용.

실행:
    python Tools/validate_motion_batch.py

출력:
    Validation/motion_validation_summary.csv   ← 단어별 핵심 지표 요약
    Validation/motion_validation_details.json  ← 전체 상세 데이터

[G] 시각 판정은 Blender에서 각 동작을 확인한 뒤 수동으로 기입:
    motion_validation_details.json 의 G 섹션 직접 편집, 또는
    motion_validation_summary.csv 의 G_* 컬럼을 스프레드시트로 작성.
────────────────────────────────────────────────────────────────
"""

import json, os, math, csv

# ================================================================
# 설정 -blend_animate.py 확정 설정과 반드시 일치시킬 것
# ================================================================
RECORDINGS_DIR   = r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\Recordings"
OUTPUT_DIR       = r"C:\MyUnityProjects\WordsInSilence\Validation"

# 확정 설정 (blend_animate.py 기준선)
SWAP_HAND_LABELS = False   # mp_left→캐릭터LEFT, mp_right→캐릭터RIGHT

# 스무딩 파라미터 (blend_animate.py 와 동일 수치)
SMOOTH_WINDOW = 5
# 주의: blend_animate.py 의 JUMP_THRESHOLD=0.5 는 Blender 월드 m 단위.
# 여기서는 어깨너비 정규화 좌표 사용.
# 근사 환산: 0.5m / char_sw(≈0.257m) ≈ 1.95 → 2.0 사용.
# 스무딩 결과는 Blender 출력과 완전히 동일하지 않음 (참고용).
JUMP_THRESHOLD_NORM = 2.0

# 검증 대상 단어 목록 (None 이면 RECORDINGS_DIR 전체 실행)
TARGET_WORDS = ['감사합니다', '나', '모르다', '빵', '먹다', '싫다']

# ── 이상 탐지 임계값 ─────────────────────────────────────────
# [B] Pose wrist 프레임간 이동 / 어깨너비 > 이 값이면 경고
WRIST_JUMP_RATIO_WARN  = 0.15
# [C][E] 프레임간 방향 변화 > 이 값(도)이면 경고
ROTATION_DELTA_WARN    = 30.0
# [E] 각속도 > 이 값(deg/s)이면 경고
ROTATION_SPEED_WARN    = 300.0
# [D] palm_plane_strength < 이 값이면 저품질 (어깨너비 정규화 외적 크기)
PLANE_STRENGTH_MIN     = 0.02
# [D] 연속 palm normal dot < 이 값이면 반전으로 분류
NORMAL_DOT_FLIP_THRESH = 0.0

# MediaPipe landmark 인덱스
P_L_SHOULDER = 11;  P_R_SHOULDER = 12
P_L_WRIST    = 15;  P_R_WRIST    = 16
H_WRIST      = 0;   H_INDEX_MCP  = 5
H_MID_MCP    = 9;   H_PINK_MCP   = 17


# ================================================================
# 벡터 유틸 (표준 라이브러리만)
# ================================================================
def get_pt(pts, idx):
    if not pts or idx >= len(pts): return None
    p = pts[idx]
    return (p.get('x', 0.0), p.get('y', 0.0), p.get('z', 0.0)) if isinstance(p, dict) else None

def v3sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def v3len(a):    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
def v3dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def v3norm(a):
    l = v3len(a)
    return (a[0]/l, a[1]/l, a[2]/l) if l > 1e-9 else (0.0, 0.0, 0.0)
def v3cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def angle_deg(a, b):
    """두 3D 벡터 사이 각도(도). 길이 0이면 0 반환."""
    la, lb = v3len(a), v3len(b)
    if la < 1e-9 or lb < 1e-9: return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, v3dot(a, b) / (la * lb)))))

def pct_val(vals, p):
    """p번째 백분위수."""
    if not vals: return 0.0
    sv = sorted(vals)
    idx = p / 100.0 * (len(sv) - 1)
    lo = int(idx); hi = min(lo + 1, len(sv) - 1)
    return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])


# ================================================================
# 스무딩 (blend_animate.py smooth_track 논리 재현, 정규화 좌표)
# ================================================================
def smooth1d(vals, window=SMOOTH_WINDOW, jump=JUMP_THRESHOLD_NORM):
    """1차원 스칼라 리스트 스무딩 (None 보간 → 아웃라이어 제거 → 이동평균)."""
    n = len(vals)
    if n == 0: return []
    f = list(vals)
    # None 선형 보간
    first = next((i for i, v in enumerate(f) if v is not None), None)
    if first is None: return [0.0] * n
    for i in range(first): f[i] = f[first]
    last = next((i for i in range(n-1, -1, -1) if f[i] is not None), 0)
    for i in range(last+1, n): f[i] = f[last]
    i = 0
    while i < n:
        if f[i] is None:
            a, b = i-1, i
            while b < n and f[b] is None: b += 1
            for k in range(i, b):
                t = (k-a)/(b-a) if b > a else 0.0
                fv = f[a] if a >= 0 else 0.0
                tv = f[b] if b < n else fv
                f[k] = fv*(1-t) + tv*t
            i = b
        else:
            i += 1
    # 아웃라이어 제거 (2회)
    for _ in range(2):
        for i in range(1, n-1):
            avg = (f[i-1] + f[i+1]) / 2
            if abs(f[i] - avg) > jump: f[i] = avg
    # 이동평균
    half = window // 2
    result = []
    for i in range(n):
        chunk = f[max(0, i-half):min(n, i+half+1)]
        result.append(sum(chunk) / len(chunk))
    return result


def smooth3d(vecs, window=SMOOTH_WINDOW, jump=JUMP_THRESHOLD_NORM):
    """(x,y,z) 리스트 스무딩. None 포함 가능."""
    xs = smooth1d([v[0] if v else None for v in vecs], window, jump)
    ys = smooth1d([v[1] if v else None for v in vecs], window, jump)
    zs = smooth1d([v[2] if v else None for v in vecs], window, jump)
    return list(zip(xs, ys, zs))


# ================================================================
# 핵심 분석
# ================================================================
def analyze_motion(word, json_path):
    """
    단일 JSON 파일을 분석해 섹션 A~E + G 구조를 반환.
    G 섹션(시각 판정)은 PENDING -Blender 확인 후 수동 기입.
    """
    with open(json_path, encoding='utf-8') as f:
        rec = json.load(f)

    frames_data = rec.get('Frames', [])
    res = {
        'word':             word,
        'json_path':        json_path,
        'total_json_frames': len(frames_data),
        'swap_hand_labels': SWAP_HAND_LABELS,
    }

    # ── 유효 프레임 추출 ─────────────────────────────────────────
    frs = []
    for fd in frames_data:
        if not fd.get('HasPose', False): continue
        pose_pts = fd.get('PosePoints') or []
        mp_l_pts = fd.get('LeftHandPoints') or []
        mp_r_pts = fd.get('RightHandPoints') or []
        has_mp_l = fd.get('HasLeftHand', False)
        has_mp_r = fd.get('HasRightHand', False)
        ts = fd.get('TimestampMs', None)

        if SWAP_HAND_LABELS:
            ch_l, hl = mp_r_pts, has_mp_r
            ch_r, hr = mp_l_pts, has_mp_l
        else:
            ch_l, hl = mp_l_pts, has_mp_l
            ch_r, hr = mp_r_pts, has_mp_r

        ls = get_pt(pose_pts, P_L_SHOULDER)
        rs = get_pt(pose_pts, P_R_SHOULDER)
        if not ls or not rs: continue
        sw = math.sqrt((ls[0]-rs[0])**2 + (ls[1]-rs[1])**2)
        if sw < 1e-4: continue

        cx = (ls[0]+rs[0])/2; cy = (ls[1]+rs[1])/2

        def hpt(idx, pts, valid):
            """Hand landmark → 어깨너비 정규화 좌표 (cx/cy 기준)."""
            if not valid: return None
            pt = get_pt(pts, idx)
            if pt is None: return None
            return ((pt[0]-cx)/sw, -(pt[1]-cy)/sw, pt[2]/sw)

        def ppt(idx):
            """Pose landmark → 어깨너비 정규화 좌표."""
            pt = get_pt(pose_pts, idx)
            if pt is None: return None
            return ((pt[0]-cx)/sw, -(pt[1]-cy)/sw, pt[2]/sw)

        frs.append({
            'ts':      ts,
            'has_l':   hl,
            'has_r':   hr,
            # Pose wrist (IK 기준 -Step C 이동량 분석용)
            'pose_lw': ppt(P_L_WRIST),
            'pose_rw': ppt(P_R_WRIST),
            # Hand landmarks (어깨너비 정규화)
            'l0':  hpt(H_WRIST,     ch_l, hl),
            'l9':  hpt(H_MID_MCP,   ch_l, hl),
            'l5':  hpt(H_INDEX_MCP, ch_l, hl),
            'l17': hpt(H_PINK_MCP,  ch_l, hl),
            'r0':  hpt(H_WRIST,     ch_r, hr),
            'r9':  hpt(H_MID_MCP,   ch_r, hr),
            'r5':  hpt(H_INDEX_MCP, ch_r, hr),
            'r17': hpt(H_PINK_MCP,  ch_r, hr),
        })

    n = len(frs)
    res['pose_frames'] = n
    if n == 0:
        res['error'] = '유효 pose 프레임 없음'
        return res

    # ── A: 검출 통계 ─────────────────────────────────────────────
    l_det = sum(1 for fr in frs if fr['has_l'])
    r_det = sum(1 for fr in frs if fr['has_r'])
    both  = sum(1 for fr in frs if fr['has_l'] and fr['has_r'])

    def max_consec_miss(flags):
        best = cur = 0
        for v in flags:
            cur = 0 if v else cur + 1
            best = max(best, cur)
        return best

    # 연속 누락 구간 목록 (start_frame, length, start_ts, end_ts)
    def consec_miss_spans(flags, ts_list):
        spans = []; start = None
        for i, v in enumerate(flags):
            if not v:
                if start is None: start = i
            else:
                if start is not None:
                    spans.append({'start': start, 'len': i-start,
                                  'ts_start': ts_list[start], 'ts_end': ts_list[i-1]})
                    start = None
        if start is not None:
            spans.append({'start': start, 'len': len(flags)-start,
                          'ts_start': ts_list[start], 'ts_end': ts_list[-1]})
        return sorted(spans, key=lambda s: -s['len'])[:5]

    ts_all = [fr['ts'] for fr in frs]
    res['A'] = {
        'l_det':  l_det,  'l_pct': 100*l_det//n,
        'r_det':  r_det,  'r_pct': 100*r_det//n,
        'both':   both,   'both_pct': 100*both//n,
        'l_max_consec_miss': max_consec_miss([fr['has_l'] for fr in frs]),
        'r_max_consec_miss': max_consec_miss([fr['has_r'] for fr in frs]),
        'l_miss_spans': consec_miss_spans([fr['has_l'] for fr in frs], ts_all),
        'r_miss_spans': consec_miss_spans([fr['has_r'] for fr in frs], ts_all),
        'note': ('LEFT 거의 미검출 -캐릭터LEFT 분석 제한'
                 if l_det < n * 0.1 else
                 '양손 미검출 -단일손 동작 가능성'
                 if both < n * 0.2 and l_det > n * 0.1 and r_det > n * 0.1
                 else ''),
    }

    # ── B: Pose wrist 이동량 (어깨너비 정규화, Step C) ──────────
    def wrist_stats(wkey):
        """
        wrist_jump_ratio = 프레임간 이동거리 / 어깨너비
        raw: Pose wrist 원본
        smoothed: SMOOTH_WINDOW 이동평균 (Blender 스무딩 근사)
        retention_ratio = smoothed_max / raw_max  (작을수록 스무딩 효과 큼)
        reduction_ratio = 1 - retention_ratio
        """
        raw = [fr[wkey] for fr in frs]
        sm  = smooth3d(raw)

        rj = []; rj_frame = []; rj_ts = []
        sj = []; sj_frame = []
        for i in range(1, n):
            if raw[i-1] and raw[i]:
                d = v3len(v3sub(raw[i], raw[i-1]))
                rj.append(d); rj_frame.append(i); rj_ts.append(frs[i]['ts'])
            d2 = v3len(v3sub(sm[i], sm[i-1]))
            sj.append(d2); sj_frame.append(i)

        def st(vals, frames, ts_list=None):
            if not vals:
                return {'max':0.0,'mean':0.0,'p95':0.0,'max_frame':None,'max_ts':None}
            mx = max(vals); mi = vals.index(mx)
            return {
                'max':  round(mx, 5),
                'mean': round(sum(vals)/len(vals), 5),
                'p95':  round(pct_val(vals, 95), 5),
                'max_frame': frames[mi],
                'max_ts':    (ts_list[mi] if ts_list else frs[frames[mi]]['ts']),
            }

        rst = st(rj, rj_frame, rj_ts)
        sst = st(sj, sj_frame)
        ret = (sst['max'] / rst['max']) if rst['max'] > 1e-9 else 1.0

        # 경고 프레임 (raw, > WRIST_JUMP_RATIO_WARN)
        warn = [{'frame': rj_frame[i], 'ts': rj_ts[i], 'ratio': round(rj[i], 4)}
                for i in range(len(rj)) if rj[i] > WRIST_JUMP_RATIO_WARN]

        return {
            'raw':              rst,
            'smoothed':         sst,
            'retention_ratio':  round(float(ret), 3),
            'reduction_ratio':  round(float(1 - ret), 3),
            'warn_frames':      warn[:10],
            'note': ('스무딩 후에도 이동량 큼' if sst['max'] > WRIST_JUMP_RATIO_WARN else
                     '스무딩 효과 미미 (retention≈1)' if ret > 0.95 else ''),
        }

    res['B'] = {
        'LEFT':  wrist_stats('pose_lw'),
        'RIGHT': wrist_stats('pose_rw'),
        'metric': 'wrist_jump_ratio = frame-to-frame_distance / shoulder_width (정규화)',
    }

    # ── C: wrist→mid9 방향 안정성 (Step A) ──────────────────────
    def fwd_stats(k0, k9, kv):
        """
        연속 프레임 wrist→mid9 방향 변화량.
        hand bone forward direction 의 안정성 지표.
        값이 크면 FK 기준 방향이 프레임마다 크게 바뀜 → 손가락 기울기 불안정.
        """
        fwds = []
        for fr in frs:
            if fr[kv] and fr[k0] and fr[k9]:
                fwds.append((fr['ts'], v3norm(v3sub(fr[k9], fr[k0]))))
            else:
                fwds.append((fr['ts'], None))

        deltas = []; warns = []
        for i in range(1, len(fwds)):
            ts_prev, fv_prev = fwds[i-1]
            ts_cur,  fv_cur  = fwds[i]
            if fv_prev and fv_cur:
                d = angle_deg(fv_prev, fv_cur)
                deltas.append(d)
                if d > ROTATION_DELTA_WARN:
                    warns.append({'frame': i, 'ts': ts_cur, 'deg': round(d, 1)})

        return {
            'mean_deg':   round(sum(deltas)/len(deltas), 2) if deltas else 0.0,
            'max_deg':    round(max(deltas), 2) if deltas else 0.0,
            'p95_deg':    round(pct_val(deltas, 95), 2) if deltas else 0.0,
            'warn_count': len(warns),
            'warn_frames': warns[:10],
        }

    res['C'] = {
        'LEFT':  fwd_stats('l0', 'l9', 'has_l'),
        'RIGHT': fwd_stats('r0', 'r9', 'has_r'),
        'metric': 'wrist→mid9 연속 프레임 방향 변화(도) -hand bone FK forward 안정성',
    }

    # ── D: Palm normal 안정성 (Step D) ──────────────────────────
    def pn_stats(k0, k5, k17, kv):
        """
        palm_plane_strength = |(index_mcp - wrist) × (pinky_mcp - wrist)|
        강도가 낮으면 손이 옆면(side-on)이라 normal 신뢰 불가.

        연속 프레임 dot < 0 → 법선 반전 후보.
        - flip_high_quality: strength >= PLANE_STRENGTH_MIN 인 반전 (실제 문제 후보)
        - flip_low_quality:  strength <  PLANE_STRENGTH_MIN 인 반전 (side-on 노이즈 후보)
        """
        normals = []
        for fr in frs:
            if fr[kv] and fr[k0] and fr[k5] and fr[k17]:
                v05   = v3sub(fr[k5],  fr[k0])
                v17   = v3sub(fr[k17], fr[k0])
                cross = v3cross(v05, v17)
                s     = v3len(cross)
                normals.append((fr['ts'], v3norm(cross) if s > 1e-9 else None, s))
            else:
                normals.append((fr['ts'], None, 0.0))

        dots = []; flip_total = 0; flip_hq = 0; flip_lq = 0
        lowq_frames = []
        for i in range(1, len(normals)):
            ts_prev, na, _  = normals[i-1]
            ts_cur,  nb, si = normals[i]
            if na and nb:
                d = v3dot(na, nb)
                dots.append(d)
                if d < NORMAL_DOT_FLIP_THRESH:
                    flip_total += 1
                    if si >= PLANE_STRENGTH_MIN: flip_hq += 1
                    else:                        flip_lq += 1
            # 저품질 프레임 기록
            if nb is not None and 0.0 < si < PLANE_STRENGTH_MIN:
                lowq_frames.append({'frame': i, 'ts': ts_cur,
                                    'strength': round(si, 5)})

        mean_strength = 0.0
        valid_s = [normals[i][2] for i in range(len(normals))
                   if normals[i][1] is not None]
        if valid_s:
            mean_strength = round(sum(valid_s)/len(valid_s), 5)

        return {
            'mean_dot':           round(sum(dots)/len(dots), 4) if dots else 0.0,
            'min_dot':            round(min(dots), 4) if dots else 0.0,
            'mean_plane_strength': mean_strength,
            'flip_total':         flip_total,
            'flip_high_quality':  flip_hq,   # 실제 반전 후보
            'flip_low_quality':   flip_lq,   # side-on 노이즈 후보
            'low_quality_frame_count': len(lowq_frames),
            'low_quality_frames': lowq_frames[:10],
            'note': ('전역 flip 설정 오류 후보 (high_quality 반전 多)'
                     if flip_hq > 3 else
                     'side-on 노이즈 (low_quality 반전)' if flip_lq > 5 else ''),
        }

    res['D'] = {
        'LEFT':  pn_stats('l0', 'l5', 'l17', 'has_l'),
        'RIGHT': pn_stats('r0', 'r5', 'r17', 'has_r'),
        'metric': 'palm_plane_strength = |cross| (어깨너비 정규화), dot = 연속 프레임 법선 내적',
        'plane_strength_min': PLANE_STRENGTH_MIN,
    }

    # ── E: 회전 변화 (Step D) -target basis 변화로 근사 ────────
    def rot_stats(k0, k9, k5, k17, kv):
        """
        실제 Blender quaternion 없이 target basis 변화로 근사.
        delta = max(wrist→mid9 각도 변화, palm_normal 각도 변화)
        angular_speed = delta / delta_time_seconds

        같은 프레임에서 wrist→mid9 변화, palm_normal dot, 검출 유효 여부를 함께 기록.
        빠른 실제 동작(speed 크고 검출 유효) vs 검출 오류(speed 크고 방향 이상)를 구분.
        """
        bases = []
        for fr in frs:
            if fr[kv] and fr[k0] and fr[k9] and fr[k5] and fr[k17]:
                fwd   = v3norm(v3sub(fr[k9], fr[k0]))
                cross = v3cross(v3sub(fr[k5], fr[k0]), v3sub(fr[k17], fr[k0]))
                pn    = v3norm(cross) if v3len(cross) > 1e-9 else None
                bases.append((fr['ts'], fwd, pn, True))
            else:
                bases.append((fr['ts'], None, None, False))

        deltas = []; speeds = []; warns = []
        for i in range(1, len(bases)):
            ts0, f0, p0, valid0 = bases[i-1]
            ts1, f1, p1, valid1 = bases[i]
            if f0 and f1:
                fd  = angle_deg(f0, f1)
                pd  = angle_deg(p0, p1) if (p0 and p1) else 0.0
                dot = v3dot(p0, p1) if (p0 and p1) else None
                delta = max(fd, pd)
                deltas.append(delta)
                dt  = ((ts1 - ts0) / 1000.0
                       if (ts0 is not None and ts1 is not None and ts1 > ts0)
                       else None)
                spd = (delta / dt) if dt else None
                if spd is not None: speeds.append(spd)

                if delta > ROTATION_DELTA_WARN or (spd and spd > ROTATION_SPEED_WARN):
                    warns.append({
                        'frame':       i,
                        'ts':          ts1,
                        'ts_prev':     ts0,
                        'delta_deg':   round(delta, 1),
                        'fwd_delta':   round(fd, 1),
                        'pn_delta':    round(pd, 1),
                        'pn_dot':      round(dot, 3) if dot is not None else None,
                        'speed_degs':  round(spd, 0) if spd else None,
                        'det_valid':   valid1,
                    })

        vspd = [s for s in speeds if s is not None]
        return {
            'mean_delta_deg':  round(sum(deltas)/len(deltas), 2) if deltas else 0.0,
            'max_delta_deg':   round(max(deltas), 2) if deltas else 0.0,
            'p95_delta_deg':   round(pct_val(deltas, 95), 2) if deltas else 0.0,
            'mean_speed_degs': round(sum(vspd)/len(vspd), 1) if vspd else 0.0,
            'max_speed_degs':  round(max(vspd), 1) if vspd else 0.0,
            'p95_speed_degs':  round(pct_val(vspd, 95), 1) if vspd else 0.0,
            'warn_count':      len(warns),
            'warn_frames':     warns[:10],
            'note': ('rotation_note: 각도 변화 클 때 det_valid와 pn_dot을 함께 확인.'
                     ' det_valid=True + pn_dot>0 이면 실제 빠른 동작.'
                     ' det_valid=False 또는 pn_dot<0 이면 검출 오류 후보.'),
        }

    res['E'] = {
        'LEFT':  rot_stats('l0','l9','l5','l17','has_l'),
        'RIGHT': rot_stats('r0','r9','r5','r17','has_r'),
        'metric': 'target basis 변화 근사: max(fwd_delta, pn_delta) per frame',
    }

    # ── F: 손가락 FK 품질 (수동 측정) ────────────────────────────
    # blend_animate.py TEST_MODE=True + diag_index_finger 실행 필요.
    # 현재 TEST_MODE는 LEFT 검지만 측정.
    # RIGHT 검지 측정을 위해서는 별도 실행 또는 스크립트 확장 필요.
    res['F'] = {
        'LEFT_index': {
            'implementation_error': {'fwd_err_deg': 'PENDING', 'roll_err_deg': 'PENDING'},
            'source_fidelity': 'PENDING',  # Blender 원본 비교 시각 판정
            'measure_method': 'TEST_MODE=True 에서 diag_index_finger(LEFT) 실행',
        },
        'RIGHT_index': {
            'implementation_error': {'fwd_err_deg': 'PENDING', 'roll_err_deg': 'PENDING'},
            'source_fidelity': 'PENDING',
            'measure_method': 'TEST_MODE=True 에서 diag_index_finger(RIGHT) 실행 (확장 필요)',
        },
        'note': ('fwd_err/roll_err는 구현 정확도만 측정.'
                 ' 값이 작아도 MediaPipe 검출 품질이나 최종 시각 결과와는 별개.'),
    }

    # ── G: 시각 판정 (수동 입력 -Blender 확인 후 기입) ─────────
    res['G'] = {
        'source_fidelity': {
            'hand_assignment':   'PENDING',   # OK / NG
            'left_palm_dir':     'PENDING',   # OK / NG / PARTIAL / NA(단일손)
            'right_palm_dir':    'PENDING',   # OK / NG / PARTIAL / NA
            'flip_180_frames':   'PENDING',   # none / L=N / R=N / LR=N
            'wrist_bend_issues': 'PENDING',   # none / mild / severe(N frames)
            'overall':           'PENDING',   # OK / OK_WITH_ISSUES / NG
        },
        'problem_classification': {
            # 각 항목: none / 해당 값 기입
            'input_problem':    'PENDING',  # detection_failure / handedness_error / occlusion / none
            'pipeline_problem': 'PENDING',  # global_setting / palm_normal_flip / side_on_normal / none
            'character_problem':'PENDING',  # skinning / rig / none
        },
        'notes': '',
    }

    return res


# ================================================================
# 보고서 출력
# ================================================================
SEP  = '─' * 66
SEP2 = '=' * 66

def print_report(res):
    print(f"\n{SEP2}")
    print(f"동작: {res['word']}  | JSON: {os.path.basename(res['json_path'])}")
    print(f"총 JSON 프레임: {res.get('total_json_frames')}  |  "
          f"유효 Pose 프레임: {res.get('pose_frames')}")
    if 'error' in res:
        print(f"[오류] {res['error']}")
        return

    A = res['A']
    print(f"\n[A] 검출 통계  (SWAP_HAND_LABELS={SWAP_HAND_LABELS})")
    print(f"  캐릭터 LEFT  (mp_{'right' if SWAP_HAND_LABELS else 'left'}): "
          f"{A['l_det']}프레임 ({A['l_pct']}%)  최대연속누락: {A['l_max_consec_miss']}f")
    print(f"  캐릭터 RIGHT (mp_{'left' if SWAP_HAND_LABELS else 'right'}): "
          f"{A['r_det']}프레임 ({A['r_pct']}%)  최대연속누락: {A['r_max_consec_miss']}f")
    print(f"  양손 동시: {A['both']}프레임 ({A['both_pct']}%)")
    if A.get('note'): print(f"  [주의] {A['note']}")

    print(f"\n[B] Pose wrist 이동량  (wrist_jump_ratio = 프레임간이동 / 어깨너비)")
    for side in ['LEFT', 'RIGHT']:
        B = res['B'][side]
        r = B['raw']; s = B['smoothed']
        print(f"  {side}")
        print(f"    raw     max={r['max']:.4f}  mean={r['mean']:.4f}  p95={r['p95']:.4f}"
              f"  (max@frame{r['max_frame']} ts={r['max_ts']}ms)")
        print(f"    smooth  max={s['max']:.4f}  mean={s['mean']:.4f}  p95={s['p95']:.4f}")
        print(f"    retention={B['retention_ratio']:.3f}  reduction={B['reduction_ratio']:.3f}")
        for w in B['warn_frames'][:3]:
            print(f"    [경고] frame={w['frame']} ts={w['ts']}ms ratio={w['ratio']:.4f}")
        if B.get('note'): print(f"    [주의] {B['note']}")

    print(f"\n[C] wrist→mid9 방향 안정성  (Hand FK forward 기준)")
    for side in ['LEFT', 'RIGHT']:
        C = res['C'][side]
        print(f"  {side}  mean={C['mean_deg']:.1f}°  max={C['max_deg']:.1f}°"
              f"  p95={C['p95_deg']:.1f}°  경고={C['warn_count']}건")
        for w in C['warn_frames'][:2]:
            print(f"    [경고] frame={w['frame']} ts={w['ts']}ms deg={w['deg']:.1f}")

    print(f"\n[D] Palm normal 안정성  (plane_strength_min={PLANE_STRENGTH_MIN})")
    for side in ['LEFT', 'RIGHT']:
        D = res['D'][side]
        print(f"  {side}  mean_dot={D['mean_dot']:.3f}  min_dot={D['min_dot']:.3f}"
              f"  mean_strength={D['mean_plane_strength']:.4f}")
        print(f"    반전: 총={D['flip_total']}  "
              f"고품질(파이프라인 오류 후보)={D['flip_high_quality']}  "
              f"저품질(side-on 노이즈)={D['flip_low_quality']}")
        print(f"    저품질 프레임 수(strength<{PLANE_STRENGTH_MIN}): "
              f"{D['low_quality_frame_count']}개")
        if D.get('note'): print(f"    [주의] {D['note']}")

    print(f"\n[E] 회전 변화  (target basis 근사 -angular_speed_deg_s 포함)")
    for side in ['LEFT', 'RIGHT']:
        E = res['E'][side]
        print(f"  {side}  mean={E['mean_delta_deg']:.1f}°  max={E['max_delta_deg']:.1f}°"
              f"  p95={E['p95_delta_deg']:.1f}°")
        print(f"    speed: mean={E['mean_speed_degs']:.0f}°/s  "
              f"max={E['max_speed_degs']:.0f}°/s  p95={E['p95_speed_degs']:.0f}°/s")
        print(f"    경고 프레임: {E['warn_count']}건")
        for w in E['warn_frames'][:3]:
            print(f"    [경고] frame={w['frame']} ts={w['ts']}ms "
                  f"delta={w['delta_deg']:.1f}° speed={w['speed_degs']}°/s "
                  f"det={w['det_valid']} pn_dot={w['pn_dot']}")

    print(f"\n[F] 손가락 FK 품질  (수동 측정 필요 -TEST_MODE=True)")
    for side, key in [('LEFT','LEFT_index'),('RIGHT','RIGHT_index')]:
        print(f"  {side}: fwd_err=PENDING°  roll_err=PENDING°"
              f"  source_fidelity=PENDING")

    print(f"\n[G] 시각 판정  (Blender 확인 후 수동 기입)")
    G = res['G']
    for k, v in G['source_fidelity'].items():
        print(f"  {k}: {v}")
    for k, v in G['problem_classification'].items():
        print(f"  {k}: {v}")


# ================================================================
# CSV / JSON 저장
# ================================================================
CSV_FIELDS = [
    'word', 'pose_frames',
    # A
    'A_l_pct', 'A_r_pct', 'A_both_pct',
    'A_l_max_miss', 'A_r_max_miss', 'A_note',
    # B LEFT
    'B_L_raw_max', 'B_L_raw_p95', 'B_L_sm_max', 'B_L_sm_p95',
    'B_L_retention', 'B_L_warn_count',
    # B RIGHT
    'B_R_raw_max', 'B_R_raw_p95', 'B_R_sm_max', 'B_R_sm_p95',
    'B_R_retention', 'B_R_warn_count',
    # C
    'C_L_mean', 'C_L_max', 'C_L_p95', 'C_L_warn',
    'C_R_mean', 'C_R_max', 'C_R_p95', 'C_R_warn',
    # D
    'D_L_min_dot', 'D_L_strength', 'D_L_flip_hq', 'D_L_flip_lq', 'D_L_lowq',
    'D_R_min_dot', 'D_R_strength', 'D_R_flip_hq', 'D_R_flip_lq', 'D_R_lowq',
    # E
    'E_L_max_delta', 'E_L_max_speed', 'E_L_p95_speed', 'E_L_warn',
    'E_R_max_delta', 'E_R_max_speed', 'E_R_p95_speed', 'E_R_warn',
    # F (수동)
    'F_L_fwd_err', 'F_L_roll_err', 'F_R_fwd_err', 'F_R_roll_err',
    # G (수동)
    'G_hand_assign', 'G_left_palm', 'G_right_palm', 'G_flip_180',
    'G_wrist_bend', 'G_overall',
    'G_input_problem', 'G_pipeline_problem', 'G_character_problem',
    'G_notes',
]


def to_csv_row(res):
    if 'error' in res:
        return {'word': res['word'], 'pose_frames': 0}
    A = res.get('A', {}); B = res.get('B', {})
    C = res.get('C', {}); D = res.get('D', {})
    E = res.get('E', {}); G = res.get('G', {})
    BL = B.get('LEFT', {}); BR = B.get('RIGHT', {})
    CL = C.get('LEFT', {}); CR = C.get('RIGHT', {})
    DL = D.get('LEFT', {}); DR = D.get('RIGHT', {})
    EL = E.get('LEFT', {}); ER = E.get('RIGHT', {})
    Gsf = G.get('source_fidelity', {}); Gpc = G.get('problem_classification', {})
    return {
        'word': res['word'], 'pose_frames': res['pose_frames'],
        'A_l_pct': A.get('l_pct',''), 'A_r_pct': A.get('r_pct',''),
        'A_both_pct': A.get('both_pct',''),
        'A_l_max_miss': A.get('l_max_consec_miss',''),
        'A_r_max_miss': A.get('r_max_consec_miss',''),
        'A_note': A.get('note',''),
        'B_L_raw_max': BL.get('raw',{}).get('max',''),
        'B_L_raw_p95': BL.get('raw',{}).get('p95',''),
        'B_L_sm_max':  BL.get('smoothed',{}).get('max',''),
        'B_L_sm_p95':  BL.get('smoothed',{}).get('p95',''),
        'B_L_retention': BL.get('retention_ratio',''),
        'B_L_warn_count': len(BL.get('warn_frames',[])),
        'B_R_raw_max': BR.get('raw',{}).get('max',''),
        'B_R_raw_p95': BR.get('raw',{}).get('p95',''),
        'B_R_sm_max':  BR.get('smoothed',{}).get('max',''),
        'B_R_sm_p95':  BR.get('smoothed',{}).get('p95',''),
        'B_R_retention': BR.get('retention_ratio',''),
        'B_R_warn_count': len(BR.get('warn_frames',[])),
        'C_L_mean': CL.get('mean_deg',''), 'C_L_max': CL.get('max_deg',''),
        'C_L_p95':  CL.get('p95_deg',''),  'C_L_warn': CL.get('warn_count',''),
        'C_R_mean': CR.get('mean_deg',''), 'C_R_max': CR.get('max_deg',''),
        'C_R_p95':  CR.get('p95_deg',''),  'C_R_warn': CR.get('warn_count',''),
        'D_L_min_dot':  DL.get('min_dot',''), 'D_L_strength': DL.get('mean_plane_strength',''),
        'D_L_flip_hq':  DL.get('flip_high_quality',''),
        'D_L_flip_lq':  DL.get('flip_low_quality',''),
        'D_L_lowq':     DL.get('low_quality_frame_count',''),
        'D_R_min_dot':  DR.get('min_dot',''), 'D_R_strength': DR.get('mean_plane_strength',''),
        'D_R_flip_hq':  DR.get('flip_high_quality',''),
        'D_R_flip_lq':  DR.get('flip_low_quality',''),
        'D_R_lowq':     DR.get('low_quality_frame_count',''),
        'E_L_max_delta': EL.get('max_delta_deg',''),
        'E_L_max_speed': EL.get('max_speed_degs',''),
        'E_L_p95_speed': EL.get('p95_speed_degs',''),
        'E_L_warn':      EL.get('warn_count',''),
        'E_R_max_delta': ER.get('max_delta_deg',''),
        'E_R_max_speed': ER.get('max_speed_degs',''),
        'E_R_p95_speed': ER.get('p95_speed_degs',''),
        'E_R_warn':      ER.get('warn_count',''),
        # F/G: 수동 입력 -초기값 PENDING
        'F_L_fwd_err': 'PENDING', 'F_L_roll_err': 'PENDING',
        'F_R_fwd_err': 'PENDING', 'F_R_roll_err': 'PENDING',
        'G_hand_assign':      Gsf.get('hand_assignment','PENDING'),
        'G_left_palm':        Gsf.get('left_palm_dir','PENDING'),
        'G_right_palm':       Gsf.get('right_palm_dir','PENDING'),
        'G_flip_180':         Gsf.get('flip_180_frames','PENDING'),
        'G_wrist_bend':       Gsf.get('wrist_bend_issues','PENDING'),
        'G_overall':          Gsf.get('overall','PENDING'),
        'G_input_problem':    Gpc.get('input_problem','PENDING'),
        'G_pipeline_problem': Gpc.get('pipeline_problem','PENDING'),
        'G_character_problem':Gpc.get('character_problem','PENDING'),
        'G_notes':            G.get('notes',''),
    }


def save_summary_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        for res in results:
            w.writerow(to_csv_row(res))
    print(f"[저장] {path}")


def save_details_json(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[저장] {path}")


# ================================================================
# 파일 목록
# ================================================================
def list_recordings(recs_dir):
    if not os.path.isdir(recs_dir): return []
    return [(fn[:-5], os.path.join(recs_dir, fn))
            for fn in sorted(os.listdir(recs_dir)) if fn.endswith('.json')]


# ================================================================
# 메인
# ================================================================
def main():
    print(SEP2)
    print("validate_motion_batch.py")
    print(f"  SWAP_HAND_LABELS = {SWAP_HAND_LABELS}  "
          f"(mp_{'right' if SWAP_HAND_LABELS else 'left'}→캐릭터LEFT, "
          f"mp_{'left' if SWAP_HAND_LABELS else 'right'}→캐릭터RIGHT)")
    print(f"  RECORDINGS_DIR   = {RECORDINGS_DIR}")
    print(f"  OUTPUT_DIR       = {OUTPUT_DIR}")
    print(SEP2)

    # 사용 가능한 JSON 전체 목록
    all_recs = list_recordings(RECORDINGS_DIR)
    print(f"\n[사용 가능한 JSON] ({len(all_recs)}개)")
    for word, _ in all_recs:
        marker = ' <-- 대상' if (TARGET_WORDS and word in TARGET_WORDS) else ''
        print(f"  {word}{marker}")

    # 검증 대상 필터
    if TARGET_WORDS is not None:
        targets = [(w, p) for w, p in all_recs if w in TARGET_WORDS]
        missing = set(TARGET_WORDS) - {w for w, _ in targets}
        if missing:
            print(f"\n[경고] TARGET_WORDS 에 있으나 JSON 없음: {sorted(missing)}")
    else:
        targets = all_recs

    print(f"\n[분석 대상] ({len(targets)}개): {[w for w, _ in targets]}\n")

    results = []
    for word, path in targets:
        print(f"  분석 중: {word} ...")
        try:
            res = analyze_motion(word, path)
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {'word': word, 'json_path': path,
                   'total_json_frames': 0, 'pose_frames': 0, 'error': str(e)}
        results.append(res)
        print_report(res)

    # 저장
    csv_path  = os.path.join(OUTPUT_DIR, 'motion_validation_summary.csv')
    json_path = os.path.join(OUTPUT_DIR, 'motion_validation_details.json')
    save_summary_csv(results, csv_path)
    save_details_json(results, json_path)

    # 간단 비교 표 출력
    print(f"\n{SEP2}")
    print("자동 수치 요약")
    print(f"  {'단어':<12} {'L%':>4} {'R%':>4} {'both%':>6} "
          f"{'BL_rmax':>8} {'BR_rmax':>8} "
          f"{'D_L_fhq':>8} {'D_R_fhq':>8} "
          f"{'E_L_max':>8} {'E_R_max':>8}")
    for res in results:
        if 'error' in res: continue
        A=res['A']; B=res['B']; D=res['D']; E=res['E']
        print(f"  {res['word']:<12} "
              f"{A['l_pct']:>4} {A['r_pct']:>4} {A['both_pct']:>6} "
              f"{B['LEFT']['raw']['max']:>8.3f} {B['RIGHT']['raw']['max']:>8.3f} "
              f"{D['LEFT']['flip_high_quality']:>8} {D['RIGHT']['flip_high_quality']:>8} "
              f"{E['LEFT']['max_delta_deg']:>8.1f} {E['RIGHT']['max_delta_deg']:>8.1f}")

    # 판정 기준 안내
    print(f"\n{SEP}")
    print("[Step 진입 기준 -3개 이상 동작에서 동일 원인 확인 시 범용 파이프라인 수정 검토]")
    print(f"  Step A (손가락 기울기): C_max>30° 2개+ 동작 or F_fwd_err>10° 2개+ 동작")
    print(f"  Step B (손가락 roll):   F_roll_err>20° 2개+ 동작, Step A 완료 후 잔존")
    print(f"  Step C (이동량):        B_sm_max>{WRIST_JUMP_RATIO_WARN:.2f} (어깨너비비) 2개+ 동작")
    print(f"  Step D (손목 회전):     D_flip_hq>3 또는 E_max_speed>{ROTATION_SPEED_WARN:.0f}°/s  2개+ 동작")
    print(f"\n  Palm/hand assignment NG: 1건이라도 발생 시 G_pipeline/input 분류 후 판단")
    print(f"  1개 동작 발생: 해당 영상/검출 문제 후보")
    print(f"  2개 이상: 공통 원인 조사")
    print(f"  3개 이상: 범용 수정 검토 → 수정 후 대표 6개 전체 회귀 테스트")
    print(f"\n[출력 파일]")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"\n[다음 단계]")
    print(f"  1. 각 동작을 blend_animate.py(JSON_PATH만 변경)로 생성 후 Blender에서 확인")
    print(f"  2. G섹션(시각 판정)을 details.json 또는 summary.csv에 직접 기입")
    print(f"  3. F섹션: TEST_MODE=True + diag_index_finger 로 좌우 검지 fwd/roll_err 측정")
    print(SEP2)


main()
