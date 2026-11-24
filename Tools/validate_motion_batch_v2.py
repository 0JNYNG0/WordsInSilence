"""
Tools/validate_motion_batch_v2.py
────────────────────────────────────────────────────────────────
blend_animate.py 파이프라인 범용성 검증 - 배치 진단 스크립트 v2

v1 대비 수정 사항 (10개):
  1. ppt() z=0 (DEPTH_SCALE=0.0 일치), hpt() HAND_DEPTH_GAIN=1.0 반영
  2. warn_count(전수) / warn_frames[:10](표시용) 분리
  3. D: palm_plane_sin=|cross|/(|u||v|), hand_scale=(|u|+|v|)/2 추가
  4. D: flip 고품질 = 양 프레임 모두 sin>=PLANE_SIN_MIN AND scale>=HAND_SCALE_MIN
  5. E: true 3x3 target basis (L/R_PALM_FLIP 적용), acos((tr(R_rel)-1)/2)
  6. E: det_valid 제거 -> landmarks_valid / basis_valid / basis_determinant 분리
  7. 영벡터 -> invalid 표시, 통계 제외, invalid_frame_count 출력
  8. 출력 파일: _v2.csv/_v2.json(auto), manual.json(없을 때만 템플릿), merged.csv
  9. 임계값 = 검토 경고 표시용 (통과/실패 결정 아님)
 10. 대상 단어: 감사합니다 나 모르다 빵 먹다 싫다

생산 코드(blend_animate.py) 수정 없음.
Blender 없이 순수 Python 표준 라이브러리만 사용.

실행:
    python Tools/validate_motion_batch_v2.py

출력:
    Validation/motion_validation_summary_v2.csv
    Validation/motion_validation_details_v2.json
    Validation/motion_validation_manual.json    (없을 때만 템플릿 생성)
    Validation/motion_validation_merged.csv
────────────────────────────────────────────────────────────────
"""

import json, os, math, csv

# ================================================================
# 설정
# ================================================================
RECORDINGS_DIR  = r"C:\Users\local-user\AppData\LocalLow\DefaultCompany\WordsInSilence\Recordings"
OUTPUT_DIR      = r"C:\MyUnityProjects\WordsInSilence\Validation"

# blend_animate.py 확정 설정
SWAP_HAND_LABELS = False   # mp_left->캐릭터LEFT, mp_right->캐릭터RIGHT
L_PALM_FLIP      = False   # effective_pn = pn
R_PALM_FLIP      = True    # effective_pn = -pn

# 스무딩 (blend_animate.py 동일)
SMOOTH_WINDOW       = 5
JUMP_THRESHOLD_NORM = 2.0
HAND_DEPTH_GAIN     = 1.0   # blend_animate.py HAND_DEPTH_GAIN

TARGET_WORDS = ['감사합니다', '나', '모르다', '빵', '먹다', '싫다']

# ── 검토 경고 임계값 (통과/실패 결정 아님 - 검토 플래그용) ────────
WRIST_JUMP_RATIO_WARN  = 0.15    # [B] Pose wrist 이동/어깨너비
ROTATION_DELTA_WARN    = 30.0    # [C][E] 프레임간 방향 변화(도)
ROTATION_SPEED_WARN    = 300.0   # [E] 각속도(deg/s)
PLANE_SIN_MIN          = 0.15    # [D] palm_plane_sin 최소
HAND_SCALE_MIN         = 0.03    # [D] hand_scale 최소 (어깨너비 비율)
NORMAL_DOT_FLIP_THRESH = 0.0     # [D] 연속 법선 dot < 이 값이면 반전 후보
BASIS_FWD_MIN_LEN      = 1e-6    # [E] forward 벡터 최소 길이
BASIS_PROJ_MIN_LEN     = 1e-6    # [E] 투영 후 법선 최소 길이

# MediaPipe landmark 인덱스
P_L_SHOULDER = 11;  P_R_SHOULDER = 12
P_L_WRIST    = 15;  P_R_WRIST    = 16
H_WRIST      = 0;   H_INDEX_MCP  = 5
H_MID_MCP    = 9;   H_PINK_MCP   = 17


# ================================================================
# 벡터 유틸
# ================================================================
def get_pt(pts, idx):
    if not pts or idx >= len(pts): return None
    p = pts[idx]
    return (p.get('x', 0.0), p.get('y', 0.0), p.get('z', 0.0)) if isinstance(p, dict) else None

def v3sub(a, b):  return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def v3len(a):     return math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
def v3dot(a, b):  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def v3norm(a):
    l = v3len(a)
    return (a[0]/l, a[1]/l, a[2]/l) if l > 1e-9 else (0.0, 0.0, 0.0)
def v3cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def angle_deg_safe(a, b):
    """None 반환 = invalid (통계 제외). 영벡터이면 0 대신 None."""
    la, lb = v3len(a), v3len(b)
    if la < 1e-9 or lb < 1e-9: return None
    return math.degrees(math.acos(max(-1.0, min(1.0, v3dot(a, b) / (la * lb)))))

def pct_val(vals, p):
    if not vals: return 0.0
    sv = sorted(vals)
    idx = p / 100.0 * (len(sv) - 1)
    lo = int(idx); hi = min(lo + 1, len(sv) - 1)
    return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])

def _dist_stats(vals):
    if not vals:
        return {'min': 0.0, 'mean': 0.0, 'p5': 0.0, 'max': 0.0, 'count': 0}
    return {
        'min':   round(min(vals), 5),
        'mean':  round(sum(vals) / len(vals), 5),
        'p5':    round(pct_val(vals, 5), 5),
        'max':   round(max(vals), 5),
        'count': len(vals),
    }


# ================================================================
# 3x3 행렬 유틸
# ================================================================
def mat3_from_cols(c0, c1, c2):
    return [[c0[0], c1[0], c2[0]],
            [c0[1], c1[1], c2[1]],
            [c0[2], c1[2], c2[2]]]

def mat3_T(m):
    return [[m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]]]

def mat3_mul(a, b):
    r = [[0.0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3): r[i][j] += a[i][k] * b[k][j]
    return r

def mat3_trace(m): return m[0][0] + m[1][1] + m[2][2]

def mat3_det(m):
    return (m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1])
          - m[0][1] * (m[1][0]*m[2][2] - m[1][2]*m[2][0])
          + m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0]))

def mat3_rotation_angle(R0, R1):
    """두 회전행렬 사이 각도(도). R_rel = R0^T @ R1."""
    R_rel = mat3_mul(mat3_T(R0), R1)
    tr = mat3_trace(R_rel)
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


# ================================================================
# 스무딩 (blend_animate.py smooth_track 재현)
# ================================================================
def smooth1d(vals, window=SMOOTH_WINDOW, jump=JUMP_THRESHOLD_NORM):
    n = len(vals)
    if n == 0: return []
    f = list(vals)
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
    for _ in range(2):
        for i in range(1, n-1):
            avg = (f[i-1] + f[i+1]) / 2
            if abs(f[i] - avg) > jump: f[i] = avg
    half = window // 2
    result = []
    for i in range(n):
        chunk = f[max(0, i-half):min(n, i+half+1)]
        result.append(sum(chunk) / len(chunk))
    return result

def smooth3d(vecs, window=SMOOTH_WINDOW, jump=JUMP_THRESHOLD_NORM):
    xs = smooth1d([v[0] if v else None for v in vecs], window, jump)
    ys = smooth1d([v[1] if v else None for v in vecs], window, jump)
    zs = smooth1d([v[2] if v else None for v in vecs], window, jump)
    return list(zip(xs, ys, zs))


# ================================================================
# 핵심 분석
# ================================================================
def analyze_motion(word, json_path):
    with open(json_path, encoding='utf-8') as f:
        rec = json.load(f)

    frames_data = rec.get('Frames', [])
    res = {
        'word': word, 'json_path': json_path,
        'total_json_frames': len(frames_data),
        'swap_hand_labels': SWAP_HAND_LABELS,
        'l_palm_flip': L_PALM_FLIP,
        'r_palm_flip': R_PALM_FLIP,
    }

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
            """Hand landmark -> 어깨너비 정규화 (HAND_DEPTH_GAIN 적용)."""
            if not valid: return None
            pt = get_pt(pts, idx)
            if pt is None: return None
            return ((pt[0]-cx)/sw, -(pt[1]-cy)/sw, pt[2]*HAND_DEPTH_GAIN/sw)

        def ppt(idx):
            """Pose landmark -> 어깨너비 정규화 (z=0, DEPTH_SCALE=0.0)."""
            pt = get_pt(pose_pts, idx)
            if pt is None: return None
            return ((pt[0]-cx)/sw, -(pt[1]-cy)/sw, 0.0)

        frs.append({
            'ts': ts, 'has_l': hl, 'has_r': hr,
            'pose_lw': ppt(P_L_WRIST), 'pose_rw': ppt(P_R_WRIST),
            'l0':  hpt(H_WRIST,     ch_l, hl), 'l9':  hpt(H_MID_MCP,   ch_l, hl),
            'l5':  hpt(H_INDEX_MCP, ch_l, hl), 'l17': hpt(H_PINK_MCP,  ch_l, hl),
            'r0':  hpt(H_WRIST,     ch_r, hr), 'r9':  hpt(H_MID_MCP,   ch_r, hr),
            'r5':  hpt(H_INDEX_MCP, ch_r, hr), 'r17': hpt(H_PINK_MCP,  ch_r, hr),
        })

    n = len(frs)
    res['pose_frames'] = n
    if n == 0:
        res['error'] = '유효 pose 프레임 없음'
        return res

    # ── A: 검출 통계 ──────────────────────────────────────────
    l_det = sum(1 for fr in frs if fr['has_l'])
    r_det = sum(1 for fr in frs if fr['has_r'])
    both  = sum(1 for fr in frs if fr['has_l'] and fr['has_r'])

    def max_consec_miss(flags):
        best = cur = 0
        for v in flags:
            cur = 0 if v else cur + 1
            best = max(best, cur)
        return best

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
            spans.append({'start': start, 'len': n-start,
                          'ts_start': ts_list[start], 'ts_end': ts_list[-1]})
        return sorted(spans, key=lambda s: -s['len'])[:5]

    ts_all = [fr['ts'] for fr in frs]
    res['A'] = {
        'l_det': l_det, 'l_pct': 100*l_det//n,
        'r_det': r_det, 'r_pct': 100*r_det//n,
        'both':  both,  'both_pct': 100*both//n,
        'l_max_consec_miss': max_consec_miss([fr['has_l'] for fr in frs]),
        'r_max_consec_miss': max_consec_miss([fr['has_r'] for fr in frs]),
        'l_miss_spans': consec_miss_spans([fr['has_l'] for fr in frs], ts_all),
        'r_miss_spans': consec_miss_spans([fr['has_r'] for fr in frs], ts_all),
        'note': '',
    }

    # ── B: Pose wrist 이동량 (z=0 적용됨) ─────────────────────
    def wrist_stats(wkey):
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
                return {'max': 0.0, 'mean': 0.0, 'p95': 0.0, 'max_frame': None, 'max_ts': None}
            mx = max(vals); mi = vals.index(mx)
            return {'max': round(mx, 5), 'mean': round(sum(vals)/len(vals), 5),
                    'p95': round(pct_val(vals, 95), 5), 'max_frame': frames[mi],
                    'max_ts': (ts_list[mi] if ts_list else frs[frames[mi]]['ts'])}

        rst = st(rj, rj_frame, rj_ts)
        sst = st(sj, sj_frame)
        ret = (sst['max'] / rst['max']) if rst['max'] > 1e-9 else 1.0

        # v2 수정: warn_count = 전수, warn_frames = [:10]
        warn = [{'frame': rj_frame[i], 'ts': rj_ts[i], 'ratio': round(rj[i], 4)}
                for i in range(len(rj)) if rj[i] > WRIST_JUMP_RATIO_WARN]
        warn_count  = len(warn)
        warn_frames = warn[:10]

        return {
            'raw': rst, 'smoothed': sst,
            'retention_ratio': round(float(ret), 3),
            'reduction_ratio': round(float(1-ret), 3),
            'warn_count':  warn_count,
            'warn_frames': warn_frames,
        }

    res['B'] = {
        'LEFT':  wrist_stats('pose_lw'),
        'RIGHT': wrist_stats('pose_rw'),
        'metric': 'wrist_jump_ratio = frame-to-frame / shoulder_width (z=0, DEPTH_SCALE=0.0)',
    }

    # ── C: wrist->mid9 방향 안정성 ────────────────────────────
    def fwd_stats(k0, k9, kv):
        fwds = []; invalid_frames = []
        for i, fr in enumerate(frs):
            if fr[kv] and fr[k0] and fr[k9]:
                fraw = v3sub(fr[k9], fr[k0])
                flen = v3len(fraw)
                if flen < 1e-6:
                    invalid_frames.append({'frame': i, 'ts': fr['ts'], 'reason': 'fwd_zero'})
                    fwds.append((fr['ts'], None))
                else:
                    fwds.append((fr['ts'], v3norm(fraw)))
            else:
                fwds.append((fr['ts'], None))

        deltas = []; warns = []
        for i in range(1, len(fwds)):
            _, fv_prev = fwds[i-1]
            ts_cur, fv_cur = fwds[i]
            if fv_prev is None or fv_cur is None: continue
            d = angle_deg_safe(fv_prev, fv_cur)
            if d is None:
                invalid_frames.append({'frame': i, 'ts': ts_cur, 'reason': 'angle_zero'})
                continue
            deltas.append(d)
            if d > ROTATION_DELTA_WARN:
                warns.append({'frame': i, 'ts': ts_cur, 'deg': round(d, 1)})

        return {
            'mean_deg': round(sum(deltas)/len(deltas), 2) if deltas else 0.0,
            'max_deg':  round(max(deltas), 2) if deltas else 0.0,
            'p95_deg':  round(pct_val(deltas, 95), 2) if deltas else 0.0,
            'warn_count':           len(warns),
            'warn_frames':          warns[:10],
            'invalid_frame_count':  len(invalid_frames),
            'invalid_frames':       invalid_frames[:5],
        }

    res['C'] = {
        'LEFT':  fwd_stats('l0', 'l9', 'has_l'),
        'RIGHT': fwd_stats('r0', 'r9', 'has_r'),
        'metric': 'wrist->mid9 연속 방향 변화(도) / hand bone FK forward 안정성',
    }

    # ── D: Palm normal 안정성 (품질 분리 강화) ─────────────────
    def pn_stats(k0, k5, k17, kv):
        """
        palm_area_raw  = |u x v|
        palm_plane_sin = |u x v| / (|u| |v|)   0=측면 1=정면
        hand_scale     = (|u| + |v|) / 2

        flip 고품질 = 양 프레임 모두:
            min(prev_sin, curr_sin)   >= PLANE_SIN_MIN
            min(prev_scale, curr_scale) >= HAND_SCALE_MIN
        """
        normals = []; invalid_frames = []
        for i, fr in enumerate(frs):
            if fr[kv] and fr[k0] and fr[k5] and fr[k17]:
                v05 = v3sub(fr[k5], fr[k0]); v17 = v3sub(fr[k17], fr[k0])
                cross = v3cross(v05, v17)
                cross_len = v3len(cross)
                u_len = v3len(v05); v_len = v3len(v17)
                palm_area_raw  = cross_len
                palm_plane_sin = (cross_len / (u_len*v_len)
                                  if (u_len > 1e-9 and v_len > 1e-9) else 0.0)
                hand_scale     = (u_len + v_len) / 2.0
                if cross_len < 1e-9:
                    nrm = None
                    invalid_frames.append({'frame': i, 'ts': fr['ts'], 'reason': 'cross_zero'})
                else:
                    nrm = v3norm(cross)
                normals.append((fr['ts'], nrm, palm_area_raw, palm_plane_sin, hand_scale))
            else:
                normals.append((fr['ts'], None, 0.0, 0.0, 0.0))

        valid_sins   = [e[3] for e in normals if e[1] is not None and e[3] > 0]
        valid_scales = [e[4] for e in normals if e[1] is not None and e[4] > 0]
        valid_areas  = [e[2] for e in normals if e[1] is not None and e[2] > 0]

        dots = []; flip_total = 0; flip_hq = 0; flip_lq = 0
        flip_hq_details = []; lowq_frames = []
        for i in range(1, len(normals)):
            ts_prev, na, _, prev_sin, prev_scale = normals[i-1]
            ts_cur,  nb, _, curr_sin, curr_scale = normals[i]
            if na is None or nb is None: continue
            d = v3dot(na, nb)
            dots.append(d)
            if d < NORMAL_DOT_FLIP_THRESH:
                flip_total += 1
                pair_sin   = min(prev_sin, curr_sin)
                pair_scale = min(prev_scale, curr_scale)
                is_hq = (pair_sin >= PLANE_SIN_MIN and pair_scale >= HAND_SCALE_MIN)
                if is_hq:
                    flip_hq += 1
                    if len(flip_hq_details) < 10:
                        flip_hq_details.append({
                            'prev_frame': i-1, 'curr_frame': i,
                            'prev_ts': ts_prev, 'curr_ts': ts_cur,
                            'dot': round(d, 4),
                            'prev_sin': round(prev_sin, 4),
                            'curr_sin': round(curr_sin, 4),
                            'pair_sin': round(pair_sin, 4),
                            'prev_scale': round(prev_scale, 4),
                            'curr_scale': round(curr_scale, 4),
                            'pair_scale': round(pair_scale, 4),
                        })
                else:
                    flip_lq += 1
            if nb is not None and (curr_sin < PLANE_SIN_MIN or curr_scale < HAND_SCALE_MIN):
                if len(lowq_frames) < 20:
                    lowq_frames.append({'frame': i, 'ts': ts_cur,
                                        'sin': round(curr_sin, 5),
                                        'scale': round(curr_scale, 5),
                                        'area_raw': round(normals[i][2], 5)})

        return {
            'palm_plane_sin_dist': _dist_stats(valid_sins),
            'hand_scale_dist':     _dist_stats(valid_scales),
            'palm_area_raw_dist':  _dist_stats(valid_areas),
            'mean_dot':            round(sum(dots)/len(dots), 4) if dots else 0.0,
            'min_dot':             round(min(dots), 4) if dots else 0.0,
            'flip_total':          flip_total,
            'flip_high_quality':   flip_hq,
            'flip_low_quality':    flip_lq,
            'flip_hq_details':     flip_hq_details,
            'low_quality_frame_count': len(lowq_frames),
            'low_quality_frames':  lowq_frames[:10],
            'invalid_frame_count': len(invalid_frames),
            'invalid_frames':      invalid_frames[:5],
            'thresholds': {'PLANE_SIN_MIN': PLANE_SIN_MIN, 'HAND_SCALE_MIN': HAND_SCALE_MIN},
            'note': '검토: flip_high_quality>3 또는 sin_dist.p5 낮음 시 palm normal 문제 후보',
        }

    res['D'] = {
        'LEFT':  pn_stats('l0', 'l5', 'l17', 'has_l'),
        'RIGHT': pn_stats('r0', 'r5', 'r17', 'has_r'),
        'metric': 'palm_plane_sin=|cross|/(|u||v|), hand_scale=(|u|+|v|)/2, flip=양프레임 모두 기준 이상',
    }

    # ── E: 회전 변화 (true target basis rotation) ──────────────
    def rot_stats(k0, k9, k5, k17, kv, flip):
        """
        Right-handed target basis per frame:
          Y = normalize(wrist->mid9)
          effective_pn = -pn if flip else pn
          Z = project(effective_pn) onto Y-perp, normalize
          X = Y x Z
        R_relative = R_prev^T @ R_current
        target_basis_rotation_delta = acos(clamp((trace(R_rel)-1)/2))

        basis_valid = landmarks OK AND fwd_len>1e-6 AND Z_len>1e-6
                      AND palm_sin>=PLANE_SIN_MIN AND hand_scale>=HAND_SCALE_MIN
        """
        bases = []; invalid_frames = []
        for i, fr in enumerate(frs):
            ts = fr['ts']
            if not (fr[kv] and fr[k0] and fr[k9] and fr[k5] and fr[k17]):
                bases.append({'ts': ts, 'R': None, 'landmarks_valid': False,
                              'basis_valid': False, 'det': None,
                              'reason': 'landmarks_missing',
                              'palm_sin': 0.0, 'hand_scale': 0.0})
                continue

            wrist = fr[k0]; mid9 = fr[k9]; p5 = fr[k5]; p17 = fr[k17]
            fwd_raw  = v3sub(mid9, wrist)
            fwd_len  = v3len(fwd_raw)
            v05 = v3sub(p5, wrist); v17v = v3sub(p17, wrist)
            cross    = v3cross(v05, v17v)
            cross_len = v3len(cross)
            u_len = v3len(v05); vv_len = v3len(v17v)
            palm_sin   = (cross_len/(u_len*vv_len) if (u_len > 1e-9 and vv_len > 1e-9) else 0.0)
            hand_scale = (u_len + vv_len) / 2.0

            if fwd_len < BASIS_FWD_MIN_LEN:
                invalid_frames.append({'frame': i, 'ts': ts, 'reason': 'fwd_zero'})
                bases.append({'ts': ts, 'R': None, 'landmarks_valid': True,
                              'basis_valid': False, 'det': None, 'reason': 'fwd_zero',
                              'palm_sin': round(palm_sin, 4), 'hand_scale': round(hand_scale, 4)})
                continue
            if cross_len < 1e-9:
                invalid_frames.append({'frame': i, 'ts': ts, 'reason': 'pn_zero'})
                bases.append({'ts': ts, 'R': None, 'landmarks_valid': True,
                              'basis_valid': False, 'det': None, 'reason': 'pn_zero',
                              'palm_sin': round(palm_sin, 4), 'hand_scale': round(hand_scale, 4)})
                continue

            Y   = v3norm(fwd_raw)
            pn  = v3norm(cross)
            eff = (-pn[0], -pn[1], -pn[2]) if flip else pn

            d_dot = v3dot(eff, Y)
            Z_raw = (eff[0]-d_dot*Y[0], eff[1]-d_dot*Y[1], eff[2]-d_dot*Y[2])
            Z_len = v3len(Z_raw)
            if Z_len < BASIS_PROJ_MIN_LEN:
                invalid_frames.append({'frame': i, 'ts': ts, 'reason': 'pn_parallel_fwd'})
                bases.append({'ts': ts, 'R': None, 'landmarks_valid': True,
                              'basis_valid': False, 'det': None, 'reason': 'pn_parallel_fwd',
                              'palm_sin': round(palm_sin, 4), 'hand_scale': round(hand_scale, 4)})
                continue

            Z = v3norm(Z_raw)
            X = v3cross(Y, Z)
            M = mat3_from_cols(X, Y, Z)
            det = mat3_det(M)
            bv  = (palm_sin >= PLANE_SIN_MIN and hand_scale >= HAND_SCALE_MIN)
            bases.append({'ts': ts, 'R': M, 'landmarks_valid': True,
                          'basis_valid': bv, 'det': round(det, 4),
                          'reason': None if bv else 'low_quality',
                          'palm_sin': round(palm_sin, 4),
                          'hand_scale': round(hand_scale, 4)})

        deltas = []; speeds = []; warns = []
        for i in range(1, len(bases)):
            b0 = bases[i-1]; b1 = bases[i]
            if not (b0['basis_valid'] and b1['basis_valid']): continue
            if b0['R'] is None or b1['R'] is None: continue
            delta = mat3_rotation_angle(b0['R'], b1['R'])
            deltas.append(delta)
            ts0 = b0['ts']; ts1 = b1['ts']
            dt = ((ts1-ts0)/1000.0
                  if (ts0 is not None and ts1 is not None and ts1 > ts0) else None)
            spd = (delta / dt) if dt else None
            if spd is not None: speeds.append(spd)
            if delta > ROTATION_DELTA_WARN or (spd and spd > ROTATION_SPEED_WARN):
                warns.append({
                    'frame': i, 'ts': ts1, 'ts_prev': ts0,
                    'target_basis_rotation_delta': round(delta, 1),
                    'speed_degs': round(spd, 0) if spd else None,
                    'landmarks_valid_curr':   b1['landmarks_valid'],
                    'basis_valid_curr':       b1['basis_valid'],
                    'basis_determinant_prev': b0.get('det'),
                    'basis_determinant_curr': b1.get('det'),
                })

        warn_count  = len(warns)
        warn_frames = warns[:10]
        vspd = [s for s in speeds if s is not None]
        return {
            'mean_delta_deg':    round(sum(deltas)/len(deltas), 2) if deltas else 0.0,
            'max_delta_deg':     round(max(deltas), 2) if deltas else 0.0,
            'p95_delta_deg':     round(pct_val(deltas, 95), 2) if deltas else 0.0,
            'mean_speed_degs':   round(sum(vspd)/len(vspd), 1) if vspd else 0.0,
            'max_speed_degs':    round(max(vspd), 1) if vspd else 0.0,
            'p95_speed_degs':    round(pct_val(vspd, 95), 1) if vspd else 0.0,
            'warn_count':        warn_count,
            'warn_frames':       warn_frames,
            'invalid_frame_count': len(invalid_frames),
            'invalid_frames':    invalid_frames[:10],
            'basis_valid_count': sum(1 for b in bases if b['basis_valid']),
        }

    res['E'] = {
        'LEFT':  rot_stats('l0','l9','l5','l17','has_l', L_PALM_FLIP),
        'RIGHT': rot_stats('r0','r9','r5','r17','has_r', R_PALM_FLIP),
        'metric': 'true target basis R: Y=fwd Z=proj(eff_pn) X=YxZ; delta=acos((tr(R_rel)-1)/2)',
        'L_PALM_FLIP': L_PALM_FLIP, 'R_PALM_FLIP': R_PALM_FLIP,
    }

    # F, G (수동 입력)
    res['F'] = {
        'LEFT_index': {
            'implementation_error': {'fwd_err_deg': 'PENDING', 'roll_err_deg': 'PENDING'},
            'source_fidelity': 'PENDING',
            'measure_method': 'TEST_MODE=True, diag_index_finger(LEFT)',
        },
        'RIGHT_index': {
            'implementation_error': {'fwd_err_deg': 'PENDING', 'roll_err_deg': 'PENDING'},
            'source_fidelity': 'PENDING',
            'measure_method': 'TEST_MODE=True, diag_index_finger(RIGHT)',
        },
    }
    res['G'] = {
        'source_fidelity': {
            'hand_assignment':   'PENDING',
            'left_palm_dir':     'PENDING',
            'right_palm_dir':    'PENDING',
            'flip_180_frames':   'PENDING',
            'wrist_bend_issues': 'PENDING',
            'overall':           'PENDING',
        },
        'problem_classification': {
            'input_problem':     'PENDING',
            'pipeline_problem':  'PENDING',
            'character_problem': 'PENDING',
        },
        'notes': '',
    }
    return res


# ================================================================
# 보고서 출력
# ================================================================
SEP  = '-' * 66
SEP2 = '=' * 66

def print_report(res):
    print('\n' + SEP2)
    print('동작: {}  | JSON: {}'.format(res['word'], os.path.basename(res['json_path'])))
    print('총 JSON 프레임: {}  |  유효 Pose 프레임: {}'.format(
          res.get('total_json_frames'), res.get('pose_frames')))
    if 'error' in res:
        print('[오류] {}'.format(res['error']))
        return

    A = res['A']
    print('\n[A] 검출 통계  (SWAP_HAND_LABELS={})'.format(SWAP_HAND_LABELS))
    print('  캐릭터LEFT  (mp_{}): {}f ({}%)  최대연속누락: {}f'.format(
          'right' if SWAP_HAND_LABELS else 'left',
          A['l_det'], A['l_pct'], A['l_max_consec_miss']))
    print('  캐릭터RIGHT (mp_{}): {}f ({}%)  최대연속누락: {}f'.format(
          'left' if SWAP_HAND_LABELS else 'right',
          A['r_det'], A['r_pct'], A['r_max_consec_miss']))
    print('  양손 동시: {}f ({}%)'.format(A['both'], A['both_pct']))

    print('\n[B] Pose wrist 이동량  (z=0, wrist_jump_ratio = 이동/어깨너비)')
    for side in ['LEFT', 'RIGHT']:
        B = res['B'][side]
        r = B['raw']; s = B['smoothed']
        print('  {}  raw: max={:.4f} mean={:.4f} p95={:.4f}  '
              'smooth: max={:.4f}  retention={:.3f}  warn_count={}'.format(
              side, r['max'], r['mean'], r['p95'],
              s['max'], B['retention_ratio'], B['warn_count']))
        for w in B['warn_frames'][:2]:
            print('    [검토] frame={} ts={}ms ratio={:.4f}'.format(
                  w['frame'], w['ts'], w['ratio']))

    print('\n[C] wrist->mid9 방향 안정성')
    for side in ['LEFT', 'RIGHT']:
        C = res['C'][side]
        print('  {}  mean={:.1f} max={:.1f} p95={:.1f}  warn={} invalid={}'.format(
              side, C['mean_deg'], C['max_deg'], C['p95_deg'],
              C['warn_count'], C['invalid_frame_count']))

    print('\n[D] Palm normal 안정성  (PLANE_SIN_MIN={} HAND_SCALE_MIN={})'.format(
          PLANE_SIN_MIN, HAND_SCALE_MIN))
    for side in ['LEFT', 'RIGHT']:
        D = res['D'][side]
        sd = D['palm_plane_sin_dist']; sc = D['hand_scale_dist']
        print('  {}  sin: min={:.3f} mean={:.3f} p5={:.3f}  '
              'scale: min={:.3f} mean={:.3f}'.format(
              side, sd['min'], sd['mean'], sd['p5'], sc['min'], sc['mean']))
        print('    flip: total={}  hq(검토)={}  lq={}  '
              'lowq_frames={}  invalid={}'.format(
              D['flip_total'], D['flip_high_quality'], D['flip_low_quality'],
              D['low_quality_frame_count'], D['invalid_frame_count']))
        for det in D['flip_hq_details'][:2]:
            print('    [flip_hq] prev={} curr={} pair_sin={:.3f} pair_scale={:.3f} dot={:.3f}'.format(
                  det['prev_frame'], det['curr_frame'],
                  det['pair_sin'], det['pair_scale'], det['dot']))

    print('\n[E] 회전 변화  (true target basis, L_PALM_FLIP={} R_PALM_FLIP={})'.format(
          L_PALM_FLIP, R_PALM_FLIP))
    for side in ['LEFT', 'RIGHT']:
        E = res['E'][side]
        print('  {}  delta: mean={:.1f} max={:.1f} p95={:.1f}  '
              'speed: mean={:.0f} max={:.0f} p95={:.0f} deg/s'.format(
              side, E['mean_delta_deg'], E['max_delta_deg'], E['p95_delta_deg'],
              E['mean_speed_degs'], E['max_speed_degs'], E['p95_speed_degs']))
        print('    warn={}  invalid={}  basis_valid={}'.format(
              E['warn_count'], E['invalid_frame_count'], E['basis_valid_count']))
        for w in E['warn_frames'][:2]:
            print('    [검토] frame={} ts={}ms delta={} speed={}  '
                  'lv={} bv={} det_c={}'.format(
                  w['frame'], w['ts'],
                  w['target_basis_rotation_delta'], w['speed_degs'],
                  w['landmarks_valid_curr'], w['basis_valid_curr'],
                  w['basis_determinant_curr']))


# ================================================================
# CSV / JSON 저장
# ================================================================
CSV_FIELDS_V2 = [
    'word', 'pose_frames',
    'A_l_pct', 'A_r_pct', 'A_both_pct', 'A_l_max_miss', 'A_r_max_miss',
    'B_L_raw_max', 'B_L_raw_p95', 'B_L_sm_max', 'B_L_sm_p95', 'B_L_retention', 'B_L_warn_count',
    'B_R_raw_max', 'B_R_raw_p95', 'B_R_sm_max', 'B_R_sm_p95', 'B_R_retention', 'B_R_warn_count',
    'C_L_mean', 'C_L_max', 'C_L_p95', 'C_L_warn', 'C_L_invalid',
    'C_R_mean', 'C_R_max', 'C_R_p95', 'C_R_warn', 'C_R_invalid',
    'D_L_sin_p5', 'D_L_sin_mean', 'D_L_scale_mean', 'D_L_min_dot',
    'D_L_flip_hq', 'D_L_flip_lq', 'D_L_lowq', 'D_L_invalid',
    'D_R_sin_p5', 'D_R_sin_mean', 'D_R_scale_mean', 'D_R_min_dot',
    'D_R_flip_hq', 'D_R_flip_lq', 'D_R_lowq', 'D_R_invalid',
    'E_L_max_delta', 'E_L_max_speed', 'E_L_p95_speed', 'E_L_warn', 'E_L_invalid', 'E_L_basis_valid',
    'E_R_max_delta', 'E_R_max_speed', 'E_R_p95_speed', 'E_R_warn', 'E_R_invalid', 'E_R_basis_valid',
    'F_L_fwd_err', 'F_L_roll_err', 'F_R_fwd_err', 'F_R_roll_err',
    'G_hand_assign', 'G_left_palm', 'G_right_palm', 'G_flip_180',
    'G_wrist_bend', 'G_overall',
    'G_input_problem', 'G_pipeline_problem', 'G_character_problem', 'G_notes',
]


def to_csv_row_v2(res, manual_g=None):
    if 'error' in res:
        return {'word': res['word'], 'pose_frames': 0}
    A  = res.get('A', {}); B = res.get('B', {}); C = res.get('C', {})
    D  = res.get('D', {}); E = res.get('E', {}); G = res.get('G', {})
    BL = B.get('LEFT', {}); BR = B.get('RIGHT', {})
    CL = C.get('LEFT', {}); CR = C.get('RIGHT', {})
    DL = D.get('LEFT', {}); DR = D.get('RIGHT', {})
    EL = E.get('LEFT', {}); ER = E.get('RIGHT', {})
    # G: manual override if provided
    if manual_g:
        Gsf = manual_g.get('source_fidelity', {})
        Gpc = manual_g.get('problem_classification', {})
        Gn  = manual_g.get('notes', '')
    else:
        Gsf = G.get('source_fidelity', {}); Gpc = G.get('problem_classification', {})
        Gn  = G.get('notes', '')

    return {
        'word': res['word'], 'pose_frames': res['pose_frames'],
        'A_l_pct': A.get('l_pct',''), 'A_r_pct': A.get('r_pct',''),
        'A_both_pct': A.get('both_pct',''),
        'A_l_max_miss': A.get('l_max_consec_miss',''),
        'A_r_max_miss': A.get('r_max_consec_miss',''),
        'B_L_raw_max':  BL.get('raw',{}).get('max',''),
        'B_L_raw_p95':  BL.get('raw',{}).get('p95',''),
        'B_L_sm_max':   BL.get('smoothed',{}).get('max',''),
        'B_L_sm_p95':   BL.get('smoothed',{}).get('p95',''),
        'B_L_retention': BL.get('retention_ratio',''),
        'B_L_warn_count': BL.get('warn_count',''),
        'B_R_raw_max':  BR.get('raw',{}).get('max',''),
        'B_R_raw_p95':  BR.get('raw',{}).get('p95',''),
        'B_R_sm_max':   BR.get('smoothed',{}).get('max',''),
        'B_R_sm_p95':   BR.get('smoothed',{}).get('p95',''),
        'B_R_retention': BR.get('retention_ratio',''),
        'B_R_warn_count': BR.get('warn_count',''),
        'C_L_mean': CL.get('mean_deg',''), 'C_L_max': CL.get('max_deg',''),
        'C_L_p95':  CL.get('p95_deg',''),  'C_L_warn': CL.get('warn_count',''),
        'C_L_invalid': CL.get('invalid_frame_count',''),
        'C_R_mean': CR.get('mean_deg',''), 'C_R_max': CR.get('max_deg',''),
        'C_R_p95':  CR.get('p95_deg',''),  'C_R_warn': CR.get('warn_count',''),
        'C_R_invalid': CR.get('invalid_frame_count',''),
        'D_L_sin_p5':    DL.get('palm_plane_sin_dist',{}).get('p5',''),
        'D_L_sin_mean':  DL.get('palm_plane_sin_dist',{}).get('mean',''),
        'D_L_scale_mean':DL.get('hand_scale_dist',{}).get('mean',''),
        'D_L_min_dot':   DL.get('min_dot',''),
        'D_L_flip_hq':   DL.get('flip_high_quality',''),
        'D_L_flip_lq':   DL.get('flip_low_quality',''),
        'D_L_lowq':      DL.get('low_quality_frame_count',''),
        'D_L_invalid':   DL.get('invalid_frame_count',''),
        'D_R_sin_p5':    DR.get('palm_plane_sin_dist',{}).get('p5',''),
        'D_R_sin_mean':  DR.get('palm_plane_sin_dist',{}).get('mean',''),
        'D_R_scale_mean':DR.get('hand_scale_dist',{}).get('mean',''),
        'D_R_min_dot':   DR.get('min_dot',''),
        'D_R_flip_hq':   DR.get('flip_high_quality',''),
        'D_R_flip_lq':   DR.get('flip_low_quality',''),
        'D_R_lowq':      DR.get('low_quality_frame_count',''),
        'D_R_invalid':   DR.get('invalid_frame_count',''),
        'E_L_max_delta':   EL.get('max_delta_deg',''),
        'E_L_max_speed':   EL.get('max_speed_degs',''),
        'E_L_p95_speed':   EL.get('p95_speed_degs',''),
        'E_L_warn':        EL.get('warn_count',''),
        'E_L_invalid':     EL.get('invalid_frame_count',''),
        'E_L_basis_valid': EL.get('basis_valid_count',''),
        'E_R_max_delta':   ER.get('max_delta_deg',''),
        'E_R_max_speed':   ER.get('max_speed_degs',''),
        'E_R_p95_speed':   ER.get('p95_speed_degs',''),
        'E_R_warn':        ER.get('warn_count',''),
        'E_R_invalid':     ER.get('invalid_frame_count',''),
        'E_R_basis_valid': ER.get('basis_valid_count',''),
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
        'G_notes':            Gn,
    }


def save_summary_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS_V2, extrasaction='ignore')
        w.writeheader()
        for res in results:
            w.writerow(to_csv_row_v2(res))
    print('[저장] {}'.format(path))


def save_details_json(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('[저장] {}'.format(path))


# ================================================================
# Manual JSON template (없을 때만 생성)
# ================================================================
MANUAL_G_TEMPLATE = {
    'source_fidelity': {
        'hand_assignment':   'PENDING',
        'left_palm_dir':     'PENDING',
        'right_palm_dir':    'PENDING',
        'flip_180_frames':   'PENDING',
        'wrist_bend_issues': 'PENDING',
        'overall':           'PENDING',
    },
    'problem_classification': {
        'input_problem':     'PENDING',
        'pipeline_problem':  'PENDING',
        'character_problem': 'PENDING',
    },
    'notes': '',
}


def create_manual_template_if_absent(words, manual_path):
    if os.path.exists(manual_path):
        print('[수동 판정] 기존 파일 보존: {}'.format(manual_path))
        return
    os.makedirs(os.path.dirname(manual_path), exist_ok=True)
    template = {
        '_description': ('수동 시각 판정 결과. auto _v2.json 과 병합해'
                         ' motion_validation_merged.csv 생성.'),
        '_instructions': 'G 섹션을 Blender 확인 후 직접 편집. 이 파일은 존재하면 덮어쓰지 않음.',
        'words': {w: {'G': dict(MANUAL_G_TEMPLATE)} for w in words},
    }
    with open(manual_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print('[생성] manual template: {}'.format(manual_path))


def save_merged_csv(results, manual_path, merged_path):
    """auto 결과 + manual G 를 병합해 CSV 저장."""
    manual_words = {}
    if os.path.exists(manual_path):
        with open(manual_path, encoding='utf-8') as f:
            mdata = json.load(f)
        for word, wdata in mdata.get('words', {}).items():
            manual_words[word] = wdata.get('G', {})

    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS_V2, extrasaction='ignore')
        w.writeheader()
        for res in results:
            mg = manual_words.get(res['word'], None)
            w.writerow(to_csv_row_v2(res, manual_g=mg))
    print('[저장] merged: {}'.format(merged_path))


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
    print('validate_motion_batch_v2.py')
    print('  SWAP={} L_FLIP={} R_FLIP={}'.format(
          SWAP_HAND_LABELS, L_PALM_FLIP, R_PALM_FLIP))
    print('  RECORDINGS: {}'.format(RECORDINGS_DIR))
    print('  OUTPUT    : {}'.format(OUTPUT_DIR))
    print(SEP2)

    all_recs = list_recordings(RECORDINGS_DIR)
    print('\n[사용 가능한 JSON] ({}개)'.format(len(all_recs)))
    for word, _ in all_recs:
        marker = ' <-- 대상' if (TARGET_WORDS and word in TARGET_WORDS) else ''
        print('  {}{}'.format(word, marker))

    if TARGET_WORDS is not None:
        targets = [(w, p) for w, p in all_recs if w in TARGET_WORDS]
        missing = set(TARGET_WORDS) - {w for w, _ in targets}
        if missing:
            print('\n[경고] TARGET_WORDS에 있으나 JSON 없음: {}'.format(sorted(missing)))
    else:
        targets = all_recs

    print('\n[분석 대상] ({}개): {}\n'.format(len(targets), [w for w, _ in targets]))

    results = []
    for word, path in targets:
        print('  분석 중: {} ...'.format(word))
        try:
            res = analyze_motion(word, path)
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {'word': word, 'json_path': path,
                   'total_json_frames': 0, 'pose_frames': 0, 'error': str(e)}
        results.append(res)
        print_report(res)

    # 출력 파일 경로
    csv_path    = os.path.join(OUTPUT_DIR, 'motion_validation_summary_v2.csv')
    json_path   = os.path.join(OUTPUT_DIR, 'motion_validation_details_v2.json')
    manual_path = os.path.join(OUTPUT_DIR, 'motion_validation_manual.json')
    merged_path = os.path.join(OUTPUT_DIR, 'motion_validation_merged.csv')

    save_summary_csv(results, csv_path)
    save_details_json(results, json_path)
    create_manual_template_if_absent([w for w, _ in targets], manual_path)
    save_merged_csv(results, manual_path, merged_path)

    # 요약 비교 표
    print('\n' + SEP2)
    print('v2 자동 수치 요약')
    print('  {:<12} {:>4} {:>4} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(
          '단어', 'L%', 'R%', 'both%',
          'BL_rmax', 'BR_rmax',
          'DL_sin_p5', 'DR_sin_p5',
          'DL_fhq', 'DR_fhq',
          'EL_max', 'ER_max'))
    for res in results:
        if 'error' in res: continue
        A=res['A']; B=res['B']; D=res['D']; E=res['E']
        DL=D['LEFT']; DR_=D['RIGHT']
        print('  {:<12} {:>4} {:>4} {:>6} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8} {:>8} {:>8.1f} {:>8.1f}'.format(
              res['word'],
              A['l_pct'], A['r_pct'], A['both_pct'],
              B['LEFT']['raw']['max'], B['RIGHT']['raw']['max'],
              DL['palm_plane_sin_dist']['p5'], DR_['palm_plane_sin_dist']['p5'],
              DL['flip_high_quality'], DR_['flip_high_quality'],
              E['LEFT']['max_delta_deg'], E['RIGHT']['max_delta_deg']))

    print('\n' + SEP)
    print('[v2 vs v1 주요 변경점]')
    print('  B: Pose z=0 적용 (v1은 z 포함) -> wrist 이동량 변화 있을 수 있음')
    print('  D: palm_plane_sin(0~1) 추가, flip_hq = 양 프레임 모두 기준 이상')
    print('  E: true 3x3 basis rotation (v1은 max(fwd_delta, pn_delta) 근사)')
    print('  E: invalid_frame_count, basis_valid_count 추가')
    print('  출력: _v2.csv/_v2.json(auto) + manual.json(템플릿) + merged.csv')
    print('\n[출력 파일]')
    print('  {}'.format(csv_path))
    print('  {}'.format(json_path))
    print('  {}'.format(manual_path))
    print('  {}'.format(merged_path))
    print(SEP2)


main()
