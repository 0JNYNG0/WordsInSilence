"""Independent evidence-guard verification helpers; never rerun public paths.

The final saved-result entry point is added once the producer schema is fixed.
No producer import is used for the raw mask/count arithmetic below.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

U = Path(__file__).resolve().parents[1]
FAMILIES = ['R_shape', 'L_shape', 'R_angles', 'L_angles', 'hand_relation', 'arm_path', 'hand_axis']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def independent_masks(a, names):
    """Rebuild frozen observed support from stored69 mask and stored7 heads."""
    direct = np.asarray(a['mask'], bool) & np.isfinite(a['x'])
    post = np.asarray(a['probabilities'])
    times = np.asarray(a['timestamp_us'], float) / 1e6
    out = {}
    for side, cols in [('R', [0, 1, 3]), ('L', [2, 4])]:
        angle = [i for i, name in enumerate(names)
                 if name.startswith(side + '_') and '_angle' in name and 'elbow' not in name]
        wrist = [names.index(side + '_hand_wrist_' + d) for d in ['x', 'y']]
        native = np.asarray([all(direct[i, wrist]) and sum(direct[i, angle]) >= 10
                             for i in range(len(times))], bool)
        supported = native.copy()
        for start, stop in zip(np.flatnonzero(native)[:-1], np.flatnonzero(native)[1:]):
            if times[stop] - times[start] <= .2:
                supported[start:stop + 1] = True
        valid = np.isfinite(post[:, cols]) & supported[:, None]
        conflict = np.sum(post[:, cols] >= .5, axis=1) > 1
        valid[conflict] = False
        out[side + '_shape'] = valid
        out[side + '_angles'] = direct[:, angle]
    out['hand_relation'] = direct[:, [names.index(n) for n in
                                     ['hands_wrist_delta_x', 'hands_wrist_delta_y', 'hands_wrist_distance']]]
    out['arm_path'] = direct[:, [names.index(side + '_' + site + '_' + d)
                                 for side in ['L', 'R'] for site in ['pose_wrist', 'elbow'] for d in ['x', 'y']]]
    out['hand_axis'] = direct[:, [names.index(side + '_palm_axis_' + d)
                                  for side in ['L', 'R'] for d in ['x', 'y']]]
    return out


def independent_scope(row, masks):
    keep = {name: np.ones_like(masks[name], bool) for name in FAMILIES}
    for stage in row['stages']:
        lo, hi = stage['frames']
        for side in stage.get('confirmed_rest_sides', []):
            keep[side + '_shape'][lo:hi + 1] = False
            keep[side + '_angles'][lo:hi + 1] = False
            for column in ([0, 1] if side == 'L' else [2, 3]):
                keep['hand_axis'][lo:hi + 1, column] = False
        if stage.get('confirmed_rest_sides'):
            keep['hand_relation'][lo:hi + 1] = False
    return keep


def independent_support(qmask, rmask, frames, links, shape=False):
    """Set/cell enumeration independent of the producer's vector reduction."""
    dim = rmask.shape[1]
    links = {(int(q), int(r)) for q, r in links if int(r) in set(frames)}
    queries = sorted({q for q, _ in links})
    reference_expected = {(int(r), d) for r in frames for d in range(dim) if rmask[r, d]}
    reference_observed = {(r, d) for r, d in reference_expected
                          if any(qmask[q, d] for q, rr in links if rr == r)}
    query_expected = {(q, d) for q in queries for d in range(dim)
                      if any(rmask[r, d] for qq, r in links if qq == q)}
    query_observed = {(q, d) for q, d in query_expected if qmask[q, d]}
    de = [sum(d == k for _, d in reference_expected) for k in range(dim)]
    do = [sum(d == k for _, d in reference_observed) for k in range(dim)]
    qe = [sum(d == k for _, d in query_expected) for k in range(dim)]
    qo = [sum(d == k for _, d in query_observed) for k in range(dim)]
    rc = len(reference_observed) / len(reference_expected) if reference_expected else None
    qc = len(query_observed) / len(query_expected) if query_expected else None
    dc = [min(do[k] / max(de[k], 1), qo[k] / max(qe[k], 1)) for k in range(dim)]
    cov = min(rc, qc) if rc is not None and qc is not None else None
    if shape and reference_expected and query_expected:
        cov = min(dc)
    return dict(expected=len(reference_expected), observed=len(reference_observed),
        query_expected=len(query_expected), query_observed=len(query_observed),
        reference_coverage=rc, query_coverage=qc, coverage=cov,
        dimension_reference_expected=de, dimension_reference_observed=do,
        dimension_query_expected=qe, dimension_query_observed=qo, dimension_coverage=dc,
        query_indices=queries, reference_indices=list(frames), unique_links=sorted(links))


def complete_path(path, nq, nr):
    p = np.asarray(path, int)
    return bool(p.ndim == 2 and p.shape[1] == 2 and np.array_equal(p[0], [0, 0]) and
        np.array_equal(p[-1], [nq - 1, nr - 1]) and len(np.unique(p[:, 0])) == nq and
        len(np.unique(p[:, 1])) == nr and all(tuple(d) in [(1, 0), (0, 1), (1, 1)] for d in np.diff(p, axis=0)))
