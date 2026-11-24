"""Pinned cached observations and weak local-event supervision; no model calls.

The native timeline is never cropped for input preparation. Bags select a fixed
readout scope from one future full-native forward; they do not create dense
per-frame labels. All normalized arrays exist in memory only.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
O = U.parent
P = O / 'word100_learned_guide_feedback_20261102'
M = O / 'word100_learned_motion_order_20261102'
R = O / 'word100_learned_rgb_motion_20261102'
N = O / 'word100_learned_motion_invariance_20261102'
V = O / 'word100_learned_body_relation_20261102'
OLD = O / 'word100_multipeople_live_feedback_20261101'
AG = O / 'word100_context_episode_query_20261103'
AC = O / 'word100_process_contract_alignment_20261102'
WORDS = ['WORD0739', 'WORD1133', 'WORD1567', 'WORD2523']
TARGETS = ['right_hand_toward_left_wrist_along_arm', 'hands_move_apart',
           'right_hand_rises_relative_left', 'right_hand_descends_face_to_torso',
           'right_hand_falls_relative_left']
FIT_IDS = list(range(1, 10))
DEV_IDS = [11, 12, 13]
INPUT_DIM = 915
TCN_RADIUS = 7

for folder in [M, R, AC, AG]:
    sys.path.insert(0, str(folder))
from episode_query import prepare_input
from motion_model import features

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def geometry_valid():
    keep = read(N / 'TRAINING_CONTRACT.json')['motion_value_indices']
    if len(keep) != len(set(keep)) or any(type(i) is not int or not 0 <= i < 73 for i in keep):
        raise ValueError('Invalid frozen motion geometry selection')
    valid = np.zeros(146, np.float32)
    valid[keep] = 1
    valid[np.asarray(keep) + 73] = 1
    return valid

def validate_role(row):
    expected = 'fit' if row['person'] in FIT_IDS else 'development' if row['person'] in DEV_IDS else None
    if row['role'] != expected or row['word_id'] not in WORDS:
        raise ValueError('Outside the frozen public source/registered-ID scope')
    if row['clip_id'] != f"NIA_SL_{row['word_id']}_REAL{row['person']:02d}_F":
        raise ValueError('Clip/registered ID mismatch')

def observation_paths(row):
    validate_role(row)
    folder = 'bank_predictions' if row['role'] == 'fit' else 'baseline_probabilities'
    return (OLD / folder / (row['clip_id'] + '.npz'),
            V / 'runs/rgb_geometry_42/predictions' / (row['clip_id'] + '.npz'))

def recipe_for(row, a, public_manifest, public_sha, code_pins):
    """Verify actual original generator, including registered DEV cached rows.

    The AG helper had a FIT-only task assertion; this task explicitly includes
    registered DEV11..13. Their own exact manifest records are checked rather
    than applying a FIT record or inventing saved window values.
    """
    keys = {'window_starts', 'window_center_us', 'window_features'}
    present = keys.intersection(a)
    if present:
        if present != keys:
            raise ValueError('Incomplete cached encoder window metadata')
        return None
    old = [r for r in public_manifest['rows'] if r['clip_id'] == row['clip_id']]
    if len(old) != 1:
        raise ValueError('Missing unambiguous encoder generator provenance')
    old = old[0]
    validate_role(old)
    for key in ['clip_id', 'word_id', 'person', 'role', 'source_sha256', 'input_sha256']:
        if old[key] != row[key]:
            raise ValueError('Generator/input provenance mismatch: ' + key)
    if Path(old['input']).resolve() != Path(row['input']).resolve():
        raise ValueError('Generator input path differs')
    if old['frames'] != len(a['frame_id']) or old['reused_input'] is not False:
        raise ValueError('No direct original generator recipe for this cache')
    return dict(route='public_inputs_16_stride2_center_interp', frames=old['frames'],
                manifest_sha256=public_sha, code_pins=code_pins,
                manifest_clip_id=old['clip_id'], role=old['role'],
                source_sha256=old['source_sha256'], input_sha256=old['input_sha256'],
                saved_window_values_verified=False,
                limitation='Window support recovered from pinned original generator; missing window vectors cannot be numerically re-interpolated.')

def fragments(observed):
    edges = np.diff(np.r_[False, observed, False].astype(np.int8))
    return [[int(s), int(e)-1] for s,e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]

def prepare_row(row, stats, rgb_stats, recipe, valid):
    """Read caches only. No checkpoint loads, encoder, detector or head forward."""
    with np.load(row['input'], allow_pickle=False) as f:
        a = {k: np.array(f[k], copy=True) for k in f.files}
    sp, rp = observation_paths(row)
    with np.load(sp, allow_pickle=False) as f:
        s = {k: np.array(f[k], copy=True) for k in f.files}
    with np.load(rp, allow_pickle=False) as f:
        r = {k: np.array(f[k], copy=True) for k in f.files}
    for auxiliary in [s, r]:
        for key in ['frame_id', 'timestamp_us']:
            if not np.array_equal(a[key], auxiliary[key]):
                raise ValueError('Cached shape/relation exact native-axis mismatch')
    n = len(a['frame_id'])
    if not np.array_equal(a['frame_id'], np.arange(n)):
        raise ValueError('Expected complete zero-based native source axis')
    if a['x'].shape != (n,69) or a['mask'].shape != (n,69) or a['mask'].dtype.kind != 'b':
        raise ValueError('Cached feature or validity dimensions changed')
    if s['probabilities'].shape != (n,7) or r['probability'].shape != (n,3):
        raise ValueError('Cached auxiliary feature shape mismatch')
    if 'posterior_available' in s:
        if not np.array_equal(s['posterior_available'], np.isfinite(s['probabilities'])):
            raise ValueError('Finite cached posterior would contradict its availability flag')
    z, mask = features(a, s['probabilities'], r['probability'])
    prepared = prepare_input(a, z, mask, stats, rgb_stats, recipe)
    if prepared['x'].shape != (n, INPUT_DIM):
        raise ValueError('Prepared input must be native rows x915')
    selected_valid = prepared['mask'] & valid[:73].astype(bool)[None,:]
    observed = prepared['available'] | selected_valid.any(1)
    fs = fragments(observed)
    anchor = np.zeros(n, bool)
    for lo,hi in fs:
        if hi > lo:
            anchor[lo:hi+1] = True
    prepared.update(geometry_valid=valid.copy(), evidence_available=observed,
                    observed_fragments=fs, motion_anchor=anchor)
    return prepared

def indices_for(prepared, frames):
    if len(frames) != 2 or any(type(f) is not int for f in frames) or frames[1] < frames[0]:
        raise ValueError('Expected inclusive fixed native endpoints')
    ids = prepared['ids']
    ix = np.flatnonzero((ids >= frames[0]) & (ids <= frames[1]))
    if not len(ix) or not np.array_equal(ids[ix], np.arange(frames[0], frames[1]+1)):
        raise ValueError('Native bag frames absent or discontinuous')
    return [int(ix[0]), int(ix[-1])]

def evidence_for(prepared, indices, witness_frames):
    lo,hi = indices
    observed = prepared['evidence_available']
    witness_rows = [int(np.flatnonzero(prepared['ids'] == f)[0]) for f in witness_frames]
    containing = [f for f in prepared['observed_fragments'] if f[0] <= lo and hi <= f[1]]
    complete = bool(hi > lo and observed[lo:hi+1].all() and
                    prepared['motion_anchor'][lo:hi+1].all() and len(containing) == 1)
    intact = bool(complete and witness_rows and all(containing[0][0] <= i <= containing[0][1] for i in witness_rows))
    return dict(evidence_complete=complete, witness_complete_one_observed_fragment=intact,
                containing_observed_fragment=containing[0] if containing else None,
                unavailable_native_frames=prepared['ids'][lo:hi+1][~observed[lo:hi+1]].tolist(),
                no_motion_anchor_native_frames=prepared['ids'][lo:hi+1][~prepared['motion_anchor'][lo:hi+1]].tolist(),
                geometry_coverage=float(prepared['mask'][lo:hi+1].mean()),
                rgb_coverage=float(prepared['available'][lo:hi+1].mean()))

def composed_support(prepared, indices):
    """Structural TCN(15 rows) + encoder(16/stride2/interp) support, not truth."""
    lo,hi = indices
    nominal = list(range(lo,hi+1))
    tcn = set()
    for start,end in prepared['observed_fragments']:
        for i in range(max(lo,start), min(hi,end)+1):
            if prepared['motion_anchor'][i]:
                tcn.update(range(max(start,i-TCN_RADIUS), min(end,i+TCN_RADIUS)+1))
    tcn = sorted(tcn)
    rgb = sorted({j for i in tcn for j in prepared['support']['row_consumed_support'][i]})
    ids, ts = prepared['ids'], prepared['ts']
    outside = sorted(set(rgb) - set(nominal))
    return dict(nominal_indices=indices, nominal_native_frames=ids[nominal].tolist(),
                nominal_timestamp_us=ts[nominal].tolist(),
                tcn_input_indices=tcn, tcn_native_frames=ids[tcn].tolist(),
                rgb_input_indices=rgb, rgb_native_frames=ids[rgb].tolist(),
                rgb_timestamp_us=ts[rgb].tolist(),
                rgb_outside_nominal_native_frames=ids[outside].tolist(),
                tcn_radius_rows=TCN_RADIUS, rgb_window_rows=16, rgb_stride_rows=2,
                scope='Structural maximum support only; not attention, causal attribution, exact onset, or nominal dense truth.')

def verify_pins(dataset):
    for path, digest in dataset['pins'].items():
        if sha(path) != digest:
            raise ValueError('Frozen file changed: ' + path)

def load_inputs(dataset):
    """Return clip->prepared arrays in memory; never persist or re-infer them."""
    if isinstance(dataset, (str, Path)):
        dataset = read(dataset)
    verify_pins(dataset)
    stats = read(M/'NORMALIZATION.json')
    rgb_stats = read(R/'RGB_NORMALIZATION.json')
    valid = geometry_valid()
    if not np.array_equal(valid, dataset['geometry_valid']):
        raise ValueError('Frozen geometry selection changed')
    result = {}
    for row in dataset['rows']:
        prepared = prepare_row(row, stats, rgb_stats, row['verified_recipe'], valid)
        if prepared['ids'].tolist() != row['native_frame_ids'] or prepared['ts'].tolist() != row['timestamp_us']:
            raise ValueError('Frozen native input axis changed')
        if prepared['observed_fragments'] != row['observed_fragments']:
            raise ValueError('Frozen actual observation fragments changed')
        result[row['clip_id']] = prepared
    return result
