"""Separate Codex input audit. Read-only sources; no model imports/inference.

Only this review directory receives output. Numeric reproducibility is not a
visual, hand-identity, linguistic, or independent-human validity certificate.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import ast
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter
import numpy as np
import av
from PIL import Image

REVIEW = Path(__file__).resolve().parent
U = REVIEW.parent
OPS = U.parent
OBSERVER = OPS / 'word100_guide_video_feedback_20261028/motion_observer.py'
ENCODER = OPS / 'word100_signrep_temporal_head_20261029/prepare_inputs.py'
spec = importlib.util.spec_from_file_location('audit_readonly_motion_observer', OBSERVER)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
HASHES = {}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    key = str(Path(p).resolve())
    if key not in HASHES:
        h = hashlib.sha256()
        with Path(p).open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        HASHES[key] = h.hexdigest()
    return HASHES[key]


def require(condition, name):
    if not bool(condition):
        raise AssertionError(name)


def bitwise(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def independent_scalar_check(obs, extracted):
    """Independent scalar positions; atan2 cross/dot angles vs adapter arccos.

    No adapter internal helper, normalization, dependency table, or angle helper
    is used to calculate these expected numbers.
    """
    xy = obs['xy_px'].astype(float)
    valid = obs['observed'] & obs['usable'] & np.isfinite(xy).all(axis=2)
    ix = {name: i for i, name in enumerate(extracted['feature_names'])}
    count, max_error = 0, 0.0
    specs = []
    for side, offset, shoulder, elbow, wrist in [('L', 33, 11, 13, 15), ('R', 54, 12, 14, 16)]:
        specs += [(side + '_elbow_angle', (shoulder, elbow, wrist)),
                  (side + '_index_pip_angle', (offset+5, offset+6, offset+7)),
                  (side + '_little_dip_angle', (offset+18, offset+19, offset+20)),
                  (side + '_thumb_ip_angle', (offset+2, offset+3, offset+4))]
    for t in range(len(xy)):
        sx = float(xy[t, 11, 0] - xy[t, 12, 0])
        sy = float(xy[t, 11, 1] - xy[t, 12, 1])
        scale = math.hypot(sx, sy)
        body_ok = bool(valid[t, 11] and valid[t, 12] and math.isfinite(scale) and scale > 1e-8)
        for side, offset, wrist in [('L', 33, 15), ('R', 54, 16)]:
            for item, j in [('hand_wrist', offset), ('pose_wrist', wrist), ('index_tip', offset+8)]:
                for d, axis in enumerate('xy'):
                    name = side + '_' + item + '_' + axis
                    ok = body_ok and bool(valid[t, j])
                    require(bool(extracted['mask'][t, ix[name]]) == ok, 'independent position mask ' + name)
                    if ok:
                        expected = (float(xy[t,j,d]) - (float(xy[t,11,d])+float(xy[t,12,d]))/2) / scale
                        err = abs(float(extracted['x'][t, ix[name]]) - expected)
                        require(err <= 1e-10, 'independent position value ' + name)
                        count += 1
                        max_error = max(max_error, err)
        for name, (a,b,c) in specs:
            ux,uy = float(xy[t,a,0]-xy[t,b,0]), float(xy[t,a,1]-xy[t,b,1])
            vx,vy = float(xy[t,c,0]-xy[t,b,0]), float(xy[t,c,1]-xy[t,b,1])
            product = math.hypot(ux,uy) * math.hypot(vx,vy)
            ok = bool(valid[t,a] and valid[t,b] and valid[t,c] and math.isfinite(product) and product > 1e-12)
            require(bool(extracted['mask'][t,ix[name]]) == ok, 'independent angle mask ' + name)
            if ok:
                expected = math.atan2(abs(ux*vy-uy*vx), ux*vx+uy*vy)/math.pi
                err = abs(float(extracted['x'][t,ix[name]]) - expected)
                require(err <= 1e-7, 'independent angle value ' + name)
                count += 1
                max_error = max(max_error,err)
    return dict(compared_cells=count, max_absolute_error=max_error,
                position_tolerance=1e-10, angle_tolerance=1e-7)


def audit_one(row, selected):
    a = row['alias']
    prepared_path = U / 'prepared' / (a + '.json')
    prepared = read(prepared_path)
    require(prepared['input'] == row, 'prepared input exact metadata')
    require(len(prepared['pins']) == 6, 'six prepared pins')
    for path, digest in prepared['pins'].items():
        require(sha(path) == digest, 'prepared pinned file hash: ' + path)
    record = read(row['record'])
    require(sha(row['record']) == row['record_sha256'] == selected['record_sha256'], 'record sha')
    require(sha(row['source']) == row['source_sha256'] == selected['archive_source']['sha256'] == record['source']['sha256'], 'public source sha')
    require(sha(row['input']) == row['input_sha256'], 'prepared npz sha')
    require(sha(row['observation']) == record['outputs'][row['observation']], 'native observation recorded sha')
    for key in ('clip_id', 'word_id', 'person', 'role', 'box', 'observation', 'record'):
        require(row[key] == selected[key], 'selection metadata ' + key)
    require(record['clip_id'] == row['clip_id'] and record['signer_id'] == row['person'] and record['word_code'] == row['word_id'], 'record identity')
    require(record['role'] == 'train_candidate', 'public original TRAIN provenance')
    require(record['source'] == selected['archive_source'], 'archive member identity')
    require(record['calibration']['box_source_pixels'] == row['box'], 'frozen per-clip source ROI')
    require(record['views']['native']['observation'] == row['observation'], 'native observation path')
    data = load_npz(row['input'])
    raw = load_npz(row['observation'])
    obs = {k: raw[k].copy() for k in ('xy_px','observed','usable','timestamp_us','frame_id')}
    n = len(obs['frame_id'])
    require(n == row['frames'] == record['frames'], 'full native frame length')
    require(set(data) == {'x','mask','rgb','window_features','window_starts','window_center_us','frame_id','timestamp_us'}, 'input fields')
    require(data['x'].shape == (n,69) and data['mask'].shape == (n,69), '69 geometry channels')
    require(data['rgb'].shape == (n,768), 'frame RGB representation width 768')
    require(data['window_features'].shape == (row['windows'],768), 'window RGB representation width 768')
    require(data['mask'].dtype == bool, 'boolean feature mask')
    for key in ('x','rgb','window_features','window_center_us','timestamp_us','frame_id'):
        require(np.isfinite(data[key]).all(), 'finite saved ' + key)
    for key in ('x','rgb','window_features'):
        require(data[key].dtype == np.float32, 'float32 ' + key)
    require(bitwise(data['frame_id'],obs['frame_id']), 'native frame ids bitwise')
    require(np.array_equal(data['frame_id'],np.arange(n)), 'complete original decode frame indices')
    require(np.diff(data['timestamp_us']).min() > 0, 'strictly increasing PTS')
    require(np.abs(data['timestamp_us']-obs['timestamp_us']).max() <= 1, 'native PTS tolerance 1us')
    require(np.isfinite(obs['xy_px'][obs['observed'] & obs['usable']]).all(), 'observed usable source positions finite')
    x0,y0,x1,y1 = row['box']
    xy = obs['xy_px']
    # Independent construction: begin false, use a conjunction of all bounds.
    inside = np.logical_and.reduce((xy[:,:,0] >= x0, xy[:,:,0] < x1, xy[:,:,1] >= y0, xy[:,:,1] < y1))
    original_usable = obs['usable'].copy()
    obs['usable'] = np.logical_and(original_usable, inside)
    expected = observer.extract_features(obs)
    require(expected['feature_names'] == row['feature_names'], 'feature order')
    require(bitwise(np.nan_to_num(expected['x']).astype(np.float32),data['x']), 'recomputed geometry bitwise')
    require(bitwise(expected['mask'],data['mask']), 'recomputed feature masks bitwise')
    require(not obs['usable'][~inside].any(), 'no outside ROI point usable')
    require(not expected['joint_observed_usable'][~inside].any(), 'adapter no outside ROI point accepted')
    require((data['x'][~data['mask']] == 0).all(), 'unknown geometry encoded zero with false mask')
    # Alter all outside-ROI coordinates radically, keeping the frozen false mask.
    # Every observed output must remain identical, including temporal channels.
    perturb = {k:v.copy() for k,v in obs.items()}
    perturb['xy_px'][~inside] = [1e6,-1e6]
    changed = observer.extract_features(perturb)
    require(np.array_equal(changed['mask'],expected['mask']), 'outside ROI perturbation cannot change mask')
    require(np.array_equal(changed['x'][expected['mask']],expected['x'][expected['mask']]), 'outside ROI perturbation cannot change observed values')
    # Check explicit spatial and previous-frame temporal dependencies too.
    valid = expected['joint_observed_usable']
    for col,name in enumerate(expected['feature_names']):
        deps = expected['dependencies'][name]
        allowed = valid[:,deps].all(axis=1)
        if '_velocity_' in name or '_motion_direction_' in name:
            allowed &= np.r_[False,valid[:-1,deps].all(axis=1)]
        require(not (expected['mask'][:,col] & ~allowed).any(), 'feature dependency ROI mask ' + name)
    scalar = independent_scalar_check(obs,expected)
    starts = list(range(0,n-15,2))
    if starts[-1] != n-16:
        starts.append(n-16)
    require(np.array_equal(data['window_starts'],starts), '16frame stride2 windows no padding')
    mid = (data['timestamp_us'][data['window_starts']] + data['timestamp_us'][data['window_starts']+15])/2
    require(np.array_equal(mid,data['window_center_us']), 'window midpoint times')
    interpolated = np.stack([np.interp(data['timestamp_us'],mid,data['window_features'][:,j]) for j in range(768)],axis=1).astype(np.float32)
    require(bitwise(interpolated,data['rgb']), 'saved RGB time interpolation bitwise')
    decoded_pts, dimensions = [], set()
    with av.open(row['source']) as container:
        for frame in container.decode(video=0):
            require(frame.pts is not None and frame.time_base is not None, 'source PTS available')
            decoded_pts.append(round(float(frame.pts*frame.time_base)*1e6))
            dimensions.add((frame.width,frame.height))
    require(len(decoded_pts) == n, 'source decoded full frame count')
    require(np.array_equal(decoded_pts,data['timestamp_us']), 'decoded source PTS exactly saved')
    require(dimensions == {(record['calibration']['source_width'],record['calibration']['source_height'])}, 'source image dimensions')
    width,height = next(iter(dimensions))
    require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, 'ROI within source dimensions')
    old = selected.get('old_input')
    if old:
        require(sha(old['input']) == old['input_sha256'], 'old36 npz frozen hash')
        prior = load_npz(old['input'])
        for key in data:
            require(bitwise(data[key],prior[key]), 'old36 bitwise reuse ' + key)
        require(row['reused_rgb'] and row['rgb_seconds'] == 0, 'reuse flags')
    else:
        require(not row['reused_rgb'], 'new44 flags')
    visual = prepared['visual']
    require(visual['static_ordinals'] == [3,6,9], 'visual static ordinal mapping')
    require([f['ordinal'] for f in visual['frames']] == list(range(13)), '13 visual sample order')
    for f in visual['frames']:
        require(abs(f['seconds']-float(obs['timestamp_us'][f['frame_id']])/1e6) < 1e-8, 'sample image timestamp mapping')
    for key,size in [('sequence',(1600,1640)),('states',(1500,1110))]:
        with Image.open(visual[key]) as image:
            require(image.size == size, 'RGB contact-sheet size '+key)
    # This is a warning inventory, never a source of positive/negative labels.
    static_roi = []
    for f in visual['frames']:
        if f['ordinal'] not in [3,6,9]:
            continue
        t = f['frame_id']
        for side,off in [('R',54),('L',33)]:
            have = obs['observed'][t,off:off+21] & np.isfinite(xy[t,off:off+21]).all(axis=1)
            outside = have & ~inside[t,off:off+21]
            if outside.any():
                static_roi.append(dict(ordinal=f['ordinal'],frame_id=t,side=side,observed_points_outside=int(outside.sum())))
    motion_roi = []
    for start,end in [(0,3),(3,6),(6,9),(9,12)]:
        first,last = visual['frames'][start]['frame_id'],visual['frames'][end]['frame_id']
        for side,off in [('R',54),('L',33)]:
            have = obs['observed'][first:last+1,off:off+21] & np.isfinite(xy[first:last+1,off:off+21]).all(axis=2)
            bad = (have & ~inside[first:last+1,off:off+21]).any(axis=1)
            if bad.any():
                motion_roi.append(dict(start_ordinal=start,end_ordinal=end,side=side,frames_with_observed_points_outside=int(bad.sum())))
    return dict(alias=a,status='PASS_NUMERIC_ONLY',frames=n,windows=len(starts),source_dimensions=[width,height],
                pins_checked=len(prepared['pins']),old_input_bitwise_reused=bool(old),
                original_usable_points_removed_by_roi=int((original_usable & ~inside).sum()),
                scalar_check=scalar,static_roi_warning_states=static_roi,motion_roi_warning_intervals=motion_roi,
                prepared_sha256=sha(prepared_path),source_sha256=sha(row['source']),
                observation_sha256=sha(row['observation']),record_sha256=sha(row['record']),input_sha256=sha(row['input']))


def audit_saved_label_roi():
    """Review the completed merge without running label construction/training."""
    paths = [U/'build_labels.py', U/'LABEL_AUDIT.json', U/'LEARNING_INPUTS.json', U/'RGB_RUBRIC.json']
    if not all(p.exists() for p in paths):
        return dict(status='NOT_YET_AVAILABLE',requirements_remain=True)
    learning, audit, rubric = read(paths[2]), read(paths[1]), read(paths[3])
    by_alias = {r['alias']:r for r in audit['rows']}
    targets = learning['all_targets']
    allowed = rubric['motion_values']
    require('diagonal' not in allowed, 'no unsigned diagonal accepted by rubric')
    require(all(d in allowed for d in ('up_left','up_right','down_left','down_right')), 'signed diagonal components defined')
    direction_counts = Counter()
    for name in ('RGB_LABELS_A.json','RGB_LABELS_B.json','RGB_LABELS_SECONDARY.json'):
        p = REVIEW/name
        if not p.exists():
            return dict(status='MISSING_REVIEW_INPUT',file=str(p))
        paths.append(p)
        for r in read(p)['rows']:
            for interval in r['motion']:
                for side in ('R','L'):
                    d=interval[side]
                    require(d in allowed,'reviewed direction enum')
                    direction_counts[str(d)]+=1
    known_cells, interval_cells, excluded_cells = 0,0,0
    no_hand_fallback_cells, partial_hand_cells = 0,0
    target_pins = {}
    for row in learning['rows']:
        data,raw = load_npz(row['input']),load_npz(row['observation'])
        z=load_npz(row['targets_file'])
        require(sha(row['targets_file'])==row['targets_sha256'],'saved target hash')
        target_pins[row['alias']]=sha(row['targets_file'])
        y,mask=z['y'],z['target_mask']
        require(y.shape==mask.shape==(len(data['frame_id']),len(targets)),'target dimensions')
        require(mask.dtype==bool and np.isfinite(y).all(),'finite target and bool mask')
        require(np.isin(y[mask],[0,1]).all(),'binary known target values')
        require(int(mask.sum())==row['known_cells'],'known target count')
        ar=by_alias[row['alias']]
        lookup={int(f):i for i,f in enumerate(data['frame_id'])}
        spans={(e['frame_id'],e['feature']):e['interval_frames'] for e in ar['observations'] if e['kind']=='interval_direction'}
        x0,y0,x1,y1=row['box']
        for i,j in np.argwhere(mask):
            fid,name=int(data['frame_id'][i]),targets[j]
            sides=('R','L') if name=='hands_overlap' else (name[0],)
            if '_motion_' in name:
                require((fid,name) in spans,'motion evidence span exists')
                lo,hi=spans[(fid,name)]
                ids=[lookup[f] for f in range(lo,hi+1)]
                interval_cells+=1
            else:
                ids=[int(i)]
            cell_fallback,cell_partial=False,False
            for t in ids:
                for side in sides:
                    off=54 if side=='R' else 33
                    points=raw['xy_px'][t,off:off+21]
                    good=raw['observed'][t,off:off+21] & np.isfinite(points).all(axis=1)
                    have=points[good]
                    if len(have):
                        require(all(x0<=float(p[0])<x1 and y0<=float(p[1])<y1 for p in have),'known label with observed hand outside ROI')
                        cell_partial |= len(have)<21
                    else:
                        cell_fallback=True
                        pw=16 if side=='R' else 15
                        p=raw['xy_px'][t,pw]
                        require(raw['usable'][t,pw] and np.isfinite(p).all() and x0<=p[0]<x1 and y0<=p[1]<y1,'known label violates wrist fallback ROI')
            no_hand_fallback_cells+=cell_fallback
            partial_hand_cells+=cell_partial
            known_cells+=1
        for drop in ar['roi_exclusions']:
            require(not mask[lookup[drop['frame_id']],targets.index(drop['feature'])],'excluded ROI target reinstated')
            excluded_cells+=1
        for conflict in ar['conflicts']:
            require(not mask[lookup[conflict['frame_id']],targets.index(conflict['feature'])],'conflicting target reinstated')
    return dict(status='PASS_SAVED_GEOMETRIC_EXCLUSIONS_ONLY',code_and_data_pins={str(p):sha(p) for p in paths},
                targets_sha256=target_pins,known_target_cells_checked=known_cells,known_interval_cells_checked_across_every_source_frame=interval_cells,
                confirmed_roi_exclusions=excluded_cells,known_target_cells_using_body_wrist_fallback=no_hand_fallback_cells,
                known_target_cells_with_partial_observed_hand=partial_hand_cells,direction_counts=dict(direction_counts),
                unsigned_diagonal_issue='NOT_APPLICABLE: frozen rubric and all actual drafts use signed diagonal categories only',
                interval_gate='Saved targets checked across every native frame in each full interval; midpoint-only issue resolved',
                limitation='A detector/wrist gate cannot prove pixel-level visibility or anatomical identity. Missing hand detector points require explicit conservative treatment; this audit does not turn them into negative labels.')


def main():
    started = time.time()
    inputs, selection = read(U/'INPUTS.json'), read(U/'SELECTION.json')
    rows, selected_rows = inputs['rows'],selection['rows']
    require(len(rows) == len(selected_rows) == 80, '80 sources')
    expected_aliases = [f'P{i:03}' for i in range(1,81)]
    require([r['alias'] for r in rows] == expected_aliases, '80 ordered aliases')
    selected = {r['alias']:r for r in selected_rows}
    require(len(selected) == 80, 'unique selected aliases')
    for key in ('clip_id','source','source_sha256','observation','input'):
        require(len({r[key] for r in rows}) == 80, 'no duplicated '+key)
    roles = dict(Counter(r['role'] for r in rows))
    require(roles == {'fit':36,'development':12,'held_person':24,'held_expression_person':8}, '36/12/32 person split')
    require(roles == inputs['roles'] == selection['roles'], 'declared role counts')
    original,novel = set(selection['words']),set(selection['novel_words'])
    require(len(original)==12 and len(novel)==4 and not original.intersection(novel), '12 original and 4 disjoint novel words')
    required_pairs = {(w,p) for w in original for p in range(1,7)} | {(w,p) for w in novel for p in (5,6)}
    require({(r['word_id'],r['person']) for r in rows} == required_pairs, 'exact word/person design')
    for r in rows:
        expected = 'fit' if r['person'] in (1,2,3) else 'development' if r['person']==4 else 'held_expression_person' if r['word_id'] in novel else 'held_person'
        require(r['role']==expected, 'person role '+r['alias'])
    require(sum(bool(r.get('old_input')) for r in selected_rows)==36, '36 reused sources')
    require({r['alias'] for r in selected_rows if r.get('old_input')} == {r['alias'] for r in rows if r['role']=='fit'}, 'old36 are all new fit, never new held validation')
    results = []
    for i,r in enumerate(rows,1):
        try:
            results.append(audit_one(r,selected[r['alias']]))
        except Exception as exc:
            results.append(dict(alias=r['alias'],status='FAIL',error=type(exc).__name__+': '+str(exc)))
        if i % 10 == 0:
            print(json.dumps(dict(audited=i,failed=sum(q['status']=='FAIL' for q in results),elapsed_seconds=round(time.time()-started,1))),flush=True)
    successful = [r for r in results if r['status']=='PASS_NUMERIC_ONLY']
    failed = [r for r in results if r['status']=='FAIL']
    try:
        label_roi_audit = audit_saved_label_roi()
    except Exception as exc:
        label_roi_audit = dict(status='FAIL',error=type(exc).__name__+': '+str(exc))
    report = dict(
        status='PASS_NUMERIC_ONLY' if not failed else 'FAIL',
        reviewer='Same Codex, separate subagent rgb_labels_a input verification; not independent human or independent model validation',
        no_new_model_inference=True,source_data_written=False,output_scope=str(REVIEW),
        input_pins={str(p):sha(p) for p in (U/'INPUTS.json',U/'SELECTION.json',U/'prepare_bundle.py',OBSERVER,ENCODER,Path(__file__))},
        numeric_scope='80 source/record/observation/input SHA pins, duplicate checks, person split, full decoded PTS, finite and shape checks, exact ROI geometry/masks, out-of-ROI counterfactual, independent scalar math, RGB interpolation and old36 bitwise reuse',
        split=dict(roles=roles,person_roles={'1':'fit','2':'fit','3':'fit','4':'development','5':'held','6':'held'},historical_warning=selection['split_note']),
        totals=dict(clips=len(rows),passed=len(successful),failed=len(failed),frames=sum(r['frames'] for r in successful),
                    old36_bitwise_reused=sum(r['old_input_bitwise_reused'] for r in successful),
                    scalar_cells=sum(r['scalar_check']['compared_cells'] for r in successful),
                    scalar_max_absolute_error=max((r['scalar_check']['max_absolute_error'] for r in successful),default=None),
                    removed_usable_points=sum(r['original_usable_points_removed_by_roi'] for r in successful),
                    static_roi_warning_states=sum(len(r['static_roi_warning_states']) for r in successful),
                    motion_roi_warning_intervals=sum(len(r['motion_roi_warning_intervals']) for r in successful),
                    unique_hashed_files=len(HASHES)),
        rgb_scope=dict(saved_arrays_and_interpolation='verified',new44_encoder_execution='not repeated; no new model inference',
                       preprocessing_code_review='encode uses original source-sized frame, neutral [124,116,104] outside fixed box, then 224x224 resize; not a claim that RGB embedding provenance is independently re-executed',
                       old36_reuse='all eight shared NPZ arrays required bitwise-identical'),
        visual_semantic_preservation=dict(status='NOT_ESTABLISHED',reason='Exact arithmetic and saved detector coordinates do not establish anatomical assignment, hidden handshape, palm visibility, or linguistic correctness. Full-source review sheets may show hands removed from input ROI.'),
        saved_label_roi_audit=label_roi_audit,
        label_merge_requirements=[
            'Exclude target masks whose evidence hand lies outside fixed ROI; do not convert exclusions or unavailable evidence to zero.',
            'For interval motion, gate the entire evidence interval, not only its midpoint. Use all decoded native frames or at least every reviewed sample; an out-of-ROI interval portion invalidates that interval target.',
            'Do not treat partially detected or absent fingers plus an in-ROI body wrist as proof that handshape/palm evidence is in ROI. Detector-based gating is a conservative proxy, not visual visibility proof.',
            'Require both relevant hands for hands_overlap, and relevant visible face evidence for face_overlap. Keep ambiguity unknown.',
            'Keep the frozen signed direction vocabulary; do not introduce an unsigned diagonal category whose components would be undefined.',
            'Persist per-target exclusion reasons and preserve full-source draft labels as separate evidence. Same-Codex agreement is not independent human truth.'
        ],
        rows=results,elapsed_seconds=round(time.time()-started,3))
    path=REVIEW/'INPUTS_INDEPENDENT.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status=report['status'],totals=report['totals'],output=str(path))),flush=True)
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
