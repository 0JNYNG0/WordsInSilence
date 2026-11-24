"""Read-only FIT annotation audit; no neural imports/forward pass or raw video read."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import ast
import hashlib
import json
import re
import numpy as np

U = Path(__file__).resolve().parents[1]
O = U.parent
A = O / 'word100_autonomous_localization_20261111'
J = O / 'word100_episode_joint_linkage_20261111'
R = O / 'word100_candidate_role_learning_20261111'
S = O / 'word100_shared_motion_learning_20261109'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
web4 = {'WORD0739', 'WORD1133', 'WORD1567', 'WORD2523'}
word = lambda r: re.search(r'WORD\d+', r['clip_id']).group()
doc = read(A / 'FIT_LABELS.json')
anns = doc['rows']
assert all(r['role'] in ('fit', 'fit_additional') for r in anns)
# FIRST metadata filter. Only selected FIT timestamp/frame_id fields are opened.
fit = [r for r in read(S / 'READY.json')['rows'] if r['role'] == 'fit']
fm = {r['clip_id']: r for r in fit}
ids = sorted({a['clip_id'] for a in anns})
assert set(ids) <= fm.keys()
issues = []
clipmeta = {}
for cid in ids:
    r = fm[cid]
    with np.load(r['path'], allow_pickle=False) as z:
        frames, ts = z['frame_id'].copy(), z['timestamp_us'].copy()
    assert np.array_equal(frames, np.arange(len(frames)))
    assert len(ts) == len(frames) and np.all(np.diff(ts) > 0)
    clipmeta[cid] = dict(frames=len(frames), timestamps=ts, source_sha256=r['source_sha256'],
                         cache_path=r['path'], cache_manifest_sha256=r['sha256'],
                         source_group=r['source_sha256'])

review_meta = {}
for folder in [O/'word100_observed_event_boundaries_20261110', J, A]:
    for r in read(folder/'REVIEW_INPUTS.json')['rows']:
        if r.get('role') in ('fit', 'fit_additional') and r['clip_id'] in ids:
            review_meta[r['clip_id']] = r

byword, byclip = defaultdict(list), defaultdict(list)
axis_side = Counter()
events, intervals, collisions = [], [], []
sheet_checks = {}
for ai, a in enumerate(anns):
    cid = a['clip_id']; meta = clipmeta[cid]; n = meta['frames']; ts = meta['timestamps']
    assert a['source_sha256'] == meta['source_sha256']
    assert a['person'] == fm[cid]['person']
    k = a['axis']*2 + (a['moving'] == 'L')
    byword[word(a)].append(a); byclip[cid].append(a); axis_side[(a['axis'], a['moving'])] += 1
    sp = a['sheet']
    if sp not in sheet_checks:
        sheet_checks[sp] = dict(exists=Path(sp).exists(), expected_sha256=a['sheet_sha256'])
        if Path(sp).exists():
            sheet_checks[sp]['actual_sha256'] = sha(sp)
            sheet_checks[sp]['match'] = sheet_checks[sp]['actual_sha256'] == a['sheet_sha256']
    for ei, e in enumerate(a['events']):
        lo, hi = e['onset'], e['offset']
        assert 0 <= lo[0] <= lo[1] <= hi[1] < n and hi[0] <= hi[1]
        strict_a, strict_b = lo[1]+1, hi[0]-1
        strict_n = max(0, strict_b-strict_a+1)
        # Sheet sampling is provenance only, not a newly visually confirmed motion witness.
        sampled = review_meta.get(cid, {}).get('frames', [])
        sampled_inside = [v for v in sampled if strict_a <= v <= strict_b]
        event = dict(event_id=f'{cid}:axis{a["axis"]}:{a["moving"]}:event{ei}',
            annotation_index=ai, clip_id=cid, word_id=word(a), person=a['person'],
            axis=a['axis'], moving=a['moving'], direction=e['direction'],
            onset_frame_range=lo, offset_frame_range=hi,
            outer_extent_frames=hi[1]-lo[0]+1,
            outer_elapsed_seconds=float((ts[hi[1]]-ts[lo[0]])/1e6),
            possible_motion_elapsed_seconds_range=[float(max(0,ts[hi[0]]-ts[lo[1]])/1e6),float((ts[hi[1]]-ts[lo[0]])/1e6)],
            strict_between_boundary_frame_count=strict_n,
            strict_between_boundary_range=[strict_a,strict_b] if strict_n else None,
            original_sheet_sample_ids_inside_strict_range=sampled_inside,
            explicit_rgb_progress_witness_frame_in_source=None,
            boundary_uncertainty_must_not_become_exact_frame_truth=True,
            source_sha256=a['source_sha256'], sheet=sp, sheet_sha256=a['sheet_sha256'])
        events.append(event)
        intervals.append((cid,k,lo[0],hi[1],'event',event['event_id']))
    for kind in ('holds','background'):
        for ii,(lo,hi) in enumerate(a[kind]):
            assert 0 <= lo <= hi < n
            intervals.append((cid,k,lo,hi,kind,f'annotation{ai}:{kind}{ii}'))

for i,x in enumerate(intervals):
    for y in intervals[i+1:]:
        if x[0:2] != y[0:2] or min(x[3],y[3]) < max(x[2],y[2]): continue
        if x[4] == y[4] == 'background': continue
        collisions.append(dict(clip_id=x[0],relation_index=x[1],kinds=[x[4],y[4]],
            interval_ids=[x[5],y[5]],overlap=[max(x[2],y[2]),min(x[3],y[3])],
            interpretation='Shared turning/terminal boundary or competing range; do not densely propagate all labels.'))

# Run only an isolated AST of the original metadata-to-spec function, no imports.
tree = ast.parse((J/'joint_core.py').read_text('utf-8'))
fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='example_specs')
scope = {}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(J/'joint_core.py'), 'exec'),scope)
rows = [dict(clip_id=cid,frame_id=range(clipmeta[cid]['frames']),annotations=byclip[cid]) for cid in ids]
specs, labels, pairs, pair_y = scope['example_specs'](rows)
labelsets = defaultdict(set); duplicate_counts=Counter()
for spec,y in zip(specs,labels):
    i,lo,hi,k=spec; key=(ids[i],lo,hi,k)
    labelsets[key].add(y);duplicate_counts[key]+=1
exact_conflicts=[dict(clip_id=k[0],a=k[1],b=k[2],relation_index=k[3],labels=sorted(v)) for k,v in labelsets.items() if len(v)>1]

word_stats=[]
for w, aa in sorted(byword.items()):
    ev=[e for e in events if e['word_id']==w]
    word_stats.append(dict(word_id=w,web4=w in web4,unique_clips=len({r['clip_id'] for r in aa}),
        relation_annotations=len(aa),record_ids=sorted({r['person'] for r in aa}),
        events=len(ev),holds=sum(len(r['holds']) for r in aa),background=sum(len(r['background']) for r in aa),
        axes=sorted({r['axis'] for r in aa}),right_only=all(r['moving']=='R' for r in aa),
        event_outer_frame_range=[min(e['outer_extent_frames'] for e in ev),max(e['outer_extent_frames'] for e in ev)] if ev else None,
        event_outer_elapsed_seconds_range=[min(e['outer_elapsed_seconds'] for e in ev),max(e['outer_elapsed_seconds'] for e in ev)] if ev else None,
        event_with_no_strict_between_boundary_frame=sum(e['strict_between_boundary_frame_count']==0 for e in ev)))

same_axis_duplicates=[dict(clip_id=cid,relation_index=k,count=n) for (cid,k),n in Counter((a['clip_id'],a['axis']*2+(a['moving']=='L')) for a in anns).items() if n>1]
source_groups=defaultdict(list)
for cid,m in clipmeta.items():source_groups[m['source_sha256']].append(cid)
sha_duplicates={h:c for h,c in source_groups.items() if len(c)>1}
original_j=read(J/'ALL_FIT_BOUNDARIES.json')['rows']
assert anns==original_j+doc['additional']
assert len(ids)==30 and len(anns)==33 and len(events)==49
assert len(byword)==10 and all(a['moving']=='R' for a in anns)
assert all(v.get('match',False) for v in sheet_checks.values())

source_paths=[A/'FIT_LABELS.json',A/'labels.py',A/'NORMALIZATION.json',A/'localization_core.py',
 J/'ALL_FIT_BOUNDARIES.json',J/'joint_core.py',J/'proposal_adapter.py',R/'prepare_train.py',S/'READY.json',
 U/'reviews/local_motion_quality_design.json']
out=dict(status='METADATA_PREFLIGHT_PASSED_WITH_CONTRACT_AND_WITNESS_LIMITS',timestamp=datetime.now().astimezone().isoformat(),
 reviewer='Separate Codex code/metadata arithmetic audit; not independent human or KSL expert truth',
 scope=dict(model_inference=0,training=0,raw_media_read=0,private_data_read=0,development_arrays_read=0,
            fit_cache_metadata_only=30,cache_fields=['frame_id','timestamp_us'],root_documents_modified=False),
 counts=dict(fit_manifest_clips=len(fit),unique_public_videos=len(ids),relation_annotations=len(anns),
             expressions=len(byword),web4_videos=sum(x['unique_clips'] for x in word_stats if x['web4']),
             other6_videos=sum(x['unique_clips'] for x in word_stats if not x['web4']),events=len(events),
             holds=sum(len(a['holds']) for a in anns),background=sum(len(a['background']) for a in anns),
             record_ids=sorted({a['person'] for a in anns}),moving_sides=sorted({a['moving'] for a in anns}),
             other6_events=sum(x['events'] for x in word_stats if not x['web4']),
             source_sha_groups=len(source_groups),sheet_hashes_checked=len(sheet_checks)),
 per_word=word_stats,
 provenance=dict(fit_rows_equal_previous27_plus_additional6=True,
     additional_is_metadata_duplicate_already_inside_rows=True,
     additional_rows_must_not_be_appended_again=True,
     annotation_notes='Observed 2D body/wrist relationship. Not handshape/contact or educational repeat count.',
     source_manifest_sha_matches=True,source_movie_rehashed=False,
     source_sha_check_scope='Annotation SHA values equal existing FIT manifest SHA; raw movies were not reopened/rehashed.',
     source_sha_duplicates=sha_duplicates,
     multiple_annotation_clips={cid:len(v) for cid,v in byclip.items() if len(v)>1},
     duplicate_same_clip_relation=same_axis_duplicates,sheet_checks=sheet_checks),
 old_normal_spec_arithmetic=dict(scope='Deterministic example_specs only; actual boundary candidate additions/upper/reverse variants are not included in this denominator.',
     specs=len(specs),class_counts=dict(Counter(labels)),unique_input_intervals=len(labelsets),
     duplicate_spec_occurrences=sum(v-1 for v in duplicate_counts.values()),same_interval_contradictory_labels=exact_conflicts,
     pair_counts=dict(Counter(pair_y)),no_candidate_head_rerun=True),
 original_contract=dict(
     zero='Only directly annotated holds/background; absent observations are not label 0.',
     one='Arithmetic subintervals/actual candidates inside annotated movement. Fragment, not failed sign.',
     two='Endpoints in reviewed onset/offset ranges; reviewed movement extent, not entire-sign completion or exact contact.',
     three='example_specs: adjacent distinct movement union. Later proposal_adapter.assign also labels contained movement plus overlapping explicit hold/background as 3.',
     runtime_candidate_flag='candidate_spans complete=True denotes a boundary-pair proposal, not verified complete movement.',
     inherited_duration='304 feature final element is frame count /100, not true duration seconds.'),
 recommended_new_contract=dict(
     same_b_c_labels=True,
     extent3='Use distinct directly annotated events only; movement+terminal hold stays unknown with metadata unless separately admitted.',
     missing='Mask unsupported relations; never manufacture hold/background supervision from missingness.',
     boundaries='Intervals are uncertain brackets. No extra frame-level exact labels.',
     progress_witness='Source labels contain no explicit per-frame RGB progress witness IDs. An arithmetic midpoint is not direct RGB confirmation. Require explicit evidence admission or state interval-level supervision honestly.',
     short_events='Some reviewed very short motions have no strict interior frame outside both boundary bands. Preserve them as unknown/sparse evidence; do not relabel as absent or require smoothing.',
     edits='Discontinuous edits supervise only separate observation-edit output, extent loss mask 0.',
     split='Group by original clip AND source SHA; all axes, full/upper views, fragments, speed variants and edits stay in same group.',
     weighting='Report 30 source videos/33 relation rows separately from derived candidate counts; balance/group correlated replicas.',
     broad_generalization='Six additional expressions broaden relation patterns but do not establish all 100-word detailed correctness or L-hand performance.',
     inherited_exposure='FIT holdout isolates new quality training only; original observer/quality initialization already saw these FIT records. Do not call unseen-person final test.'),
 boundary_overlap_rows=collisions,events=events,
 no_strict_interior_events=[e['event_id'] for e in events if e['strict_between_boundary_frame_count']==0],
 generation_code_review='Pending new CONTRACT/model/prepare files; this audit does not authorize training.',
 pins={str(p):sha(p) for p in source_paths})
target=U/'reviews/local_quality_label_preflight.json'
target.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:out[k] for k in ['status','counts','per_word','old_normal_spec_arithmetic','no_strict_interior_events']},ensure_ascii=False,indent=2))
