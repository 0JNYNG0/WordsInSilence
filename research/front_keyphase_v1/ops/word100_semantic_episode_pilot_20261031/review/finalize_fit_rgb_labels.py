"""Metadata enrichment and structural checks only; no inference/coordinate reads."""
import json
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'LABELS_FIT.json'
data = json.loads(PATH.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected = ['P033','P034','P035','P045','P046','P047','P063','P064','P065','P069','P070','P071']
assert [r['alias'] for r in data['clips']] == expected
counts = Counter()
core_frames = Counter()
for clip in data['clips']:
    a = clip['alias']
    mp = ROOT / 'visuals' / a / 'MANIFEST.json'
    m = json.loads(mp.read_text(encoding='utf-8'))
    assert m['alias'] == a
    t = {r['frame_id']:r['timestamp_us'] for r in m['selected']}
    clip['time_unit'] = 'microseconds from native MANIFEST timestamp_us'
    clip['sampled_frames_directly_viewed_in_sheet'] = sorted(t)
    clip['manifest_sha256'] = sha(mp)
    clip['source_sha256_from_manifest'] = m['source_sha256']
    clip['native_frame_count_from_manifest'] = m['frames']
    occupied = set()
    major = 0
    other = 0
    for e in clip['events']:
        assert e['state'] in ('up','down','other')
        assert e['support'] == 'visible'
        s, end = e['core_start_frame'], e['core_end_frame']
        sb, eb = e['start_bracket'], e['end_bracket']
        assert 0 <= sb[0] <= sb[1] <= s < end <= eb[0] <= eb[1] < m['frames']
        assert all(v in t for v in [s,end,*sb,*eb,*e['evidence_frames']])
        frames = set(range(s,end+1))
        assert not occupied.intersection(frames)
        occupied |= frames
        e['core_start_timestamp_us'] = t[s]
        e['core_end_timestamp_us'] = t[end]
        e['start_bracket_timestamp_us'] = [t[f] for f in sb]
        e['end_bracket_timestamp_us'] = [t[f] for f in eb]
        e['evidence_timestamp_us'] = [t[f] for f in e['evidence_frames']]
        e['support_scope'] = 'RGB sampled anchors; intervening native frames were not all inspected'
        counts[e['state']] += 1
        core_frames[e['state']] += end-s+1
        major += e['state'] != 'other'
        other += e['state'] == 'other'
    assert major <= 2 and 1 <= other <= 2
    for q in clip['unknown_intervals']:
        s, end = q['start_frame'],q['end_frame']
        assert s <= end and s in t and end in t
        assert not occupied.intersection(range(s,end+1))
        q['start_timestamp_us'] = t[s]
        q['end_timestamp_us'] = t[end]
        q['kind'] = 'unlabelled_selection_gap; not other'

seen = data['provenance']['actual_viewed_new_files']
assert len(seen) == len(set(seen)) == 19
data['provenance']['viewed_file_hashes'] = [
    {'file':str(ROOT/'visuals'/p),'sha256':sha(ROOT/'visuals'/p)} for p in seen
]
data['summary'] = {
    'clips':len(data['clips']), 'event_windows':sum(counts.values()),
    'class_windows':dict(counts), 'inclusive_core_frames':dict(core_frames),
    'new_whole_clip_sheets_viewed':12,'new_full_frame_stills_viewed':7,
    'sampled_native_frames_in_new_sheets':576,
    'note':'Structural PASS only; provisional RGB interval judgment is not independently validated truth.'
}
PATH.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report = {
    'status':'PASS_FIT_RGB_STRUCTURE_ONLY',
    'labels_file':str(PATH),'labels_sha256':sha(PATH),
    'checks':['12 authorized aliases','states/support','max two major events and one-two other windows','bracket/core order','no overlapping cores','all anchors in native manifest','exact manifest timestamps','highlighted unknown gaps do not overlap cores','viewed-file hashes'],
    'not_verified':['native unviewed intermediate RGB frames','independent human annotation correctness','educational correctness','ROI input support'],
    'summary':data['summary'],
}
(ROOT/'review'/'LABELS_FIT_STRUCTURE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
