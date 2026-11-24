"""Serialize actual visual review notes; this script does not classify images."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

A = Path(__file__).resolve().parent
M = A / 'PARTIAL_VIEWS_MANIFEST.json'
OUT = A / 'REVIEW_PARTIAL_0739.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# All these pages were opened with view_image by predicate_contract_review.
VIEWED = set('''
9c62310ebf51da7d5658_p1 19fca20714bfb502e2be_p1 d27d7560f9bd9fb1eb08_p1 d27d7560f9bd9fb1eb08_p2
4eb9a6f97ebc453124d2_p1 909f31f29cd9e393096d_p1 00744f3a04db868ee546_p1
91dff4da3faa91fe6775_p1 0cc4a40452c97c1f31a8_p1 5817e47e090339f78753_p1 5817e47e090339f78753_p2
1791a7404c62ff546ac8_p1 2972ea0024f13ac7d8c9_p1 c0d33f2c9b1d56c01045_p1 c0d33f2c9b1d56c01045_p2
018b7211d316a34134dd_p1 e148d3f3249ac3787860_p1 73ccbd2b35eab8f46206_p1 73ccbd2b35eab8f46206_p2
1a2bc3e16d267c017641_p1 d8b907c11c5cce58ddb3_p1 1b607159a2063ed1276f_p1
d5732fc6e3891bed9fd2_p1 3d3eee808fa3ec75a09d_p1 c48da279242134a58fdf_p1
cc8b45eb835755d54e40_p1 1196c77b356e5f4c162b_p1 c495ff64388983cbf55d_p1 c495ff64388983cbf55d_p2
'''.split())

# Each note was authored after inspecting the named person's actual retained
# images. The manifest's construction kind and old labels are not a label rule.
NOTES = {
1: {
 'setup': [48,54,60], 'terminal': [69,73,79,82],
 'front': 'f48–60에서 오른손이 입·코 앞에 유지된다. 작은 몸의 변화는 있지만 손이 얼굴을 벗어나 상체로 내려오는 모습은 이 조각에 없다.',
 'back': 'f69부터 손은 이미 가슴 부근이다. f69–82에는 손목·주먹의 방향과 팔꿈치 높이가 변하지만, 얼굴 앞에서 가슴으로 내려오는 과정은 보이지 않는다.',
 'join': 'f60의 얼굴 앞 손 다음 표시가 f69의 가슴 부근 손이다. 뒤에 남은 손목 회전과 팔꿈치 내림을 이 두 위치 사이의 실제 하강 영상으로 대체할 수 없다.'},
2: {
 'setup': [52,57,61], 'terminal': [72,74,80,83],
 'front': 'f52–61의 오른손은 입·코 앞에 있고, 상체에 도착하는 하강은 아직 표시되지 않는다.',
 'back': 'f72–83은 가슴 높이에서 시작하여 주먹 방향이 바뀌고 팔꿈치가 내려간다. 끝에는 주먹이 어깨 부근에 있으나 얼굴 앞에서 이곳까지 이어진 하강은 이 조각에 없다.',
 'join': 'f61의 얼굴 앞 모습 뒤에 f72의 가슴 부근 모습이 바로 나온다. 보이는 뒤 조각의 주먹 방향 변화는 face→upperbody 연결 이동이 아니다.'},
3: {
 'setup': [57,62,68], 'terminal': [80,82,89,93],
 'front': 'f57–68에서 오른손은 얼굴 앞에 남아 있다. 손가락 주변의 미세 변화가 있어도 얼굴과 가슴 사이를 내려오는 과정은 보이지 않는다.',
 'back': 'f80–82에는 가슴 부근 손과 수평에 가까운 팔이 보인다. 이후 f83–93에는 주먹이 세워지고 팔꿈치가 내려가며 손은 어깨 부근으로 조금 올라간다. 얼굴에서 내려온 경로는 없다.',
 'join': 'f68의 얼굴 앞 손에서 f80의 가슴 부근 손으로 표시가 바로 바뀐다. f80–93의 실제 팔·손목 변화는 남지만, 누락된 얼굴→가슴 하강의 진행 근거는 아니다.'},
4: {
 'setup': [74,80,87], 'terminal': [100,104,110,112],
 'front': 'f74–87 내내 오른손이 코·입 앞에 유지된다. 몸 자세의 작은 변화는 보이나 상체 도착까지의 손 하강은 보이지 않는다.',
 'back': 'f100–104에서 손은 이미 가슴 부근이며, 이후 f105–112에는 손목 방향이 달라지고 팔꿈치가 내려간다. 얼굴 앞 출발 모습이나 그곳에서 내려온 길은 이 조각에 없다.',
 'join': 'f87 얼굴 앞 모습 다음은 f100의 가슴 부근 모습이다. 뒤쪽 팔꿈치 내림은 관측되지만 얼굴 앞 손이 내려오는 중간 경로를 보여 주지는 않는다.'},
5: {
 'setup': [40,46,53], 'terminal': [65,68,71],
 'front': 'f40–53에서 오른손이 입·코 앞에 유지된다. 마지막에도 얼굴 부근이며 가슴에 닿는 하강 과정은 없다.',
 'back': 'f65–68은 손이 가슴 윗부분에 있는 상태다. f69–77에는 팔꿈치가 내려가고 손·주먹 방향이 달라지며 끝의 주먹은 턱 아래에 가까워진다. 이 방향 변화와 작은 상승은 얼굴→가슴 하강의 다른 발생을 보여 주지 않는다.',
 'join': 'f53의 얼굴 앞 손에서 f65의 상체 부근 손으로 바로 바뀐다. 뒤쪽에 가슴에서 턱 아래 쪽으로 움직이는 손은 보이지만, 편집 사이의 얼굴→가슴 진행 모습은 남아 있지 않다.'},
6: {
 'setup': [59,64,70], 'terminal': [81,84,90],
 'front': 'f59–70에서 오른손은 입·코 앞이다. 약간의 몸 흔들림과 손 변화가 있으나 가슴으로 내려오는 경로는 없다.',
 'back': 'f81–84부터 손은 가슴 부근이다. f85–90에는 손목·주먹 방향과 팔꿈치 높이가 달라지고 손이 조금 올라가지만, 얼굴 앞에서 내려오는 구간은 보이지 않는다.',
 'join': 'f70의 얼굴 앞 손 다음은 f81의 가슴 부근 손이다. 뒤에 남은 주먹 방향 변화와 팔꿈치 이동만으로 연결 하강을 확인할 수 없다.'},
7: {
 'setup': [40,45,51], 'terminal': [64,65,71,75],
 'front': 'f40–51에서 몸의 높이는 조금 달라져도 손은 계속 얼굴 앞 관계를 유지한다. 화면상 몸 움직임을 손의 얼굴→상체 하강으로 세지 않았다.',
 'back': 'f64–75는 가슴 부근 손에서 시작한다. f64–70에 팔꿈치가 내려가고 주먹 방향이 달라지며 f71–75에는 가슴 부근의 세워진 주먹이 남는다. 얼굴에서 내려오는 실제 경로는 없다.',
 'join': 'f51의 얼굴 앞 손 다음은 f64의 가슴 부근 손이다. f64 이후 실제 손목·팔 변화는 보존했지만 이를 앞의 빠진 하강 과정으로 보지 않았다.'},
9: {
 'setup': [40,46,53], 'terminal': [63,64,70,76],
 'front': 'f40–53에서 오른손이 입·코 앞에 유지된다. 손이 얼굴을 떠나 상체까지 내려오는 장면은 이 조각에서 관측되지 않는다.',
 'back': 'f63–65는 이미 가슴 높이의 손과 가로 방향 팔이다. f66–76에는 손목이 돌아가고 팔꿈치가 낮아져 세워진 주먹이 어깨 부근에 남는다. 얼굴 앞에서 상체로 내려오는 또 다른 이동은 보이지 않는다.',
 'join': 'f53 얼굴 앞 손 다음 f63은 이미 가슴 부근이다. 뒤 조각에서 실제 방향 변화가 있지만, 얼굴→가슴 사이의 진행 장면이 표시되지 않아 양 끝 자세만으로 연결 과정을 확인하지 않았다.'}
}

manifest = json.loads(M.read_text(encoding='utf-8'))
views = [v for v in manifest['views'] if 'WORD0739' in v['query_id'] and
         v['construction_kind_audit_only'] in ('prefix','suffix','middle_splice')]
assert len(views) == 24
rows=[]
seen_pages=set()
for v in views:
    person=v['recorded_person_id']; n=NOTES[person]
    kind=v['construction_kind_audit_only']
    assert v['role']=='fit' and person in NOTES and person != 8
    fm=v['ordered_frame_map']; frame_to_index={e['source_native_frame']:e['output_index'] for e in fm}
    assert len(frame_to_index)==len(fm)
    assert all(e['image_available'] and e['pixel_operation']=='copy' for e in fm)
    page_out=[]; indices=[]
    for p in v['review_pages']:
        stem=Path(p['path']).stem
        assert stem in VIEWED and sha(p['path'])==p['sha256']
        seen_pages.add(stem)
        page_out.append({'path':p['path'],'sha256':p['sha256']})
        indices.extend(p['output_indices'])
    assert indices==[e['output_index'] for e in fm]
    def witnesses(frames):
        return [{'source_frame':f,'output_index':frame_to_index[f]} for f in frames if f in frame_to_index]
    fragments=[]
    for j,(lo,hi) in enumerate(v['retained_contiguous_native_fragments']):
        isfront=lo<=n['setup'][0]<=hi
        text=n['front'] if isfront else n['back']
        fragments.append({'fragment_index':j,'source_frame_bounds':[lo,hi],
          'output_index_bounds':[frame_to_index[lo],frame_to_index[hi]],
          'visible_evidence':text,'queried_process_supported_in_fragment':False})
    reason=n['front'] if kind=='prefix' else n['back'] if kind=='suffix' else n['join']
    rows.append({
      'view_id':v['view_id'],'query_id':v['query_id'],'clip_id':v['clip_id'],
      'recorded_person_id':person,'role':'fit','physical_query':v['physical_query'],
      'pixel_view_key':v['pixel_view_key'],'source_path':v['source_path'],'source_sha256':v['source_sha256'],
      'reviewed_pages':page_out,'reviewed_output_indices':indices,
      'observed_process_evidence':0,'visible_chain':'partial','evidence_reason':reason,
      'remaining_evidence_by_fragment':fragments,
      'setup_witnesses':witnesses(n['setup']),'motion_witnesses':[],
      'terminal_witnesses':witnesses(n['terminal']),
      'motion_witness_scope':'Only actual visible face-to-upperbody descent belongs in this field; observed later wrist rotation/elbow change is described separately above.',
      'seam_output_indices_audit':v['seam_output_indices'],
      'whole_word_correctness':None,'learner_error':None,'required_repetitions':None,
      'zero_scope':'The displayed retained images do not support an observed face-to-upperbody process. This does not claim that the source signer omitted the movement, or that this is an incorrect sign.',
      'not_assessed':['whole guide clause','whole sign','linguistic acceptability','exact start/end boundary','continuous video playback timing','fine finger joint correctness']
    })
assert seen_pages==VIEWED and len(seen_pages)==29
assert len({r['query_id'] for r in rows})==8
assert all(r['whole_word_correctness'] is None and r['learner_error'] is None for r in rows)
result={
 'at':datetime.now(timezone.utc).isoformat(),
 'status':'DIRECT_EDITED_VIEW_REVIEW_COMPLETE_NOT_EXPERT_GOLD',
 'reviewer':'Codex /root/predicate_contract_review; same-Codex non-expert visual review',
 'input_manifest':{'path':str(M),'sha256':sha(M)},
 'method':'Opened every listed new review sheet using view_image and inspected all displayed retained images. Authored per-person observations after viewing; this serialization helper only attaches exact frame indices and hashes.',
 'label_definition':{'1':'visible queried relational process is directly supported within the displayed retained material',
   '0':'visible retained material has only fragments/other movements and does not support that observed process; not absence in the original person or a sign error',
   'null':'insufficient or ambiguous visible material to make this physical-view judgment'},
 'limitations':['24 dependent views from 8 FIT source clips, not 24 independent sign performances',
   'Same-Codex physical annotation, not independent human/KSL expert judgment',
   'Sheet inspection of all retained source images, not actual edited-video speed or perceptual playback validation',
   'No requirement to hold still, no fixed body-height threshold, no repetition count or whole-action completeness truth',
   'ID08 normal high terminal variation remains unmodified in original review; no partial counterviews were fabricated',
   'All three edit types were inspected; no rule maps edit kind, missing source frames, or endpoints alone to a language label'],
 'inspection_failure_preserved':{'tool_chunk_id':'1dfe0d','error':'StopIteration in read-only metadata inspection',
   'cause':'Inspection attempted nonexistent kind instead of construction_kind_audit_only',
   'correction':'Read documented manifest key; no source files, sheets, labels, or models changed'},
 'counts':{'views':len(rows),'source_clips':8,'queries':8,'reviewed_pages':len(seen_pages),
   'reviewed_output_image_slots':sum(len(r['reviewed_output_indices']) for r in rows),
   'unique_source_images':len({(v['clip_id'],e['source_native_frame']) for v in views for e in v['ordered_frame_map']}),
   'positive':0,'unsupported_observed_process':24,'unknown':0},
 'new_inference':0,'new_training':0,'existing_labels_modified':False,'whole_goal_complete':False,
 'rows':rows}
assert not OUT.exists(), 'Preserve existing review instead of overwriting'
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'counts':result['counts']},ensure_ascii=False))
