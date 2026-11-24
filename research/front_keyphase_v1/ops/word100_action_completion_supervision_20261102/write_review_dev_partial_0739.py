"""Serialize direct displayed-image observations; no model or automatic labeler."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

A = Path(__file__).resolve().parent
M = A / 'DEV_PARTIAL_VIEWS_MANIFEST.json'
OUT = A / 'REVIEW_DEV_PARTIAL_0739.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

VIEWED = set('''
3512fdbdc24f54cfd923_p1 ad0e7e8765e8c7671887_p1 4f2b810480f03b4f8177_p1
39e906bed7bd232b3c08_p1 39e906bed7bd232b3c08_p2
09512928ffd610da1430_p1 09512928ffd610da1430_p2
059f719cf158fd45cb79_p1 66518fc1898dad95961e_p1 66518fc1898dad95961e_p2
aca04029b2bd1bc27645_p1 b2c2e2826d3ede5730e2_p1
c8e78d268f4d108472a6_p1 c8e78d268f4d108472a6_p2
0ad0c8401ff8803229f4_p1 53a9e62ae7694bd5be4c_p1
64c56ec785d6b3eebffe_p1 64c56ec785d6b3eebffe_p2
'''.split())

# These are per-view observations authored after opening all listed sheets.
# Construction names index the notes only; they do not generate their labels.
NOTES = {
 (11, 'prefix'): (0, 'partial', [59,63,66,68], [], [],
  'f59–68에서 오른손은 코·입 앞에 남아 있다. 약간의 자세 변화는 있지만 얼굴 앞을 떠나 가슴으로 내려오는 진행 모습이나 도착 모습은 이 화면에 없다.'),
 (11, 'suffix'): (0, 'partial', [], [], [78,81,83],
  'f78부터 손은 이미 가슴 앞이다. 뒤에는 손목·주먹 방향이 달라지고 팔꿈치가 내려가며 손이 어깨 옆으로 올라간다. 이 실제 움직임은 남지만 얼굴 앞에서 가슴으로 내려오는 과정은 보이지 않는다.'),
 (11, 'middle_splice'): (0, 'partial', [59,63,66,68], [], [78,81,83],
  'f68의 얼굴 앞 손 다음 표시가 f78의 가슴 앞 손이다. 앞뒤 관계는 보이나 그 사이를 내려오는 진행 모습이 없다. 뒤의 손목 방향 변화와 팔꿈치 내림을 빠진 얼굴→가슴 이동으로 세지 않았다.'),
 (11, 'speed_0_8'): (1, 'yes', [63,66,68], [(68,78)], [78,81],
  '실제 표시된 연속 이미지에서 f68 얼굴 앞→f69–73 입·턱 아래→f74–78 가슴 앞 이동이 보이고 f78–81의 도착 관계도 남는다. 이는 순서대로 남은 이미지의 과정 근거이며 실제 0.8배속 영상의 인지·모델 성능 검증은 아니다.'),
 (11, 'speed_1_25'): (1, 'yes', [63,66,68], [(68,78)], [78,81],
  '이 시트의 두 페이지도 직접 열람했다. f68 얼굴 앞에서 f69–73 아래 얼굴·목 부근을 거쳐 f74–78 가슴 앞으로 이동한다. 0.8배 조건과 같은 순서의 원픽셀이며 표시 시각만 다르다. 실제 빠른 재생이나 추론 정확도를 시험한 결과가 아니다.'),
 (11, 'intact_context_trim'): (1, 'yes', [66,68], [(68,78)], [78,81],
  'f66–68의 얼굴 앞 손, f69–77의 하강 진행, f78–81의 가슴 앞 손이 모두 이어서 보인다. 앞뒤 여백이 짧아도 이 물리 과정의 근거는 남는다. 전체 수어가 완료되었다는 뜻은 아니다.'),
 (11, 'missing_rgb'): (None, 'uncertain', [59,63,66,67], [], [79,81,83],
  'f59–67의 얼굴 앞 손과 f79–83의 가슴 앞 손은 보이나 f68–78 화면이 회색이다. 하강이 진행되었는지 현재 화면만으로 확인할 수 없다. 앞뒤 자세나 원본 기억으로 가려진 경로를 채우지 않아 미확인으로 남긴다. 이후 손목·팔 움직임은 별개로 보인다.'),
 (12, 'prefix'): (0, 'partial', [43,49,55,60], [], [],
  'f43–60에서 오른손은 입·코 앞에 유지된다. 얼굴과 손 주변에 작은 변화가 있지만 가슴 앞에 이르는 하강은 표시되지 않는다.'),
 (12, 'suffix'): (0, 'partial', [], [], [70,73,75,80],
  'f70부터 손은 상체 앞이고 팔은 옆으로 벌어져 있다. f76–80에는 주먹 방향과 팔꿈치가 달라지지만 손이 얼굴에서 내려오는 과정은 이 조각에 없다.'),
 (12, 'middle_splice'): (0, 'partial', [43,49,55,60], [], [70,73,75,80],
  'f60의 얼굴 앞 손 다음에 f70의 상체 앞 손이 바로 표시된다. 두 위치는 관측되나 연결 하강의 중간 모습은 없다. f76 이후 실제 방향 변화는 보존하되 표적 하강의 진행으로 대체하지 않았다.'),
 (13, 'prefix'): (0, 'partial', [68,72,76,78], [], [],
  'f68–78에서 고개가 조금 숙여진 채 오른손이 얼굴 앞에 남아 있다. 이 화면에는 가슴까지 내려오는 진행이나 도착 모습이 없다.'),
 (13, 'suffix'): (0, 'partial', [], [], [89,93,99,102],
  'f89부터 오른손은 가슴 앞이다. f94–105에서 주먹이 세워지고 팔꿈치가 낮아지는 실제 변화는 보이지만 얼굴 앞에서 상체로 내려오는 또 다른 이동은 보이지 않는다.'),
 (13, 'middle_splice'): (0, 'partial', [68,72,76,78], [], [89,93,99,102],
  'f78의 얼굴 앞 모습 다음은 f89의 가슴 앞 모습이다. f89 이후 손목·팔 변화는 남았지만 얼굴에서 상체로 내려오는 중간 진행은 표시되지 않는다. 양 끝 자세의 존재만으로 연결 과정을 확정하지 않았다.'),
}

manifest = json.loads(M.read_text(encoding='utf-8'))
views = [v for v in manifest['views'] if 'WORD0739' in v['query_id']
         and v['construction_kind_audit_only'] != 'native']
assert len(views) == len(NOTES) == 13
rows, seen_pages = [], set()
for v in views:
    key = (v['recorded_person_id'], v['construction_kind_audit_only'])
    label, chain, setups, motions, terminals, reason = NOTES[key]
    assert v['role'] == 'development' and v['recorded_person_id'] in (11,12,13)
    fm = v['ordered_frame_map']
    by_frame = {e['source_native_frame']: e for e in fm}
    assert len(by_frame) == len(fm)
    pages, indices = [], []
    for p in v['review_pages']:
        stem = Path(p['path']).stem
        assert stem in VIEWED and sha(p['path']) == p['sha256']
        seen_pages.add(stem)
        pages.append({'path':p['path'],'sha256':p['sha256']})
        indices.extend(p['output_indices'])
    assert indices == [e['output_index'] for e in fm]

    def witness(frame):
        e = by_frame[frame]
        assert e['image_available'] and e['pixel_operation'] == 'copy'
        return {'source_frame':frame,'output_index':e['output_index'],
          'source_timestamp_us':e['source_native_timestamp_us'],
          'view_timestamp_us':e['view_timestamp_us']}

    motion_rows = []
    for lo, hi in motions:
        entries = [witness(f) for f in range(lo, hi + 1)]
        assert [e['output_index'] for e in entries] == list(range(entries[0]['output_index'],entries[-1]['output_index']+1))
        motion_rows.append({'source_frames':[e['source_frame'] for e in entries],
          'output_indices':[e['output_index'] for e in entries],
          'source_timestamps_us':[e['source_timestamp_us'] for e in entries],
          'view_timestamps_us':[e['view_timestamp_us'] for e in entries],
          'scope':'Visible face-to-upperbody relational change across this interval, not a dense frame label or exact event boundary.'})

    fragments=[]
    for j,(lo,hi) in enumerate(v['retained_contiguous_native_fragments']):
        entries = [e for e in fm if lo <= e['source_native_frame'] <= hi]
        contains_full_motion = any(lo<=a and b<=hi for a,b in motions)
        fragments.append({'fragment_index':j,'source_frame_bounds':[lo,hi],
          'output_index_bounds':[entries[0]['output_index'],entries[-1]['output_index']],
          'source_timestamp_us_bounds':[entries[0]['source_native_timestamp_us'],entries[-1]['source_native_timestamp_us']],
          'view_timestamp_us_bounds':[entries[0]['view_timestamp_us'],entries[-1]['view_timestamp_us']],
          'visible_evidence':reason,
          'unavailable_source_frames':[e['source_native_frame'] for e in entries if not e['image_available']],
          'queried_process_supported_in_fragment':None if label is None else contains_full_motion})
    row={
      'view_id':v['view_id'],'query_id':v['query_id'],'clip_id':v['clip_id'],
      'recorded_person_id':v['recorded_person_id'],'role':'development',
      'physical_query':v['physical_query'],'pixel_view_key':v['pixel_view_key'],
      'source_path':v['source_path'],'source_sha256':v['source_sha256'],
      'reviewed_pages':pages,'reviewed_output_indices':indices,
      'observed_process_evidence':label,'visible_chain':chain,'evidence_reason':reason,
      'remaining_evidence_by_fragment':fragments,
      'setup_witnesses':[witness(f) for f in setups],
      'motion_witnesses':motion_rows,'terminal_witnesses':[witness(f) for f in terminals],
      'motion_witness_scope':'Only visible face-to-upperbody descent belongs here; later wrist rotation/elbow change is distinct evidence.',
      'seam_output_indices_audit':v['seam_output_indices'],
      'unavailable_output_indices':[e['output_index'] for e in fm if not e['image_available']],
      'whole_word_correctness':None,'learner_error':None,'required_repetitions':None,
      'zero_scope':'0 means these retained images do not support the queried process; not a source signer error, a whole-sign omission, or a missing-frame negative.',
      'not_assessed':['whole guide clause','whole sign','linguistic acceptability','exact onset/offset','actual edited-video playback speed','fine finger correctness'],
      'training_allowed':False,'evaluation_only':True,
    }
    if key[1].startswith('speed_'):
        row['speed_scope']={'all_pages_directly_viewed':True,'ordered_pixel_evidence_only':True,
          'same_ordered_pixels_as_other_speed':True,'actual_speed_performance_tested':False,
          'evidence_not_an_independent_new_performance':True}
    rows.append(row)

assert seen_pages == VIEWED and len(VIEWED) == 18
assert sum(r['observed_process_evidence']==1 for r in rows) == 3
assert sum(r['observed_process_evidence']==0 for r in rows) == 9
assert sum(r['observed_process_evidence'] is None for r in rows) == 1
result={
 'at':datetime.now(timezone.utc).isoformat(),
 'status':'DIRECT_DEVELOPMENT_VIEW_REVIEW_COMPLETE_NOT_EXPERT_GOLD',
 'reviewer':'Codex /root/predicate_contract_review; same-Codex non-expert visual review',
 'input_manifest':{'path':str(M),'sha256':sha(M)},
 'method':'All 18 new sheets opened with view_image and every displayed image slot inspected. Per-view observations authored before model predictions. This helper only attaches exact saved mappings, hashes and timestamps.',
 'label_definition':{'1':'Displayed material directly supports the queried relational process.',
   '0':'Visible retained fragments do not support that process; not linguistic incorrectness or missing frames alone.',
   'null':'Current images are insufficient or ambiguous; original-video knowledge does not fill unavailable frames.'},
 'counts':{'views':len(rows),'source_clips':3,'queries':3,'reviewed_pages':len(seen_pages),
   'reviewed_output_image_slots':sum(len(r['reviewed_output_indices']) for r in rows),
   'unavailable_image_slots':sum(len(r['unavailable_output_indices']) for r in rows),
   'unique_source_frame_ids':len({(v['clip_id'],e['source_native_frame']) for v in views for e in v['ordered_frame_map']}),
   'positive':3,'unsupported_observed_process':9,'unknown':1},
 'limitations':['13 dependent views from 3 public development videos, not 13 independent performances.',
   'Repeated-development evaluation labels only, never FIT/training or candidate tuning targets.',
   'Same-Codex physical annotation is not independent human/KSL expert gold.',
   'Actual continuous edited-video speed and model speed robustness are not established by sheet inspection.',
   'All three removal forms and all four sentinel forms were directly inspected; no edit-kind-to-label rule.',
   'Only the bounded face-to-upperbody physical process is assessed; no whole-word completion, required repetition count, or educational correctness.'],
 'new_model_inference':0,'new_training':0,'old_native_reviews_modified':False,
 'training_allowed':False,'independent_final_test':False,'whole_goal_complete':False,
 'rows':rows}
assert not OUT.exists(), 'Do not overwrite a completed direct review'
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'counts':result['counts']},ensure_ascii=False))
