"""Count and pin only this agent's recorded visual review; no model/detector."""
from pathlib import Path
import json,hashlib,collections,datetime
U=Path(__file__).resolve().parent
M=U/'review_all_flags_materials'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ledger=U/'review_all_flags_protocol_ledger.jsonl'
rows=[json.loads(x) for x in ledger.read_text('utf-8').splitlines()]
partial_path=U/'review_all_flags_protocol_PARTIAL_DETAIL_45_48.json'
partial=read(partial_path)['rows']
expected={}
pages=[]
for page in sorted(M.glob('batch_2_overview_v3_*.json')):
 p=read(page);pages.append(p)
 for r in p['rows']:
  assert r['condition_review_id'] not in expected
  expected[r['condition_review_id']]=r
assert len(pages)==55 and len(expected)==272
assert len(rows)==228 and len({r['window_id'] for r in rows})==228
assert {r['window_id'] for r in rows}=={r['window_id'] for r in expected.values()}
imgs={};condition_rows=[];unique_frames=set();unique_overlay=set()
for r in rows:
 m=read(r['material_manifest'])
 assert sha(r['material_manifest'])==r['material_manifest_sha256']
 assert m['record_sha256']==r['record_sha256'] and m['source_sha256']==r['source_sha256']
 assert r['conditions']==m['job']['conditions']
 assert not r['every_source_frame_viewed'] and not r['expert_semantic_guarantee'] and not r['training_allowed_by_this_row']
 assert not r['additional_frames_needed'] and not r['unresolved_essential_geometric_cut']
 for im in r['viewed_images']:
  assert sha(im['path'])==im['sha256'];imgs[im['path']]=im['sha256']
 for f in r['actual_viewed_frames']:unique_frames.add((r['clip_id'],f))
 # The first two ledger rows predate v3 and contain complete explicitly viewed
 # detailed pages. Every displayed detailed row has native/masked/sensor overlays.
 overlay=r.get('actual_overlay_compared_frame_ids',r['actual_viewed_frames'])
 for f in overlay:unique_overlay.add((r['clip_id'],f))
 nested=r.get('condition_reviews')
 if nested:
  assert len(nested)==len(r['conditions'])
  for c in nested:
   e=expected[c['condition_review_id']]
   assert e['window_id']==r['window_id'] and e['condition']==c['condition']
   assert c['observation_ko'] and not c['unresolved_essential_geometric_cut']
   assert set(c['actual_overlay_compared_frame_ids']) & set(c['condition']['focus_frame_ids'])
   condition_rows.append(dict(condition_review_id=c['condition_review_id'],window_id=r['window_id'],comparison=c['condition']['comparison'],side=c['condition']['side'],review_note=c['observation_ko']))
 else:
  assert r['window_id'] in ('W0002','W0005')
  assert r['observation_ko']
  for i,c in enumerate(r['conditions'],1):
   key=r['window_id']+f'_C{i:02d}'
   assert expected[key]['condition']==c
   assert set(overlay)&set(c['focus_frame_ids'])
   condition_rows.append(dict(condition_review_id=key,window_id=r['window_id'],comparison=c['comparison'],side=c['side'],review_note=r['observation_ko'],note_scope='Earlier detailed ledger note covers all listed conditions; explicit sampled overlay frames intersect every condition.'))
for r in partial:
 assert sha(r['material_manifest'])==r['material_manifest_sha256']
 for im in r['viewed_images']:
  assert sha(im['path'])==im['sha256'];imgs[im['path']]=im['sha256']
 base=next(x for x in rows if x['window_id']==r['window_id'])
 for f in r['actual_viewed_frames']:unique_frames.add((base['clip_id'],f))
 for f in r['actual_overlay_compared_frame_ids']:unique_overlay.add((base['clip_id'],f))
assert len(condition_rows)==len({r['condition_review_id'] for r in condition_rows})==272
assert {r['condition_review_id'] for r in condition_rows}==set(expected)
# Manual finding labels from actual visual notes, not thresholds or auto-PASS.
inside_sensor={
 'W0002_C02','W0005_C01','W0008_C01','W0017_C02',
 'W0032_C01','W0032_C02','W0053_C02','W0071_C02',
 'W0101_C02','W0104_C02','W0110_C01','W0143_C02','W0146_C02',
 'W0164_C01','W0224_C01','W0236_C02','W0239_C01','W0245_C01',
 'W0269_C02','W0272_C02','W0284_C02','W0296_C02','W0365_C01',
 'W0446_C01','W0470_C01',
}
sensor={r['condition_review_id'] for r in condition_rows if r['comparison']=='mask_to_sensor'}
assert inside_sensor<=sensor
counts=dict(collections.Counter(r['comparison'] for r in condition_rows))
for c in condition_rows:
 c['finding_category']='INSIDE_ROI_SENSOR_LOSS_OR_ASSIGNMENT_RISK' if c['condition_review_id'] in inside_sensor else ('BOUNDARY_REMAINDER_SENSOR_LOSS_RISK' if c['comparison']=='mask_to_sensor' else 'NO_NEW_ESSENTIAL_GEOMETRIC_CUT_OBSERVED_IN_SAMPLES')
result=dict(
 status='BATCH2_ALL_FLAGGED_CONDITIONS_SAMPLED_VISUAL_REVIEW_COMPLETE',
 at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer='roi_training_protocol',
 batch=2,window_count=len(rows),condition_count=len(condition_rows),overview_pages_actually_viewed=55,
 overview_pages_covered_in_ledger=len({im for im in imgs if 'overview_v3_' in im}),
 viewed_image_count=len(imgs),unique_clip_frame_count=len(unique_frames),unique_clip_overlay_frame_count=len(unique_overlay),
 comparison_counts=counts,inside_roi_sensor_or_assignment_risk_conditions=len(inside_sensor),boundary_remainder_sensor_risk_conditions=len(sensor-inside_sensor),
 missing_windows=0,missing_conditions=0,duplicate_windows=0,duplicate_conditions=0,hash_or_link_errors=0,
 new_essential_geometric_cut_observed=0,unresolved_additional_frame_requests=0,
 conditions=condition_rows,
 evidence=[dict(path=str(ledger),sha256=sha(ledger)),dict(path=str(partial_path),sha256=sha(partial_path))],
 findings_ko=[
  '하단 ROI 때문에 새로 잘린 필수 움직임은 실제 열람한 표본에서 관찰되지 않았다. 대부분 한손 표현의 반대 휴식손 또는 두손 표현 사이의 손 내림·재진입/종료 구간이었다.',
  'ROI 안에 실제 손이 있어도 sensor 보기에서 추가 손실·좌우배정 차이가 있는 조건은 별도 위험으로 유지한다. 이를 정상 수행 또는 정보 보존 PASS로 바꾸지 않는다.',
  '초대권의 받침손, 수영·후원·추출·무산의 겹침/접촉, 여러 복합표현의 손 재진입 구간은 sensor 관측의 잔존 한계이다. native/masked 관측과 구별해야 한다.',
  '숫자 표현 중 초기·종료에 개요가 몰린 4개 영상은 p02 상세만 추가 확인했다. 주동손의 가슴·얼굴 앞 동작과 아래 정지한 반대손을 구분했다.',
 ],
 limitations=[
  'Every flagged condition was reviewed through explicit sampled RGB/overlays; every source frame was NOT viewed.',
  'This is visual geometric/observation review, NOT expert linguistic semantic preservation, classification performance, or new-user generalization proof.',
  'Sensor risks remain; this batch alone never authorizes training or produces overall READY.',
  'No detector run, model inference, training, source/frozen rule modification or user/DEV data examination occurred in this review.',
 ],training_allowed_by_this_review=False,overall_ready_claim=False,
)
out=U/'review_all_flags_protocol_RESULT.json'
assert not out.exists()
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('conditions','findings_ko','limitations','evidence')},ensure_ascii=True))
