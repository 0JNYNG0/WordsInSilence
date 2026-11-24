"""Serialize manually inspected nominal-span observations, not model predictions."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone

U=Path(__file__).resolve().parent
AB=U.parent/'word100_action_completion_supervision_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
M=U/'FIRST_RUNTIME_INTERVAL_REVIEW_MANIFEST.json'
manifest=read(M)
native=read(AB/'PROCESS_REVIEW_MANIFEST.json')
# Each entry follows actual view_image inspection of every listed new page.
# The label describes the visible near-to-apart physical process, not completion
# of the whole sign, final hold, or the old annotation's exact endpoint.
notes={
1:dict(label=1,setup=[33,36,38],motion=[38,42],terminal=[41,42],reason='명목 구간 30–42 안에서 올린 양손이 33·36·38에 서로 가깝고, 38–42의 연속 화면에서 양쪽으로 벌어진다. 41·42에는 두 손 사이의 분리가 보인다. 회색 43 이후의 더 벌어진 모습이나 유지 동작을 가져오지 않아도 가까움→벌어짐이 관측된다.'),
2:dict(label=1,setup=[51,54],motion=[54,58],terminal=[58],reason='명목 48–58에서 51·54의 모인 양손과 54–58의 벌어지는 연속 과정, 58의 분리된 배치가 보인다. 59–62의 회색 후속 화면을 정답 근거에 포함하지 않았다. 충분히 오래 유지했는지 또는 단어가 끝났는지를 판정한 것이 아니다.'),
3:dict(label=1,setup=[62,66,67],motion=[67,70],terminal=[70],reason='명목 56–70에서 양손이 62·66·67에 가까이 있고 67–70에 벌어진다. 70에서 손 사이가 분리된 모습까지 실제 보인다. 변화가 짧아도 실제 연속 관측이 있어 1이며, 회색 71 이후의 더 큰 벌어짐이나 이전 주석의 끝점 77을 요구하지 않는다.'),
4:dict(label=1,setup=[84,87],motion=[87,89],terminal=[89],reason='명목 74–89의 대부분은 가까운 양손 배치 유지이지만, 마지막 87–89에서 실제 손끝 사이가 벌어진다. 작게 나온 시트만으로 확정하지 않고 기존 큰 원프레임 87·89도 직접 확인했다. 87의 서로 근접한 손끝과 89의 분리된 손끝 및 그 사이 88이 보인다. 짧은 물리 변화의 관측이지 최종 자세나 단어 전체 완료 판정은 아니다.'),
5:dict(label=1,setup=[59,62],motion=[62,67],terminal=[66,67],reason='명목 55–67에서 59·62의 모인 배치가 62–67의 연속 벌어짐으로 이어지고 66·67에는 두 손이 분리되어 있다. 68 이후 회색 화면을 끌어오지 않고도 해당 과정이 보인다.'),
6:dict(label=1,setup=[61,66],motion=[66,72],terminal=[71,72],reason='명목 56–72 안에서 양손 근접 배치가 61·66에 보이고, 66–72에 연속으로 손이 벌어진다. 71·72의 양손 분리는 명목 안에 있다. 두 번째 페이지의 76은 회색 참고 화면일 뿐 과정 정답에 사용하지 않았다.'),
7:dict(label=1,setup=[41,44],motion=[44,49],terminal=[48,49],reason='명목 35–49 안에서 41·44의 서로 가까운 양손이 44–49에 벌어지고, 48·49의 분리 상태가 보인다. 50 이후 회색 참고 화면이 없어도 이 좁은 물리 과정은 관측된다.'),
8:dict(label=1,setup=[50,54],motion=[54,58],terminal=[57,58],reason='명목 45–58에서 50·54의 양손 근접 배치, 54–58의 연속 벌어짐, 57·58의 분리된 손 배치를 확인했다. 회색 후속 장면의 더 큰 벌어짐이나 유지 시간은 요구하거나 가져오지 않았다.'),
9:dict(label=1,setup=[62,66],motion=[66,70],terminal=[69,70],reason='명목 57–70 안에서 62·66의 가까운 두 손이 66–70의 연속 화면에서 벌어진다. 69·70에는 손 사이가 분리된 배치가 실제 보인다. 회색 71 이후의 자세를 정답에 포함하지 않았고 단어 전체 동작 완료나 필요 거리의 충분함은 판단하지 않았다.')}
rows=[]
for m in manifest['rows']:
 if m['word_id']!='WORD2523':continue
 n=notes[m['person']];axis={e['source_frame']:e for e in m['display_axis']}
 def witness(frame):
  e=axis[frame];assert e['nominal']
  return dict(source_frame=frame,source_timestamp_us=e['source_timestamp_us'],output_index=e['proposal_output_index'])
 pages=[]
 for p in m['review_pages']:
  assert sha(p['path'])==p['sha256']
  pages.append(dict(p,actually_viewed=True))
 extra=[]
 if m['person']==4:
  old=next(x for x in native['rows'] if x['id']==m['query_id'])
  for f in [87,89]:
   e=next(x for x in old['frames'] if x['frame']==f);assert sha(e['path'])==e['sha256']
   extra.append(dict(e,actually_viewed=True,purpose='Magnify visible near/separated hand relation within nominal span.'))
 frames=list(range(n['motion'][0],n['motion'][1]+1))
 motion=[witness(f) for f in frames]
 rows.append(dict(review_id=m['review_id'],case_id=m['case_id'],query_id=m['query_id'],role='fit',person=m['person'],
  word_id=m['word_id'],physical_query=m['physical_query'],source_sha256=m['source_sha256'],input_sha256=m['input_sha256'],
  output_indices=m['nominal_output_indices'],source_frames=m['nominal_source_frames'],source_timestamps_us=m['nominal_source_timestamps_us'],
  label=n['label'],observed_process_evidence=n['label'],visible_chain='yes',reason=n['reason'],
  reviewed_pages=pages,additional_native_images_reviewed=extra,
  reviewed_nominal_output_indices=m['nominal_output_indices'],reviewed_context_frames=[e['source_frame'] for e in m['display_axis'] if not e['nominal']],
  setup_witnesses=[witness(f) for f in n['setup']],
  motion_witnesses=[dict(source_frames=frames,source_timestamps_us=[e['source_timestamp_us'] for e in motion],output_indices=[e['output_index'] for e in motion],
    label=1,scope='Directly visible continuous outward change within nominal; not exact onset/offset.')],
  terminal_witnesses=[witness(f) for f in n['terminal']],
  nominal_scope_reviewed=True,label_scope='nominal_visible_process_only',label_inferred_from_whole_view=False,
  rgb_context_outside_nominal_possible=True,context_frames_used_to_supply_missing_process=False,
  process_definition='Raised hands mutually near then visibly moving apart to separated arrangement; no final hold or minimum excursion criterion.',
  whole_word_correctness=None,learner_error=None,required_repetitions=None,exact_completion_boundary=None))
assert len(rows)==9
out=dict(status='DIRECT_NOMINAL_PROCESS_REVIEW_COMPLETE',at=datetime.now(timezone.utc).isoformat(),
 reviewer='Codex direct image inspection; not an independent human or Korean Sign Language expert.',
 prior_results_known_not_blinded=True,model_predictions_used_as_labels=False,
 manifest=dict(path=str(M),sha256=sha(M)),rows=rows,
 counts=dict(queries=9,actually_viewed_new_pages=10,nominal_frame_slots=sum(len(r['source_frames']) for r in rows),
 context_frame_slots=sum(len(r['reviewed_context_frames']) for r in rows),additional_existing_native_images_viewed=2,
 observed_process_1=9,observed_process_0=0,observed_process_unknown=0),
 limits=['The physical near-to-apart process is narrow. All nine positive labels do not prove whole-word correctness.',
 'No exact movement boundary, 3D contact, required repetition count, sufficient extent, or educational tolerance truth.',
 'Context pages were inspected but do not supply evidence missing from the selected nominal span.',
 'Small changes count when visibly continuous; no blanket positivity from the word or previously positive full clip.'],
 new_training=0,new_inference=0,existing_labels_changed=False)
dest=U/'REVIEW_FIRST_2523.json'
with dest.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(path=str(dest),sha256=sha(dest),counts=out['counts'])))
