"""Close this bounded development cycle; never mark the whole product finished."""
from episode import *
from datetime import datetime,timezone,timedelta

def main():
 frozen=read(U/'FROZEN.json');wf=read(U/'WEB_FROZEN.json')
 for f in [frozen,wf]:
  for kind in ['pins','protected']:
   for p,h in f[kind].items():assert sha(p)==h,p
 assert read(U/'WEB_RECOVERY_EXECUTION_COMPLETE.json')['worker_finished']
 assert not any((U/p).exists() for p in ['WEB_RUNNING.json','EVALUATION_RUNNING.json','EMPIRICAL_RUNNING.json'])
 ev=read(U/'EVALUATION.json');web=read(U/'WEB_RESULT.json');ui=read(U/'REPORT_UI_CHECK.json')
 assert sha(U/'RESULT_KO.html')==ui['report_sha256']
 for p in ['review/STORED_RESULT_VERIFICATION.json','review/EMPIRICAL_RESULT_VERIFICATION.json','review/WEB_RESULT_VERIFICATION.json','review/REPORT_CLAIM_RESOLUTION.json','GUIDE_EPISODE_LINKS.json']:assert (U/p).exists()
 timestamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
 summary=dict(status='BOUNDED_SIX_STAGE_DEVELOPMENT_CYCLE_COMPLETE_NOT_PRODUCT_COMPLETE',at=timestamp,
   active=None,unit=U.name,goal='한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 배워, 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고 중요한 차이와 허용 가능한 차이를 구분한다.',
   stages=dict(S1='16 public inputs/source/use-history verified; previously used ID9 not called fresh',S2='58 sampled-RGB physical intervals;42fit/16development;31fit supported',S3='one4803parameter motion observer160epochs×3seeds; existing4/16,new7/16,geometry8/16; noadoption',S4='full native-frame proposals+individual-reference DTW; auto core3/16→7/16; boundary7/8 outside brackets; lexical stage identity unverified',S5='fit3 LOPO descriptive range; dev3inside1outside,allincomplete;112recordedcontrols pass;no pedagogical verdict',S6='4isolated public CPUreplays624frames+3refs each+browsermedia/paint verified'),
   known_windows=dict(n=16,unknown=8,old_correct=4,new_correct_each_seed=7,geometry_correct=8,new_percent=43.75,old_percent=25.,delta_percentage_points=18.75,new_conditional_correct='7/8',geometry_conditional_correct='8/8'),
   automatic_core=dict(n=16,unknown=9,old_correct=3,new_correct_each_seed=7,geometry_correct=4,new_percent=43.75,old_percent=18.75,delta_percentage_points=25.,new_conditional_correct='7/7',whole_clip_boundary_validated=False),
   new_candidate_adopted=False,old_model_preserved=True,production_changed=False,private_data_used=False,
   latency=dict(path='CPU fullframe MediaPipe/fixed assignment+ROI/new motion head/3referenceDTW/browser',end_to_media_range_seconds=[min(r['capture_end_to_media_and_paint_ack_seconds'] for r in web['rows']),max(r['capture_end_to_media_and_paint_ack_seconds'] for r in web['rows'])],head_range_seconds=[min(r['episode_seconds'] for r in web['rows']),max(r['episode_seconds'] for r in web['rows'])],is_total_game_latency=False,is_isolated_idle_benchmark=False),
   limitations=['Labels are sampled same-Codex RGB drafts, not independent human linguistic truth','Previously consumed development clips4expressions; no new-person/environment generalization claim','R image-plane movement only; source phase identity, full shape/contact/repetition semantics remain unverified','Synthetic coordinate/time invariance is not detector invariance under real distance/light/blur','Some control transformed predictions were not persisted: recorded112 checks not all independently reproven','New candidate loses to geometry in oracle windows; no immediate selection of mixed model based on these scores'],
   next_single_action='Freeze a small core-stage support bundle that separates preparation/core/return using guide-form compatibility+existing shape/arm evidence, and recover or explicitly withhold missing hand-position evidence. Compare fixed observer/geometry/episode without consuming past held/fresh or changing thresholds on these outcomes.',
   review_roles=dict(root='implementation/training/integration/UI/root pin verification',rgb_labels_a='fitRGB+DEVcrosscheck/inputsupport+web implementation and own separate arithmetic review',rgb_labels_b='DEV RGB+code review+independent saved training/result/report arithmetic',guide_stage_map='source history+DTW contracts+independent saved path/range arithmetic'),
   execution_failures='Input numeric invariance assertion from float32 transform fixed before freeze using float64 raw transform; original failure kept. Chrome attempt exited before replay; same scientific conditions retried with separate launcher/evidence paths and adjusted local execution environment. Annotation/review-script formatting expectation failures retained.',
   evidence={p:sha(U/p) for p in ['FROZEN.json','EVALUATION.json','EMPIRICAL_RANGE.json','CONTROLS.json','FINAL_GATE.json','WEB_FROZEN.json','WEB_RESULT.json','GUIDE_EPISODE_LINKS.json','RESULT_KO.html','REPORT_UI_CHECK.json']})
 write(U/'COMPLETION.json',summary)
 report='''개발 단계에서도 허용을 판단할 비교 기준은 필요합니다. 이번에는 사람마다 몸 크기와 영상 속 위치가 다른 영향을 줄이고, 여러 참고가 보이는 차이를 경험적 범위로 만드는 1~6단계 개발 묶음을 완료했습니다. 전문가 검토를 기다리며 개발을 중단하지 않았습니다. 다만 전체 피드백 AI와 교육적 정오 검증이 완성된 것은 아닙니다.

한 일
1) 순종·잘하다·조상·이집트의 공개16영상을 fit12/dev4로 고정했습니다. ID9도 과거 사용이 있어 새 평가로 이용하지 않았습니다.
2) 원본 RGB 표본과 PTS를 보고 상승·하강·기타58구간을 기록했습니다. 애매한 경계와 미주석 구간은 정답으로 채우지 않았습니다.
3) 지원되는31개 학습창으로 작은 공통 시간 판정부(4803파라미터)를 고정160epoch,3seed 학습했습니다. 단어이름이나 개인자료는 넣지 않았습니다. 기존 손모양 출력·C/T3-F는 변경하지 않았습니다.
4) 시작/끝을 알려준 구간과 전체 영상 자동 구간을 분리해 비교했습니다. DTW 뒤 대표점만 점수에 남겨 짧은 추가 동작을 버리는 코드 문제를 고쳤고, 모든 정렬 경로·각 참고·미관측 분모를 남겼습니다.
5) 각 표현의 참고3개 중 하나씩 빼고 비교해 경험 범위를 만들었습니다. 개발4개 중 관측부분3개는 범위 안,잘하다1개는 밖입니다. 모두 일부 관측 부족이라 전체 정오를 판정하지 않습니다. 순서/짧은삭제는 합성 통제이며 실제 잘못된 수어 정답이 아닙니다.
6) 격리 웹에서 공개4영상624프레임을 실제로 다시 검출하고, 새 이동판정부·참고3개DTW·두 영상·슬라이더까지 확인했습니다. 현재 운영 웹은 그대로입니다.

실제 개선과 판단
- 시작/끝을 알려준16구간: 기존4/16(25%)→새7/16(43.75%), +18.75%p. 관측충분8개만 보면4/8→7/8. 나머지8개는 보류이며 분모에서 숨기지 않았습니다. 단순 이동 계산은8/16으로 새모델보다 좋았습니다.
- 전체 영상에서 구간 자동 제안 후 주석 핵심구간16개 대조: 기존3/16(18.75%)→새7/16(43.75%),+25%p. 충분한7개에서3/7→7/7,나머지9개보류. 이 수치가 전체 단계경계를 정확히 찾았다는 뜻은 아닙니다.
- 3seed의 집계점수는 같았고, 자동 프레임출력은 seed42대비6·7프레임 달랐습니다.
- 새후보는 알려준 구간에서 단순계산보다 낮아 미채택입니다. 점수를 보고 뒤늦게 조합하거나 임계값을 맞추지 않았습니다.
- 웹의 촬영종료→화면·두영상준비는2.097~3.119초. 이동판정부 자체는0.039~0.052초. 현재노트북CPU/공개4영상각1회/준비된앱·참고캐시 기준이며, 손모양head·SignRep·C/T3-F까지 포함한 전체게임시간이 아닙니다. 전용무부하시험도 아닙니다.

남은 중요한 문제
- 상승14창의35프레임은 원본 손검출이 빠졌습니다. 다른5개 손내림/정지창91프레임은 손이 검출되어도 ROI밖이었습니다. 후자를 잘못된 행동으로 지적하지 않습니다.
- 주요사건8개중7개 시작점이 가능한 경계에서1~7프레임 벗어났습니다(늦음5·빠름2). 정확한 자동 단계경계는 아직 미검증입니다.
- 같은 ‘상승’이라도 준비동작인지 수어의 핵심동작인지는 다릅니다. 새 GUIDE_EPISODE_LINKS는 설명의 일부와 시간후보를 연결하지만 이를 정답 단계로 확정하지 않습니다. 주먹·양손역할·팔따라이동·접촉·반복을 이3종이동출력만으로 이해했다고 부르지 않습니다.
- 참고3개 범위는 작습니다. 범위밖이라고 수어오류가 아니며, 실제 다른배경·조명·카메라거리의 강건성은 추가검증대상입니다.

검증과 보존
저장학습/점수별도검산52130개,DTW/범위3447개 불일치0. 통제112행 PASS는 계산·분모·보류 계약검사입니다. 일부변형예측배열을저장하지않아112개모델동일성을전부독립재증명한것은아닙니다. 같은Codex의표본RGB검토/별도산술검증과사람전문가의수어정답검증을구분합니다. C/T3-F·운영웹/기존DTW·원본보호해시는유지됐고사용자·멘토자료접근/학습/튜닝/선택은0입니다.

다음 우선순위
새 모델을 더 붙이기보다, 준비/핵심/마무리의 시간 근거와 설명 표현형을 먼저 맞춥니다. 손목·팔 위치와 손모양 근거를 분리한 채 관측 부족을 해결/보류하고, 단순이동계산·작은시간모델이각각잘하는조건을새결과전동결한비교로검증합니다. 이전held/fresh점수나개인촬영에맞춰조정하지않습니다. 손모양·양손관계·순서가함께검증된범위만향후실제피드백에연결합니다.
'''
 (U/'RESULT_KO.txt').write_text(report,encoding='utf8')
 root=OPS.parents[2]
 context=root/'WordsInSilence_PROJECT_CONTEXT.md';txt=context.read_text(encoding='utf8')
 current='''\n\n### 2026-10-31 공통 구간 관측·경험적 범위 6단계 개발 묶음 완료 — 새 후보 미채택

- `word100_semantic_episode_pilot_20261031` 완료, active=null. 최신사용자의 허용기준개발/S1~6계속승인에 따라 전문가출시검증대기없이 경험적 비교 개발을 진행. C/T3-F/운영/개인자료보존. 전체피드백AI완료는아님.
- 공개16(4표현fit12/dev4),표본RGB58창(42/16),입력지원fit31/dev8. 작은4803parameter공통R화면방향head160epoch×3seed. 알려준구간기존4/16→새7/16(+18.75%p),기하8/16으로후보미채택. 자동core3/16→7/16(+25%p),unknown9. seed집계동일이나프레임출력6/7차이. 정확경계7/8이bracket밖1~7f,자동단계경계채택X.
- 손미검출상승14창35f/정상하단ROI제외other5창91f분리. 모든원본프레임/3개별참고/미관측분모유지;DTW대표한점축약으로짧은query누락문제수정. fit3LOPO경험범위dev3within/1outside모두incomplete;교육정오X. ID9과거사용으로새평가추론X.
- 공개4격리웹624f/12쌍,촬영종료→두영상·표시2.097~3.119초,head0.039~0.052초.CPU/1회씩/캐시준비/보고작업겹침;전체게임TCN·SignRep·손모양head미포함. 첫Chrome실패처리0후기록보존·별도launcher동일조건복구. 프로세스종료/서버닫힘확인.
- 학습점수별도52130,DTW범위3447검사불일치0;통제112PASS는일부변형예측미저장독립한계명시. WEB/REPORT_UI및해시검산완료. source→episode는물리후보이며준비/핵심단계정체성미검증(GUIDE_EPISODE_LINKS). COMPLETION/RESULT_KOhtml·txt가최신근거.
'''
 nextaction='''\n\n### 2026-10-31 다음 핵심 작업 — 핵심 단계와 관측 부족을 분리해 고정 비교

현재6단계개발실행은완료했다. 새모델을계속바꾸지말고공개fit/dev에서준비/핵심/마무리의시간근거와사전표현형을맞춘작은지원묶음부터고정한다. 같은상승을설명단계로자동확정하지않는다. 손목·팔위치근거와손모양근거를분리하고상승의실제손미검출및시작지연을다룬다. 단순기하·작은시간후보가각각잘하는조건은새결과전계약으로비교하며이번점수에맞춘즉석혼합/임계값완화금지. 기존C/T3F/운영·개인자료보존,이미완료3seed/경험범위/4웹재실행금지. 전문가검토는출시교육정오별도,현재개발중단조건아님.
'''
 for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',current),('## 9. 다음 단일 작업',nextaction)]:
  assert heading in txt;txt=txt.replace(heading,heading+body,1)
 context.write_text(txt,encoding='utf8')
 statefile=OPS/'STATE.json';state=read(statefile);state['current_stage']='SEMANTIC_EPISODE_DEVELOPMENT_CYCLE_COMPLETE';state['activity_state']='IDLE_REPORTED_BOUNDED_DEVELOPMENT';state['next_single_action']=summary['next_single_action'];state[U.name]=summary;statefile.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
 handoff=f'''# WordsInSilence 최신 인수인계\n\n{timestamp}\n\n목표: {summary['goal']}\n\n현재단위 `{U.name}`, `BOUNDED_SIX_STAGE_DEVELOPMENT_CYCLE_COMPLETE_NOT_PRODUCT_COMPLETE`, active=null. 사용자승인1~6개발실행은완료했으나전체교육피드백AI완료아님. 전문가검토를현재개발중단전제로삼지않음.\n\n{current}\n\n{nextaction}\n\n최신근거: `research/front_keyphase_v1/ops/{U.name}/COMPLETION.json`, `RESULT_KO.html`, `RESULT_KO.txt`, `FROZEN.json`, `CURRENT_EXECUTION.json`. 학습3회/평가/경험범위/112통제/4웹실행완료,중복추론/재채점불필요.\n\n실행: train88968/evaluate완료/empirical94563 exit0. 웹67321은Chrome시작전실패(추론0), WEB_FIRST_FAILURE보존. 별도launcher+로컬실행권한환경복구65619exit0,4run624프레임완료. WEB_RECOVERY_EXECUTION_COMPLETE server/chrome/worker종료true,lock없음. REPORT_UI_CHECK의sha최종HTML일치.\n\n역할: root구현/학습/최종검토, rgb_labels_a fitRGB+DEV일부교차/입력분석/웹구현과자체산술검산, rgb_labels_b devRGB/학습코드/저장결과별도검산, guide_stage_map 출처/DTW계약·저장산술별도검산. 같은Codex표본주석·산술검증은독립사람수어정답아님.\n\n환경: CoreUltra7 258V/32GB/Arc140V, Python `word100_signrep_feasibility_20261029/runtime/Scripts/python.exe`. 이번4803parameterhead·MediaPipe는CPU,SignRep GPU기존경로보존미사용. 기존C/T3F/운영웹/DTW/동결원본보존,사용자·멘토자료사용0,download/camera/external/Git0. 최초오류보존후조건동일복구.\n'''
 (root/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
 (U/'CURRENT_EXECUTION.json').write_text(json.dumps(dict(status=summary['status'],active=None,completed_at=timestamp,training_complete=True,web_complete=True,report_complete=True,next_single_action=summary['next_single_action']),ensure_ascii=False,indent=2),encoding='utf8')
 print('COMPLETED_DOCUMENTS',root)

if __name__=='__main__':main()
