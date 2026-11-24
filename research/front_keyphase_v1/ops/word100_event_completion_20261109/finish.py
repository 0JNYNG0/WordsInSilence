from core import *
import re,datetime
def main():
 assert read(E/'VERIFIED.json')['status']=='PASS' and read(E/'GUARD_VERIFIED.json')['status']=='PASS';assert not (E/'RESULT.json').exists()
 result=read(E/'GUARD_V2_COMPARISON.json');primary=result['models']['final_42']['by_kind'];held=read(E/'HELDOUT.json');verify=read(E/'GUARD_VERIFIED.json');timeinfo=read(E/'COMPARISON.json')['models']['final_42']['additional_forward_seconds']
 nextplan=dict(objective='Freeze the separated completion guard and verify the complete public replay path without changing default web',steps=['Use existing public RTMW raw arrays to verify quarantine -> existing TCN/feedback -> completion identity guard -> DTW display, preserving candidate provenance and no duplicate RTMW extraction','Audit availability of public recordings not already used for FIT or repeated DEV; evaluate unchanged weights/thresholds on eligible fresh normal, partial-action and observation-gap cases','Improve remaining ambiguous action boundaries and fine handshape labels using FIT only, separating stroke from optional arrival; do not tune on historical failures','Only after full replay and fresh-scope evidence, expose an explicitly selected research web candidate; keep default C/T3F/current TCN intact'],no_rerun=['six completed head trainings','237 cached fixed condition comparisons','saved arithmetic verification'],limits=['No live webcam or private data','No external downloads','No default replacement','No expert correctness claim'])
 write(E/'NEXT_PLAN.json',nextplan)
 out=dict(status='COMPLETE_LIMITED_COMPLETION_GUARD_GAIN_ISOLATED_ADAPTER_READY',unit=E.name,steps=dict(public_fit_review='36 clips / 9 IDs / 4 expressions / 576 sampled RGB frames; 7 ambiguous; not dense labels or expert truth',training='6 fixed progression head runs on frozen geometry encoders; no existing checkpoints changed',comparison='237 historical conditions per seed; 61 new FIT-heldout partial controls; initial and posthoc repairs retained',integration='opt-in consumer adapter and13 no-camera integration/contract tests; current web not changed or started'),primary_seed=42,final_primary=primary,heldout_summary=held['summary'],heldout_partial=dict(wrong_core=held['partial_wrong_core'],total=held['partial_total'],scope='Core classifier only; weakly located sample boundaries, not whole sign correctness'),additional_neural_forward=timeinfo,first_failed_full_replacement=dict(normal_uncertain_before=3,after=8,retained=True),posthoc_repairs=True,fresh_independent_evaluation=False,verification=dict(initial=13857,guard=verify['checks'],contract_tests=13,who='Same Codex separate verification, not independent human truth'),default_web_changed=False,isolated_runtime_adapter=str(E/'runtime_adapter.py'),private_data_access=False,protected_preserved=True)
 write(E/'RESULT.json',out)
 report='''이번 결과: 끝까지 하지 않았는데 “모두 비슷하다”고 표시하던 사례를 막았습니다.
다만, 모든 누락과 손모양 오류를 해결한 것은 아닙니다. 현재 웹 기본판은 유지했습니다.

무엇을 했나
1. 공개 참고 36편(9명 × 4표현)에서 실제 RGB 화면 576시점을 확인했습니다.
   “핵심 모습으로 가는 중”과 “핵심 모습이 보임”을 나눴습니다. 7시점은 불확실해 정답으로 쓰지 않았습니다.
   16시점씩 본 표본 주석이며, 모든 프레임의 경계를 확인했다는 뜻은 아닙니다.
   같은 표시가 이어진 사이만 낮은 가중치로 활용했고, 다른 표시 사이를 임의 정답으로 채우지 않았습니다.
2. 기존 팔·손목 관측 신경망을 고정하고 작은 진행/핵심모습 판정부를 학습했습니다.
   사람을 나눠 확인하는 3회와 고정 난수 3회, 총 6회입니다. C·T3-F와 기존 RTMW TCN은 학습하지 않았습니다.
3. 과거 공개 개발 12영상(3명 × 4표현)의 237개 조건을 비교했습니다.
   237은 서로 다른 사람이나 영상 수가 아닙니다. 같은 영상의 속도·가림·모양·순서를 바꾼 시험 조건이 포함됩니다.
4. 개선된 보조 판정을 별도 소비기 연결부로 구현하고 연결 규칙 13개를 확인했습니다.
   현재 웹8814에 자동 적용하거나 카메라를 실행하지 않았습니다. 실제 웹 전체경로와 새로운 공개 표본 확인은 다음 단계입니다.

중간에 발견하고 고친 문제
- 처음에는 새 판정부가 정한 구간으로 손모양 검사까지 바꿨습니다. 누락 오판은 줄었지만 정상12영상의 보류가3→8로 늘었습니다.
- 반복 동작을 처음 한 번만 손모양 검사에 보내는 문제와, 완료 판단이 불확실해지면 기존 모양/방향 차이까지 사라지는 문제를 보완했습니다.
- 마지막에는 역할을 분리했습니다. 기존 손모양·방향 검사와 비교 구간을 유지하고, 새 판정부는 “이 구간이 다른 동작으로 잡혔다”는 분명한 충돌이 있을 때만 완료 표시를 보류합니다.
- “팔을 쓸어내리는 동작”과 그 뒤의 선택적인 도착 자세는 기존 설명에서도 한 움직임입니다. 이를 서로 다른 행동으로 처리하던 연결 오류도 바로잡았습니다. 단어 이름을 보고 예외 처리한 것이 아니라 기존 움직임 설명의 연결 관계를 공통으로 사용했습니다.
- 처음 실패한 결과와 이후 수정 결과를 모두 보존했습니다. 새 결과를 독립 평가로 포장하지 않습니다.

최종 비교(대표42, 나머지 두 학습 결과에서도 핵심 결과 동일)
항목                                      기존 웹 계산  최종 보조 판정
끝 동작을 제거한6시험: 잘못된 모두비슷함        2/6          0/6
마지막 관절만 가린6시험: 잘못된 모두비슷함       2/6          0/6
관절가림을 실제 동작누락이라고 한 오판           0            0
정상 원본12영상: 차이 경고 / 확인보류             0 / 3        0 / 3
실제 상반신crop12영상: 차이 경고 / 확인보류        0 / 6        0 / 6
정상 속도변경48조건                            기존 결과 유지(각 조건 단위로 확인)
정상 양손교차3영상                             기존 결과 유지
기존 손모양변형·역방향·복합 시험 각3개            차이 안내 각각3/3 유지
손을 전부 가린12시험                            정상 판정 없이 확인보류 유지

“0/6”은 빠진 동작을 모두 정확히 진단했다는 뜻은 아닙니다.
빠진 동작 또는 다른 동작으로 잘못 연결된 부분에 “잘했다”고 표시하지 않게 된 것입니다.
손이 가려져 안 보이는 경우와 실제로 생략한 경우는 계속 구분해서 기록합니다.
또 앞 동작을 뒤 동작으로 착각하는 모든 경우를 해결한 것은 아닙니다. 관측·두 판정부의 충돌을 이용해 확정 표시를 막는 한정된 개선입니다.

범용성
판정에는 특정 단어·사람만을 위한 예외나 별도 문턱값이 없습니다. 설명의 동작 관계와 공통 관측 출력을 사용합니다.
다만 이 새 진행 판정부가 배운 범위는4표현입니다. 모든100단어의 피드백 성능을 검증한 것은 아닙니다.

아직 남은 문제
- 사람을 나눠 확인한 새 진행 판정부에서 정상 원본의 확인시점569개 중495개가 세부상태와 일치했습니다. 동작 종류만 보면524개였습니다. 수어 정답률이 아닙니다.
- 학습에서 제외한 사람의 중도 종료61조건 중4개에서는 판정부가 핵심 모습에 도달했다고 잘못 표시했습니다. 이는 전체 피드백 결과가 아니라 진행 판정부 자체의 한계이며, 표본 경계 주석의 정밀도도 추가 확인이 필요합니다.
- 손끝 모으기·엄지·정확한 접촉·정확한 반복 횟수와 자연스러운 실제 오동작의 교육적 정오는 아직 충분히 검증되지 않았습니다.
- 정상 원본의 기존 보류3개, crop 보류6개를 없앤 작업은 아닙니다.
- 새 분기 신경망 계산은 중앙값 약8.51ms,95백분위12.31ms였습니다. 촬영·관절 추출·TCN·DTW·화면 표시 전체 응답시간이 아닙니다.
- 이번은 이미 여러 번 사용한 개발 자료의 회귀 확인입니다. 구현 보완도 결과 진단 뒤 이뤄졌으므로 새 독립 성능 수치로 볼 수 없습니다.

검증과 보존
초기 저장검산13,857항목, 보조 판정/수정 이력 검산19,558항목, 연결 규칙13항목 통과.
구현·실행·별도 산술검산은 같은Codex입니다. 독립 전문가나 농인 강사의 정답 검증을 받았다는 뜻이 아닙니다.
모듈 이름 충돌 오류1건과 선택사항 검사조건 오류1건은 최초 오류를 보존한 뒤 수정했습니다. 학습·점수·원본을 덮어쓰지 않았습니다.
C·T3-F, 현재 RTMW TCN, 현재 웹, 원본과 과거 동결 결과 유지. 사용자·멘토 촬영 접근/학습/튜닝/후보선택 없음.

다음 한 작업
이 보조 판정을 고정하고 공개 RTMW 원본 배열로 실제 전체 분석 경로에 연결해 확인합니다.
그다음 기존 학습·반복 개발검사에 쓰지 않은 공개 촬영을 확보할 수 있는지 먼저 확인해, 같은 기준을 바꾸지 않고 재검증합니다.
그 확인 전에는 현재 웹 기본 모델을 교체하지 않습니다.

근거 파일
REVIEW_LABELS.json / INPUTS.json / runs/*/COMPLETE.json / COMPARISON.json / FOLLOWUP.json
GUARD_COMPARISON.json / GUARD_V2_COMPARISON.json / HELDOUT.json / VERIFIED.json / GUARD_VERIFIED.json
runtime_adapter.py / CONTRACT_TESTS.json / NEXT_PLAN.json
'''
 (E/'RESULT_KO.txt').write_text(report,encoding='utf-8')
 write(E/'CURRENT_EXECUTION.json',dict(status=out['status'],training=None,evaluation=None,active_process=None,default_web_changed=False,report=str(E/'RESULT_KO.txt')))
 summary='公開FIT9名36動画のRGB576時点を確認'.replace('公開FIT9名36動画のRGB576時点を確認','공개FIT9명36영상 RGB576시점 확인(7불확실), 진행/핵심모습 작은head6고정학습 완료')
 summary+='. 첫전체구간교체는정상보류3→8회귀하여미채택. 기존관측창·모양·방향은유지하고분명한다른사건근거만완료보류하는분리보조판정으로수정. 같은동작의선택적도착자세를다른사건으로취급한연결오류도기존설명계약으로수정. 237과거공개조건×3seed: 사건삭제6/관절가림6 각각거짓전체비슷함2→0,가림누락오판0; 정상원본12 경고0/보류3 및crop12 경고0/보류6·속도48조건·교차3·기존오류각3경고유지. 새독립평가아님(사후연결수정). 신규head사람분리569확인시점495세부일치/524사건일치,중도종료61중잘못된core4잔존. 추가분기8.51ms는전체속도아님. 격리consumer연결부/13계약검사구현; 현재웹미교체·카메라없음. 13857+19558저장검산PASS(같은Codex). C/T3F/현재TCN/원본보존·개인자료0.'
 nexttext='완료6학습/237조건/후속저장검산재실행금지. 최종completion_guard_v2와42고정: 공개RTMW원본에서quarantine→기존TCN/피드백→추가완료충돌검사→DTW표시까지격리전체경로검증. 이어기존FIT/반복DEV에노출되지않은공개촬영확보가능성감사후고정평가. 새DEV문턱/seed/단어예외튜닝금지. 정상/속도/교차유지와중도종료·관측부족구분을확인한뒤명시선택연구웹검토;기본웹·C/T3F/현재TCN유지,개인자료/카메라/다운로드없음. 이전전체구간교체실패와사후수정이력보존. '+str(E.relative_to(O.parent.parent.parent))+'/NEXT_PLAN.json'
 root=O.parent.parent.parent;ctx=root/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text(encoding='utf-8-sig');title='### 2026-11-09 진행·완수 분리 보완 — 잘못된 완료 표시 감소, 정상 결과 유지'
 for section,body in [(8,summary+' 근거 '+str(E.relative_to(root))+'/RESULT_KO.txt.'),(9,nexttext)]:
  pat=r'(^## '+str(section)+r'\.[^\n]*\n)';text,n=re.subn(pat,lambda m:m.group(1)+'\n'+title+'\n\n'+body+'\n',text,count=1,flags=re.M);assert n==1
 ctx.write_text(text,encoding='utf-8')
 (root/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n다음 한 작업: '+nexttext+'\n\n학습·추론·검산 모두 종료. 현재 웹 기본8814의 코드/모델은 그대로이며 서버·카메라를 이번에 시작하지 않았다. 격리 adapter는 구현/모의연결검사 완료일 뿐 실제웹 전체경로 검증·새 공개평가·교육정오검증 완료 아님. 결과 확인 전 운영교체 금지. 독립검증은 별도 산술코드 의미이며 독립사람정답 아님.\n\n주요 결과 '+str(E/'RESULT_KO.txt')+'\n현재 상태 '+str(E/'CURRENT_EXECUTION.json')+'\n',encoding='utf-8')
 state=read(O/'STATE.json');entry=dict(status=out['status'],report=str(E/'RESULT_KO.txt'),result=str(E/'RESULT.json'),verification=str(E/'GUARD_VERIFIED.json'),execution=str(E/'CURRENT_EXECUTION.json'),summary=summary,next=nexttext,default_web_changed=False,private_access=False,active=None)
 state.update(current_stage=E.name,next_single_action=nexttext,last_updated=datetime.datetime.now().isoformat(timespec='seconds'));state[E.name]=entry
 order=['current_stage','next_single_action','last_updated',E.name];state={**{k:state[k] for k in order},**{k:v for k,v in state.items() if k not in order}};write(O/'STATE.json',state)
 write(E/'DELIVERY_CHECK.json',dict(status='PASS',root_sync=True,checks=['result and next plan parse','current state key equals unit','latest context8/9 match','handoff current unit','no live process'],paths={str(p):sha(p) for p in [E/'RESULT_KO.txt',E/'RESULT.json',E/'NEXT_PLAN.json',ctx,root/'NEW_CHAT_HANDOFF.md',O/'STATE.json']}));print('RESULT_AND_CONTEXT_SYNCED')
if __name__=='__main__':main()
