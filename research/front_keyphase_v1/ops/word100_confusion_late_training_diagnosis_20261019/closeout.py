from common import *
import re

def main():
 verification=read(U/'VERIFICATION.json');assert verification['status'].startswith('PASS');summary=read(U/'SUMMARY.json');contract=read(U/'CONTRACT.json')
 for n,h in verification['pins'].items():assert sha(U/n)==h
 for p,h in contract['protected_pins'].items():assert sha(p)==h
 page=(U/'RESULT_KO.html').read_text(encoding='utf8')
 for link in re.findall(r'(?:href|src)="([^"]+)"',page):assert (U/link).is_file(),link
 for p in U.glob('*.json'):read(p)
 at=now();key=U.name
 next_action='공개 반복 혼동·학습 후반 진단 결과를 사용자와 검토. 첫 후보는 원본 순서·899회 노출·29단계를 유지하는 균등 배치 분할(현재 32×28+3 → 31×29). 계획 확인 전 추가 학습·통계 재추정·운영 교체 없음. 기존 C·웹·DTW 유지, 사용자·멘토 자료 학습/튜닝/선택 금지.'
 sp=O/'STATE.json';state=read(sp);state.update(updated_date_kst=at,updated_at=at,current_stage='WORD100_CONFUSION_LATE_TRAINING_DIAGNOSIS_COMPLETE',activity_state='AWAITING_USER_RESULT_REVIEW',next_single_action=next_action)
 state[key].update(status='COMPLETE_VERIFIED_NO_TRAINING_C_RETAINED',at=at,training_allowed=False,training_runs=0,inference_runs=0,mediapipe_runs=0,running=False,
  saved_predictions=3600,history_epochs=960,public_observation_records=345,public_rgb_videos=25,visual_sample_frames=200,total_wrong_repeated=62,unique_error_clips=17,id12_error_count=46,
  C_late_adjacent_max_pp=13/3,result=str(U/'RESULT_KO.html'),verification=str(U/'VERIFICATION.json'),next_plan=str(U/'NEXT_PLAN.json'),cause_status='Observed error concentration/late fluctuation; batch/BN/LR causal attribution unproven',next_single_action=next_action)
 sp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업';assert h8 in s and h9 in s
 e8=f'''### 2026-10-19 공개 반복 혼동·학습 후반 진단 완료 — 추가 학습 없음
ops/{key}. C/Q/R10/R01×3 저장 sensor 예측 3600개, 학습 이력 960회차, 공개 관측345개(DEV300+관련TRAIN45), RGB25영상 표본200장 분석. 오답62건/고유17영상; ID12 46건, ID11 15건, ID13 1건. 상위5영상 동대문10/12·속다8/12·백육십이8/12·상처7/12·흡연실5/12. 백육십이/흡연실 사용손 주석구간 관측100%, 손 누락만으로 설명불가. 정상 내려놓은 손 미관측과 실제 동작손 구분. C 후반61~80회 인접 최대13/300=4.33%p 변화, 12실행 전부 후반 평균정확도/CE는 앞41~60회보다 개선. 고정 C 추론의 무작위성이 아님. 일정학습률0.001, MaskedBN 갱신0.1, 마지막3개 배치도 동일통계갱신/1step이라는 코드 사실 확인; 정확도 원인인지는 가설. 다음 제안은 ceil(N/B) 균등분할(899→31×29), 원본·순서·step 유지, 다른변경과 섞지 않는 3seed 비교. 아직 학습/추론/통계재추정0, 기존15checkpoint 핀보존. 같은Codex 별도산술검산PASS, 언어전문가/독립에이전트 판정아님. RESULT_KO.html·VERIFICATION.json·NEXT_PLAN.json.

'''
 e9=f'''### 2026-10-19 후반 안정화 단일 후보 계획 검토 대기
{next_action} 단위 ops/{key}. 다음 단계·세부 근거는 NEXT_PLAN.json. 균등 배치가 인식률을 높인다는 검증은 아직 없고, 마지막 배치 통계 의존도 자체를 모두 제거하는 변경도 아님. 손모양·단계 구분 후보는 후순위 별도 비교. 반복사용 DEV 결과를 새 사용자 일반화 증거로 확대하지 않는다.

'''
 p.write_text(s.replace(h8,h8+'\n\n'+e8,1).replace(h9,h9+'\n\n'+e9,1),encoding='utf8')
 h=f'''# WordsInSilence 최신 인수인계

갱신: {at}

현재 완료: ops/{key}. 사용자 승인 범위인 공개 반복 혼동 및 학습 후반 원인 분석 완료. 추가 학습은 사용자 결과 검토 전 금지.

- 기존 C 유지. 사용자/멘토 자료 학습·튜닝·선택·분석 사용0. 추가 학습0/모델 추론0/MediaPipe 실행0/다운로드0/외부전송0/Git0. 공개 영상 ZIP에서 작은 원본25개를 로컬 근거 캐시로 읽음.
- 4조건(C/Q/R10/R01)×3 저장 예측3600개, 이력960epoch 분석. 오답62건=고유17개 DEV영상, ID12에46건. 상위5 동대문10/12, 속다8/12, 백육십이8/12, 상처7/12, 흡연실5/12. 정답2위37/62, 상위5위52/62. 모델12개를 사용자12명 시험으로 말하지 않음.
- 공개 관측345개(DEV300+관련TRAIN45), RGB25영상×8표본=200장 검토. 주요 동작손이 관측되는 오답 존재. 백육십이/흡연실 주석 내 오른손 관측100%; 이는 좌표정확도100%가 아님. 상처의 교대/반복 및 단어간 미세형태/단계차이는 시각단서, 전문가 정오판정 아님.
- C에도 후반변동: 61~80회 인접 최대13/300=4.33%p. 12실행 모두 후반 평균정확도·손실이 앞 구간보다 개선, 단순 지속과적합 설명 부족. 같은seed 곡선 상관0.63~0.95; 공통 학습과정 가설. 개별epoch logits없어서 개별영상의 회차별반전은 미확인.
- 코드사실: 고정lr0.001, MaskedBN momentum0.1(패딩 제외), 마지막batch3도 통계10%갱신/1step. 직접 인과는 미확인. eval에서dropoutOFF. C선택규칙(양쪽DEV CE최소)은 유지, 정답수높은epoch로 재선택 안함.
- 다음 제안: generic ceil(N/B) 균등 배치(현재 32×28+3→31×29) 한 조건. 원본899·순서·29step·모델·학습률·관측/시간일정 유지. 먼저 누락/중복/일정 검산, 사용자 확인 후 3seed 비교. BN재보정/LR감쇠/EMA/단계pooling 동시도입 금지. 효과 미보장.
- 원본15모델checkpoint 핀 동일. 별도 산술 검산 PASS; 같은 Codex 검산, 독립사람/에이전트 평가 아님. RESULT_KO.html/txt, NEXT_PLAN.json, VERIFICATION.json, FINAL_AUDIT.json.
- 진행중 작업/잠금 없음. 실행 inspect_inputs exec66672 종료0; saved/late/verify/report 완료0. 도구 미리보기 파일명1회오입력 후 실제 WORD2370 감히시트 확인; 원본/실행 실패 아님.

다음 한 작업: {next_action}
'''
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(h,encoding='utf8')
 write(U/'FINAL_AUDIT.json',{'at':at,'status':'COMPLETE_VERIFIED_NO_TRAINING_C_RETAINED','original_checkpoint_pins':contract['protected_pins'],
  'result_pins':{n:sha(U/n) for n in ['CONTRACT.json','SAVED_ANALYSIS.json','INPUT_QUALITY.json','LATE_MECHANISM.json','VISUAL_REVIEW.json','NEXT_PLAN.json','VERIFICATION.json','SUMMARY.json','RESULT_KO.html','RESULT_KO.txt','late_curves.png','repeated_errors.png']},
  'document_pins':{str(x):sha(x) for x in [p,sp,ROOT/'NEW_CHAT_HANDOFF.md']},'running':False,'training':0,'inference':0,'user_data_used':False,'production_changed':False,'next_action':next_action})
 print('COMPLETE_VERIFIED_NO_TRAINING_C_RETAINED')
if __name__=='__main__':main()
