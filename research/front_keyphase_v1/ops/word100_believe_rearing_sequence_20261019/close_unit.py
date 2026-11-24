from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,shutil,re
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'FINAL_AUDIT.json').exists()
    a=read(U/'ANALYSIS.json');v=read(U/'INDEPENDENT.json');q=read(U/'LAYOUT_QA.json');plan=read(U/'NEXT_PLAN.json')
    assert v['status']==q['status']=='PASS' and plan['training_authorized_this_turn'] is False
    for p,h in a['pins'].items():assert sha(p)==h,p
    for name in ['SCOPE.json','CONCLUSION.json','NEXT_PLAN.json','FIRST_DIAGNOSTIC_CORRECTION.json','ORDER_CLOCK_CORRECTION.json']:read(U/name)
    stamp=datetime.now(timezone(timedelta(hours=9))).isoformat();key=U.name
    backup=U/'before';backup.mkdir(exist_ok=False)
    docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
    old={str(p):sha(p) for p in docs}
    for p in docs:shutil.copy2(p,backup/p.name)
    next_action='검토한 공개 믿다·사육장 진단과 NEXT_PLAN.json을 사용자와 확인한다. 후속 제안은 공개 TRAIN100단어의 단계/순서 민감도 및 입력 보존 감사 후, 기존 T3-F 전체 고정+GRU16 순서 보조 후보를 조건부3seed로 비교하는 계획이다. 아직 추가학습/후보구현 승인·실행 없음. C/T3-F/웹 유지, 사용자·멘토 자료 학습/튜닝/선택 금지, ID14~16 미사용 유지.'
    state=read(O/'STATE.json');state['current_stage']='WORD100_BELIEVE_REARING_PUBLIC_DIAGNOSIS_COMPLETE_PLAN_REVIEW'
    state['updated_date_kst']=stamp[:10];state['next_single_action']=next_action
    state[key]={'status':'PUBLIC_DIAGNOSIS_COMPLETE_NEXT_PLAN_ONLY','at':stamp,'unit':key,
      'scope':{'public_TRAIN_videos':18,'views':['masked','sensor'],'RGB_sample_videos':6,'RGB_sample_frames':96,'saved_DEV_videos':6,'saved_DEV_predictions':72},
      'normal_predictions':{'C_TRAIN':36,'T3F_TRAIN':36,'TRAIN_denominator_with_views':36,'saved_DEV_all_correct':72,'actual_public_pair_confusion_reproduced':False},
      'sensor_active_seconds':{'믿다':[1.266667,1.5,1.766667],'사육장':[2.733333,3.033333,3.366667]},
      'sensor_T3F_diagnostics':{'reversed_thirds_original_label_top1':{'믿다':'8/9','사육장':'9/9'},'rearing_reordered_min_score':98.72313320308446,'rearing_prefix_half_top1':'5/9','rearing_suffix_half_top1':'6/9','both_words_double_sequence_top1':'18/18'},
      'interpretation':'Partial/order changes can retain high class scores. Not valid-sign correctness or actual-user cause proof. T3-F order-sensitive margins decline despite retained top1. No raw-duration hard rule.',
      'verification':{'stored_logits':64800,'source_model_web_pins_unchanged':104,'arithmetic':'same Codex separate scalar code, not independent agent/expert'},
      'first_correction':'Reverse-thirds first moved dt with frames; original evidence retained; ONLY36pairs corrected to retain original time. Report excludes first-order condition.',
      'training_runs':0,'user_recordings_read':0,'production_changes':0,'new_mediapipe_runs':0,
      'next_plan':str(U/'NEXT_PLAN.json'),'report':str(U/'RESULT_KO.html'),'candidate_training_authorized':False}
    context=docs[0].read_text(encoding='utf-8-sig')
    title='### 2026-10-19 믿다·사육장 공개 단계·순서 진단 완료 — 추가학습 전 계획 검토'
    assert title not in context
    body=f'''{title}
갱신 {stamp}. ops/{key}. 사용자 의문(추가 단계·순서·길이 차이)을 공개 TRAIN 두단어×9ID=18영상의 masked/sensor36입력으로 진단. RGB 미리정한ID01/05/09×2단어6영상96장 시각검토. sensor 활동길이 믿다1.27~1.77초(중앙1.50),사육장2.73~3.37초(3.03). 정상입력 C/T3-F각36/36정답, 관련DEV6영상×3seed×2모델×2view저장72예측모두정답. 공개정상입력에서 이번믿다↔사육장혼동은재현안됨, 실사용단일원인미확정. 진단상 세덩어리역순에도 sensorT3-F사육장9/9(최소98.72점),믿다8/9원래단어1위; 사육장앞절반5/9·뒤절반6/9(masked뒤절반0/9)로관측의존도함께공개. 순서변경시pairlogit차중앙약8감소(C약0.5)라순서무시주장금지. 반복연결18/18원래단어1위. 이는인위적민감도진단이며수어정오·완수정확도아님. dt만변경시점수거의불변이나시간입력없다는뜻아님. 최초역순진단dt채널까지이동한문제의증거보존후36쌍만원래시간유지로복구. 저장64800logits/입력36/기여도36별도산술검산PASS;원본·모델·웹104파일+RGB6영상해시보존. 추가학습0/사용자자료읽기0/MediaPipe재추출0/운영변경0. 기존 C/T3-F유지. 다음은공개100단어공통감사와고정T3-F+작은GRU16(새4000변수)후보계획,미구현·학습전검토대기. SCOPE/ANALYSIS/ORDER_CLOCK_CORRECTION/INDEPENDENT/CONCLUSION/NEXT_PLAN/RESULT_KO. Codex로컬구현+별도산술코드이며독립사람/agent검토아님.

'''
    h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';assert h8 in context
    context=context.replace(h8,h8+'\n'+body,1)
    h9='## 9. 다음 단일 작업\n';assert h9 in context
    context=context.replace(h9,h9+'\n### 2026-10-19 공개 순서 진단·후속 계획 검토\n'+next_action+'\n\n',1)
    handoff=f'''# WordsInSilence 최신 인수인계

갱신: {stamp}

현재 목표: 믿다·사육장의 구분 단계/순서/길이 의문을 공개 자료로 진단 완료. 후속 계획 사용자 검토 대기.

- 단위: research/front_keyphase_v1/ops/{key}
- 범위: TRAIN ID01~09 두단어18영상×masked/sensor36입력. RGB6영상96장 시각검토, 관련DEV6영상의기존저장72예측만재사용. 사용자/멘토녹화읽기0, ID14~18원시접근0.
- 결과: 정상 TRAIN36/36 양모델정답, 저장DEV72예측모두1위. 공개정상자료에서 이번두단어오답은재현되지않았으며 실사용단일원인미확정. 학습자료높은점수를일반화성능으로보면안됨.
- 길이: sensor활동믿다1.27~1.77초(중앙1.50),사육장2.73~3.37초(3.03). 전체영상은준비/마무리포함범위가겹침. 초단위하드규칙제안안함.
- 순서진단: 세덩어리역순 sensorT3-F 사육장9/9원래단어1위(최소98.72),믿다8/9. 사육장앞절반5/9·뒤절반6/9(masked뒤절반0/9). 전체2회연결18/18. 인위적민감도진단이지수어정오/완수율아님. T3-F순서변경pairlogit차약8감소(C약0.5)라순서전혀못본다고설명금지. dt만2배/.5배점수변화극소,실제속도재촬영시험아님.
- 구조: C프레임별TCN후전체평균;T3-F앞중간뒤상대시간3요약. 단어분류라필수단계누락/순서오류거절보장없음. 원본관절누락도이번원인으로확정안함.
- 검산: 저장64800logits/메타36/기여도36별도스칼라검산PASS,104근거·모델·웹파일+RGB6해시보존. 같은Codex의별도코드,독립agent/전문가아님.
- 정정: 첫역순입력에서dt까지섞여내부0발생. FIRST_DIAGNOSTIC_CORRECTION에기록,최초ANALYSIS보존. 좌표/마스크만재배열하고시간유지하여36쌍만복구. 보고는ORDER_CLOCK_CORRECTION결과만사용. 정상입력및완료DEV재채점없음.
- 제안 NEXT_PLAN: 먼저공개TRAIN100단어의공통순서민감도/입력단계보존감사. 이후검토승인시 T3-F 전체47236동결+GRU16순서보조(새4000변수,0출력초기화)만3seed비교. 아직새후보구현/학습안함. 단어라벨학습만으로완수검증보장안됨,실제수행판정에는전문가확인정상/생략/순서오류시험별도필요.
- 이번실행종료: diagnose19933/public_visuals95656정상0,순서시각정정/독립검산/보고/Chrome QA완료. 진행중학습없음.
- C/T3-F/웹/DTW불변. 고정C seed20261013 epoch77 SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab; T3-F동일seed epoch69 SHAb0faa0df5055204295c60b78c596a0b490cd63ef14f30225582a85ea39855846.
- 웹 http://127.0.0.1:8765/practice.html?model=t3f . 최근서버기록PID43256/require_escalated,이번프로세스확인·재시작·카메라없음. 기본M0유지.
- 이전사용자평가: word100_t3f_web_paired_review_20261019. 같은12촬영C10/12→T3-F12/12(+16.67%p),믿다T3-F53.66/사육장44.15. 이자료를이번진단에다시읽거나학습/튜닝/선택에사용하지않음.
- 이전최종공개ID17/18평가사용됨. 정상ID14~16총299미사용유지. 재탐색에보류자료사용금지.

다음 한 작업: {next_action}
'''
    for p,h in old.items():assert sha(p)==h,'Concurrent documentation edit'
    docs[0].write_text(context,encoding='utf8');docs[1].write_text(handoff,encoding='utf8');docs[2].write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert read(docs[2])[key]['training_runs']==0
    report=(U/'RESULT_KO.html').read_text(encoding='utf8')
    for href in re.findall(r'(?:src|href)="([^"]+)"',report):
        if not href.startswith(('http:','https:','#')):assert (U/href).exists(),href
    files={str(p.relative_to(U)):sha(p) for p in U.iterdir() if p.is_file()}
    (U/'FINAL_AUDIT.json').write_text(json.dumps({'status':'PASS','at':stamp,'report_links_valid':True,'docs_synced':[str(p) for p in docs],'protected_source_pins_unchanged':len(a['pins']),'training_runs':0,'user_recordings_read':0,'files':files},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('PASS: source protection, local report links, documentation sync. No training.')
if __name__=='__main__':main()
