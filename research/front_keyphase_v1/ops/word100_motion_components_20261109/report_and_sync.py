from common import *
import re,datetime

def main():
 decision=read(U/'PUBLIC_DECISION.json');ev=read(U/'EVALUATION_COMPLETE.json');primary=ev['results']['motion_only_42'];private=read(U/'PRIVATE_POSTHOC.json');integ=read(U/'INTEGRATION.json');browser=read(U/'BROWSER_VERIFIED.json')
 assert browser['status']=='PASS' and read(U/'RUNTIME_CHECK.json')['passed'] and read(U/'SERVER_READY.json')['state']=='ready'
 result=dict(status='COMPLETE_PARTIAL_LEARNED_MOTION_GAIN_AND_RESEARCH_COVERAGE_FIX',goal='Reusable handshape, body-relative movement and evidence coverage; no word-specific numerical exceptions',public_data=dict(fit_clips=36,fit_ids=list(range(1,10)),dev_clips=12,dev_ids=[11,12,13],word_count=4,additional_fit_motion_pairs=71,fit_pairs=105,augmented_fit_samples=420,dev_pairs=32,source='public Korean sign RGB and cached RTMW coordinates; same Codex sparse visual review; historical DEV'),training=dict(new_shared_motion_readout=True,models=2,seeds=3,epochs=240,primary='motion_only_42',primary_parameters=4083,base_conv_gru_frozen=True,uses_word_id=False,uses_person_id=False,old_finetuned_candidate_replaced=False),normal=primary['normal'],controls={v:{k:s[k] for k in ['n','raw_correct','withheld']} for v,s in primary.items()},speed=ev['timing']['motion_only_42'],public_decision=decision,web=dict(url='http://127.0.0.1:8814/',component_coverage_applied=True,new_direction_model_applied=False,tcns_unchanged=True,feedback_base_unchanged=True,examples=integ['examples'],browser_checks=browser['count'],camera=False),private_posthoc=private,next='Disambiguate automatic movement windows under combined shape+direction errors and collect within-hand fingertip convergence labels across expressions; do not tune on repeated DEV or private captures',active_training=False,active_evaluation=False)
 write(U/'RESULT.json',result)
 report='''2026-11-09 공통 움직임 판정부와 확인 범위 보완

이번에 바꾼 원칙
단어별로 점수나 예외를 바꾸지 않는다. 몸을 기준으로 손이 위/아래로 움직이는지, 반대팔을 따라 어느 쪽으로 이동하는지, 두 손 사이가 벌어지는지 등을 공통 부품으로 계산한다. 단어 설명은 필요한 부품과 순서를 연결한다. 한 손 안의 손끝 모양은 두 손 사이의 거리와 별개다.
공통 코드라는 것과 모든 수어에서 정확도가 검증됐다는 것은 다르다. 이번 성능 확인 범위는4단어다.

실제로 한 학습
현재 시간 합성곱+GRU의 손모양/단계 가중치는 보존했다. 같은 손모양인데 이동하는 상황을 별도로 볼 수 있도록, 신체 상대 위치의9개 시간표본과 변화량을 읽는 작은 공통 신경망을 지도학습했다. 기존 특징을 함께 넣는 조건도 비교했다. 새 판정부에 단어ID·사람ID·정답 방향을 입력하지 않았다.
공개 수행자 기록ID01~09의36영상, 움직임 정답105쌍(추가71쌍)을 사용했다. 전체/상반신과 정방향/역방향 학습 변형을 포함한420항목이며420개의 독립 촬영이 아니다. 조건2개×고정seed3개×240epoch. 대표seed42는 결과 전 고정했다.
검사는 학습에 넣지 않은 기록ID11~13의12영상,32구간쌍이다. 기존에도 개발에 사용한 자료이므로 새로운 독립시험이라고 부르지 않는다. 시각 정답은 같은Codex가 공개RGB를 본 것이며 수어 전문가의 정답 검증은 아니다.

좋아진 점 — 구간을 미리 지정한 비교
                         기존 공통계산     직전 미세조정      이번 공통신경망
움직임/유지 구분             26/32              28/32              31/32
실제로 이동한 구간           20/21              19/21              21/21
별도 이동 없이 유지          6/11               9/11              10/11
팔을 따라 쓸어내리는 구간      3/3                1/3                3/3
32는 단어 수나 전체 영상 인식률이 아니라, 사람이 앞뒤 시점을 확인한 움직임 비교 항목 수다. 새3seed 모두 정상31/32였고,한seed는 확신 기준에서1항목을 더 보류했다. 남은1오류는 손 사이가 조금 변하는 유지 구간을 벌어지는 동작으로 본 것이다.
좌우 반전+손 연결 변경, 좌표 평행이동/크기 변경, 시간만0.75/1.25배 변경에서도 대표결과는31/32. 실제 촬영 속도 변이 전체를 증명한 것은 아니다. 역순30/32. 중간2프레임 누락은23정답/9보류, 중간절반 누락은32개 모두 보류했다. 보류를 정답으로 세지 않았다.

왜 새 모델을 아직 화면에 교체하지 않았나
화면이 자동으로 고른 구간으로 연결하면 정상 팔쓸기3/3과 단순 역방향3/3은 구분했다. 그러나 손모양과 방향을 동시에 바꾼 합성시험3개 중1개에서는 새 모델이 방향을 반대로 읽었다. 기존 방향 계산은 그3개를 모두 잡았다. 따라서 단일 구간 개선을 전체 자동 피드백 완성이라고 주장하지 않고 기존 방향 검사를 유지한다. 두 화면카드가 같은 이동을 보여주므로6카드 결과를6독립영상처럼 세지 않았다.
속도는 미리 계산된6구간을 새 작은 판정부가 읽는 부분만 중앙값약0.043ms였다. 관절검출·구간선택·녹화·DTW·화면표시를 포함한 사용자 응답시간은 아니다.

이집트 손끝 문제에서 확인한 점과 실제 적용한 것
현재 손모양 출력은 주로 긴 손가락 펴기/닫기 등7특징이며, 각 손의 손끝 모으기를 따로 배운 출력이 없다. 두 손을 벌렸다는 결과 또는 손가락 펴기 결과로 마지막 손끝 모양까지 맞았다고 할 수 없다.
공개6영상의18시점,36손 확대화면을 확인했다. 납작하게 모인 긴 손가락은 보이지만 엄지와 모든 손끝 관계는 일관되게 보이지 않아 새 확정 정답을 만들지 않았다. 저장된 국립국어원 설명23905에는‘두 손 각각 손끝을 모은다’가 있고5990에는 그 구절이 없다. 두 표현 변이를 구분하지 않고 모든 예시에 강제하지 않는다. 현재 실시간 사이트 본문 재접속은 성공하지 않아 기존 저장본을 근거로 삼았다.
보완한 공통 처리: 필요한 특징마다 손·구간·확인 가능성을 구분한다. 다른 특징이 맞아도 미확인 특징을 대신 채우지 못한다. 확인한 차이는 유지하며 불확실을 오답으로 바꾸지 않는다. 임의 단어ID와 양손 조합61검사를 통과했다.
연구 화면8814에 이 처리와 설명별 확인항목 연결을 적용했다. 공개 저장결과93조건을 재생해 관련 표시21개가 변경됐고, 다른3단어 결과 및 기존 차이 경고는 보존했다. 공개 예제6개도 갱신했다. 새 학습모델과 기존TCN/Conv+GRU 가중치는 교체하지 않았다.

최근 사용자 촬영 사후 확인
공개 결론을 확정한 뒤 승인된 최근 이집트143프레임 결과만 확인했다. 기존‘오른손 펴기·왼손 펴기 특징이 비슷’에서‘두 손의 이동과 손끝 모으기는 따로 확인해야 해요. 마지막 손끝 모양은 아직 확인하지 못했어요’로 바뀐다. TCN순위와DTW는 동일하다. 이것은 손끝 오류를 정확히 검출했다는 뜻이 아니라, 확인하지 않은 부분을 비슷하다고 넘기던 표시를 고친 것이다. 개인 촬영은 학습·튜닝·후보 선택에 사용하지 않았다.

검증과 보존
입력/변환924확인, 저장결과 별도산술266확인, 공통 확인범위61검사, 공개 런타임 연결, 실제브라우저29검사 완료. 같은Codex의 별도 코드 검산이며 독립 전문가 검증이 아니다. 최초 런타임 검사에서 공통모듈 이름 충돌로 pilot경로를 못 찾았고, 격리된 검사 실행으로 해결했다. 실패 기록을 보존했다. C·T3-F·현재RTMW TCN·기존Conv/GRU/단계모델 보존. 학습/평가 모두종료.8814는준비완료·카메라꺼짐.

다음 핵심 작업
1. 복합 오류 때 자동 구간이 어떻게 바뀌는지 기존 저장 결과로 원인을 구분한다. 신경망과 관절의 명확한 이동 방향이 모순될 때 좋은 동작으로 승격하지 않는 공통 연결을 준비한다. 새 경계값을 기존DEV에 맞춰 반복하지 않는다.
2. 손끝 모으기/펴기/주먹을 구분할 실제로 보이는 공개 정답과 설명 변이를 여러 표현에서 연결한다. 이집트 전용 예외나 단어 이름으로 만든 정답을 쓰지 않는다.
3. 공개 학습/검증 분리를 유지해 충분한 정답이 있는 특징만 학습하고, 정상·반대방향·손모양만변경·복합오류·관측부족을 함께 확인한 뒤 새 모델의 웹 적용을 판단한다.
'''
 (U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
 current='''### 2026-11-09 공통 움직임 학습 개선·확인 범위 웹 보완

word100_motion_components_20261109. 사용자 최신지시: 모든 보완은 공통 신체관계/손특징으로 구현하고 단어명 수치예외 금지. 설명별 필요한 부품연결은 허용하되 전체단어검증주장 금지. 공개4단어36FIT(ID01~09)/12기존DEV(11~13). 추가RGB 움직임71쌍으로FIT105쌍/변형420표본. 단어/사람ID 없는9시간표본+신체상대변화 신경망2조건×3seed×240epoch,42사전고정. 기존Conv/GRU/손모양/단계 가중치보존. 직접구간32쌍에서기존기하26→새31,이동20/21→21/21,유지6/11→10/11,팔쓸기기존3/3·직전미세조정1/3→새3/3. 새3seed정상31/32. 상반신/미러손연결/스케일·이동/시간늘림동일,역순30/32,2프레임누락23정답9보류,절반누락32보류. 기존노출DEV와같은CodexRGB주석·주어진구간평가이므로전문가/독립일반화/전체자동정오로승격금지.

자동구간정상팔쓸기3/3·역순3/3이나손모양+역순복합3개중1개새모델실패(기존계산3/3검출). 새방향모델웹미적용.6관계이미계산한작은판정부CPU중앙0.043ms는전체응답시간아님. 이집트공개6영상18시점36손확대,긴손가락모임은보이나엄지/모든손끝정답불충분. 저장사전23905suffix/5990차이보존,손끝모으기전용학습0. 공통component_coverage부품은필요특징의손·구간·출처연결/관측확인을분리하고미확인특징을다른성공으로채우지않음. 8814 consumer+CARDS+6예제반영,모델/TCN/DTW보존.93공개저장조건21표시변경,다른3단어/기존차이보존. 공개결론동결후사용자최근이집트143프레임사후확인:마지막초록→손끝모양확인불가,TCN/DTW동일;오류검출완료주장금지. 개인학습튜닝선택0.924입력/266산술/61범용계약/런타임/29브라우저통과. 같은Codex별도검산. 최초import경로충돌보존·격리검사해결. C·T3F/현재TCN/기존Conv+GRU보존,새HTML없음,RESULT_KO.txt/RESULT.json/FINAL_CHECK.json. 모든학습평가종료,8814ready카메라꺼짐.
'''
 nextstep='''### 2026-11-09 공통 움직임·확인 범위 보완 이후

다음 단일작업은 기존저장AUTO_WINDOW_CHECK의 복합손모양+역방향1실패에서 자동구간/현재상대위치시계열/분류기 판단을 분리하고,명확한관절이동과신경망판단모순을긍정으로승격하지않는공통연결을준비하는것. 새모델은실험후보보존·웹방향계산교체금지,동일DEV점수로가중치/문턱반복튜닝금지. 손끝모으기는별도의부족한감독정답과설명변이문제이므로여러표현의보이는공개사례/엄지포함양음성을먼저확보,open_four0/주먹/두손거리로정답만들지않음. 핵심목표는학습된동작순서/손모양/신체상대관계로새사용자에게근거있는피드백,모든단어공통알고리즘;단어설명별필요부품연결과단어특화예외를혼동금지. 이번6학습/7모델9조건/93저장재생완료재실행금지. 최근개인촬영은공개결론이후사후확인만,학습튜닝선택0.8814새로고침가능·카메라사용자버튼만. 다음에C·T3F나현재TCN을교체하지않는다.
'''
 path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=path.read_text('utf-8-sig')
 for n,block in [(8,current),(9,nextstep)]:
  m=re.search(r'^## '+str(n)+r'\. .*$',s,re.M);assert m;s=s[:m.end()]+'\n\n'+block+s[m.end():]
 path.write_text(s,encoding='utf-8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+current+'\n'+nextstep,encoding='utf-8')
 state=read(O/'STATE.json');unit=dict(status=result['status'],report=str(U/'RESULT_KO.txt'),result=str(U/'RESULT.json'),verification=str(U/'FINAL_CHECK.json'),active=None,web='http://127.0.0.1:8814/',new_direction_model_deployed=False,coverage_fix_deployed=True,private_training=False,normal_interval_accuracy='31/32 vs26/32 geometry; not whole-sign accuracy',next=nextstep.split('\n\n',1)[1].strip())
 state={U.name:unit,**state};state['current_stage']='MOTION_COMPONENT_STUDY_COMPLETE_COVERAGE_FIX_IN_RESEARCH_WEB';state['updated_date_kst']='2026-11-09';state['updated_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();write(O/'STATE.json',state)
 checked=0
 for p,h in read(PREV/'FROZEN.json')['source_pins'].items():assert sha(p)==h;checked+=1
 for f,h in decision['pins'].items():assert sha(U/f)==h;checked+=1
 for r in read(U/'READY.json')['rows']:assert sha(r['path'])==r['sha256'];checked+=1
 ckpt=J/'tcn_runs/finetune_C_20261013';assert sha(ckpt/'model.pt')==read(ckpt/'COMPLETE.json')['checkpoint_sha256'];checked+=1
 assert sha(D/'consumer.py')==integ['code_sha256'] and sha(D/'CARDS.json')==integ['cards_sha256'];checked+=2
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md']:
  assert U.name in p.read_text('utf-8')[:35000];checked+=1
 assert next(iter(read(O/'STATE.json')))==U.name;checked+=1
 write(U/'FINAL_CHECK.json',dict(status='PASS',final_hash_checks=checked,protected_models_unchanged=True,private_learning=False,direction_model_deployed=False,coverage_research_applied=True,camera=False,active_training=False,counts=dict(input=924,arithmetic=266,coverage=61,browser=29),report_sha256=sha(U/'RESULT_KO.txt'),result_sha256=sha(U/'RESULT.json'),document_hashes={str(p):sha(p) for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']},verification_owner='Codex same operator separate checks; not expert independent validation'))
 print('REPORT_AND_SYNC_PASS',checked)
if __name__=='__main__':main()
