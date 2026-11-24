from common_action import *
import datetime,shutil
def main():
 assert read(U/'VERIFICATION.json')['status']=='PASS' and read(U/'CONTRACT_CHECK.json')['status']=='PASS'
 assert not (U/'RESULT.json').exists()
 now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
 scores=read(U/'EVALUATION_COMPLETE.json')['models'];full=read(U/'FULLPATH_SUMMARY.json');causes=read(U/'COMPONENT_CAUSE_AUDIT.json')['summary'];hp=read(U/'HISTORY_BACKFILL_VERIFICATION.json')
 report=f'''공통 행동 항목 학습·실제 분석 비교와 과거 기록 보완 — 2026-11-10

결론
승인한 네 단계(항목 감사→공개 정답 검증→학습→실제 경로 비교)를 마쳤다.
손모양 확인은 조금 좋아졌지만 방향·동작 구간까지 함께 안정화되지는 않았다. 새 모델은 기본 웹에 넣지 않았다.
과거 연구 기록의 누락된 큰 단계12개를 보완하고 이전 완료 보고서292개를 찾을 수 있는 근거 색인을 만들었다.
전체 피드백 AI 완성이나 모든 수어의 성능 향상을 뜻하지 않는다.

1. 어떤 정답으로 무엇을 학습했나
공개12표현72영상의 기존 시간별 손모양 주석을 다시 연결했다. 학습은60영상(기록ID01~09), 비교는12영상(ID11~13)이다.
그중4표현의 움직임 정답은 학습105구간쌍, 비교32구간쌍이다. 손목 높이 변화, 전완 세워짐, 반대팔을 따른 이동, 두 손 사이 거리 변화, 양손 상대 높이 변화를 구분한다.
학습의420항목은 전체/상반신·구간방향 변형을 포함한 수이며420개의 독립 영상이 아니다.
12표현의 손모양 정답이 있다는 것과12표현 모두의 움직임·반복·완료 정답이 있다는 것은 다르다.
이번에는 새 정답을 대량 생성한 것이 아니라 검토된 정답들을 하나의 공통 학습에 연결했다. 공개3영상의 시간순 RGB 시트도 다시 확인했다.
엄지와 모든 손끝 접촉, 반복의 필수 횟수처럼 확인되지 않은 항목은 정답을 만들어 붙이지 않았다. DTW로 옮긴 단계 주석은 기존 낮은 가중치의 추정으로 남겼다.
정답의 시각 검토자는 같은 Codex다. 한국수어 전문가가 붙인 독립 정답으로 표현하지 않는다. 비교 자료는 과거에도 사용한 개발 자료이며 새 사용자 최종시험이 아니다.

2. 두 학습 방법을 실제로 비교
A: 현재 시간 합성곱+GRU를 보존하고 새 공통 움직임 판정부만 학습.
B: 새 판정부와 함께 시간 합성곱+GRU 관측부도 미세조정. 기존 손모양·단계 출력층/기준흐름 가중치는 고정하고 관측 보존 손실을 추가.
각3개 고정 초기값, 각30회 학습. 총6회 모두 완료. 대표42는 점수 전에 고정했으며 결과가 가장 좋은 초기값을 고르지 않았다.
목표 단어는 단계 설명을 선택할 때만 사용한다. 관측된 손모양·움직임에는 목표 단어를 넣지 않았다. 현재 TCN은 이번에 학습하지 않았다.
중요한 점: 출력층의 가중치를 고정해도 그 앞의 관측부를 바꾸면 출력과 구간이 달라질 수 있다. 이번 전체 연결 비교에서 이 영향을 함께 확인했다.

3. 이전 → 현재 (대표42, 전체화면의 같은 확인 항목)
손모양: 기존221/228 → 판정부만221/228 → 함께 미세조정224/228.
나머지 두 미세조정 초기값은225/228,225/228이었다. 228은 손모양 확인 항목 수이며 단어 인식률이나 사용자 전체 정답률이 아니다.
단계: 기존72/84 → 판정부만72/84 → 함께 미세조정72/84. 다른 두 미세조정은71/84.
움직임/유지: 기존 공통 계산26/32, 이전 별도 움직임 후보31/32 → 이번 판정부만31/32 → 함께 미세조정30/32(다른 초기값31/32,30/32).
따라서 손모양 일부 향상은 확인했지만 단계·움직임을 모두 개선한 모델로 판단하지 않는다. 확신 부족과 관측 부족도 분모에서 빼지 않았다.

4. 실제 자동 구간으로 연결했을 때
각후보69조건 ×6개 모델 =414개 새 비교. 이전 모델은 저장 결과를 재사용했다.
69조건 = 정상12 + 느림12 + 빠름12 + 손관절가림12 + 마지막동작삭제6 + 마지막동작관절가림6 + 손모양만변경3 + 방향만변경3 + 둘다변경3.
같은 공개 원본에 만든 변형이며69명의 실제 오류 사례가 아니다. 앞60개는 원시RTMW→관측검사→TCN/새피드백→DTW→완료검사→최종관측검사 경로, 뒤9개는 저장한 편집 관절 입력 경로다.
대표B: 정상12 중 모든 필수 확인항목 비슷함9→8, 미확인3→4. 정상 원본에 새 차이 경고는0.
느린12조건에서는 새 방향 경고0→1. 정상 속도를 유지해야 하는 기준을 통과하지 못했다.
손모양만/방향만/둘다변경은 각각3/3에서 어떤 차이 표시는 남았다. 하지만 이것만으로 성공이라고 하면 안 된다.
손모양만 바꾼3개에서 새 방향 판정이 정상 방향을 지지한 경우는0/3이었다. 둘다바꾼3개에서도 방향 자체를 맞게 구분한 것은2/3이었다.
즉 손모양 차이를 잡았다는 사실이 잘못된 방향 설명까지 정당화하지 않는다. 이 문제를 따로 집계했다.
원본 정상 교차에는 새 차이 경고를 만들지 않았고, 마지막동작삭제6·마지막관절가림6·전체손관절가림12의 거짓전체비슷함은 모두0으로 유지했다.
기존 TCN 후보와 입력 프레임 수는 유지했다. 새 모델의 실패를 숨기려고 비교 구간이나 관측 부족 항목을 분모에서 빼지 않았다.

5. 왜 이 결과가 나왔나
확인된 사실: 손모양만 바꿔도 움직임 판정이 달라졌다. 따라서 공유 특징 안에서 모양과 방향의 영향이 섞인 문제가 있다.
또 미세조정으로 단계 후보창이 달라져, 정상 후반 동작을 앞 동작과 충돌한다고 보류한 사례가 생겼다.
학습량 부족 하나로만 설명할 수 없다. 학습한 정보의 분리와 실제 비교창 연결도 문제다. 특정 내부 가중치가 유일한 원인이라는 인과관계까지 증명한 것은 아니다.
기존 방향 계산을 새 판정부로 무조건 바꾸지 않으며, 이미 본 개발 점수로 기준·초기값·학습 횟수를 다시 맞추지 않았다.
이번 후보의 공통 관측 출력은 구현·검사했지만, 정확한 손끝 접촉/모든 신체 관계/반복횟수/모든 단어의 교육적 정오 판단을 새로 완성하지 않았다.

6. 속도와 검증
대표B의 저장 RTMW 입력부터 분석 결과 계산까지 중앙 {full['joint_finetune_42']['timing']['median']:.3f}초, 95백분위 {full['joint_finetune_42']['timing']['p95']:.3f}초였다.
촬영·RTMW 추출·모델 최초 준비·파일저장·화면 표시 시간은 제외한 계산 시간이다. 사용자 체감 전체 대기시간 개선으로 주장하지 않는다.
저장 결과 별도 산술검산1920개와 목표 누출/몸 기준 측정/시점 연결 계약검사156개 통과. 같은 Codex의 별도 검산이다.
첫 검산에서는 손가락을 가렸으면 팔꿈치·손목으로 보는 전완 움직임도 모두 보류해야 한다고 잘못 가정했다. 최초 오류와 코드를 남긴 뒤 손에 의존하는 항목만 검사하도록 바로잡았다. 손가락 미관측과 별개로 관측된 팔 근거는 유지하되 전체수어 정답으로 승격하지 않는다. 모델·주석·예측·기준을 바꾸거나 재학습하지 않았다.

7. 과거 연구 이력 보완
Docs/PROJECT_DIRECTION_HISTORY_KO.txt: 실제 진행은11월10일까지인데 기록이11월2일 중간에 멈춘 부분을 보완했다.
RGB 시간보완, YOLO 손, MANO/HaMeR, AIHub좌표 검증, E3 인식 비교, 학습된 시간별 피드백, 영상 중심 UI, RTMW 전환, 관절 재사용, 공통 입력 보호, 모양/방향, 완료 판정의 목표·이유·결과·적용 여부를 연결했다.
Docs/RESEARCH_REPORT_INDEX_KO.txt: 기존 완료보고서292개의 제목·경로·SHA256 색인. 폴더명이 본문에 없던245개는 인용 누락 후보이지 연구 내용245개가 전부 누락됐다는 뜻은 아니다.
현재처럼 쓰인 오래된 M0/MediaPipe/SignRep 설명은 당시 스냅샷으로 표시하고 최신 경로를 앞에 구분했다. 최초 실패·미채택 기록과 원본 보고서는 보존했다.

다음 작업 목표
공개 FIT 자료에서 '같은 움직임·다른 손모양'과 '같은 손모양·다른 움직임'을 짝지어, 어느 관측이 어느 질문의 근거인지 분리한다.
방향은 손가락 모양에 영향을 받지 않는 손목·팔 상대 흐름을 중심으로 보게 하고, 손모양은 손가락 관측으로 판정한다. 접촉·회전 등 다른 항목까지 단순 손목 이동으로 대체하지 않는다.
기존 단계/완료를 읽는 경로는 보존한 상태에서 새 관측을 연결하고, 자동으로 선택된 창에서도 학습 때와 같은 질문을 하는지 확인한다.
새 조건은 공개 FIT에서 고정하고, 정상 속도·교차·주요 오류·관측 부족을 함께 통과해야 적용 검토한다. 이번 점수로 특정 단어 예외나 문턱을 맞추지 않는다.
가려진 모든 손끝의 확정 정답을 기다리느라 관측 가능한 항목의 개발을 멈추지도 않는다. 다만 손끝 모으기나 반복 필수 횟수는 별도 근거가 있어야 배웠다고 보고한다.

보존
기존 C·T3-F, 현재 RTMW TCN, 기본 웹8814, 원본 자료 유지. 사용자·멘토 촬영 접근/학습/튜닝/선택0. 카메라/다운로드/외부전송/운영교체0.
명확한 전체 성능 개선이 없어 새 HTML은 만들지 않았다. 이번 단위의 완료와 최종 피드백 제품의 완성은 다르다.
'''
 (U/'RESULT_KO.txt').write_text(report,'utf-8')
 summary='공통항목 감사·기존 공개12표현72영상 주석 연결(60FIT/12역사DEV)·새head만/Conv+GRU미세조정2조건×3seed 학습·414자동경로비교 완료. 대표42손모양221→224/228,단계72/84유지,움직임기존별도후보31→30/32. 정상전체비슷함9→8/12,느림새방향경고1;모양만바꿈에서방향지지0/3,복합변경방향2/3로결합문제확인. 누락/관절가림거짓완료0유지,TCN동일. 새가중치웹미교체. 1920산술+156계약PASS(같은Codex),최초검산가정오류보존/결과변경없음. 과거누락12큰단계보완·292보고서근거색인. 개인자료0.'
 nxt='완료6학습/414비교 재실행금지. 공개FIT의같은이동-다른모양/같은모양-다른이동대조정답을검증하고,모양관측과손목·팔방향관측을분리하며기존단계/완료경로를보존하는조건을사전고정한다. 새학습이기존단계창을바꾸는영향과자동창감독불일치를함께검증. 역사DEV/ID10점수로문턱·seed·단어예외튜닝금지. 보이는손끝관계만추가주석,전문가규칙/숨은접촉추정금지. C/T3F/현재RTMWTCN/웹8814보존,사용자촬영학습·튜닝·선택금지.'
 write(U/'RESULT.json',dict(status='COMPLETE_COMPONENT_GAIN_FULLPATH_REGRESSION_NOT_DEPLOYED',summary=summary,report=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFICATION.json'),contracts=str(U/'CONTRACT_CHECK.json'),history=hp['history'],history_index=hp['index'],new_training_runs=6,fullpath_comparisons=414,default_web_changed=False,private_used=False,next=nxt))
 write(U/'NEXT_PLAN.json',dict(next=nxt,reason='Same latent used for shape and motion can mix causes; frozen output weights did not preserve automatic phase windows after encoder fine-tuning.',no_new_dev_tuning=True,whole_project_complete=False))
 history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
 with history.open('a',encoding='utf-8') as f:f.write('\n\n2026-11-10 — 공통 정답 연결·관측부 공동학습과 전체 경로 회귀 확인\n'+summary+'\n왜: 무엇을 어떻게 모았는지 배우는 근거와 실제 출력 연결을 점검하고, 손모양·관계·움직임을 공통 기능으로 가르치려는 사용자 승인.\n방법: 단어명을 관측에 넣지 않고 몸 기준 관계와 기존 시간 특징을 함께 사용; 출력부만 학습과 관측부 미세조정을 고정 비교.\n판단: 부분 손모양 향상은 있었으나 모양 변경이 방향 판정까지 바꾸고 정상 단계창도 달라졌으므로 웹 미교체. 이는 모든 공개 원본이 불량하거나 좌표가 완벽해야만 한다는 결론이 아니다.\n다음: '+nxt+'\n근거: '+str(U/'RESULT_KO.txt')+'\n')
 goal=ROOT/'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt';shutil.copyfile(goal,U/'before'/goal.name);gt=goal.read_text('utf-8')
 gt=gt.replace('개정: 2026-11-02 / 상태:', '원 설계: 2026-11-02 / 최신 실행 대조: 2026-11-10 / 당시 상태:',1).replace('최신 실행 현황 (2026-11-02','과거 실행 현황 (2026-11-02',1).replace('2. 현재 상태와 이번 정정','2. 2026-11-02 당시 상태와 정정 (과거 설계 기록)',1)
 first=gt.index('\n\n')+2;gt=gt[:first]+'최신 기준 (2026-11-10)\n목표는 아래1절 그대로다. 현재 웹은RTMW관절→TCN+관절전용Conv/GRU→학습흐름DTW이며SignRep RGB는기본실행경로에없다. 아래 과거계획의8804·SignRep·개별참고대조 설명은 현재구현으로 읽지 않는다.\n'+summary+'\n다음: '+nxt+'\n근거: '+str(U/'RESULT_KO.txt')+'\n\n'+gt[first:];goal.write_text(gt,'utf-8')
 context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=context.read_text('utf-8');title='### 2026-11-10 공통 관측 공동학습·자동경로 비교와 연구 이력 보완'
 for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:
  assert marker in txt;txt=txt.replace(marker,marker+'\n\n'+title+'\n\n'+body+' 근거 '+str(U/'RESULT_KO.txt')+'\n',1)
 context.write_text(txt,'utf-8')
 sp=O/'STATE.json';st=read(sp);st['current_stage']=U.name;st['next_single_action']=nxt;st['last_updated']=now;st[U.name]=read(U/'RESULT.json');write(sp,st)
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n최근 승인: 공통 기능의4단계(주석검증·학습·비교 포함) 전부 진행, 발표용 과거 누락 기록 보완.\n\n마지막 완료\n'+summary+'\n\n다음 한 작업\n'+nxt+'\n\n모든 학습/비교 프로세스 종료. 현재기본웹코드/서버/카메라 변경 없음. 신경망 전체완성이나 모든단어 교정정확도 향상 아님. 기본웹은RTMW관절전용계보이며SignRep RGB는실행하지 않는다. E완료충돌검사는격리 비교에만있음.\n\n검산오류: 손가락가림이면 전완움직임도모두보류한다는잘못된검산가정수정. 첫코드/로그보존, 모델·예측·주석/문턱변경0. 새후보DEV재튜닝금지.\n\n결과 '+str(U/'RESULT_KO.txt')+'\n검산 '+str(U/'VERIFICATION.json')+'\n원인 '+str(U/'COMPONENT_CAUSE_AUDIT.json')+'\n이력 '+str(history)+'\n근거색인 '+str(ROOT/'Docs/RESEARCH_REPORT_INDEX_KO.txt')+'\n','utf-8')
 write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_NOT_DEPLOYED',pid=None,time=now,training_complete=True,comparison_complete=True,history_backfilled=True))
 write(U/'FINAL_DOCUMENT_CHECK.json',dict(status='PASS',current_stage=read(sp)['current_stage'],files={str(p):sha(p) for p in [context,sp,ROOT/'NEW_CHAT_HANDOFF.md',history,goal,U/'RESULT_KO.txt',ROOT/'Docs/RESEARCH_REPORT_INDEX_KO.txt']},original_backfill_audit_was_pre_current_entry=True,source_reports_untouched=True))
 print(summary)
if __name__=='__main__':main()
