from bilateral_core import *
from datetime import datetime,timezone,timedelta
import urllib.request

def main():
 f=read(U/'FINAL_RESULT.json');v=read(U/'VERIFICATION.json');ui=read(U/'BROWSER_VERIFIED.json');deploy=read(U/'DEPLOYMENT.json')
 text='''부분 동작·완료 분리 / 양손 공통 보완 결과

결론
승인받은 정답 보강→학습→설명 연결 비교→공개 검증·선별 웹 적용을 수행했다.
부분 동작을 완료로 착각하는 문제는 줄었고 왼손 구간 구분은 나아졌다. 그러나 정상적인 오른손 전체 동작까지 미완으로 보류하는 문제가 늘어 새 판정부는 웹에 넣지 않았다.
웹에는 검증한 표시 기능만 반영했다. 현재 촬영의 AI 판정 성능이 이번 연구 후보의 성능으로 바뀌었다고 말하면 안 된다.

1. 정답을 어떻게 나눴나
기존 오른손 중심 정답에 실제 공개 원본 12영상의 왼손을 확인했다. 6영상은 학습, 다른 기록ID의 6영상은 개발 비교다. 좌우 반전으로 만든 한국수어를 정상 정답으로 쓰지 않았다.
합계 학습32영상, 실제 이동61개. 이로부터 전체 이동/앞부분/뒷부분/중간부분/유지/여러 이동을 나눈1827개 입력을 만들었다. 1827개의 서로 다른 영상이라는 뜻이 아니다.
방향 정답과 행동 시작·끝의 관측 정답을 분리했다. 움직이는 손과 상대손도 따로 기록했다. 손끝 접촉·손가락 모양·교육적으로 옳은 수어의 최종 완료를 이 손목 정답으로 대신하지 않았다.
주석은 같은 Codex가 원본 큰 몸통·팔 화면을 확인한 것이며 독립 사람/농인 전문가의 정답은 아니다. 경계는 몇 프레임의 불확실한 범위로 보존했다. 실제 왼손 보강은 두 표현의 높이/양손 상대 높이에 한정된다. 모든 종류의 왼손 동작 검증 완료가 아니다.

2. 무엇을 학습했나
기존 시간합성곱+양방향GRU 움직임부의 가중치를 복사해, 현재 후보 앞뒤160ms와 해당 후보를 함께 보는 별도 판정부를 지도 미세조정했다. 방향·시작/끝·전체/부분/복수 구간을 각각 출력한다.
100단어899영상으로 배운 공통부와 기존 손모양·방향 모델은 그대로 보존했다. 100단어 전체를 이번에 다시 학습한 것은 아니다. 새 판정부에는 단어 ID, 정답 방향, 수행자 ID를 입력하지 않는다. 같은 상대 신체좌표 계산에 주체손/상대손을 전달한다.
초기값42/43/44의3회,각60epoch를 점수확인 전에 고정했다. 이전/새판정부만/공통연결만/둘다를 비교했다. 비교 후 임계값이나 정답을 고쳐 좋은 숫자를 만들지 않았다.

3. 이전과 현재 — 지정한 구간의 부품 비교
다른 기록ID의 공개8영상에서 만든100구간(전체19,부분50,그 외31)을 평가했다. 같은 영상의 파생 구간은 통계적으로 독립된 사람100명이 아니다.
- 부분50개를 완료라고 착각: 이전9개 → 새0/1/1개.
- 실제 왼손 전체이동12개를 완료로 확인: 이전8개 → 새10/8/9개.
- 실제 오른손 전체이동7개를 완료로 확인: 이전7개 → 새4/2/3개. 이것이 웹 채택을 막는 중요한 퇴행이다.
- 왼손 구간종류62개: 이전52/52/51개 → 새59/57/55개.
슬래시로 나열한 세 숫자는3회 학습 결과다. 가장 좋은 한 번만 골라 발표하지 않았다. 이 숫자는 새 사용자의 수어 전체 정답률이 아니다.

4. 자동 구간→최종 안내 비교
기존 공개12영상에서 만든78조건을3회·3방식으로 비교했다. 정상12+속도24(양손 교차 포함),기존 모양/방향 오류9,관측삭제18,행동삭제6,역재생9다. 옛 TCN·DTW 결과와 입력은 저장한 것을 재사용했다.
결합안에서 정상/속도36조건의 새 차이 경고0,기존오류9 유지,관측부족18의 정상 승격0은 지켰다.
하지만 정상12조건 중 모든 필수 카드가 비슷하던9조건이 새4/5/6조건으로 줄었다. 정상 속도 변화에서도 보류가 늘었다. 경고가 없다는 사실만으로 정상 성능을 유지했다고 보고할 수 없다.
역재생9조건에서 남아 있던 거짓비슷함1개는 새0/0/1개로 줄었지만3회 모두 해결하지 못했다. 행동삭제 조건의 경고수 증가에는 다른 종류의 경고가 섞일 수 있어 누락 정확도 개선으로 세지 않았다.
별도6표현·기록ID14의24조건에서도 알려진21후보의 종류구분은14/14/13→17/15/14로 나아졌지만 실제 전체이동8개 확인은6→2/3/2로 나빠졌다. 정답이 불명확한305후보는 분리해 공개했다. 이것들도 이전에 노출한 개발자료이며 독립 최종시험이 아니다.

5. 웹에 실제로 반영한 부분
최신 연구 웹8814의 index.html/app.js/style.css만 바꿨다. 촬영·RTMW·TCN·DTW·기존 신경망과 판정은 변경하지 않았다.
- 위 시간띠는 주요 동작 후보, 아래 시간띠는 확인 결과다. 준비/연결/마무리는 빈 곳의 가능한 해석일 뿐, 확정한 세부 단계로 꾸미지 않았다.
- 청록은 확인한 특징이 비슷함,주황은 차이 근거,빗금은 근거 부족,빈 곳은 검사 밖이다. 관측 부족은 오답과 구분한다.
- 카드를 누르면 손모양/이동방향의 확인 상태와 '행동 전체 완료:미확인'을 따로 보여준다. 일부 특징이 맞았다는 것을 전체 정답으로 읽지 않도록 했다.
- 관측 근거가 있는 손은 실제 영상으로 확대할 수 있다. 이동 방향 근거가 있는 구간은 관측 손목 궤적을 점선 화살표로 보여준다. 끊기거나 크게 튀는 좌표는 화살표로 연결하지 않는다. 교정할 손 위치를 만들어 넣는 기능이 아니다.
- 누락/연결 불확실 구간에서는 참고만 보여주고 없는 사용자 프레임을 생성하지 않는다.
공개 저장결과6개,두영상재생/카드/확대/화살표/누락/세화면크기/카운트다운·로딩 모의상태 등35개 브라우저 검사를 통과했다. 사용자 촬영 시험을 했다는 뜻은 아니다. 화면 전환 중 이전 카드가 눌리는 경우도 방어했다.

6. 속도·보호·검증
새 후보의 추가 계산만 중앙값 약0.069초,95백분위 약0.178초였다. RTMW·촬영종료·TCN·DTW·화면출력 전체시간이 아니다. 새 모델은 미배포이므로 실제 웹 대기시간 개선이라고 주장하지 않는다.
원본/동결 모델·결과 보호,분리,출력 산술 등4248개 별도 저장결과 검산을 통과했다. 구현과 별도 검산 모두 같은 Codex이며 독립 사람 검증이 아니다. C·T3-F,공통100단어부,기존 손모양·방향 보존. 사용자·멘토 자료 접근/학습/튜닝/후보선택0,다운로드0,Git0,카메라0.
초기 fit_additional 구분 오류,검산 코드 오타,브라우저 제한환경 실행실패,화면 전환을 기다리지 않던 검사 오류는 첫 기록을 남기고 수정했다. 모델 평가조건이나 점수를 고친 것은 아니다.

남은 핵심
한 방향이 보이는 것과 중요한 행동을 끝까지 한 것은 다르다. 이를 분리해 배우는 효과는 확인했지만, 실제 정상 동작의 끝을 일관되게 인정하는 능력이 아직 부족하다. 새 후보의 전체 웹 적용과 확정적인 자동 교정은 보류한다.
다음은 실제 FIT에서 정상 완료/다음 동작으로 이어짐/중간 끊김을 같은 관계축의 대비 정답으로 보강하고, 경계 불확실성을 학습에 직접 반영하는 작업이다. 자세한 범위와 통과 조건은 NEXT_PLAN_KO.txt를 따른다. 이번 완료 학습/평가를 반복 실행하지 않는다.
'''
 (U/'RESULT_KO.txt').write_text(text,'utf-8')
 nxt='''다음 단일 작업: 정상 완료를 잃지 않는 시작·끝 판정부

근거: 직접 지정한 정상 오른손 전체7개도7→4/2/3으로 감소했다. 자동 구간 찾기만의 문제가 아니다. 예컨대 순종 기록11 전완상승81~89는 전체구간 종류확률이약.99인데 끝확률은약.68로 완료를 거부했다. 다른 경우는 구간종류부터부분으로 혼동했다. 이런 서로 다른 실패를 한 문턱 조정으로 덮지 않는다.

1. FIT에서 동일 관계축의 정상완료/진행중/다음행동전환을 대조 정답으로 보강한다. 이미 본 DEV의실패 장면을학습에넣지않는다. 기록ID와원본해시를먼저분리하고 왼손의전완/상대팔방향 등현재부족한축도원본을확인한다. 실제자료가없으면좌우반전으로정상수어정답을만들지않는다.
2. 시작·끝은단일정확프레임이아닌검토범위로감독한다. 전체이동종류와끝근거를구분하고 정상적인다음동작연결을미완으로혼동하지않는지확인한다. 기존모양·방향·공통100단어는고정하고필요한작은판정부만비교한다. 무조건더낮은문턱으로완료를늘리지않는다.
3. 직접구간과자동구간을분리해완료재현율/부분오완료/정상보류를함께본다. 실제반대방향시도가여럿이면모두보존하며맞는조각만선택하지않는다. 편집점주변급변은자연속도변화와구분해인위편집의부산물을교정성공으로세지않는다.
4. 고정한3회학습전부에서정상완료퇴행0·부분오완료증가0·정상속도교차경고증가0·기존모양방향오류손실0·관측부족정상승격0을확인한다. 평가노출이없는공개자료는주석/원본중복감사후별도로선정한다. 자연학습자오류정답이없으면합성시험성공과실제교정정확도를분리한다.

현재 새모델 미채택. 웹에는이번검증한표시3파일만적용. 이다음작업은미착수이며완료결과재학습/재추론/재채점금지. 사용자자료학습/튜닝/선택금지,C·T3-F와원본보존.
'''
 (U/'NEXT_PLAN_KO.txt').write_text(nxt,'utf-8')
 now=datetime.now(timezone(timedelta(hours=9))).isoformat();title='2026-11-11 부분/완료·양손 보완 4단계 실행 — 표시만 웹 반영'
 summary='공개왼손12영상검토(학습6/개발6),합계FIT32영상61이동·1827파생입력으로별도Conv+GRU판정부3회학습.부분50오완료9→0/1/1,왼손전체12확인8→10/8/9이나오른손전체7확인7→4/2/3퇴행.78조건정상속도추가경고0·기존오류9유지·관측부족18정상승격0이나정상보류증가.추가6표현전체8도6→2/3/2.새판정부미채택.검증35개를통과한2종시간띠·특징/완료분리·손확대·관측궤적표시만8814연구웹3파일에적용.4248저장검산PASS(동일Codex).C·T3-F/공통100단어/기존판정유지,개인자료0.독립최종평가아님.'
 nextone='정상완료/진행중/다음행동전환의FIT대조정답과경계불확실성감독을보강.직접구간오류와자동연결오류분리,부분오완료와정상보류를함께평가.다음계획은미착수.이번완료학습/평가재실행금지.'
 path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';content=path.read_text('utf-8-sig')
 for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nextone)]:
  assert heading in content
  content=content.replace(heading,heading+'\n\n### '+title+'\n\n'+body+'\n근거 '+str(U/'RESULT_KO.txt')+'\n계획 '+str(U/'NEXT_PLAN_KO.txt')+'\n',1)
 path.write_text(content,'utf-8')
 with (ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf-8') as h:h.write('\n\n['+title+']\n이유: 부분에서방향이보이는것과전체완료를같이취급해오완료가생겼고기존새연결이오른손중심이었다.두감독을나누고원본왼손을보강했다.\n'+summary+'\n전체목표변경없음.새판정부의거절이많아지는퇴행을숫자로공개하고검증한표시만적용했다.자동최종교정완성주장없음.\n다음 '+nextone+'\n근거 '+str(U)+'\n')
 state=read(O/'STATE.json');state.update(current_stage=U.name,last_updated=now,next_single_action=nextone,active_web_test=D.name)
 server=dict(listening=False,camera_used=False)
 try:
  status=json.load(urllib.request.urlopen('http://127.0.0.1:8814/api/status',timeout=3));server.update(listening=True,state=status.get('state'),pid=read(D/'SERVER.json').get('pid'))
 except Exception as ex:server['check_error']=type(ex).__name__
 write(U/'SERVER_CHECK.json',server)
 state[U.name]=dict(status='COMPLETE_DISPLAY_APPLIED_MODEL_REJECTED',steps_executed=[1,2,3,4],report=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFICATION.json'),ui_verification=str(U/'BROWSER_VERIFIED.json'),web_display_applied=True,model_adopted=False,training_running=False,private_data=0,next=str(U/'NEXT_PLAN_KO.txt'),server=server)
 state[D.name]['display_followup']=str(U/'DEPLOYMENT.json');state[D.name]['latest_server_check']=server;write(O/'STATE.json',state)
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계\n\n'+title+'\n'+summary+'\n\n현재단위 '+str(U)+'\n상세 RESULT_KO.txt / FINAL_RESULT.json / DEPLOYMENT.json.\n다음 '+nextone+'\n최신웹 http://127.0.0.1:8814/ / 서버 '+json.dumps(server,ensure_ascii=False)+'\n학습/평가프로세스없음.이번완료결과재실행금지.사용자자료접근·학습·튜닝·선택금지,카메라는사용자버튼으로만.운영C·T3-F/원본/과거동결결과보존.\n','utf-8')
 write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_DISPLAY_APPLIED_MODEL_REJECTED',pid=None,training_running=False,server=server))
 # Final protection check exempts exactly the three explicitly deployed UI files.
 allowed={r['path']:r['after'] for r in deploy['changes']};count=0
 for p,h in read(U/'PROTECTIONS.json').items():assert sha(p)==allowed.get(p,h),p;count+=1
 for p,h in allowed.items():assert sha(p)==h
 for name in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md']:assert U.name in (ROOT/name).read_text('utf-8')
 write(U/'DELIVERY_CHECK.json',dict(status='PASS',protected_files=count,allowed_web_changes=3,model_changes=0,documents_synchronized=True,server=server,independent_human=False))
 event('complete',protected=count,server=server)
if __name__=='__main__':main()
