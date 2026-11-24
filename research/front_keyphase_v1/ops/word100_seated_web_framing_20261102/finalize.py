from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
assert (R/'AGENTS.md').exists()
check=read(U/'SAVED_VERIFICATION.json');inputs=read(U/'INPUT_VALIDATION.json');ui=read(U/'LIVE_UI_VERIFICATION.json');server=read(U/'SERVER_ACTIVE.json')
assert check['status']==inputs['status']==ui['status']=='PASS' and server['new_health']=='ok'
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
summary=('앉은웹캠범위 보완을 별도8804에 연결. 사용자입력은 실제화면전체,초기손목/추정하단 필수조건과 추가가로선 제거. '
         '공개48관측614계약검사, 실제손내림이 처음1초 화면밖인 공개4표현ID12세로크롭을 신규전체추론하여9인비교 완료. '
         '기존경계 산술은4편모두전체거절,새경로는4편모두항목결과 제공. 13구간 중 유사3/차이후보4/근거부족6; '
         '91구간별항목 유사61/차이후보5/부족22/참고쉼3. 교육정확도아님. 전체요청10.06~16.62초. '
         '단일시점/원좌표/PTS/전창보존,누락분모유지. C/T3F/기존8802/학습·참고은행·임계값보존,개인자료사용0/추가학습0.')
next_action=('사용자는 http://127.0.0.1:8804/ 에서 평소 앉은웹캠 조건으로 재시험 가능. 8802는이전판그대로. '
             '새사용자영상은추론/명시요청진단만,학습/튜닝/선택금지. 준비손화면밖을입력오류로취급X. '
             '주요동작의특징별근거부족과좌표차이경고는공개자료로만후속분석;단어예외/문턱완화X. '
             '이번4편은실제카메라분포/정확도검증이아닌공개영상세로크롭입력호환성시험. '
             'GPU성능불안정은타프로그램부하분리필요;이번10~17초로안정성해결주장X. 완료4편추론재실행금지.')
report='앉은 웹캠 촬영 범위 보완 완료 — 2026-11-02\n\n'
report+='새 테스트 주소: http://127.0.0.1:8804/\n기존8802는이전판으로유지. 카메라는사용자가직접켜기.\n\n'
report+='이전: 브라우저녹화는전체화면이었으나 분석에서휴식손목/어깨비율하단으로사용자영상도사용범위를제한. 그 하단이화면밖이면전체거절.\n'
report+='현재: 사용자화면전체를그대로검출하고실제센서범위내관절만사용. 임의가로선없음. 초기/종료손내림이화면밖이어도거절하지않음. 어깨정규화는기존frame별계산유지. 공개참고의기존준비범위/모델/분모/임계값유지.\n'
report+='확인가능한손모양·팔관계는비교하고빠진근거는특징별보류. 추정좌표나모델확률로실제미관측을정답처럼채우지않음. UI에비교된항목수와보류항목수를나누어표시.\n\n'
report+='공개검증: 48저장관측의화면내좌표·공통특징값·프레임·PTS·한시점누락의존성·반대손보존614검사PASS. '\
        '새공개4영상은소스첫초어깨중심+어깨폭1배높이로세로크롭,resize없음,frame/PTS보존. 촬영조건근사이며실제앉은카메라/다양한환경성능정답이아님.\n'
for row in check['records']:
    report+=f"- {row['label']}: 처음1초 보이는손{row['first_second_any_visible_hand']}/{row['first_second_frames']}frame, 이전정책{row['old_reason']}(하단{row['old_proposed_bottom']}>화면{row['actual_sensor'][1]}), 새9인비교완료 {row['seconds']:.2f}초.\n"
report+='이전정책은새크롭관측에고정경계함수만계산한대조이며,기존모델재추론을반복한비교가아님.\n'
report+=f"13주요구간: {check['stage_counts']}\n91구간별특징항목: {check['feature_counts']}\n"
report+='구간전체보류6개에도비교가능한항목이존재하면따로표시. 위수치는교육정답률/사용자정확도가아님. '\
        '순종/잘하다손모양일부근거부족,조상위치차이후보잔존. 정상동작오경고해결/3초속도달성/실제의미구분완성주장없음.\n'
report+='63저장배열/원장검사 및 실제브라우저4표현×9참고/동기재생/가독성/모바일폭검증PASS. 구현과추가검산모두Codex이며독립사람교육정답아님.\n'
report+='첫자동UI시험이새업로드응답전이전jobID를읽은오류보존. 실제sourceSHA로이미완료2편을찾아재사용,추론중복0. 나머지2편후완료;표시문구검증은저장결과만재사용.\n\n'
report+='적용: 신규8804 연구테스트경로만. 실패한속도후보/새GPU실행방법혼합없음. 현재GPU한개순차+동결기존모델. '\
        'C/T3F,8802원본,참고/정규화/임계값그대로. 개인·멘토영상접근0/재추론0/학습0/튜닝0/외부전송0/새다운로드0.\n\n다음: '+next_action+'\n'
(U/'RESULT_KO.txt').write_text(report,encoding='utf8')
complete=dict(status='READY_SEATED_SENSOR_DEMO_WITH_PARTIAL_FEEDBACK_LIMITS',updated_at=now,summary=summary,
              demo_url='http://127.0.0.1:8804/',server=server,public_clips=4,public_observations=48,
              new_model_training=0,user_inputs_used=0,models_changed=False,old8802_preserved=True,
              stage_counts=check['stage_counts'],feature_counts=check['feature_counts'],
              accuracy_claim=False,scope='input route fix, not semantic accuracy or stable latency claim',
              next_action=next_action,result=str(U/'RESULT_KO.txt'),
              code_sha256={n:hashlib.sha256((U/n).read_bytes()).hexdigest() for n in ['adapter.py','sensor_boundary.py','run_server.py','app.js','index.html']})
with (U/'COMPLETION.json').open('x',encoding='utf8') as f:json.dump(complete,f,ensure_ascii=False,indent=2)
p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig')
for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','앉은웹캠 전체화면 시험 준비완료',summary),
                          ('## 9. 다음 단일 작업','새8804 재시험과 특징별 보류 확인',next_action)]:
    assert marker in t;t=t.replace(marker,marker+'\n\n### 2026-11-02 '+title+'\n'+body,1)
p.write_text(t,encoding='utf8')
p=O/'STATE.json';d=read(p);d['current_stage']=complete['status'];d['updated_date_kst']='2026-11-02';d['next_single_action']=next_action;d[U.name]=complete
p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_action+
    '\n\n실행: '+json.dumps(server,ensure_ascii=False)+'\n근거: '+str(U/'RESULT_KO.txt')+
    '\n모든새공개평가/브라우저검증종료,8804만사용자테스트용상주. 기존8802유지. 사용자의11/1촬영은본단위읽지않음;새입력사용시추론전용. '
    '이전GPU지연/7경고는word100_warning_latency_20261101,입력거절원인은word100_live_framing_diagnosis_20261101에보존. '\
    '이번수정은실제센서와가상신체하단분리이며,개인맞춤계수/단어예외없음.\n',encoding='utf8')
with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:
    f.write('\n\n'+now+' 앉은실사용웹캠과공개참고의입력범위구분\n'+summary+'\n원인:공개전체영상의범위준비조건을웹촬영필수조건으로사용하여초기손내림이안보이면거절. '
            '방향:사용자센서전체/특징별관측보류,공개참고기준보존. 개인촬영으로튜닝하지않고별도공개크롭4편으로검증.\n근거:'+str(U/'RESULT_KO.txt')+'\n')
assert read(O/'STATE.json')[U.name]['demo_url']==complete['demo_url']
print(complete['status'])
