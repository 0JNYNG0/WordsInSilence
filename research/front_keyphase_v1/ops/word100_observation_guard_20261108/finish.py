from prepare import *
import shutil,html
from datetime import datetime,timezone,timedelta
def main():
    assert not (U/'RESULT.json').exists()
    v=read(U/'VERIFICATION.json');e=read(U/'ENGINEERING_CHECKS.json');i=read(U/'INTEGRATION_CHECKS.json');visible=read(U/'VISIBLE_CROP_CHECK.json')
    assert all(x['status']=='PASS' for x in [v,e,i,visible])
    full=read(U/'EVAL_full.json');crop=read(U/'EVAL_crop.json');plan=read(U/'INPUT_PLAN.json');D=O/'word100_action_feedback_demo_20261108';server=read(D/'SERVER.json');R=O.parents[2]
    words={r['word_id']:r['label'] for r in plan['rows']};byword=[]
    for w in dict.fromkeys(r['word'] for r in crop['rows']):
        rr=[r for r in crop['rows'] if r['word']==w];byword.append(dict(word=w,label=words[w],n=len(rr),before=sum(r['baseline']['correct'] for r in rr),after=sum(r['guarded']['correct'] for r in rr)))
    deployment=dict(server=server,camera_opened_by_agent=False,consumer_before=sha(U/'before/consumer.py'),consumer_after=sha(D/'consumer.py'),guard_pins=read(U/'GUARD_FROZEN.json')['pins'],model_weights_changed=False,public_verified=True,actual_user_retest_pending=True)
    write(U/'DEPLOYMENT.json',deployment)
    result=dict(status='INPUT_GUARD_VALIDATED_AND_DEPLOYED_USER_RETEST_PENDING',at=datetime.now(timezone(timedelta(hours=9))).isoformat(),server=server,
        changes=['Exclude weak RTMW hand estimates only with inconsistent hand geometry or repeated boundary/downward-arm evidence','Preserve raw arrays and all timestamps; no coordinate interpolation','Use same filtered observations for TCN, feedback and result skeleton','Uncertain hand evidence cannot remain a similar-status card when overlapping its comparison span'],
        scope=dict(calibration_clips=14,calibration_recorded_ids=[1,2],cropped_evaluation_clips=22,cropped_evaluation_recorded_ids=[5,6,11,12,13],cropped_frames=sum(r['frames'] for r in crop['rows']),words=8,full_dev_clips=300,full_dev_words=100,private_used=False,training=False,download=False),
        full=full['summary'],crop=crop['summary'],per_word=byword,engineering_checks=e['n'],integration_checks=len(i['checks']),visible_crop_centers=len(visible['rows']),saved_output_verification=v['n'],protected_hashes=v['protected_count'],
        outside_hand_proxy=dict(before=v['proxy_before'],after=v['proxy_after'],unit='hand-frame, not image/frame count or human truth'),
        limitations=['Existing development material and artificial upper-body framing; not an independent real-webcam accuracy test','Candidate only addresses weak geometrically inconsistent hand estimates; coherent high-scoring swaps/finger inaccuracies remain possible','32 positive centers and3 crossing windows preserved, not every normal sign or camera condition','Raw activation ratios are not calibrated visibility probabilities','Feedback automatic educational correctness and omitted-stage diagnosis remain unvalidated','Guard duration is coordinate-only CPU overhead, not full camera-to-screen latency'],
        preserved='C/T3F,8812,TCN/feedback weights,original recordings,raw observation arrays,frame IDs/timestamps',
        implementation='Codex local',verification='Same Codex separate stored-output arithmetic/hash code plus public RGB review; no independent expert annotation',report=str(U/'RESULT_KO.txt'),visual=str(U/'BEFORE_AFTER.jpg'),next='User refreshes8814 and retests. No automatic camera opening. If a new failure is reported, reproduce with public data; no user training/tuning/candidate selection.')
    write(U/'RESULT.json',result)
    lines=['화면 밖 반대 손의 잘못된 관절 입력 보완 — 적용 완료','',
       '현재 테스트 화면: http://127.0.0.1:8814/ (새로고침 후 재촬영)',
       '단어 이름을 이용하지 않는 공통 처리이며, 모델을 새로 학습하거나 가중치를 바꾸지 않았다. 기존 C·T3-F와8812 보존. 사용자 촬영본 학습·튜닝 없음.','',
       '무엇을 고쳤나',
       '이전: 화면 밖 손을 RTMW가 화면 안에 잘못 그려도 그 좌표가 TCN과 피드백에 들어갔다.',
       '현재: 팔에 비해 손가락이 지나치게 길게 뻗는 등 기하학적 모순이 있고 관측 근거도 약한 손, 또는 손을 내린 팔과 화면 경계에서 약한 추정이 연속되는 경우를 구분한다.',
       '점수가 낮다는 이유만으로 손을 지우지 않는다. 양손이 가깝다는 이유만으로 지우지 않는다. 단어/왼손/오른손별 예외도 없다.',
       '불확실한 손은 분석 입력과 뼈대 표시에서 함께 제외한다. 원본과 시간 순서는 그대로 두며, 없는 손 모양을 만들어 메우지 않는다. 해당 비교 구간에 불확실한 손이 포함되면 비슷하다고 확정하지 않는다.','',
       '공개 검증 결과 — TCN 단어1위, 피드백 정오 정확도가 아님',
       '규칙 작성: 공개8단어14영상, 기록ID01·02. 분류 점수를 보고 기준을 고르지 않았다.',
       '고정 후 검사: 다른 기록ID05·06·11·12·13의22영상,8단어,총2586프레임. 실제 RGB를 상반신 웹캠처럼 잘라 RTMW로 관측했다.',
       '상반신22영상: 정답1위14/22 → 22/22. 원래 맞던 영상이 틀리게 바뀐 경우0.',
       '기존 전체 화면 개발300영상(100단어×3기록ID):293/300 → 293/300. 정상 입력에 변경이 없어 저장된 기존 추론을 재사용했다.']
    lines += [f"  {r['label']}: {r['before']}/{r['n']} → {r['after']}/{r['n']}" for r in byword]
    lines += ['', '보존 확인',
       '기존 RGB 주석의 정상 손모양32지점을 새 상반신 화면에서도 유지했다. 정상 교차3개 시간창의 양손 관절도 유지했다.',
       '순종의 펼침→접힘 등 확인 지점의 값은 보존했다. 전체 동작의 모든 짧은 변화가 검증된 것은 아니다.',
       '공학검사182개,연결검사9개,저장결과 별도검산765개와 보호해시72개 통과. 같은Codex 검산이며 독립 한국수어 전문가 검증은 아니다.',
       '', '속도',
       f"새 좌표 점검 계산 중앙값 {crop['summary']['guard_ms_median']:.2f}ms (약0.002초/영상). 추가 RGB 분석·재검출 없음. 실제 웹캠에서 종료부터 화면 표시까지의 시간은 이번에 다시 측정하지 않았다.",
       '', '남은 문제',
       '실제 노트북 웹캠 재촬영으로 개선을 확인해야 한다. 공개영상 상반신 잘라내기는 실제 방의 조명·옷·카메라 거리까지 재현하지 않는다.',
       '그럴듯하게 잘못 붙은 좌표나 손가락 한두 개의 오차까지 모두 해결한 것은 아니다. RTMW의 낮은 원시 점수를 가시성 확률이라고 해석하지 않는다.',
       '전체 수어가 올바른지, 빠진 동작이 있는지를 확정하는 교육용 피드백의 기존 한계는 남아 있다.',
       '화면 밖이었던 손의 가짜 화면 안 검출은 기존좌표 기준 기하 proxy로1143→13손-프레임이었으나, 독립 사람이 붙인 가시성 정답은 아니다.',
       '', '실행 기록',
       '공개 남아 REAL02 원본은 첫PTS가0.066016초였다. 초기 시간동일성 검사에서 이를 잡았고, 기존 추출기와 같이 첫시각을 뺀 상대시간으로 수정했다. 동일136프레임·동일원본SHA256 확인, 완료된13개 추출은 재실행하지 않았다. 최초 오류는FIRST_EXTRACTION_FAILURE.json에 보존.',
       '현재8814 소비자(consumer.py)에 적용,서버 재시작 및 ready/error null/카메라 비활성 확인. 자동종료·녹화·모델 가중치는 변경하지 않았다.',
       '근거: GUARD_FROZEN.json/EVAL_full.json/EVAL_crop.json/ENGINEERING_CHECKS.json/INTEGRATION_CHECKS.json/VISIBLE_CROP_CHECK.json/VERIFICATION.json/DEPLOYMENT.json.']
    text='\n'.join(lines)+'\n';(U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
    table=''.join(f"<tr><td>{r['label']}</td><td>{r['before']}/{r['n']}</td><td>{r['after']}/{r['n']}</td></tr>" for r in byword)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>보이지 않는 손의 관절 보완</title><style>body{font:17px/1.65 system-ui,sans-serif;background:#f4f6fa;color:#1d293b;margin:0}main{max-width:1160px;margin:24px auto;padding:24px;background:white;border-radius:18px}h1{font-size:30px}img{width:100%;height:auto}table{border-collapse:collapse;min-width:420px}td,th{padding:8px 28px;text-align:left;border-bottom:1px solid #d8deea}.metrics{display:flex;gap:24px;flex-wrap:wrap}.metric{background:#e9f5f1;padding:16px;border-radius:12px}.metric b{font-size:28px}a{color:#1858be}.note{background:#fff4d8;padding:16px;border-radius:10px}</style><main><h1>없는 손이 있는 것처럼 들어가던 문제를 보완했어요</h1><p>보이는 손은 유지하고, 근거가 약하고 연결이 모순된 추정은 별도로 구분합니다. 손 모양을 만들어 채우지는 않습니다.</p><div class="metrics"><div class="metric">상반신 공개22영상 · TCN정답1위<br><b>14/22 → 22/22</b></div><div class="metric">기존 전체화면 공개300영상<br><b>293/300 유지</b></div></div><p><a href="http://127.0.0.1:8814/">수정된 테스트 화면 열기</a> · 새로고침 후 직접 재촬영하세요.</p><img src="BEFORE_AFTER.jpg" alt="실제 공개 영상,이전 관절,보완 후 관절 비교. 순종의 가짜 왼손은 사라지고 정상 교차는 보존"><p>위2행: 순종의 잘못된 왼손 추정 제외. 아래행: 수어의 정상 양손 움직임 유지. 보라색=왼손,초록색=오른손.</p><table><tr><th>단어</th><th>이전1위</th><th>보완 후1위</th></tr>'''+table+'''</table><p>공개8단어22편을 상반신 촬영처럼 잘라 검사한 결과입니다. 실제 웹캠 정확도나 교육적 피드백 정오 정확도가 아닙니다. 기준을 만든 기록ID01·02의14영상은 이22편에 포함하지 않았습니다.</p><p>정상 손모양32확인지점과 교차3시간창 보존. 좌표점검 약0.002초 추가(전체 처리시간 아님). 기존 모델 가중치 변경·사용자 촬영 학습 없음.</p><p class="note">남은 확인: 실제 웹캠 재촬영,그럴듯한 좌우 오검출,세밀한 손가락 오차. 자동 수어 정오·누락 동작 판정의 기존 한계는 남아 있습니다.</p><a href="RESULT_KO.txt">자세한 검증 기록</a></main></html>'''
    (U/'RESULT_KO.html').write_text(page,encoding='utf-8')
    for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
        assert not (U/'before'/p.name).exists();shutil.copy2(p,U/'before'/p.name)
    state_text='''### 2026-11-08 화면 밖 손의 관절 입력 보완 — 공개 검증 통과·8814 적용

word100_observation_guard_20261108. 기록ID01/02의 공개8단어14영상에서 기하+상대관측점수+팔내림/경계/시간지지 공통 규칙을 정하고 동결했다. 단어/분류 점수 미사용. 다른ID05/06/11/12/13의 상반신 RGB22편2586프레임에서 TCN1위14/22→22/22,회귀0. 순종1/4→4/4. 기존 전체화면DEV300은 관절 변경0으로 저장된293/300 그대로. 기존 공개자료의 새촬영범위 조건이며 독립 실제사용자 정확도가 아님. 정상 손모양32 crop확인지점/교차3창 보존,공학182+연결9+별도검산765/보호72해시PASS. 같은Codex 검산,전문가정답 아님.

보완은 약하고 기하적으로 모순되거나 경계에서 손내림과 함께 반복되는 추정만 제외한다. 양손 근접 자체나 낮은 점수 단독 제외 없음. 원본/전체시각 보존,보간/평활화0. TCN·피드백·표시가 같은 입력을 쓰고 카드 구간의 보류 관절을 유사 근거로 올리지 않는다. CPU좌표점검 중앙2.06ms(전체지연아님). 새학습/사용자학습튜닝/다운로드0. 기존 C·T3-F/8812/현재모델 가중치 보존. 8814 consumer.py 변경·서버재시작,ready/error null/카메라비활성 확인. 실제웹캠 재촬영은사용자확인남음. PTS초기오류/수정기록보존. 근거RESULT_KO.txt/html,RESULT.json,DEPLOYMENT.json.

'''
    next_text='''### 2026-11-08 관절 보완 적용 후 실제 웹캠 확인

8814를 새로고침해 사용자 본인이 순종/양손 단어를 재촬영한다. 에이전트 카메라 자동개방·사용자 자료 학습/튜닝/후보선택 금지. 새 오류가 있으면 공개 자료로 재현하여 입력 문제와 모델 문제를 나눈다. 이번22crop/300full 고정비교·추출 재실행 금지. 그럴듯한 좌우 오검출/세밀한 손가락 오차와 자동 교육적 정오/동작누락 판정은 미완이며 과장하지 않는다. 입력이 해결됐다는 이유로 자동 전체정답·교정을 활성화하지 않는다. 필요성이 확인되기 전 재학습/새모델 추가 없음.

'''
    p=R/'WordsInSilence_PROJECT_CONTEXT.md';txt=p.read_text(encoding='utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n';assert txt.count(h8)==txt.count(h9)==1;p.write_text(txt.replace(h8,h8+'\n'+state_text,1).replace(h9,h9+'\n'+next_text,1),encoding='utf-8')
    state=read(O/'STATE.json');state['current_stage']='RTMW_OBSERVATION_GUARD_DEPLOYED_USER_RETEST_PENDING';state['updated_date_kst']='2026-11-08';state['next_single_action']='8814 사용자 재촬영 확인. 오류가 남으면 공개 자료로 재현. 완료된22crop/300full 평가 재실행/평가점수 튜닝/사용자 학습 금지. 자동 교육적 정오/누락 판정 미완.'
    state={U.name:dict(status=result['status'],report=str(U/'RESULT_KO.txt'),visual=str(U/'RESULT_KO.html'),server=server,crop_before=14,crop_after=22,crop_n=22,full_before=293,full_after=293,full_n=300,training=False,model_changed=False,private_used=False,verification=str(U/'VERIFICATION.json')),**state}
    state['word100_action_feedback_demo_20261108']['server']=server;state['word100_action_feedback_demo_20261108']['observation_guard_followup']=str(U/'RESULT.json');state['word100_webcam_recording_fix_20261108']['server']=server;write(O/'STATE.json',state)
    (R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+state_text+next_text+'서버: '+json.dumps(server,ensure_ascii=False)+'\n이번 배포 consumer.py만수정,모델/녹화/자동종료/기존UI는보존. 실제녹화·카메라는사용자버튼으로만. 결과화면은확인한일부특징비교이며전체수어정답이아님. 현재모델은RTMW TCN finetune_C_20261013과관절전용피드백,런타임SignRep RGB없음.\n',encoding='utf-8')
    write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_DEPLOYED',training_running=False,server=server))
    checks={str(p):dict(contains_unit=U.name in p.read_text(encoding='utf-8'),sha256=sha(p)) for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']};assert all(x['contains_unit'] for x in checks.values());write(U/'DOCUMENT_SYNC.json',dict(status='PASS',checks=checks))
    print(json.dumps(dict(status=result['status'],server=server,per_word=byword),ensure_ascii=False))
if __name__=='__main__':main()
