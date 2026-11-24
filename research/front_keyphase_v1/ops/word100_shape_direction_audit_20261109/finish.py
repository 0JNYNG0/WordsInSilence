from audit import U,O,D,J,G,read,write,sha
import numpy as np,shutil,json

def main():
    assert not (U/'RESULT.json').exists()
    a=read(U/'AUDIT.json');f=read(U/'FROZEN.json');checks=[]
    def ck(name,x):checks.append(dict(name=name,passed=bool(x)))
    for p,h in f['pins'].items():ck('unchanged '+p,sha(p)==h)
    for row in a['rows']:
        ck(row['clip_id']+' '+row['kind']+' hash',sha(row['prediction'])==row['sha256'])
        z=np.load(row['prediction']);ck('complete timeline '+row['clip_id']+' '+row['kind'],len(z['phase'])==len(z['timestamp_us'])==len(z['frame_id']))
        for c in row['cards']:
            lo,hi=c['span'] or row['main_indices'];idx=slice(lo,hi+1)
            ck(row['clip_id']+' '+row['kind']+' '+c['id']+' numeric summary',abs(float(z['attributes'][idx,1].mean())-c['open_right_mean'])<1e-6 and abs(float(z['different_if1'][idx,1].max())-c['difference_if_right_open_max'])<1e-6)
    for kind,counts in a['supported_counts'].items():
        for cid,num in counts.items():ck(kind+' '+cid+' count',sum(next(c for c in r['cards'] if c['id']==cid)['status']=='supported' for r in a['rows'] if r['kind']==kind)==num)
    source=(D/'consumer.py').read_text(encoding='utf-8');ck('assessment before DTW',source.index('e=assess(')<source.index('alignment=dtw('));ck('one guarded channel input',"A.convert(observed['xy_px']" in source and "out=self.channels(a['tcn280']" in source)
    pins={**read(G/'INPUT_PLAN.json')['protected'],**read(J/'FROZEN.json')['pins']}
    for p,h in pins.items():ck('preserved '+p,sha(p)==h)
    verification=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',n=len(checks),checks=checks,verifier='Same Codex separate saved-array/hash/code-order verification, not independent expert ground truth',rerun_inference=False)
    write(U/'VERIFICATION.json',verification);assert verification['status']=='PASS'
    result=dict(status='FEEDBACK_SHAPE_DIRECTION_GAPS_CONFIRMED_AUDIT_ONLY',date_kst='2026-11-09',
        user_report=dict(sunjong_top1_now=True,trial_count_unspecified=True,fist_substitution_not_warned=True,reverse_stroke_reach_wrist_similar=True,recording_not_opened=True),
        confirmed=['Observation guard is shared by TCN and feedback inputs; unchanged weights','Similarity decisions precede DTW; DTW cannot cause the audited decisions','Right-open and left-closed feature expectations exist for brush_arm, so handshape is not absent from the implementation','Positive expected-feature evidence is not required for supported; lack of threshold-crossing difference is treated as supported','reach_wrist explicitly checks only L_closed_four plus learned phase4; no direct forearm-relative signed progress/endpoint-contact check','Final-card hold marked optional does not justify accepting wrong direction','Difference head trained with flipped expectation labels at existing FIT observations, not a varied set of real wrong-handshape/wrong-direction performances'],
        diagnostic=a['supported_counts'],diagnostic_scope='3 public development clips,12 fixed feedback passes: normal,closed-hand coordinate substitution,reversed main,both. Artificial coordinate controls, not learner-error accuracy.',
        nuance='All3 modified cases per condition retained brush_arm support; altered reach_wrist cases became optional0/3 supported. This reproduces permissive stroke assessment but not the exact user-specific reach_wrist display.',
        similarity_percent_available=False,model_internal_scores_not_similarity_percent=True,
        deployment_changed=False,training=False,private_pixels_or_keypoints_used=False,
        next=['Require positive expected-handshape support and handle contradictory/uncertain features explicitly','Add body-relative movement direction and ordered start/end evidence, separate from optional holding duration','Validate joint shape/direction conditions versus normal speed/position/shorter-stroke differences on public data before training or deployment'],
        verification=str(U/'VERIFICATION.json'),report=str(U/'RESULT_KO.txt'))
    write(U/'RESULT.json',result)
    report='''관절 보완 이후 손모양·이동방향 피드백 점검

질문에 대한 답
관절 보완은 TCN과 세부 피드백 양쪽에 같은 입력으로 적용되어 있다. 모델 가중치가 새로 학습된 것은 아니다.
사용자가 순종1위를 확인했다고 보고했다. 횟수/정확도 통계로 확장하지 않으며, 사용자 영상이나 좌표는 열지 않았다.

확인한 문제
1. 손모양을 아예 확인하지 않는 것은 아니다. 잘하다의 첫 카드는 오른손 네 손가락 펼침과 왼손 네 손가락 접힘을 기대한다.
   하지만 현재 최종 판정은 동작 단계 후보가 있고 차이 판정부가 기준을 넘지 않으면 supported(화면:비슷)를 준다.
   기대 손모양이 실제로 확인됐다는 긍정 근거를 필수로 요구하지 않는다. 불확실함이나 서로 충돌하는 내부 판단이 비슷함으로 바뀔 수 있다.
2. 손목 쪽까지 이어가기 카드는 단계 후보와 왼손 접힘만 명시적으로 비교한다.
   오른손이 왼팔의 팔꿈치 쪽에서 손목 쪽으로 이동했는지 직접 측정/검증하는 규칙이나 독립 지도 정답이 연결돼 있지 않다.
   마지막 위치에서 멈추는 시간이 선택사항이라는 것과, 이동 방향이 선택사항이라는 것은 다르다.
3. 이 판정은 DTW보다 먼저 이루어진다. DTW는 그다음 재생할 영상 시점을 연결한다.
   따라서 현재 문제를 DTW가 손목 접촉 시점을 억지로 맞춰서 일으켰다고 설명할 근거는 없다. 2D접촉 자체도 확정하는 코드가 아니다.

공개 진단
잘하다 공개 개발영상 REAL11/12/13을 고정하고, 현재 보완된 관절 입력으로 검사했다. TCN 재평가는 하지 않았다.
정상,주요 구간의 오른손을 공개 영상에서 확인한 닫힌 손 좌표로 교체,주요 구간 역순,둘을 함께 적용한4조건 총12회.
합성 좌표 시험이며 사람이 자연스럽게 틀리게 수행한 오류 정답 영상은 아니다.
첫 카드(펴진 손으로 팔 따라 쓸기)는 정상3/3뿐 아니라 변경한 세 조건 각각3/3도 supported였다.
특히 닫힌 손만 적용한 REAL11/13에서는 내부 접힘 출력 평균이약0.96/0.86인데도,차이 출력이기준0.65에못미쳐 supported로넘겼다.
이 숫자는 손모양 신경망의 내부 출력이며 사용자 동작 유사도96%/86%가 아니다.
REAL12의 닫힘 출력은 더 애매하여 관측모델 자체의 한계도 남는다. 최종 규칙 하나만 고치면 모든 오류가 해결된다고 주장할 수 없다.
변경 조건의 두 번째 손목 카드들은 optional로 나왔다. 사용자의 두 번째 카드까지 동일하게 재현한 것은 아니다. 사용자 실제 영상에서 무슨 신호였는지는 확인하지 않았다.

퍼센트
현재 교육적으로 검증된 전체 동작 유사도/정답률 퍼센트는 없다. TCN후보 확률,단계 점수,손모양 점수,DTW비용은 각각 다른 값이며 그대로 얼마나 잘 따라했는지의 백분율이 될 수 없다.
직전22/22는 공개 상반신 시험의TCN단어1위 결과이지,손모양/방향 피드백 정확도가 아니다.

다음 우선순위
첫째,단계 후보를 찾은 것과 기대 손모양이 확인된 것을 분리한다. 기대 손모양의 긍정 근거가 없거나 반대 특징과 충돌하면 비슷하다고 넘기지 않는다.
둘째,반대 팔을 기준으로 어느 쪽에서 어느 쪽으로 이동했는지 앞·중간·뒤를 함께 확인한다. 같은 지점을 스쳐간 것과 같은 방향으로 진행한 것을 구분한다.
셋째,정상 속도/위치/짧은 이동거리 차이는 유지하면서 정상·손모양만 다름·방향만 다름·둘다 다름·관측부족을 공개 자료로 비교한다.
이때 단어마다 예외를 넣지 않고 설명에 연결된 공통 수형/신체관계/방향 기능으로 구성한다. 필요성이 확인된 특징/시간 정답을 보완한 뒤 학습한다.

이번 작업 범위
원인 검토와 공개 합성 진단만 완료. 코드/화면/판정 기준/모델 가중치 변경 없음,새 학습 없음,개인자료/카메라/다운로드/외부전송 없음.
기존C·T3-F,현재8814관절보완 유지. 같은Codex 별도 저장결과 산술/코드순서/해시 검산,독립 전문가 검증 아님.
근거: FROZEN.json/AUDIT.json/VERIFICATION.json 및 현재 consumer.py,evidence.py,CARDS.json,train.py.
'''
    (U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
    R=O.parents[2];before=U/'before';before.mkdir(exist_ok=True)
    for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,before/p.name)
    state_text='''### 2026-11-09 손모양·역방향 피드백 감사 — 최종 판정 공백 확인

word100_shape_direction_audit_20261109. 사용자 순종1위 성공 보고(횟수불명),잘하다 주먹/역방향에도 비슷 표시 보고. 코드확인: 관절guard는TCN/피드백공유. assess가DTW보다먼저라 DTW로인한동일판정아님. 첫카드는R_open_four/L_closed_four 기대하지만 긍정 근거 필수 없이 단계후보+차이임계미초과를supported로처리. 두번째reach_wrist는L_closed_four+phase4만명시확인,반대팔대비방향/접촉직접검증없음. optional은마지막유지시간면제이며방향면제근거아님.

공개잘하다ID11/12/13의정상/닫힌손좌표교체/주요구간역순/둘다12조건진단. 첫카드각조건3/3supported로변경에도허용;일부닫힘신호강하지만최종차이head기준미달확인. 두번째변경조건은optional로사용자두번째표시까지동일재현아님. 합성좌표진단이지자연오류정확도아님. 사용자영상/좌표열람0,학습/모델/웹변경0. 같은Codex별도저장결과검산. 전체유사도%없고내부확률을유사도%로해석금지. 직전22/22는TCN1위이며피드백정확도아님. 근거RESULT_KO.txt/RESULT.json/AUDIT.json/VERIFICATION.json.

'''
    next_text='''### 2026-11-09 피드백의 긍정 근거·움직임 방향 보완

다음은 단계후보/수형긍정근거/모순/불확실을분리해 차이를확신못한것을비슷하다고넘기는최종규칙을보완하고,반대팔대비방향과앞·중간·뒤순서를공통특징으로연결하는작업이다. 정상속도/위치/짧은거리차이와수형만오류/방향만오류/복합/관측부족을공개자료에서함께확인한다. 단어전용예외/개인튜닝/전체정답활성화금지. 이번3영상12조건진단은완료되어재실행금지. 현재8814관절guard/TCN유지,추가학습및판정배포는이번감사에서실행하지않음.

'''
    p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n';assert t.count(h8)==t.count(h9)==1;p.write_text(t.replace(h8,h8+'\n'+state_text,1).replace(h9,h9+'\n'+next_text,1),encoding='utf-8')
    state=read(O/'STATE.json');state['current_stage']='RTMW_GUARD_USER_TCN_SUCCESS_FEEDBACK_SHAPE_DIRECTION_AUDIT_COMPLETE';state['updated_date_kst']='2026-11-09';state['next_single_action']='분류 성공과 세부 교정 구분. 수형 긍정근거/모순/불확실 공통 판정,반대팔 대비 방향·순서 근거를 보완하고 공개 정상/오류/관측부족 대조. 이번 감사에서 새학습·웹변경 없음.'
    state['word100_observation_guard_20261108']['user_followup']='2026-11-09 user reports Sunjong top1 after retest, count unspecified; no measured user accuracy claim.'
    state={U.name:dict(status=result['status'],report=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFICATION.json'),private_used=False,training=False,web_changed=False,diagnostic_conditions=12),**state};write(O/'STATE.json',state)
    (R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+state_text+next_text+'8814 서버 마지막기록PID23472,기존관절guard유지. 이번서버재시작/카메라개방없음. actual최근프로세스상태필요시확인후재개. C·T3-F/8812보존,사용자·멘토학습튜닝선택금지. 사용자질문에같은답반복하지않고현재피드백허용판정의확인된공백부터해결.\n',encoding='utf-8')
    write(U/'DOCUMENT_SYNC.json',dict(status='PASS',files={str(p):sha(p) for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']}));print(json.dumps(dict(status=result['status'],verification_checks=len(checks)),ensure_ascii=False))
if __name__=='__main__':main()
