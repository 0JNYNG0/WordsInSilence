"""Publish the checked result and synchronize the project's current handoff."""
from pathlib import Path
from datetime import datetime, timezone
import json, sys
sys.dont_write_bytecode=True
import run
U=run.U
ROOT=U.parents[3]

def main():
    run.checkpins()
    e=run.read(U/'EVALUATION.json')
    v=run.read(U/'review/RESULT_VERIFICATION.json')
    sv=run.read(U/'review/SAFETY_RESULT_VERIFICATION.json')
    safety=run.read(U/'safety_audit/RESULT.json')
    assert v['status']=='PASS' and sv['status']=='PASS'
    assert not (U/'RUNNING.json').exists()
    assert e['total']['possible']==1926120 and e['availability']['possible']==437640
    assert safety['hypothesis_outcomes'][0]['different_known_stage_counterexample_exists']
    status='COMPLETE_CONTEXT_VERIFIED_ALIGNMENT_NOT_SOLVED'
    next_action=('기존 경로의 공통 관측과 상위 동작 범위를 유지하며, 손목·양손·팔의 상대 이동 궤적을 시간순으로 비교하는 공통 후보 하나를 검증한다. '
                 '정지 길이를 바꿔도 같은 경로는 유지하고, 짧은 경로 차이가 전체 평균에 묻히지 않도록 단조 min-max 비교를 연구 후보로 삼는다. '
                 '관측 공백은 궤적으로 이어 붙이지 않는다. '
                 '짧은 정지 유무를 필수 동작 오류로 만들지 않는다. 정상 속도/정지 길이 차이와 가시성이 유지된 단계 누락·순서 변형을 분리해 '
                 '조건을 고정한 뒤 비교한다. 기존 8개 물리 하위 단계 차이를 교육 오답으로 학습시키지 않는다. '
                 '이번 완료 결과 재실행·새 모델 순회·사용자 자료 튜닝은 하지 않는다.')
    report='''공통 비교 근거 보완과 후속 검증 결과 — 2026-11-01

현재 문제는 무엇인가
직전 새 DTW 정렬 후보는 동작 시점 대응 평가가 나빠져 적용하지 않았다. 피드백에 그대로 쓰면 서로 다른 순간을 비교할 위험이 있어 중요한 문제다.
그러나 기존 C·T3-F나 운영 웹을 바꾼 것은 아니다. 프로젝트 전체가 망가진 상황은 아니다.
이전 82.16%→39.71%는 작은 개발 묶음의 물리적 동작 단계 대응 지표다. 단어 인식률이나 사용자의 수어 정답률이 아니다.

이번에 실제로 한 일
1. 기존 global 정렬과 각 참고의 관측값을 고정했다. 새 정렬 후보의 나쁜 결과를 기준에 섞지 않았다.
2. 비교가 불완전한 각 특징에 대해 가장 가까운 앞뒤 관측 시점을 연결했다. 중간 손 모양을 추측해서 채우지는 않았다.
3. 참고 영상마다 어느 시점·특징을 비교할 수 있는지 따로 보관했다. 한 참고에서 부족해도 다른 참고에 근거가 있는지 찾을 수 있다.
4. 저장 기록 전체를 다른 검산 코드로 확인했다. 이어서 관측이 충분하다는 표시만으로 같은 동작 단계라고 믿어도 되는지까지 대조했다.

검증 범위와 결과
공개 4개 표현의 16개 영상, 기존 36개 비교 쌍을 사용했다. 저장된 세 seed 출력을 각각 연결해 108개 기록을 만들었다.
이는 새 영상 108개나 새 학습 3회가 아니다. 같은 영상·경로의 저장 출력 재사용이다.
기존 양쪽 시간축의 1,926,120개 특징-시점 비교 기회를 모두 보존했다. 원래 수치와 미관측 상태, 부분 지원을 없애지 않았다.
직접 완전 지원 732,837셀, 앞뒤 문맥을 따로 표시하는 1,193,283셀을 누락 없이 구분했다.
이 숫자는 여러 참고와 세 seed를 반복 포함하므로 사람 수나 독립적인 정답 수가 아니다.
고유 비교 영상/seed 48묶음의 별도 목록에서는 437,640셀 중 172,224셀에 적어도 한 참고의 완전한 비교값이 있었다.
813셀은 부분 지원만 있었고, 264,603셀은 공통 비교값이 없었다. 본인 정보가 없는 경우와 참고 정보가 없는 경우는 원인별로 보존했다.
20,322셀은 한 참고가 미지원이어도 다른 참고에는 비교값이 있었다. 새 좌표를 복원했거나 정답률이 오른 숫자가 아니다.
기능 자체 검사 65항목, 별도 합성 반례 14종, 공개 저장 결과 전수 검산을 통과했다.

쉬운 예
순종 P033의 약 1.43초 오른쪽 엄지 특징은 참고 P034에서는 대응 관측값이 없지만 P035에는 있었다.
이제 어느 참고를 통해 실제 비교값을 볼 수 있는지 연결되어 있다. P034의 누락을 없애거나 P035와 같았다고 판정하지 않는다.
약 1.30~1.37초에는 비교 불가와 부분 지원이 섞여 있었다. 앞뒤 관측이 있다는 사실과 중간 모양을 확인했다는 것은 구분한다.

추가 대조에서 확인한 핵심 한계
기존 개발 단계 기록에는 참고 기준으로 가중 합계 360프레임 기회 중 8프레임 상당이 서로 다른 물리적 하위 단계에 연결되어 있었다.
이번 추가 감사에서 이 8프레임 상당은 모두 70개 특징 차원에 양쪽 비교값이 있는 상태였다.
즉, 값이 충분히 있다는 것만으로 같은 순간끼리 잘 맞췄다고 보장할 수 없다. 560(8×70)은 특징 셀 가중량이며 560건의 오류가 아니다.
잘하다의 한 예는 손이 왼팔을 따라 이동하는 순간(P048, 3.066667초)과 손을 왼팔 위에 유지하는 순간(P045, 2.066667초)의 연결이다.
조상의 한 예는 왼손이 올라가는 순간(P066, 3.333333초)과 높은 위치에 유지하는 순간(P065, 2.333333초)의 연결이다.
두 자료 모두 정상 참고로 사용하던 영상이다. 이 차이를 사용자가 틀린 수어를 했다는 뜻으로 바꾸면 안 된다.
짧은 멈춤 유무나 개인의 속도 차이일 수 있어, 이 8개를 반드시 고쳐야 할 교육 오답으로 삼지 않는다.
또 앞뒤 관측으로 둘러싸인 구간에서는 이번 자료의 해당 반례가 없었지만, 이것만으로 안전하거나 중간 모양이 맞다고 증명한 것은 아니다.
추가 감사는 기존에 저장된 단계 표시와 대응 기록을 연결했다. 새 정렬이나 기존 점수 재계산을 한 것이 아니다.

완료된 부분과 아직 남은 부분
완료: 원 비교값·미관측·전체 범위를 보존하는 공통 근거 연결, 참고별 지원 목록, 앞뒤 문맥 기록, 독립 코드 검산.
미완료: 중요한 동작끼리 더 잘 정렬하는 성능 개선, 허용 가능한 차이와 잘못된 수어의 구분, 실제 사용자에게 줄 교정의 교육적 정확도.
이번 결과만으로 인식률·피드백 정확도가 개선됐다고 보고하지 않는다. 관측 표시를 자동 교정 승인으로 쓰지 않는다.
단어 이름을 조건으로 둔 예외는 없다. 다만 공개 4개 표현에서 확인한 결과를 모든 수어에서 입증했다고 확대하지 않는다.
자연스러운 손내림·교차에 대한 교육 정확도는 별도 검증하지 않았다. 합성 사례의 좌우·관측값 보존과 공개 입력의 불변성을 확인한 범위다.

다음 한 핵심 작업
'''+next_action+'''
구체적인 다음 검증은 중요한 손 모양·팔과의 관계·이동 방향·앞뒤 순서를 함께 보고,
단순한 정지 길이와 속도 차이를 오류로 강요하지 않는지를 목표로 한다. 입력이 안 보인 부분은 계속 별도로 남긴다.
실제 수어의 의미가 틀렸다는 주장은 기술적으로 바꾼 영상의 차이 탐지와도 구분한다.

보존 및 실행
새 학습·모델 추론·정렬 검색·운영 교체 없음. C·T3-F와 운영 웹/DTW는 보호 파일 해시로 확인했다.
사용자·멘토 촬영은 접근하거나 학습·튜닝·선택에 사용하지 않았다. 원본·과거 동결 결과도 보존했다.
다운로드·카메라·외부 전송·Git 작업 없음. 명확한 성능 개선을 입증한 결과가 아니므로 HTML은 만들지 않았다.
구현: Codex semantic_feedback, 실행·통합: root, 방법/안전성 감사: semantic_model, 독립 산술 검산: semantic_data_audit.
이는 같은 Codex의 별도 코드 검산이며 독립된 한국수어 전문가의 정답 검토가 아니다.

핵심 근거 파일(이 결과 폴더)
PLAN_KO.txt / FROZEN.json / SELFTEST.json / RESULT_MANIFEST.json / EVALUATION.json
review/SYNTHETIC_SUPPLEMENT.json / review/RESULT_VERIFICATION.json / review/POSTEXECUTION_REVIEW.json
safety_audit/PROSPECTIVE_CONTRACT.json / safety_audit/RESULT.json / review/SAFETY_RESULT_VERIFICATION.json
NEXT_PLAN_KO.txt
'''
    (U/'RESULT_KO.txt').write_text(report,encoding='utf8')
    run.update(U/'CURRENT_EXECUTION.json',dict(status=status,active=None,exec19107_exit_code=0,evaluation_exit_code=0,safety_audit='complete_verified'))
    evidence_files=['PLAN_KO.txt','NEXT_PLAN_KO.txt','FROZEN.json','SELFTEST.json','RESULT_MANIFEST.json','EVALUATION.json','RESULT_KO.txt',
                    'review/RESULT_VERIFICATION.json','review/POSTEXECUTION_REVIEW.json','review/SYNTHETIC_SUPPLEMENT.json',
                    'safety_audit/PROSPECTIVE_CONTRACT.json','safety_audit/RESULT.json','review/SAFETY_RESULT_VERIFICATION.json']
    completion=dict(status=status,at_utc=datetime.now(timezone.utc).isoformat(),active=None,
        context_records=108,unique_query_seed_groups=48,public_words=4,public_clips=16,
        context_verification=v['status'],safety_verification=sv['status'],alignment_improvement=False,
        educational_accuracy=None,production_changed=False,private_or_mentor_access=False,
        new_alignment=0,new_inference=0,new_training=0,html_created=False,
        protected_files=len(run.read(U/'FROZEN.json')['protected']),next_single_action=next_action,
        evidence_pins={str(U/f):run.sha(U/f) for f in evidence_files})
    run.new(U/'COMPLETION.json',completion)
    statefile=ROOT/'research/front_keyphase_v1/ops/STATE.json'
    state=run.read(statefile)
    state['current_stage']='ALIGNMENT_CONTEXT_VERIFIED_NO_ALIGNMENT_OR_EDUCATIONAL_IMPROVEMENT_CLAIM'
    state['next_single_action']=next_action
    state[U.name]=dict(completion,unit=str(U),report=str(U/'RESULT_KO.txt'))
    run.update(statefile,state)
    contextfile=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
    context=contextfile.read_text(encoding='utf-8-sig')
    entry8=('\n### 2026-11-01 앞뒤 관측·다중참고 근거 보완 완료 — 정렬·교육 성능 개선은 미입증\n\n'
        '`word100_alignment_context_review_20261101` 완료. 기존global36경로×저장3seed=108기록/48query묶음에 '
        '앞뒤DIRECT_COMPLETE anchor/FIT참고occurrence/개별참고지원목록을추가. 공개4표현16영상이며새독립평가아님. '
        '양축1,926,120셀원값·미관측·분모보존,직접완전732,837/문맥1,193,283. 고유query437,640중direct172,224/partial813/none264,603; '
        '다른참고미지원동반direct20,322는관측복원/정답향상아님. 자체65+합성14+전수독립검산PASS. '
        '추가saved-stage감사/별도검산PASS: DEV360프레임기회중physical다른종류8은70차원모두DIRECT. '
        '이동/상승↔유지인정상참고간물리하위단계차이로교육오답아님. 관측충분≠같은단계/정상. '
        'BOUNDED반례0도안전성입증아님. 정렬·인식·교육정오향상주장X. '
        'C/T3F·운영·사용자/멘토·원본보존,새정렬/추론/학습0,HTML없음. RESULT/COMPLETION/검산참조. 실행종료.\n')
    entry9=('\n### 2026-11-01 다음 단일 작업 — 핵심 움직임·순서와 선택적 멈춤 분리\n\n'+next_action+' '
        '현재context/안전성감사는완료결과이므로재실행X. 원값/시점/분모보존은완료했지만동작판정정확도향상아님. '
        '동일관측상태와물리same-kind,언어정오를혼동하지않는다. '
        '다음후보는결과전조건·normal-speed/hold길이변형/가시단계누락·순서변형·전체공개범위를고정하고기존대응보존과비교한다. '
        '이번4표현dev를새독립평가라고부르지않고개발/추가사람·표현검증을구분. '
        '목표는신체조건차이를허용하면서중요동작을비교하는것이며모든미세정지일치를강요하는것이아니다.\n')
    for heading,entry in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry8),('## 9. 다음 단일 작업',entry9)]:
        assert context.count(heading)==1
        context=context.replace(heading,heading+'\n'+entry,1)
    contextfile.write_text(context,encoding='utf8')
    handoff=f'''# WordsInSilence 최신 인수인계

현재 `{U.name}` 완료. 실행중 없음. 사용자는 심각성 설명과 검토된 단계별 보완·자율 진행을 승인했다.
기존global경로와원관측고정,108기록/48다중참고묶음의앞뒤관측·참고occurrence·개별참고지원목록연결완료.
공개4표현16영상/36쌍×저장3seed이며새학습/추론/정렬0. 양축1,926,120셀원값/미관측/분모불변;
direct732,837/context1,193,283. 고유query437,640=direct172,224+partial813+none264,603. 다른참고미지원동반direct20,322는좌표복원이나정답향상아님.
자체65/별도합성14/전수코드검산PASS. `review/RESULT_VERIFICATION.json`과`review/SAFETY_RESULT_VERIFICATION.json`.
별도안전성감사는기존saved stage trace만join. DEV360기회중다른physicalkind8은70차원모두DIRECT였다.
잘하다이동↔유지,조상상승↔유지사례이며정상참고간물리구분이다. 사용자잘못/필수교육오류로학습금지. BOUNDED반례0도안전성입증X.
DIRECT/BOUNDED는관측근거/전후경계이며같은단계·정상증명X. query주석은감사전용,runtime입력X.

다음 한 작업: {next_action}
다음후보아직구현/실행하지않음. 현재완료결과재실행X. 새정렬후보39.71%FAIL은미채택그대로,기존82.16%는저장물리단계지표재사용이며TCN정확도아님.
이번완료는근거연결기능과한계검증이며정렬·교육성능향상X. HTML없음. `RESULT_KO.txt`/`COMPLETION.json`/`FROZEN.json`/`safety_audit/`참조.
구현semantic_feedback/실행root/방법·안전감사semantic_model/별도검산semantic_data_audit. 같은Codex검산≠수어전문가정답.
C·T3F/운영웹·DTW/원본·과거동결보존. 사용자·멘토촬영접근·학습·튜닝·선택없음. 카메라/다운로드/외부전송/Git/운영교체없음.
공통코드이며특정단어예외X,모든수어성능을미리보장X. 목표변천사는 `Docs/PROJECT_DIRECTION_HISTORY_KO.txt`.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
    history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
    with history.open('a',encoding='utf8') as stream:
        stream.write('\n\n2026-11-01 관측 근거와 동작 의미의 구분을 명확히 함\n'
            '기존global경로를유지한108저장비교에앞뒤관측·참고단계문맥·개별다중참고지원목록을추가하고원수치/미관측/비교범위보존을별도검산했다. '
            '추가saved-stage감사에서이동/상승과유지가연결된8프레임상당은70특징모두관측지원이있었다. '
            '따라서관측값이있다는것만으로같은단계라고믿을수없다. 동시에이들은정상참고끼리의물리하위단계차이여서곧바로잘못된수어나교정대상으로취급하면안된다. '
            '원래목표는정확한좌표나모든멈춤의복제가아니라중요한손모양·관계·이동·순서를이해하고허용변이와구분하는것이다. '
            '다음방향은선택적짧은멈춤을강요하지않으면서핵심이동방향/도달/전후순서를검증하는공통후보하나. '
            '이번에는관측근거연결기능완료,정렬/교육정확도개선미입증,추가학습0/운영미적용. '
            '근거 '+str(U/'RESULT_KO.txt')+'\n')
    run.checkpins()
    documents=[statefile,contextfile,ROOT/'NEW_CHAT_HANDOFF.md',history,U/'RESULT_KO.txt',U/'COMPLETION.json',U/'CURRENT_EXECUTION.json']
    assert run.read(statefile)[U.name]['status']==status
    assert run.read(U/'CURRENT_EXECUTION.json')['active'] is None
    assert U.name in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
    assert '완료 — 정렬·교육 성능 개선은 미입증' in contextfile.read_text(encoding='utf8')
    run.new(U/'FINAL_DOCUMENT_VERIFICATION.json',dict(status='PASS',status_synchronized=status,
        active_execution=None,protected_hashes_verified=True,documents={str(p):run.sha(p) for p in documents}))
    print(status)

if __name__=='__main__':main()
