from prepare import *
from datetime import datetime,timezone,timedelta
import shutil
ROOT=U.parents[3];OPS=U.parent
def update(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    assert not (U/'COMPLETION.json').exists()
    from run import checkpins
    frozen=checkpins();e=read(U/'EVALUATION.json');v=read(U/'review/RESULT_VERIFICATION.json');mv=read(U/'review/MATH_RESULT_VERIFICATION.json')
    assert e['failure_count']==0 and v['status'].startswith('PASS') and mv['status'].startswith('PASS')
    assert not (U/'RUNNING.json').exists()
    now=datetime.now(timezone(timedelta(hours=9))).isoformat();ctrl={x['control']:x for x in e['controls']}
    example=next(x for x in e['rows'] if x['id']=='control_P033_speed_125_P033_s02__R_wrist')
    summary=dict(natural=e['natural'],speed125=ctrl['speed_125'],controls=ctrl,
        new_training=0,new_model_inference=0,production_changed=False,rotation_or_bend_validated=False,
        math_checks=mv.get('checks'),data_checks=v.get('checks'),educational_accuracy=None)
    report=f'''점 간격 때문에 생기던 차이를 줄이는 다음 작업 완료

아주 쉽게 설명
손이 지나간 자리에 점을 찍었다고 생각하면 된다. 느린 동작에는 점이 많고 빠른 동작에는 점이 적을 수 있다.
이전에는 점과 점을 비교했다. 이번에는 관측된 점을 차례로 이은 길을 비교하여, 같은 길의 점 개수·간격이 다르다는 이유만으로 차이가 커지는 영향을 줄였다.
중간에 손이 보이지 않으면 그 구간까지 선으로 메우지 않는다. 실제 카메라가 보지 못한 움직임을 복원한 것이 아니다.

사용자가 강조한 구분
주먹과 팔 자세를 유지한 채 몸 때문에 조금 흔들리는 것과, 주먹을 돌리거나 손목을 꺾는 별도 동작은 구분해야 한다.
위치·손모양·손방향·손목굽힘은 별개 정보다. 이번 후보는 위치 길만 비교한다. 기존 손가락 각도·손축·모양 출력과 미관측 표시를 그대로 보존했다.
손목의 길이 같다고 손동작 전체가 같거나 주먹이 그대로였다고 판정하지 않는다. 주먹회전·손목꺾임 자동판별이 완료됐다는 뜻도 아니다.

이전 결과 / 현재 결과
- 정확히 같은 직선에서 점을 더 찍거나 덜 찍기: 이산 점 비교에서 생길 수 있던 차이를 없애는 성질을 합성 기하에서 확인했다.
- 공개 좌표를 빠르게 바꾼 시험: 비교 가능한123성분 중 {ctrl['speed_125']['numerically_reduced']}성분의 수치가 감소했다. 모든123성분에 여전히 양수 차이는 남았다. 속도를 바꾸며 꺾임의 관측점도 빠질 수 있어 모든 값을0으로 강제하지 않았다.
- 예시 순종 P033 오른손목 첫구간: 이전 {example['old_distance']:.6f}, 현재 [{example['lower']:.6f}, {example['upper']:.6f}]. 같은 저장 위치 성분의 기하 값이며 인식률/교육정확도/%점수가 아니다.
- 같은 자세의 정지시간을 늘린 시험: 완전관측123성분 모두0 유지. 실제 변경된 부모 구간은39개이며 나머지84개는 다른 미변경 구간이다. 이전도0이었으므로 새 개선으로 세지 않는다.
- 짧은 구간 삭제·첫구간 순서반전: 각각 실제 변경된80기회 중 완전관측39기회에서 양수차이가 모두 남고41기회는 보류했다. 실제 사람이 틀린 수어를 한 영상의 오류탐지율이 아니다.
- 실제 다른 사람 영상끼리: 자연585성분 중293개 전체관측,292개 보류 그대로. 그293개 중 {e['natural']['numerically_reduced']}개만 수치가 줄었다. 더 정확한 시점 연결이나 정상/오류 구분이 확인됐다고 말하지 않는다.
- 기존 순서교환95기회의 변경 일부가 부모범위밖인 문제와, 미제공10기회도 그대로 남겼다. 범위를 작게 만들어 성공으로 세지 않았다.

검증 범위
공개4표현16영상36쌍의 기존 자료를 재사용했다. 이미 개발에 사용한 자료이며 새 독립 사람 평가가 아니다.
전체2825저장기록과 미제공10기회를 보존했다. 완전관측1417개는 수치 하한·상한을 인증했고 미관측1408개는null을 유지했다.
구현 자체검사217, 독립 합성 수학검산592검사/75곡선, 공개 수학검산 {mv.get('checks')}검사, 출처·범위·집계검산 {v.get('checks')}검사 모두PASS.
별도Codex 코드검산이지 독립 한국수어 전문가의 교육적 정오 확인은 아니다.
새 모델추론·학습0, 기존 DTW 정렬 재실행0, 기존 C·T3-F·운영·사용자/멘토자료·원본·과거동결 보존.
이번 비교 계산 합계 {e['timing']['geometry_seconds']:.3f}초, 읽기·저장 등을 포함한 실행 {e['timing']['wall_seconds']:.3f}초. 촬영 종료부터 사용자 피드백까지의 대기 시간 측정이 아니다.

손회전·손목굽힘에 대한 실제 남은 한계
저장 손축은 손목에서 중지의 시작 관절로 향하는 화면2D 방향이다. 손바닥의3D법선이 아니다. 주먹을 전완 축으로 돌리면 그 값은 비슷하게 남을 수 있다.
직접적인 손목굽힘각은 현재 입력에 없다. 팔꿈치각이나 두 검출기 손목 위치 차이는 손목굽힘 정답을 대신하지 못한다.
손축과 전완 방향을 함께 보는 화면 내 관계 근거는 쓸 여지가 있지만 아직 검증된3D회전·손목굽힘 판정부가 아니다.
현재16영상의 손바닥앞뒤 표본도 충분하지 않다. 이런 부족을 가리고 새 학습을 진행하지 않았다.

현재 판단과 다음
점 간격 영향에 대한 계산 보완은 완료했다. 수치가 낮아지는 것은 연속 경로 비교의 수학적 성질이므로 실제 피드백 정확도가 높아졌다는 증거는 아니다. 새 운영 교정 기준으로 채택하지 않았다.
거리식을 계속 바꾸는 대신, 다음은 같은 주요 동작 안에서 위치·손모양·손방향·팔관계 근거를 별도 칸으로 연결한다. 위치가 비슷하다는 이유로 회전/손모양 차이 또는 미확인을 덮지 않는지 검증한다.
회전·손목굽힘은 현재 자료에서 실제 확인 가능한 범위와 판단할 수 없는 범위를 나누어 보고한다. 단어별 임시 예외를 만들지 않는다.
NEXT_PLAN_KO.txt가 다음 단일 작업이며 아직 실행하지 않았다. 이번은 새 HTML 없이 저장 결과와 이 요약을 남겼다. http://127.0.0.1:8800/ 는 이전 단계 보고서 그대로다.

근거: EVALUATION.json / RESULT_MANIFEST.json / FROZEN.json
review/RESULT_VERIFICATION.json / MATH_RESULT_VERIFICATION.json / POSTEXECUTION_REVIEW.json
review/FEATURE_PRESERVATION.json
일반수학 방법: Alt–Godau(1995) polygonal Frechet free-space. https://webspace.science.uu.nl/~kreve101/asci/ag-cfdbt-95.pdf
'''
    (U/'RESULT_KO.txt').write_text(report,encoding='utf8')
    status='COMPLETE_VERIFIED_CONTINUOUS_GEOMETRY_NOT_EDUCATIONAL_VALIDATION'
    files=['RESULT_KO.txt','NEXT_PLAN_KO.txt','EVALUATION.json','RESULT_MANIFEST.json','FROZEN.json','review/RESULT_VERIFICATION.json','review/MATH_RESULT_VERIFICATION.json','review/POSTEXECUTION_REVIEW.json','review/FEATURE_PRESERVATION.json']
    pins={str(U/f):sha(U/f) for f in files}
    next_action='기존 목표 표현·부모 구간에 위치/손모양/2D손방향/팔관계 근거를 별도 채널로 연결해 위치 일치가 주먹회전·손목꺾임·손모양 차이와 미관측을 덮지 않는지 공개 저장자료로 검증한다. 3D회전/직접손목굽힘 미검증을 명시하고 새 거리식·학습 순회는 하지 않는다. NEXT_PLAN_KO.txt 참조.'
    new(U/'COMPLETION.json',dict(status=status,at=now,summary=summary,pins=pins,active=None,html_created=False,
        user_mentor_access=False,old_model_or_path_recalculation=0,new_training=0,production_changed=False,
        implementer='root + semantic_feedback',method_math_verifier='semantic_model',source_verifier='semantic_data_audit',
        human_ksl_truth_verified=False,next_single_action=next_action))
    update(U/'CURRENT_EXECUTION.json',dict(status=status,active=None,exec_session_completed=93005,next_action=next_action))
    (U/'before').mkdir(exist_ok=True)
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']:
        backup=U/'before'/p.name
        if not backup.exists():shutil.copy2(p,backup)
    state=read(OPS/'STATE.json');state.update(current_stage='CONTINUOUS_TRAJECTORY_COMPLETE_NOT_ADOPTED',active=None,updated_date_kst=now,updated_at=now,
        activity_state='IDLE_REPORTED_RESEARCH_COMPARISON',next_single_action=next_action,resume_command=next_action)
    state[U.name]=dict(status=status,unit=str(U),active=None,at=now,report=str(U/'RESULT_KO.txt'),summary=summary,
        new_training=0,new_model_inference=0,production_changed=False,private_access=False,next_single_action=next_action)
    update(OPS/'STATE.json',state)
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8-sig')
    e8='''\n### 2026-11-01 연속 이동 경로 비교 완료 — 점 간격 영향 감소, 교육 정확도 미검증
`word100_continuous_trajectory_20261101` 완료. 기존2825기록/미제공10 범위·좌표·마스크·경로·모양출력보존. 연속L∞Frechet 후보1개를고정1회실행해완전1417의하상한인증/미관측1408null유지. 자연585=293완전/292보류,17수치감소. speed125123완전중104감소지만123모두양수;실편집omit/reverse각39양수유지/41보류. 기존hold/slow123zero유지. 범위밖swap95한계그대로. 수치감소는교육정확도개선X.
자체217/독립합성592검사/공개수학26866검사PASS, 별도출처·범위·집계검산PASS. 새학습/모델추론/기존정렬재실행0. C/T3F·운영·원본·사용자/멘토보존. 실행exec93005종료0,새HTML없음/8800은이전단계보고서그대로. RESULT_KO.txt/COMPLETION.json/review검산참조.
사용자추가원칙: 작은위치흔들림과주먹회전·손목꺾임은다른특징이다. 손목위치일치로다른변화를정상승격하거나삭제금지. 현재palm_axis는손목→중지MCP2D축,손바닥법선/직접손목굽힘각/3D주먹회전판별완료아님. 손모양·방향원근거와미관측별도보존.
\n'''
    e9='''\n### 2026-11-01 다음 한 작업 — 손모양·손방향·팔관계와 이동을 별도로 연결
연속경로후보검증까지완료. 다음은새거리식/임계값순회보다기존목표표현설명·부모사건에위치/손모양/2D손방향/팔관계의독립근거를연결한다. 위치가같아도모양/방향이변하는기술대조와현재공개관측으로지울신호가없는지확인. 주먹축회전/손목굽힘의실제판별성능은미검증이며관측불가를정상처리금지. 기존global시점/참고개별유지,사용자/멘토접근·튜닝/운영교체X.
세부 `research/front_keyphase_v1/ops/word100_continuous_trajectory_20261101/NEXT_PLAN_KO.txt`. 다음작업아직미실행. 이번완료기하/과거모델재실행X. 새학습을강행하지않고표현의중요특징을읽을근거와부족을구분한다.
\n'''
    for pre,entry in [('## 8.',e8),('## 9.',e9)]:
        line=next(x for x in s.splitlines() if x.startswith(pre));s=s.replace(line+'\n',line+'\n'+entry,1)
    p.write_text(s,encoding='utf8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

현재 `{U.name}` 완료. 계산/학습 실행중 없음(exec93005종료0). 8800은직전hold단계읽기전용보고서서버(exec35320)이며이번새HTML아님. 중복실행금지,현재CURRENT_EXECUTION/완료파일확인.
이번은동일경로점간격과실제경로차이를분리하는연속분절선후보1개. 이전2825결과+missing10해시재사용,완전1417수치인증/unknown1408null유지. 자연585중293완전/292보류,17수치감소. speed125104/123감소지만123모두양수;실편집omit/reverse각39양수/41보류. hold/slow123zero유지(기존도zero),swap95범위밖변형/10미제공보존. 숫자감소≠교육/정렬정확도향상.
자체217,독립수학합성592/75곡선,공개수학26866PASS,출처·범위·집계PASS. 같은Codex검산≠독립수어정답. 새학습/추론/기존정렬재실행0. C·T3F·운영·W:/sources/원본·과거동결·사용자/멘토보존. 새다운로드·카메라·외부전송·Git없음.
사용자강조: 주먹과팔유지중작은흔들림과별도주먹회전/손목꺾임을구분. 위치곡선0이손전체동작일치아님. 기존palm_axis는2D손목→중지MCP축,직접손목각/3D법선없음. 회전/굽힘학습·실제판별완료아님;원69특징/모양출력/미관측보존. 세부review/FEATURE_PRESERVATION.json.
근거 {U} 의 RESULT_KO.txt / COMPLETION.json / EVALUATION.json / review/RESULT_VERIFICATION.json / MATH_RESULT_VERIFICATION.json / POSTEXECUTION_REVIEW.json.
구현root+semantic_feedback,독립수학semantic_model,출처semantic_data_audit.
다음한작업(아직미실행): {next_action}
보고서HTML새작성없음. 진짜피드백/교육적오류/모든단어효과아직미검증. 목표변천사는 Docs/PROJECT_DIRECTION_HISTORY_KO.txt.
''',encoding='utf8')
    with (ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:
        f.write(f'''\n\n2026-11-01 점 간격에 따른 차이와 실제 이동 경로 차이 분리\n손목이가는길을점끼리비교하던이산거리와,같은저장점들을이은연속선분거리를고정비교했다. 정확히같은직선의재표본차이를없애는계산성질검증,공개빠른속도123성분중104수치감소/123양수잔차유지. 실제편집된구간삭제/순서반전각39완전관측의차이는보존했다. 자연실제사람293중17만수치감소로교육정확도개선은주장하지않고운영미적용. 현재목표는변경없다. 사용자확인대로손목위치가같아도주먹회전/손목꺾임은별개이므로원래손모양·2D손축·각도근거를삭제하지않았다. 3D회전/직접손목각의충분한정답이없는한계도확인. 다음은거리식순회를멈추고위치·모양·방향·팔관계근거를같은주요동작에별도연결해중요차이가한수치에묻히지않게하는작업이다. 새학습/추론0,사용자/멘토미사용. 근거 {U}/RESULT_KO.txt.\n''')
    assert read(OPS/'STATE.json')[U.name]['status']==status
    new(U/'FINAL_VERIFICATION.json',dict(status='PASS',at=now,pins_unchanged=len(frozen['pins'])+len(frozen['protected']),
        completion_sha256=sha(U/'COMPLETION.json'),context_state_handoff_history_synchronized=True,new_training=0,new_inference=0))
    print(status)
if __name__=='__main__':main()
