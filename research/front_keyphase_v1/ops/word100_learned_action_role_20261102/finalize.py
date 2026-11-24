from role_model import *
from evaluate_roles import NORMAL
ROOT=O.parents[2]
def main():
    v=read(A/'SAVED_VERIFICATION.json');d=read(A/'CAUSE_DIAGNOSIS.json');p=read(A/'FROZEN_TRAINING.json');training=read(A/'TRAINING_RESULT.json')
    assert len(training['runs'])==6 and v['verified_arrays']==528 and v['preservation']
    values={}
    for name in ['local_42','local_43','local_44','temporal_42','temporal_43','temporal_44']:
        r=read(A/'evaluations'/f'{name}.json');normal=[s for s in r['sequences'] if s['kind'] in NORMAL];reverse=[s for s in r['sequences'] if s['kind']=='actual_reverse_main' and s['precedences']]
        values[name]=dict(native_role=[r['metrics']['native']['correct'],r['metrics']['native']['known']],normal_order_confirmed=sum(s['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED' for s in normal),
          main_reverse_detected=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in reverse),normal_false_conflicts=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in normal))
    now=clock();status='ROLE_PARTIAL_PROGRESS_SEQUENCE_NOT_READY';nextdir=O/'word100_learned_action_process_20261102';nextdir.mkdir(exist_ok=True)
    nextplan=dict(at=now,status='AUTHORIZED_NOT_TRAINED',goal='Continue the same supervised neural feedback goal; preserve observed handshape/body/motion and distinguish physically presented actions from similar passing configurations.',
      evidence=dict(role_verification=str(A/'SAVED_VERIFICATION.json'),causes=str(A/'CAUSE_DIAGNOSIS.json'),normal_order='10->12 of21 eligible normal conditions;7cycle conditions separately unsupported for fixed-count/order.',local_reverse='2of3 supported fixed-order contrasts detected; one interleaves.',core_loss='7mapped points from3original moments classified as transit; cannot ignore these when declaring feedback ready.',rest_detail='33of68role errors are transit-vs-rest, which both remain visible movement/rest evidence. Do not spend the next unit exclusively resolving this low-priority distinction.'),
      steps=[
       'Use FIT01-09 only to assemble a bounded action-process supervision table: directly observed core configurations, transitions, and direction/from-to relations tied to supported guide clauses. Audit which targets are direct/partial/unknown. Do not promote context-only source cards to dense semantic truth. Retain continuous crossings and uncertain boundaries.',
       'Reuse existing SignRep RGB768 features and body-relative trajectories. Compare a separate RGB-augmented role readout against a joint action-process readout initialized from existing trained guide/state parts. Fine-tune only candidate parts; freeze successful observation outputs. Give explicit supervised order/direction contrasts using original, actual full/local reverse and normal speed inputs. Do not hard-force expected order or learn from DEV corrections.',
       'Freeze exact candidates/targets/primary/seeds and pilot gates before training. FIT recordedIDs01-09; headvalidation07-09 disclosed prior-observer exposure; DEV11-13 exposed development only. Preserve A/T/H results, no H14/15 reuse, ID16 or private/mentor data. If FIT signal cannot separate process, diagnose data/representation before more training rather than lower confidence thresholds.',
       'Evaluate complete feedback evidence jointly: normal usable order, meaningful from-to/order changes, short core configuration preservation, missing images/joints, movement cuts, guide phrase/time alignment and runtime. A state/order match alone cannot certify a cut movement. Separate rest-vs-transit errors from false core suppression. Only passing scope goes to a separate demo; then measure full capture-to-feedback latency. No production/C/T3F/default replacement.'
      ],
      scientific_limits='This is an evidence-informed next hypothesis, not proof RGB or multi-task supervision will succeed. Do not select seed44 because its DEV result is higher. All raw time/evidence remains. Four-word feasibility does not establish all-KSL performance.',
      inputs='Existing public FIT216 prepared variants and original reviewed source clips; no new external download or private input.',whole_goal_complete=False)
    exclusive(nextdir/'PLAN.json',nextplan)
    nexttext='FIT 공개자료에서 원래 손 자세와 실제 행동 과정의 감독 신호를 분리하고, 기존 RGB 특징·몸 상대 움직임을 직접 연결한 역할/행동과정 후보의 조건을 고정한다. 유지된 관측 출력은 보존. 정상 순서확인·중요 역순/이동삭제·짧은동작·누락을 공동 검증하며 대기/전환 미세구분만 계속하지 않는다. A528/T/H완료비교 반복·H14/15재튜닝·개인/멘토자료 금지. 조건 통과 범위만 별도 데모와 전체속도 시험으로 간다.'
    current='공개48영상485시점을 실제 시간순 검토하고, 준비/전환·수행·손내림 정답665시점과 보류11시점을 구분했다. FIT216입력으로 역할판정부2조건×3seed의6학습과2사전학습, 새528비교를 완료했다. 고정 temporal42에서 정상 순서 확인은 같은 현재자세 기준10→12/21로 늘고 주요 역순은2/3를 구분했다. 별도7반복조건은 고정 순서/횟수 판정 대상이 아니다. 정상 새 순서오경고0, 영상누락 시간축 보존, 이전 손자세/손모양/관계/이동 관측 보존. 그러나 정상9조건은 계속 보류이며 수행 정답7변형시점(원본3시점)을 전환으로 잘못 보았다. 역할3종 정상314/382, 약80.17%종류별평균으로 전체 채택 기준 미달. 웹 기본교체/전체속도재실행/HTML 없음. 전체 신경망 피드백 목표 active.'
    report=f'''현재 목표와 이번 작업 결과 — {now}

목표: 여러 사람의 공개 한국수어 영상과 설명을 배운 신경망이 새 사용자의 손모양·신체 관계·움직임·순서를 보고 설명 문구와 사용자 시간대에 연결된 피드백을 제공한다. 전체 목표 완료 아님.

왜 했나
같은 손 자세가 주요 동작을 수행할 때뿐 아니라 준비하거나 다른 자세로 가는 중에도 나타났다. 자세의 등장 순서를 모두 같은 의미로 세면 정상 영상에서도 순서가 겹친다고 판단했다.

무엇을 바꿨나
손 자세/모양/관계/이동 결과는 고정하고 별도 역할 신경망을 지도학습했다. 기존 SignRep 기반 RGB+관절 시간특징과 몸 기준 좌표·이동량을 사용했다. 수행/전환/손내림을 분리하되 전환을 무의미하거나 잘못된 행동이라고 하지 않는다. 반복 동작의 교차 과정도 보존했다. 이번은 별도 판정부 학습이며 새 SignRep 본체 미세조정은 하지 않았다. 앞서 미세조정한 특징/관측 부품을 재사용했다.

자료와 실행
공개4표현×기록ID12개의48원본,485새검토시점+191이전손내림시점. 정답665, 보류11. FIT ID01~09: 정답500/보류8; DEV ID11~13: 정답165/보류3.
FIT의 실제 속도/전체역순/국소역순/관절누락 변형216입력. 역할2조건×3seed=6학습, 별도 사전학습2회. 새528역할비교, 저장된 이전 자세 결과 재사용. train exec88953 exit0, evaluate+verify exec23122 exit0. 새 공개 영상 추출 반복 없음.
사전검증은 같은 관측기에서 이미 봤던 FIT사람을 역할판정부만 제외한 검사다. 최종 독립 사람 일반화 성능으로 부르지 않는다. DEV도 이미 여러 차례 사용한 개발 자료다.

이전 → 현재 (미리 고정한 temporal42)
정상 순서 확인: 10/21 → 12/21. 추가 확인2, 이전 확인을 잃은 조건0.
주요구간 역순: 순서 기준이 있는3조건 중2조건 발견. 현재자세 기준과 비교하여 '잘하다' 역순의 보류가 물리적 순서 차이로 바뀌었다.
정상 새 순서오경고: 0. 별도7반복조건은 구성 자세를 확인하되 시작/종료 순서나 필수횟수를 억지 확정하지 않았다.
영상이 없는 조건 {v['image_gap_cases']}개: 시간범위를 그대로 남기고 순서 확인 보류.
기존 자세 확률528배열: 저장값과 완전 동일. 기존 모양/관계/이동 모델·C·T3F·기존 데모 미교체.

아직 부족한 점
정상9조건은 순서를 충분히 확인하지 못한다. 수행 장면161개 중154개는 역할을 맞혔지만7개는 전환으로 잘못 보았다. 이7개는 원본3시점의 속도변형 등이 포함된 수치다. 이를 무시하고 전체 피드백이 좋아졌다고 하지 않는다.
역할3종은 정상314/382, 종류별 재현율 평균80.17%. 68오류 중33개는 손내림과 전환끼리의 혼동이다. 이 미세구분보다 실제 수행을 전환으로 빼버리는7개와 지나가는 자세를 수행으로 오인하는 문제를 우선한다.
전체 비교 결과가 좋은 seed를 사후 선택하지 않는다. 고정 주후보의 통과조건 미달로 웹에 적용하지 않았다.
새 판정부만 약{v['head_median_seconds']:.4f}초. 영상 관측·RGB 특징 준비·DTW를 포함한 전체시간이 아니다. 이번 전체 업로드 속도 시험은 하지 않았다.

실제 이해 범위
현재는4표현의 물리적 자세와 행동 과정 일부다. 원본 설명의 모든 문구가 모든 프레임에서 검증된 상태는 아니다. 완전한 접촉/깊이/정밀 엄지/필수 반복횟수/전체 수어 정오를 판정하지 않는다. 역할 주석은 Codex 시각검토이며 독립 한국수어 전문가 정답과 다르다. 저장결과 산술검산도 같은 Codex 별도 코드다.

다음
{nexttext}
새 계획: {nextdir/'PLAN.json'}
이번에는 HTML을 추가하지 않았다. 기존 결과와 새 수치는 아래 고정파일로 추적 가능하다.
ROLE_TARGETS.json / FROZEN_TRAINING.json / TRAINING_RESULT.json / EVALUATION.json / SAVED_VERIFICATION.json / CAUSE_DIAGNOSIS.json

최초 오류/한계 보존
시트에서 프레임 번호50을53으로 잘못 읽은 입력 검사를 실행 전 수정했다. LABEL_FIRST_FAILURE.txt에 첫 오류를 남겼고, 정답 작성 뒤 해시를 고정했다. 준비 exec52325의 종료코드는 세션 만료로 재확인하지 못했으며 완료 로그·48/485/16산출물과 해시를 별도로 확인했다. 완료 작업을 다시 추출하지 않았다.
사용자·멘토 촬영 학습/튜닝/선택 없음. H14/15잠금평가 재사용 없음. 새외부다운로드/외부전송/Git/운영교체 없음.
'''
    (A/'RESULT_KO.txt').write_text(report,encoding='utf8')
    exclusive(A/'FINAL_ARTIFACTS.json',dict(at=now,status=status,values=values,pins={str(p):sha(p) for p in [A/'RESULT_KO.txt',A/'ROLE_TARGETS.json',A/'SAVED_VERIFICATION.json',A/'CAUSE_DIAGNOSIS.json',nextdir/'PLAN.json']},training_exit=0,evaluation_verification_exit=0,whole_goal_complete=False))
    write(A/'CURRENT_EXECUTION.json',dict(at=now,status=status,active=None,finished_sessions={'88953':0,'23122':0},next=str(nextdir/'PLAN.json'),whole_goal_complete=False))
    ss=read(O/'STATE.json');ss['current_stage']=status;ss['updated_date_kst']='2026-11-02';ss['word100_learned_action_role_20261102']=dict(status=status,at=now,active=None,result=str(A/'RESULT_KO.txt'),verification=str(A/'SAVED_VERIFICATION.json'),normal_eligible_order_before=10,normal_eligible_order_after=12,normal_eligible=21,local_reverse_detected=2,local_reverse_eligible=3,next=str(nextdir/'PLAN.json'));ss[nextdir.name]=dict(status='AUTHORIZED_NOT_TRAINED',plan=str(nextdir/'PLAN.json'),active=None);ss['next_single_action']=nexttext;write(O/'STATE.json',ss)
    cpath=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';c=cpath.read_text(encoding='utf-8-sig')
    for number,title,text in [(8,'2026-11-02 행동역할 학습으로 일부 순서 개선·전체 미완료',current),(9,'2026-11-02 관측 보존·행동과정 공동 감독',nexttext+' 다음 단위 '+nextdir.name)]:
        lines=c.splitlines(True);idx=next(i for i,l in enumerate(lines) if l.startswith(f'## {number}.'))+1;lines.insert(idx,f'\n### {title}\n\n{text}\n\n');c=''.join(lines)
    cpath.write_text(c,encoding='utf8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'# WordsInSilence 최신 인수인계\n\n{now}\n\n{current}\n\n다음: {nexttext}\n\n{nextdir/"PLAN.json"}\n\nA사전2/본6학습·528비교·별도검산 완료. exec88953/23122 exit0. 진행중 학습·추출·평가 없음. 기존 웹서버 미시작. 같은Codex 시각주석/별도산술 검증이며 독립전문가 정답 아님. 새 후보를 바로 운영에 넣지 않는다. 전체목표 active.\n',encoding='utf8')
    for filename in ['FEEDBACK_GOAL_AND_PLAN_KO.txt','PROJECT_DIRECTION_HISTORY_KO.txt']:
        f=ROOT/'Docs'/filename
        with f.open('a',encoding='utf8') as out:out.write(f'\n\n[{now}] 행동역할 학습 단계\n{current}\n근거: {A/"RESULT_KO.txt"}\n다음: {nexttext}\n목표 변경이 아니라 관측된 자세와 수행 과정을 분리해 학습하는 방법 보완이다.\n')
    assert read(O/'STATE.json')['current_stage']==status
    assert all(sha(p)==h for p,h in read(A/'FINAL_ARTIFACTS.json')['pins'].items())
    print(status,values)
if __name__=='__main__':main()
