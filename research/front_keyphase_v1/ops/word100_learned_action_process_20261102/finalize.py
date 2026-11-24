"""Finalize verified evidence and handoff. Does not mark the whole goal done."""
from process_model import *
ROOT=O.parents[2]

def main():
    assert not (B/'FINAL.json').exists();v=read(B/'VERIFIED.json');d=read(B/'DIAGNOSIS.json');tr=read(B/'TRAINING_RESULT.json');p=read(B/'FROZEN.json')
    assert len(tr['runs'])==6 and v['verified_arrays']==528
    assert v['status']!='PASS_TO_ISOLATED_RUNTIME','Finish the authorized runtime verification before finalizing a passing candidate.'
    n=v['normal_role'];core=n['groups']['1'];before=len(v['normal_order_before']);after=len(v['normal_order_after']);rev=len(v['local_reverse_detected']);now=clock()
    # No post-hoc candidate promotion or altered threshold in this closure.
    audit=read(B/'SEQUENCE_GAIN_AUDIT.json')
    nexttext='다음은 단순 자세/준비·수행 구분을 더 반복하지 않고, 주요 구간 안에서 설명의 어느 행동을 수행하고 있는지 배우는 감독 근거를 확인한다. 공개 FIT01~09의 기존 이동 구간·시간순 시각주석·출처 문구를 연결해 수행 과정 단위 정답과 불확실 경계를 구분한다. 한 자세가 앞뒤 다른 동작 중에 다시 나타나는 경우를 보존한다. DTW 대응은 약한 연결 근거로만 쓰고 자동 정답으로 바꾸지 않는다. 근거가 충족될 때만 실제 행동 구간 판정부를 학습·비교한다. 새 DEV 수정정답/문턱맞춤·H재사용 없이 정상 주요동작 보존·국소역순·이동삭제·영상누락을 함께 검증한다.'
    current=f'공개 FIT216입력으로 영상직접연결/자세·이동과정 공동학습 두 조건의 사전2회+본6회 학습과 새528비교·별도산술 검증을 완료했다. 고정 joint42 역할3종은314→{n["correct"]}/{n["known"]}지만 주요수행은154/{core["known"]}로 같고 주로 손내림 분류가 나아졌다. 정상 순서 계산상 {before}→{after}/21이나 새1조건은 수행 정답f74를 전환으로 빼는 오류와 겹쳐 개선으로 인정하지 않는다. 국소역순은2→{rev}/3로 감소했다. 정상 순서오경고0, 기존 자세528배열·이동삭제/영상누락 차단 보존. 새 후보 미채택·웹/C·T3F미교체, 전체목표 active.'
    nextdir=O/'word100_action_episode_supervision_20261102';nextdir.mkdir(exist_ok=True)
    exclusive(nextdir/'PLAN.json',dict(at=now,status='AUTHORIZED_INPUT_AUDIT_NEXT',whole_goal_complete=False,
      goal='Same goal: supervised, fine-tuned neural feedback mapping observed action episodes to sourced guide phrases and user time. No word-specific exceptions.',
      evidence=[str(B/'VERIFIED.json'),str(B/'SEQUENCE_GAIN_AUDIT.json')],
      root_cause='A broad performed/transit/rest role and a physical pose do not identify which guide action owns a passing or returning pose.77/78endpoint contrasts do not solve full episode sequencing.',
      steps=[
       'Audit public FIT01-09 existing visual time annotations and physical movement episodes against supported guide clauses. Produce action-process IDs with reviewed time bounds and uncertain edges, retaining continuous transitions. No dense truth from DTW or context-only source cards.',
       'Keep observed physical configuration separate from performed action episode. The same configuration may occur in multiple episodes. Do not relabel true core as transit to manufacture order. Normal rest detail is not the primary target.',
       'If supervision supports the distinction, freeze a bounded candidate initialized from existing guide/time neural parts; preserve prior observations and fine-tune using original/speed/full-local reverse/gap inputs and source-time maps. No new external model or private footage. If supervision cannot identify episodes, add only necessary public FIT temporal review before training.',
       'Evaluate core episode assignment, guide phrase/time support, normal variations, local order, cuts and gaps together. Audit apparent gains for core loss. Reuse old outputs read-only. Only functionally passing scope enters an isolated demo and full-latency check.'
      ],limits='Recorded IDs, repeated development, four expressions, same-Codex physical annotation; not independent KSL expert truth. H14/15 locked; no DEV correction-training or seed selection.'))
    comparison='\n'.join(f'{name}: 원본 역할 {q["native_role"][0]}/{q["native_role"][1]}, 정상 순서 {q["normal_order"]}/21, 국소역순 {q["local_reverse_detected"]}/3, 정상 순서오경고 {q["normal_false_conflicts"]}' for name,q in d['condition_summaries'].items())
    report=f'''현재 목표: 공개 한국수어 다인 영상과 설명을 학습한 신경망으로, 새 사용자의 손모양·관계·움직임·순서를 읽고 설명 문구와 사용자 시간대에 연결된 피드백을 제공하는 소규모 데모. 전체 목표는 아직 진행 중.

이번에 한 일
이전 모델이 잘 읽던 손 자세를 고정하고, 준비·전환과 실제 수행을 구분하는 신경망 복사본에 기존 SignRep 영상특징을 직접 추가했다. 두 조건을 비교했다.
1) 영상정보를 추가한 역할 판정부
2) 같은 입력에 자세와 관측된 앞뒤 이동 관계를 함께 가르친 판정부(고정 주후보 joint42)
둘 다 지도학습으로 기존 판정부 가중치를 미세조정했다. 2번은 설명/자세 판정부의 복사본도 함께 조정했다. 보호된 실제 관측 결과와 SignRep 영상추출 본체는 이번에 수정하지 않았다.

왜 이렇게 했나
손이 같은 자세를 보여도 주요 동작을 한 경우와 다른 동작으로 지나가는 경우가 있다. 기존에는 이 둘을 혼동하여 정상 순서를 확인하지 못하는 장면이 남았다. 이미지와 앞뒤 움직임을 더 직접 활용하면 구분에 도움이 되는지 확인했다.

이전 → 현재 (결과를 보기 전에 고정한 후보)
정상 순서 확인: {before}/21 → {after}/21
주의: 새로 확인된1조건은 실제 수행 정답인74프레임을 전환으로 잘못 빼는 현상과 겹친다. 따라서 이 수치 증가를 검증된 순서 개선으로 인정하지 않는다. 원본 시간과 관측 좌표는 보존되며, 오류는 순서를 해석하는 별도 판정부에 있다.
이전 확인을 잃은 조건: {len(v['normal_lost'])}; 새로 확인한 조건: {len(v['normal_gained'])}
주요 구간의 물리적 역순 발견: 2/3 → {rev}/3
정상 새 순서충돌 경고: 0 → {len(v['normal_false_order'])}
실제 수행 역할 맞힘: 154/161 → {core['correct']}/{core['known']}
역할3종 전체: 314/382 → {n['correct']}/{n['known']}; 종류별 재현율 평균 {n['macro_recall']*100:.2f}%
증가는 주로 손내림 분류에서 나왔다. 중요한 수행 구분은154/161로 같고, 국소역순 탐지는 오히려 줄었으므로 원하는 핵심 피드백 성능 개선이라고 할 수 없다.
영상이 비어 있는 {v['image_gap_cases']}조건은 원래 시간 전체를 보존하고 순서 확인을 보류했다.
팔 이동을 지운 시험은 자세가 남아 있어도 동작을 완료했다고 하지 않았다. 이 차단은 기존 움직임 모델의 근거를 유지한 것이며 이번에 새로 개선된 이동 정확도가 아니다.
별도 반복7조건은 시작/끝 순서나 필수 반복 횟수를 억지 확정하지 않는다.

두 조건×3회 결과
{comparison}
더 좋은 개발 점수가 나온 seed를 사후 선택하지 않았다. 영상 추가 자체의 효과와 추가 학습 횟수의 효과를 분리하는 비교는 아니며, 두 새 조건 사이의 공동학습 효과를 비교한 것이다.

자료와 검증
기존 공개4표현·학습 기록ID01~09의216변형 입력(36원본). 역할 정답3068, 자세 정답2492, 앞뒤 물리적 구성 관계216쌍은 변형에 반복 포함된 수이며 독립 영상/사람 수가 아니다. 기존 sparse 주석만 사용하고 모르는 프레임을 임의로 채우지 않았다.
개발 기록ID11~13의88조건×6모델=528새 비교. 같은 사람의 속도/역순/누락 변형이므로 528개의 독립 영상 실험으로 부르지 않는다.
사전 점검07~09도 기존 영상/자세 부품에 이미 노출된 자료다. 최종 독립 일반화 검증이 아니다. H14/15잠금시험은 다시 사용하지 않았다.
현재자세528확률배열은 이전 저장값과 완전히 동일. 이전 모델·입력 해시와 시간범위도 별도 코드로 확인했다. 같은 Codex의 저장결과 산술검증이며 독립 수어 전문가 검토는 아니다.

속도
새 부품만 중앙 {v['head_median_seconds']:.4f}초. 영상 읽기·MediaPipe·SignRep·DTW 전체시간이 아니다. 채택기준을 충족하지 않아 새 운영 연결/전체 속도시험을 하지 않았다. 기존 데모는 그대로 보존했다.

남은 문제와 다음
{nexttext}
자세 사이의 앞뒤 관계를 묻는 보조 문제는77/78을 맞혔지만, 전체 행동을 어느 설명 구간에 연결할지는 해결하지 못했다. 이 둘은 같은 시험이 아니다. 구체적인 다음 입력 점검: {nextdir/'PLAN.json'}
수어 전체 정오, 정확한 접촉·깊이·엄지 및 필수 반복 횟수는 이 결과로 보장하지 않는다. 앞뒤 두 자세를 맞힌 것만으로 중간 과정 완료를 증명할 수도 없다. 설명 문구의 일부는 참고 수준이며 모든 프레임의 교육 정답이 아니다.

실행 상태
prepare session74812 exit0 / train session55307 exit0. 최종 평가·검증 종료는 CURRENT_EXECUTION.json 참고.
새 외부 다운로드/외부전송/Git/운영교체 없음. 사용자·멘토 촬영 학습·튜닝·선택 없음. C·T3F보존.
최초 사전 파일읽기 오류(cp949)와 없는 선택적 경로 확인은 INITIAL_INSPECTION_NOTES.json에 보존했고 실제 학습 전에 UTF8·확인된 경로로 처리했다.
개선이 충분하지 않으므로 새 HTML은 만들지 않았다.

근거: FROZEN.json / SUPERVISION_AUDIT.json / SUPPORT_TABLE.json / TRAINING_RESULT.json / EVALUATION.json / VERIFIED.json / DIAGNOSIS.json
'''
    (B/'RESULT_KO.txt').write_text(report,encoding='utf8')
    exclusive(B/'FINAL.json',dict(at=now,status=v['status'],primary=p['primary'],whole_goal_complete=False,html_created=False,operational_swap=False,
      pins={str(path):sha(path) for path in [B/'RESULT_KO.txt',B/'FROZEN.json',B/'VERIFIED.json',B/'DIAGNOSIS.json',B/'SUPPORT_TABLE.json',B/'TARGET_BEHAVIOR.json',B/'SEQUENCE_GAIN_AUDIT.json',nextdir/'PLAN.json']},next_action=nexttext))
    exec_record=read(B/'CURRENT_EXECUTION.json');exec_record.update(at=now,status=v['status'],active=None,next_action=nexttext);write(B/'CURRENT_EXECUTION.json',exec_record)
    s=read(O/'STATE.json');s['current_stage']=v['status'];s[B.name]=dict(status=v['status'],at=now,active=None,result=str(B/'RESULT_KO.txt'),verification=str(B/'VERIFIED.json'),normal_order_before=before,normal_order_after=after,apparent_gain_not_accepted=True,whole_goal_complete=False);s[nextdir.name]=dict(status='AUTHORIZED_INPUT_AUDIT_NEXT',active=None,plan=str(nextdir/'PLAN.json'));s['next_single_action']=nexttext;write(O/'STATE.json',s)
    f=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';c=f.read_text(encoding='utf-8-sig')
    for num,title,txt in [(8,'2026-11-02 영상·행동과정 공동학습 비교 완료',current),(9,'2026-11-02 실제 수행 구간 감독 근거 확인',nexttext+' 근거 '+str(B/'DIAGNOSIS.json'))]:
        lines=c.splitlines(True);idx=next(i for i,l in enumerate(lines) if l.startswith(f'## {num}.'))+1;lines.insert(idx,f'\n### {title}\n\n{txt}\n\n');c=''.join(lines)
    f.write_text(c,encoding='utf8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+current+'\n\n'+str(B/'RESULT_KO.txt')+'\n\n다음: '+nexttext+'\n\n진행중 학습/평가 없음. 완료한 비교 재실행 금지. H14/15잠금, 사용자·멘토 제외, C·T3F/기존 웹 보존. 전체 목표 active. 같은 Codex 시각주석·산술검산과 독립 전문가 정답을 구분.\n',encoding='utf8')
    for name in ['FEEDBACK_GOAL_AND_PLAN_KO.txt','PROJECT_DIRECTION_HISTORY_KO.txt']:
        with (ROOT/'Docs'/name).open('a',encoding='utf8') as f:f.write('\n\n['+now+'] 영상·행동과정 공동 감독 비교\n'+current+'\n기존 목표 유지: 여러 사람의 동작과 설명을 배운 신경망 피드백. 목표 변경이나 전체 완성이 아니다.\n근거: '+str(B/'RESULT_KO.txt')+'\n다음: '+nexttext+'\n')
    assert all(sha(path)==h for path,h in read(B/'FINAL.json')['pins'].items());assert read(O/'STATE.json')['current_stage']==v['status']
    print(current,flush=True)

if __name__=='__main__':main()
