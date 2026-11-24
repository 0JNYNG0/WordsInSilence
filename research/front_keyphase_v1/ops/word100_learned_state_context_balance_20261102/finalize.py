"""Record verified progress and remaining semantic-role problem, not completion."""
import shutil,re,json
from state_model import *

def main():
    assert not (T/'FINAL_ARTIFACTS.json').exists();v=read(T/'SAVED_VERIFICATION.json');at=clock()
    assert v['status']=='RETAIN_OLD_DEMO_DIAGNOSE_REMAINING'
    reports={n:read(T/'evaluations'/f'{n}.json') for n in ['old_phase','previous_temporal_42','current_42','anchored_context_42']}
    normal=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']
    def count(n,kinds,rest=False):
        pts=[p for p in reports[n]['points'] if p['kind'] in kinds and (p['y']==9 if rest else p['y']<9)]
        return [sum(p['y']==p['pred'] for p in pts),len(pts)]
    values={n:dict(native=count(n,['native']),normal=count(n,normal),full_reverse=count(n,['actual_reverse']),local_reverse=count(n,['actual_reverse_main']),normal_rest=count(n,normal,True)) for n in reports}
    visual=dict(at=at,files=[str(T/'WORD0739_transition_diagnosis.jpg'),str(T/'WORD1133_transition_diagnosis.jpg')],
      observations=[
       'WORD0739ID12 frames62/64 visibly close the right hand and pass an upright/diagonal forearm configuration before outward arm movement at66/68/77. Frame86 is a later upright closed-hand configuration. A sparse major-state ordering is not an all-frame physical-state ordering.',
       'WORD1133ID12 frames54/57/59/60 approach the left arm; frames65/72 then78/84 show the later proximal-to-distal arm relation. Closeness to wrist during approach is not completion of the described brushing movement.'
      ],scope='Direct visual diagnosis of two exposed DEV examples only. No new training labels, no contact/depth ground truth, not proof for all uncertain cases.',
      correction='Current physical configuration and role in an action need different outputs. Do not mark every transit as irrelevant, or assume all unknown intervals lack meaning.',
      guide_warning='Existing source_cards include context_only and full_span_semantic_verified=false. Retain restrictions; never treat them as dense or fully verified semantic labels.')
    exclusive(T/'VISUAL_CAUSE_NOTE.json',visual)
    nextdir=O/'word100_learned_action_role_20261102';nextdir.mkdir(exist_ok=True)
    nextplan=dict(at=at,status='AUTHORIZED_PREPARATION_NOT_TRAINED',goal='Same neural feedback goal: preserve current visible configuration while learning which described action is being approached, performed, held or finished.',
      evidence=dict(saved_verification=str(T/'SAVED_VERIFICATION.json'),visual_note=str(T/'VISUAL_CAUSE_NOTE.json')),
      steps=[
       'Review FIT01-09 native chronology with existing source descriptions. Prepare sparse physical action-role/transition targets, including similar configurations during approach, main movement and return. Unknown remains masked; an intermediate movement is not automatically unimportant.',
       'Preserve the current-state and independent shape/relation/movement outputs. Evaluate a separate guide/action-role head using body-relative relations, observed movement and existing RGB/time features; extra context must not rewrite the observed handshape. Direct annotations and weak DTW correspondence remain distinct.',
       'Freeze candidate, source labels and role/order metrics before training. Include actual local reorders, cut actions, speed changes and sensor/image gaps. Fit only public recordedIDs01-09; use07-09 as disclosed head-level validation and11-13 only exposed development. No H14/15 reuse or ID16/private input.',
       'Check normal usable feedback, true important-order contrasts, missing evidence, source-phrase timing and full latency together. Preserve successful observation components even when whole-sequence gating fails. Integrate only passed scope into a separate demo; no production/default replacement.'
      ],
      model_selection_disclosure='T current_42 is a candidate for protected observation because this development comparison exposed context regressions. This is a development-informed next hypothesis, not an independently validated final model or silent replacement of the frozen primary.',
      not_allowed=['New external models/downloads','Word-specific score exceptions','Treat transit as meaningless by definition','Turn sparse state ordering into universal every-frame ordering','Private or mentor training/tuning/selection','Repeat locked H evaluations','Claim whole sign correctness from phase confidence'],
      no_new_training=True,whole_goal_complete=False)
    exclusive(nextdir/'PLAN.json',nextplan)
    rows=[r for r in read(old.P/'INPUT_MANIFEST.json')['rows'] if r['role']=='fit']
    exclusive(nextdir/'SOURCE_PLAN.json',dict(rows=[{k:r[k] for k in ['clip_id','word_id','person','source','source_sha256','frames']} for r in rows],existing_physical_states=str(S/'PHYSICAL_STATE_TARGETS.json'),source_guides=str(old.P/'GUIDE_SUPPORT.json'),new_targets_assigned=False,new_extraction=False))
    summary=('공개 FIT 36원본의 실제 국소 순서변경 입력을 추가해 총216학습입력을 검증하고, 원본 대기자세191시점(1시점 보류)을 확인했다. '
      '현재자세/보호된 시간문맥 두 조건×3seed의6회 지도 미세조정과2회 사전학습을 완료했다. '
      '현재자세 비교모델은 원래 DEV 확인시점85/96→96/96, 정상변형 포함201/219→219/219. 고정 주후보(추가 시간문맥)는94/96·215/219이며 일부 불확실한 현재 판단을 잘못 바꿨다. '
      '주요부분 역순 자세25/30→27/30. 하지만 전체시간의 자세를 그대로 순서로 읽으면 준비·전환에서 지나가는 비슷한 자세가 섞여 정상 순서 확인21→7/28로 줄었다. '
      '정상 순서충돌 경고0, 주요 역순의 잘못된 순서확인0이지만 단지 보류를 늘린 부분이므로 전체 피드백 통과로 보지 않는다. '
      '704비교(새528/이전저장176) 별도 산술검산. 기존 웹/C·T3F 미교체, 사용자·멘토자료 학습·튜닝·선택 없음. 전체 목표 active.')
    nexttext=('word100_learned_action_role_20261102에서 현재 자세와 설명 속 행동 역할을 분리한다. FIT 공개 영상의 접근·실제 수행·유지·복귀를 원본과 출처 문구로 확인하되 중간 동작을 무조건 버리지 않는다. '
      '좋아진 현재자세/기존 모양·관계·이동 출력은 보존하고, 같은 자세가 어느 행동 과정인지 배우는 별도 판정부의 정답·비교조건을 먼저 고정한다. '
      'DTW는 대응/정렬 근거이며 준비의 비슷한 자세를 주요 동작 완료로 바꾸지 않는다. 새 학습은 아직 없음. '
      'T/H 완료비교 재실행·H14/15 재튜닝·ID16·개인자료 금지. 전체 목표는 소규모 신경망 피드백 데모이며 완료 아님.')
    lines=['현재 목표와 이번 결과', '',
      '현재 지속 목표는 공개 한국수어 영상과 출처 설명을 이용해 신경망을 지도학습·미세조정하고, 새 사용자에게 설명의 구간과 연결된 피드백을 주는 소규모 데모다. 기존 목표를 다시 중복 등록하지 않았다.', '',
      '이번에 실제로 한 일', '공개 원본36개에서 주요부분만 순서를 바꾼36영상을 실제로 다시 관측했다. 기존 속도/역순/관절누락 입력과 합쳐216입력을 썼다.',
      '대기자세 원본192장면을 보고191개만 정답으로 정했다. 143개는 학습,48개는 개발용. 애매한1개는 제외했다. 손 검출 실패를 대기 정답으로 쓰지 않았다.',
      '현재 자세를 읽는 신경망과, 그 판단이 불확실할 때만 추가 앞뒤 흐름을 쓰는 신경망을 각각3회 학습했다. 기존SignRep 기반 표현은 보존하고 자세판정부의 가중치를 미세조정했다. SignRep 본체를 새로 전체 학습한 것은 아니다.', '',
      '비교 (정답이 직접 확인된 시점 수 / 전체 수)',
      '조건 | 원래영상 자세 | 정상 속도·대기변형 포함 | 전체 역순 | 주요부분만 역순']
    for name,ko in [('old_phase','이전 기준'),('previous_temporal_42','직전 시간문맥 후보(미채택)'),('current_42','이번 현재자세 비교모델'),('anchored_context_42','이번 고정 주후보: 추가 시간문맥')]:
        z=values[name];lines.append(ko+' | '+' | '.join(f'{z[k][0]}/{z[k][1]}' for k in ['native','normal','full_reverse','local_reverse']))
    lines+=['', '96/96이나219/219는 4표현의 제한된 물리적 자세 확인 시점이다. 전체 수어 인식률, 모든 프레임 정확도, 독립 사용자219명의 결과가 아니다. 같은 원본의 여러 변형이 포함된다.',
      '원래 영상 대기자세는48/48이나 속도·인코딩 변형까지 포함한 대기자세는 '+str(values['anchored_context_42']['normal_rest'])+' 이다. 대기와 동작의 구분도 완전히 해결된 것은 아니다.', '',
      '왜 아직 웹을 바꾸지 않았나',
      '주먹을 세운 자세는 주요 동작일 때도 나오고, 다른 동작으로 옮겨가는 중에도 잠깐 나올 수 있다. 손 자세를 맞췄다고 그 설명의 동작을 이미 완료한 것은 아니다.',
      '전체시간을 지우지 않고 모두 살펴보게 바꿨지만, 이 두 경우를 섞으면 정상 영상의 순서를 충분히 확인하지 못한다. 정상21건을 확인하던 비교에서 이번은7건만 확인했다. 새 오류 경고0건만 보고 성공이라고 하지 않는다.',
      '순서변경을 볼 수 있는3대조 중1개만 충돌을 확인했고2개는 보류했다. 교대 반복 표현은 애초에 고정 시작/종료/반복횟수 규칙을 강제하지 않았다.',
      '자세를 잘 읽는 학습 결과는 파일로 보존했다. 전체 피드백 판단은 통과하지 않아 기존 데모와C·T3F를 교체하지 않았다.', '',
      '다음 실제 작업',nexttext,'',
      '속도: 추가 판정부만 중앙 '+f"{v['head_median_seconds']:.4f}"+'초. 이번 전체 촬영→결과 속도 시험은 아직 하지 않았다. 이 수치를 사용자 대기시간으로 말하면 안 된다.',
      '검증 범위: FIT기록ID01~09, 개발ID11~13. 기록ID 분리이며 영상 전체의 생체 동일인 독립을 증명한 것은 아니다. 이 개발자료는 이미 여러 번 본 자료다. 주석과 별도 산술검산은 모두Codex이며 독립 수어 전문가 평가가 아니다.',
      '구현/실행: Codex 로컬. 추출·검토 준비·학습·비교·별도검산 종료코드0. 새 모델 다운로드, 외부 전송, 운영 기본 교체 없음. 사용자·멘토 촬영은 학습/튜닝/선택에 사용하지 않았다.',
      'DTW: 학습의 대응 근거와 실행의 학습된 흐름 정렬에 사용한다. 모든 장면을 강제로 정답 순서에 구겨 넣으면 실제 순서 오류가 숨겨지므로 원래 시간의 관측과 행동순서도 별도로 확인한다.',
      '현재 웹 서버 실행 상태는 이번 단위에서 확인하거나 시작하지 않았다. 기존 데모 보존은 현재 서버 실행 보장이 아니다. 전체 목표는 아직 진행 중이다.']
    (T/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    exclusive(T/'FINAL_ARTIFACTS.json',dict(at=at,status='STATE_IMPROVED_ACTION_ROLE_STILL_UNRESOLVED',values=values,
      pins={str(T/n):sha(T/n) for n in ['INPUT_VERIFICATION.json','FROZEN_TRAINING.json','EVALUATION_PROTOCOL.json','TRAINING_RESULT.json','SAVED_VERIFICATION.json','DIAGNOSIS.json','VISUAL_CAUSE_NOTE.json','RESULT_KO.txt']},
      implementation='Codex',verification='SameCodex separate saved arithmetic and visual diagnosis, not independent expert',exits=dict(prep=0,review=0,train=0,evaluate_verify_diagnose=0),whole_goal_complete=False))
    write(T/'CURRENT_EXECUTION.json',dict(at=at,status='STATE_IMPROVED_ACTION_ROLE_STILL_UNRESOLVED',active=None,finished=True,whole_goal_complete=False,next=str(nextdir/'PLAN.json'),old_demo_replaced=False))
    root=O.parents[2];before=T/'before';before.mkdir(exist_ok=True)
    paths=['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md','Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt','Docs/PROJECT_DIRECTION_HISTORY_KO.txt']
    for name in paths:shutil.copy2(root/name,before/Path(name).name)
    shutil.copy2(O/'STATE.json',before/'STATE.json')
    ctx=root/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text(encoding='utf-8-sig')
    for section,title,body in [(8,'현재자세 개선·행동역할 구분의 필요성 확인',summary),(9,'자세와 행동역할을 나누는 신경망 감독',nexttext)]:
        text=re.sub(rf'(?m)^(## {section}\. [^\n]*\n)',lambda m:m.group(1)+f'\n### 2026-11-02 {title}\n\n{body}\n\n',text,count=1)
    ctx.write_text(text,encoding='utf8')
    state=read(O/'STATE.json');state['current_stage']='STATE_IMPROVED_ACTION_ROLE_STILL_UNRESOLVED';state['next_single_action']=nexttext
    state[T.name]=dict(status='STATE_IMPROVED_ACTION_ROLE_STILL_UNRESOLVED',active=None,result=str(T/'RESULT_KO.txt'),verification=str(T/'SAVED_VERIFICATION.json'),values=values,next=str(nextdir/'PLAN.json'),whole_goal_complete=False)
    state[nextdir.name]=dict(status='AUTHORIZED_PREPARATION_NOT_TRAINED',plan=str(nextdir/'PLAN.json'),source_plan=str(nextdir/'SOURCE_PLAN.json'),active=None);write(O/'STATE.json',state)
    (root/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+nexttext+'\n\n현재 완료: '+str(T)+'\n다음: '+str(nextdir)+'\n\n진행중 학습/추출/평가 없음. 완료한 T704비교/H잠금평가 반복금지. 기존 웹 서버는 미시작·미교체. 신경망 피드백 목표 active. 사용자·멘토자료 접근/학습/튜닝/선택 금지.\n',encoding='utf8')
    goal=root/'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt';txt=goal.read_text(encoding='utf-8-sig');txt=re.sub(r'(?s)최신 실행 현황 .*?(?=\n1\. 사용자가 확정한 목표)',f'최신 실행 현황 ({at})\n{summary}\n다음: {nexttext}\n',txt,count=1);goal.write_text(txt,encoding='utf8')
    with (root/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:f.write('\n\n'+at+' — 목표 유지, 자세 판정과 행동 역할의 분리\n'+summary+'\n목표를 바꾼 것이 아니라 영상 속 자세와 설명의 행동을 같은 것으로 취급하던 부분을 분리하는 보완이다. 준비·전환도 중요한 동작일 수 있으므로 일괄 삭제하지 않는다.\n다음: '+nexttext+'\n')
    assert read(O/'STATE.json')[nextdir.name]['status']=='AUTHORIZED_PREPARATION_NOT_TRAINED'
    print(summary)

if __name__=='__main__':main()
