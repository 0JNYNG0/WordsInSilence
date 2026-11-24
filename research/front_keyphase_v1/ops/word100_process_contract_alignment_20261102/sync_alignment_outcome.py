"""Root-only truthful result report and handoff after saved verification."""
from pathlib import Path
import json
import sys
import numpy as np

U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
AB=O/'word100_action_completion_supervision_20261102'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
sync_flow.U=U


def main():
    t=read(U/'TRAIN_RESULT.json');e=read(U/'EVALUATION.json');v=read(U/'EVALUATION_VERIFIED.json');d=read(U/'DECISION.json')
    assert v['status'].startswith('PASS')
    old=read(AB/'PROCESS_EVALUATION.json')['summaries']['development']['overall']
    s=e['summaries']['development'];pair=d['comparison'];view=pair['reviewed_view'];auto=pair['runtime_proposal']
    adopted=d['failed']==0
    title='과정 학습·실행 조건 정렬 후보 학습과 고정 비교 완료'
    summary=(f'전체goal active. AC 실제FIT262(217edited+45runtime)/118nominal(75positive37partial6unknown), '
        f'47,980param seed42 final80epoch320steps 단일후보학습완료({t["seconds"]:.2f}s). '
        'v2누락감독·실제runtime문맥·global/local 연결과정target를함께맞춤,SignRep/C/T3F/기존웹고정. '
        f'새357질의(297view+60auto)각1회평가·저장검산{v["total_checks"]}PASS. '
        f'DEV같은75known raw {old["raw_correct"]}→{s["raw"]["raw_correct"]}; '
        f'정상근거확인{view["old_true_confirmed"]}→{view["new_true_confirmed"]}/30, '
        f'부분오확인{view["old_false_confirmation"]}→{view["new_false_confirmation"]}/45, '
        f'미확인오통과{view["old_unknown_confirmation"]}→{view["new_unknown_confirmation"]}/5, '
        f'자동정상{auto["old_true_confirmed"]}→{auto["new_true_confirmed"]}/15. '
        f'기존21gate {d["passed"]}통과/{d["failed"]}미달; {d["status"]}. '
        '현재웹미교체/선택span정확시간독립정답미검증/전체수어정오·필수반복판정아님. '
        'RGB실제support344입력검산/nominal밖context별도표시,unknownnegative금지. '
        '동일DEV11~13반복개발결과,새사람독립시험아님. 사용자/멘토/H14/15/ID16미사용. '
        '같은Codex물리검토·별도저장검산이며전문가정답아님. HTML추가없음/old완료평가재추론0.' )
    nxt=(
        'AC DECISION/EVALUATION_VERIFIED/RESULT_KO.txt 실제결과확인. 완료학습1·357평가와AB구학습/평가재실행금지. '
        +('고정gate통과후새선택구간의정확한물리근거와원시간을공개DEV에서검토하고,통과범위만별도실제demo연결·전체latency검증. ' if adopted else
          '저장값으로globalprocess/pointproposal/localprocess/동시실패·정상회귀·부분오확인을분리하고 실제근거가변경을지지하는다음한작업을정한다. 임계값/epoch/seed검색으로구제하지않는다. ')
        +'기존C/T3F/8804/8808/8809보존,개인촬영튜닝금지. nominal학습정답118은FIT진단이며DEV새candidate정답으로복사금지. '
        '전체goal미완료:중요과정구별·정확시간·정상허용·실제피드백데모/전체속도검증을이어간다.')
    active=dict(training=None,extraction=None,evaluation=None,completed_training_session=33143,
                new_candidate_adopted=False,gate_eligible=adopted,demo_port_preserved=8809)
    context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text(encoding='utf8')
    for heading,body in [('### 2026-11-02 '+title,summary),('### 2026-11-02 '+title+' 이어받기',nxt)]:
        needle=heading+'\n'
        if needle in text:
            start=text.index(needle)+len(needle);ends=[i for i in [text.find('\n### ',start),text.find('\n## ',start)] if i>=0]
            end=min(ends) if ends else len(text);text=text[:start]+'\n'+body+'\n\n'+text[end:]
    context.write_text(text,encoding='utf8')
    sync_flow.sync(title,summary,nxt,'ALIGNED_PROCESS_CANDIDATE_EVALUATED_RUNTIME_PENDING' if adopted else 'ALIGNED_PROCESS_CANDIDATE_EVALUATED_NOT_ADOPTED',active)
    timings=[r['runtime_head_seconds'] for r in e['rows'] if r['role']=='development' and r['origin']=='runtime_proposal']
    report=(
      '현재 목표와 진행 위치\n'
      '여러 사람의 공개 한국수어와 동작 설명을 미리 배운 신경망이 새 영상의 손모양·관계·움직임·순서를 분석하고, 설명 문구와 해당 시간에 연결된 피드백을 주는 것이 목표다. '
      '기존 4표현 학습형 실제 데모는 있으며, 이번에는 중요한 이동 과정이 실제로 이어졌는지를 확인하는 부분을 보완했다. 전체 목표가 완성된 상태는 아니다.\n\n'
      '이번에 바꾼 점\n'
      '1. 직접 확인해 둔 손 위치와 움직임 정답 중 학습에 전달되지 않은 항목을 연결했다.\n'
      '2. 실제 검사 때 사용하는 전체 영상 문맥을 학습에도 포함했다.\n'
      '3. 작은 구간도 단순히 움직였는지가 아니라, 출발에서 도착으로 해당 과정이 이어졌는지를 배우도록 했다. '
      '짧은 구간 안에서 동작이 실제로 보이면 정답으로 유지했고, 주변 장면에만 보이는 동작을 그 구간의 정답으로 복사하지 않았다.\n'
      '4. 판정하려는 구간과 RGB가 참고한 앞뒤 범위를 따로 보존했다. 전후 문맥은 쓸 수 있지만 정확히 그 순간에 직접 확인한 것처럼 말하지 않는다.\n\n'
      f'실제 실행: 공개 학습 입력262조건과 작은 구간118개를 연결하고, 후보1개를80epoch 학습했다({t["seconds"]:.2f}초). '
      '기존과 같은 모델 크기·학습 횟수·판정 기준을 사용했다. 357질의를 한 번 평가하고 저장 결과만 별도 검산했다. '
      '조건 수는 서로 다른 사람 수가 아니며 같은 원본의 변형이 포함된다.\n\n'
      '이전 → 이번 (같은 공개 개발 자료)\n'
      f'- 정해진 비교 범위의 단순 분류: {old["raw_correct"]}/75 → {s["raw"]["raw_correct"]}/75\n'
      f'- 정상 과정의 연결 근거 확인: {view["old_true_confirmed"]}/30 → {view["new_true_confirmed"]}/30\n'
      f'- 일부 과정만 있는데 잘못 통과: {view["old_false_confirmation"]}/45 → {view["new_false_confirmation"]}/45\n'
      f'- 근거 부족인데 잘못 통과: {view["old_unknown_confirmation"]}/5 → {view["new_unknown_confirmation"]}/5\n'
      f'- 전체 영상에서 자동으로 찾은 정상 과정 확인: {auto["old_true_confirmed"]}/15 → {auto["new_true_confirmed"]}/15\n\n'
      f'사전에 유지한 기준21개 중 {d["passed"]}개 통과, {d["failed"]}개 미달이다. '
      +('다음 실제 실행·시간 연결 검증에 들어갈 수 있으나 아직 웹을 교체하지 않았다.\n' if adopted else '기준을 충족하지 못해 새 후보를 현재 웹에 적용하지 않았다.\n')
      +'여러 보완을 함께 바꾼 비교이므로, 어느 하나만의 효과라고 해석하지 않는다. 단순 분류 점수는 전체 수어 정답률이나 교육 피드백 정확도가 아니다.\n\n'
      f'속도: 저장 입력이 준비된 뒤 새 판정부 부분만 DEV 자동15질의 중앙값 {float(np.median(timings)):.4f}초였다. '
      '관절 검출·RGB 추출·버튼부터 결과까지의 전체 시간은 이번에 다시 측정하지 않았다. 기존 공개1편의 분석8.77초/버튼9.55초 기록이 개선됐다는 뜻이 아니다.\n\n'
      '남은 일: 선택한 정확한 구간의 근거, 정상적인 차이의 허용과 중요한 누락 구별, 새 수행자 검증, 실제 데모와 전체 처리시간을 계속 검증해야 한다. '
      '이번 DEV는 반복해 본 개발 자료이고, 물리 관측 정답은 Codex 검토로 독립 한국수어 전문가 판단이 아니다.\n'
      '기존 C·T3-F와 기존 웹은 보존했고 사용자·멘토 촬영은 학습·튜닝·선택에 사용하지 않았다.\n')
    (U/'RESULT_KO.txt').write_text(report,encoding='utf8')
    (U/'PROGRESS_KO.txt').write_text(report,encoding='utf8')
    print(summary)

if __name__=='__main__':main()
