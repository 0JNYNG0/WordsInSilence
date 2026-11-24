"""Root-only record of actual comparison, scope correction and structural fix."""
from pathlib import Path
import json
import hashlib
import sys

U=Path(__file__).resolve().parent
O=U.parent
AC=O/'word100_process_contract_alignment_20261102'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
sync_flow.U=U


def main():
    decision=read(AC/'DECISION.json')
    verification=read(AC/'EVALUATION_VERIFIED.json')
    assert verification['evaluation_sha256']==sha(AC/'EVALUATION.json')
    code=read(U/'CODE_REVIEW.json')
    assert code['status'].startswith('PASS') and code['current_code_sha256']==sha(U/'relation_support.py')
    tests=read(U/'STRUCTURAL_CHECKS.json'); replay=read(U/'PUBLIC_SAVED_SUPPORT_REPLAY.json')
    assert tests['status'].startswith('PASS') and replay['status'].startswith('PASS')
    scope=read(AC/'PARTIAL_ERROR_SCOPE_DIAG.json')
    assert scope['status']=='POST_EVALUATION_DIRECT_SCOPE_AUDIT_NOT_RESCORING'
    inventory_path=U/'FIT_TARGET_SEMANTICS_INVENTORY.json'
    inventory_verification=U/'FIT_TARGET_SEMANTICS_VERIFIED.json'
    inventory_complete=inventory_path.exists() and inventory_verification.exists()
    if inventory_complete:
        invcheck=read(inventory_verification)
        assert invcheck['status'].startswith('PASS')
        assert invcheck['inventory_sha256']==sha(inventory_path)
        for p,h in invcheck['source_pins'].items():
            assert sha(p)==h,p
    summary=(
      '전체학습형피드백goal active. 직전질문설명turn은상태변경없는no-progress로확인후실제안전작업진행. '
      'AC학습terminal80epoch/320step/74.01초확인,새357평가1회와456509별도저장검산완료. '
      '같은DEV시험조건 정상근거3→10/30·자동1→7/15,부분오확인1→2/45·unknown0/5유지,raw68→65/75. '
      '고정21gate8통과13미달로미채택,기존웹/C/T3F불변. '
      '실제원화면에서1133DEV첫6candidate가손목도착전끝나는범위문제확인. '
      'FIT44과정중기존후보전체chain포함14/44 vs동일terminalrun범위44/44는미채점구조진단뿐; '
      '누락30중29는RGB문맥에motion전체가있어뒤영상완전미관측이아닌nominal관절/시간범위불일치. '
      'false2는3frame/0.10초편집이며움직임꼬리가남음:direct관측연속성0이지수어의미오류정답아님. 기존라벨/점수수정0. '
      'AD공통relation-support compact범위부품구현,짧은구간/겹침/반복/관측gap보존. '
      '초기목록복사의최악3차증가를범위저장+maxpairs256명시보류로수정,초기6파일보존. '
      f'구조{tests["valid_cases"]+tests["invalid_cases"]}검사·공개FIT45/{replay["checks"]}구조검산·별도{code["checks"]}검토PASS. '
      'AD새학습/추론/성능평가/웹적용0. '
      +('FIT정답의관측/부분/unknown 의미분리목록및별도검산완료. ' if inventory_complete else 'FIT정답의관측/부분/unknown 의미분리목록retention agent진행중. ')
      +'새성능시험전정답범위고정필요:globalhigh20/30상한때문에범위만넓혀기존gate24/30통과불가. '
      '개인/멘토/H14/15/ID16금지·새다운로드외부전송Git0. DEV반복개발/같은Codex검토이며독립전문가검증아님.')
    nxt=(
      'AD CODE_REVIEW/STRUCTURAL_CHECKS/PUBLIC_SAVED_SUPPORT_REPLAY/NEXT_SUPERVISION_CONTRACT 실제내용확인. '
      +('완료FIT_TARGET_SEMANTICS_INVENTORY/VERIFIED와GUIDE_SCOPE_ADDENDUM을읽고 ' if inventory_complete else 'retention_direction_review 실제상태/이미생성된FIT_TARGET_SEMANTICS파일먼저확인해중복하지말고마무리한뒤 ')
      +'기존FIT01~09공개직접주석에서보이는특징/과정추정/관측부족을분리한후속학습정답계약을준비한다. '
      '기존P GUIDE_SUPPORT의출처문구/시간관측/손모양·관계정답이실제로연결되는범위를감사하고context_only·verifiedfalse를새정답으로승격금지. '
      'falseDEV2를새학습정답으로옮기지않고,source편집자체/프레임개수=교육오류금지. '
      '가이드의어떤중요특징차이를직접확인할수있는지기존출처근거로정하고,정상속도/손내림/교차/반복보존과함께평가계약동결후한번의정당화된학습/비교만. '
      'unknown을loss에서빼는것만으로보류학습했다고하지않는다. '
      'AD소비부는compact *_range/lazy만사용하고예산초과UNRESOLVED처리·추가모델연산예산검증. 넓은envelope를한과정/정확끝으로채택금지. '
      'AC/AB완료학습평가재실행금지,기준하향/seed/epoch검색금지. C/T3F/8804/8808/8809·개인보호유지. '
      '전체목표미완료:지원표현의중요차이/정상허용/관측부족/사용자원시간연결·실제데모·전체지연·분리평가계속.')
    active=dict(training=None,evaluation=None,extraction=None,new_candidate_adopted=False,
      current_component='compact_relation_support_only',
      semantics_audit=None if inventory_complete else 'retention_direction_review',
      existing_demos_preserved=[8804,8808,8809])
    title='구간 보존 구조 보완·직접 관측과 의미 오류 구분'
    sync_flow.sync(title,summary,nxt,'INTERVAL_SUPPORT_IMPLEMENTED_SEMANTIC_SUPERVISION_PREPARATION',active)
    result=(
      '전체 목표는 아직 완료되지 않았습니다.\n\n'
      '이번 실제 학습 비교\n'
      '공개 자료를 사용한 후보1개를80epoch 학습하고,357개 질의를한번 비교했습니다. 같은 저장 결과를 별도 코드로 검산했습니다.\n'
      '- 정상 과정을 근거까지 확인: 이전3/30 → 이번10/30\n'
      '- 전체 영상에서 자동으로 고른 과정 확인: 이전1/15 → 이번7/15\n'
      '- 부분 과정만 남긴 편집 시험에서 잘못 확인: 이전1/45 → 이번2/45\n'
      '- 관측 부족 시험의 잘못 확인:0/5 →0/5\n'
      '- 단순 이진 분류 정답:68/75 →65/75\n'
      '사전 기준21개 중8개 통과·13개 미달로 이번 후보는 웹에 적용하지 않았습니다. '
      '이 수치는 TCN 단어 인식률이 아닙니다. 30조건은 독립30사람이 아니라 공개12영상의 구간·속도·여백 변형을 포함합니다.\n\n'
      '원인을 더 확인한 결과\n'
      '1. 실제로 끝까지 수행한 동작인데도 비교할 작은 구간을 너무 일찍 잘랐습니다. '
      '팔을 따라 손목으로 가는 예에서, 손목에 도착하기 전 화면까지만 비교했습니다. 원본에는 이후 동작이 있습니다.\n'
      '2. 앞뒤 이미지 문맥은 일부 남아 있었습니다. 따라서 모델이 뒤 영상을 완전히 못 본 것이라기보다, '
      '이미지가 참고한 범위와 관절·시간 비교 범위가 어긋났습니다.\n'
      '3. 남은 오통과2개는 중간3프레임(약0.10초)을 편집으로 뺀 영상이었습니다. 이후 움직임은 남아 있습니다. '
      '이 검사는 연결 과정을 직접 관측했는지를 확인한 것으로, 그 수행이 수어의 의미를 해친다는 정답은 없습니다. '
      '이 한계를 숨기지 않고 기존 점수와 함께 기록했습니다. 기존 점수를 바꿔 성공 처리하지 않았습니다.\n\n'
      '바로 고친 공통 부품\n'
      '도착 관계가 처음 보인 한 프레임을 동작의 끝이라고 확정하지 않고, 그 관계가 보이는 시간 범위를 보존하도록 구현했습니다. '
      '짧은 동작·관계 겹침·반복을 남기고, 영상이 없는 구간은 직접 관측으로 연결하지 않습니다. '
      '너무 많은 후보로 계산이 커지는 문제도 작은 범위 표현과 명시적 예산 초과 보류로 고쳤습니다. '
      f'구조검사{tests["valid_cases"]+tests["invalid_cases"]}건과 공개 학습 자료45개의 저장 결과 재생을 통과했습니다. '
      '이는 부품의 구조 검증이며 새 인식률 향상이나 실제 웹 적용은 아닙니다.\n\n'
      '다음 학습에서 구분할 것\n'
      '① 영상에서 직접 보인 손모양·관계·움직임 ② 앞뒤 흐름으로 추정한 부분 ③ 중요한 가이드 특징과 실제로 다른 부분을 분리합니다. '
      '짧은 편집 흔적이나 손이 잠깐 안 보이는 것만으로 틀린 수어라고 가르치지 않습니다. '
      '현재 확신 높은 정상 판정은20/30이므로 구간만 넓히는 수정으로 사전기준24/30을 통과할 수 없습니다. '
      '따라서 구조 수정만으로 성능시험을 반복하지 않고, 기존 공개 정답의 의미와 관측 범위를 정리한 후 필요한 감독을 결정합니다.\n\n'
      '최근 과정 실험의 정답335개도 목록화했습니다: 직접 과정154·부분 관측165·미확인16개입니다. '
      '미확인은 관측불가9개와 관계 경계가 애매한7개로 나뉩니다. 이335개에는 교육적 오류나 가이드 문구와의 모순을 직접 가르치는 정답이 없습니다. '
      '프로젝트 전체에 손모양·가이드 단계 학습이 없다는 뜻은 아닙니다. 이미 있는 그 정답들과 출처 설명이 정확히 연결되는 범위를 다음으로 확인합니다.\n\n'
      '기존 C·T3-F와 데모를 보존했습니다. 사용자·멘토 촬영은 학습·튜닝·선택에 사용하지 않았습니다. '
      '이번 검토는 Codex의 직접 영상 관찰과 별도 코드 검산이며 독립 한국수어 전문가 검증이 아닙니다. '
      '전체 사용자 처리시간은 이번에 다시 측정하지 않았습니다. HTML은 추가하지 않았습니다.\n')
    (U/'RESULT_KO.txt').write_text(result,encoding='utf8')
    (U/'PROGRESS_KO.txt').write_text(result,encoding='utf8')
    print(summary)


if __name__=='__main__':main()
