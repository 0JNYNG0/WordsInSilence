"""Report completed fixed comparisons; no new model or detector execution."""
from common import *

def main():
    ev=read(U/'EVALUATION_COMPLETE.json'); fv=read(U/'RESULT_VERIFIED.json')
    mv=read(U/'shape_recovery/MOTION_SAVED_RESULT_VERIFIED.json')
    m=read(U/'motion_recovery/PHYSICAL_COMPARISON.json')
    assert fv['status']==mv['status']=='PASS' and not ev['any_pass'] and m['decision']=='NOT_ADOPTED'
    protected=[]
    for p,h in read(U/'AUTHORIZATION.json')['protected'].items():
        assert sha(p)==h,p
        protected.append(dict(path=p,sha256=h,unchanged=True))
    write(U/'PROTECTED_VERIFIED.json',dict(status='PASS',files=protected,no_inference=True))
    lines=['공개자료 보완·고정 비교·아바타 시험 — 현재 결과','',
      '핵심: 이번 비교 회차는 검산까지 끝났다. 프로젝트 전체나 웹 피드백 완성이 끝난 것은 아니다. 새 손모양/움직임 모델은 기존 기능을 모두 보존하는 조건을 통과하지 못해 웹에 넣지 않았다. 아바타는 공개1단어를 실제로 움직이는 시험본까지 만들었다.','',
      '1. 무엇을 바로잡았나',
      '“지금 설명하는 행동이 아니다”와 “손이 움직이지 않는다”는 다른 뜻이다. 준비하거나 팔을 내리는 동안에도 손은 움직인다. 이전301개 부정 정답은 설명소속 정답으로 보존하고, 물리적인 이동/유지는 별도의 직접 영상 정답과 출력으로 분리했다. 이 계약 문제가 과거 실패의 유일한 원인이라는 주장은 하지 않는다.',
      '또한 설명 연결에 원본영상 SHA·영상ID·손·실제시간·설명항목ID가 모두 일치해야 하도록 검사했다. 불확실한 관측이나 문자열 false를 확정 근거로 받아들이지 않으며, 이동이 보였다는 이유만으로 행동 완료/교육적 정답을 선언하지 않는다.19개 가짜입력 계약검사 통과는 실제 수어 정확도가 아니다.','',
      '2. 손모양 학습 결과',
      '공개12학습영상의36시점에서 손가락 상태200항목을 추가 검토했다. 기존861+새200=1061개의 FIT 손가락 정답이며, 모호한110항목은 강제 정답으로 넣지 않았다. 펴짐/일부굽힘/접힘을 구분하고 엄지의 미지원 상태·벌림·정확접촉은 지원 밖으로 남겼다.',
      '공통100단어 FIT899 유지, 기존에 선별한 추가자료56 포함 총955학습입력을 재사용했다. 이는 모든955영상에 손가락 세부정답이 있다는 뜻이 아니다.',
      '기반신경망 고정+손가락 판정부만 학습 / 기반까지 미세조정의 두 조건을 각각3초기값으로 고정6회 학습했다. 같은 일반수어 DEV36영상의389개 손가락 정답에서 비교했다.389는 영상수가 아니며,3초기값의 점수를 합쳐 새로운 자료 수로 세지 않는다.']
    for s in [42,43,44]:
        old=read(PREV/'scores'/f'A_{s}.json')['finger_by_variant']['normal']
        fixed=ev['models'][f'frozen_{s}']['finger_by_variant']['normal'];tune=ev['models'][f'finetune_{s}']['finger_by_variant']['normal']
        lines.append(f"초기값{s}: 이전{old['correct']}/389 → 기반고정{fixed['correct']}/389, 미세조정{tune['correct']}/389.")
    lines+=['기반고정은 기존 공유부품을 보존했지만 새손가락 판정이 나빠졌다. 미세조정도 채택기준 미통과다. 원래 C·T3-F나 운영TCN을 이 점수로 교체하지 않았다.',
      '반면 학습자료 자체1061항목에서는 기반고정1038~1044, 미세조정1009~1013정답이었다. 새200항목도191~196개를 맞혔다. 따라서 현재 학습사례를 못 외운 문제만으로 설명할 수 없고, 다른 수행기록에 적용하는 일반화가 중요한 문제다. 추가 epoch만 늘릴 근거는 없다. 어떤 관측/체형/표현 차이가 원인인지는 별도검토가 필요하다.',
      '남은 자료범위: 엄지 굽힘/접힘 정답이 부족하며 왼손 중지·약지·소지의 부분굽힘은 한 표현에 편중돼 있다. 새 일반화 성능이나 교육적 허용경계가 검증됐다고 하지 않는다.','',
      '3. 실제 이동과 자세 유지의 분리',
      '공개24영상(4표현×기록ID02~07)에서115개의 짧은 시간창을 검토했다. 양손230항목 중 이동142, 유지86, 불확실2. 창 안에 이동이 있었는지의 정답이며 그 창의 중심 순간/동작완료/수어정오 정답은 아니다.',
      '353가중치 양손공통 보조신경망을6개 기록ID제외×3초기값=18회 학습했다. 상대몸좌표를 쓰고 단어 이름이나 사람ID는 모델입력에 넣지 않았다. 대조군은 각5개 학습기록에서만 문턱을 정한 단순 손목변화 규칙이다.']
    for row in m['seeds']:
        p=row['primary'];lines.append(f"초기값{row['seed']}: 원본정순 규칙{p['rule']}/228 → 신경망{p['MLP']}/228.")
    lines+=['기본조건의 유지86항목과 양손교대창38항목은 모두 맞췄다. 그러나 화면조건·시간방향까지 포함하면 고유2개 원본손 항목에서 새손실이 생겼다. 여러 실패층은 이같은장면의 반복집계이며 수십개 다른 실패가 아니다.',
      '하나는 관측이 충분한 왼손이동인데 정순에서는 멈춤, 역순에서는 이동으로 읽었다. 다른 하나는 잠시 유지하다가 뒤쪽에 시작한 짧은 이동을 놓쳤다. 이들은 관절이 없어 생긴 문제가 아니다. 두방법이 공통으로 놓친 작은이동2항목도 남았다.',
      '실제상반신자르기 조건에서는 알려진228항목 중106개가 관측부족이었다. 분모에서 빼지 않았으며 이런 항목은 없는 정보의 문제로 별도 구분했다. 새과제점수와 과거C6 설명소속 점수를 빼서 개선율을 만들지 않는다.','',
      '관측부족106항목을 다시 나누면 준비47+복귀41=88개, 유지검토6개, 주요동작창12개다. 따라서106개 전부가 중요한수어동작의미검출이라는뜻은아니다. 주요동작창은86손항목중12개가관측조건미충족이다. 이 구간종류는 원본검토용표본표시이지 모델이자동으로구분한결과가아니다.','',
      '4. 설명 연결 및 원래 작업 그래프',
      '원본시간에 연결한 손가락 근거는51→59항목으로 늘었다. 이는 자동으로 새사용자의 구간을 잘 찾았다는 증명이 아니다. 이집트 원본59프레임을 주요구간에 억지로 붙이지 않았다.',
      'P1 원인감사/P2 직접정답/P3 고정학습비교 완료. P3 모델채택 실패. P4 원본연결과 입력계약 보완 완료, 실제 자동구간 통합검증은 미완료. P5 새평가/P6 웹적용은 앞단계 미통과로 실행하지 않았다. 원래 그래프 G2는 보완중, G3~G6는 남아 있다.',
      '검토한1273입력(955학습+318개발/준비)에 새로 손대지 않은 최종평가가 확보됐다고 볼 수 없다. 다른공개자료가 전혀 없다는 뜻은 아니다. 봉인자료/최종8영상은 열지 않았다.','',
      '5. 아바타',
      '사용자가 키포인트ZIP을 삭제했을 가능성을 알려주고 기존5뷰로 소수시연을 요청했다. 공개조상 REAL02의 동일127프레임·30fps 영상5개로 추정카메라/근사3D를 만들고 기존캐릭터 팔·손가락에 옮겼다. 정면RTMW127프레임은 재사용, 다른4뷰508프레임만 추가추출했다.',
      '손가락 굽힘방향/FBX가져오기시점/코를몸통축으로쓴오류를 발견하여수정했다. 각최초결과는보존했다. 원본카메라보정값이없으므로 재투영일치가 정확한3D복원의증명은아니다. 사람과다른머리/목비율에따른얼굴위치, 엄지·접촉·자기충돌은교육용채택전남은문제다.',
      ('사용자가 사람모습이아니어도 사람과비슷한체형/잘보이는손가락을원해 보유모델을점검하고 로컬생성비율마네킹에같은공개동작을옮긴예시를만들었다. 최종게임외형이나얼굴접촉해결을뜻하지않는다. 새모델다운로드나사용자촬영은사용하지않았다.' if (O/'avatar_motion_pilot_20261113/proportion_examples/RESULT_KO.txt').exists() else '사용자는 사람모습이아니어도 사람과비슷한체형/잘보이는손가락을원하며 예시를요청했다. 보유한정적캐릭터와비율확인용대안을확인하는후속시각비교를진행한다.다운로드나사용자촬영은사용하지않는다.'),
      '초기수정미리보기: ../avatar_motion_pilot_20261113/PREVIEW_KO_BODY_AXIS.html','',
      '6. 검증과 속도',
      '손모양 저장결과 별도286889검사, 움직임 저장결과 별도10906검사 통과. 통과는 산술/입력/분리 검증이며 모델채택 통과가 아니다. 담당은Codex root와서로다른Codex에이전트다. 독립사람/한국수어전문가검수라고하지않는다.',
      '새 손모양 후보의 캐시관절→신경망 계산은CPU4에서 영상당 중앙값 약7ms였다. 촬영·RTMW·전송·화면시간을 제외한 값이고 이전측정과 동일조건이아니므로 사용자전체대기시간 개선이라고말하지않는다.',
      'C·T3-F/현재TCN/기존피드백 모델의 보호해시확인. 운영웹교체·개인촬영학습튜닝·다운로드·외부전송·Git작업없음.','',
      '다음: 손모양은 같은상태가 다른표현/기록에서관측되는차이를FIT내에서먼저감사한다. 물리이동은 시간반전에영향받지않는이동유무와 순서를보존해야하는방향판정을분리하고, 작은실제이동/정상흔들림대조근거를보강한뒤 새비교를동결한다. 같은개발점수를보며문턱이나seed를바꿔성공시킬계획은아니다. 새부품과설명구간이함께검증된뒤에만웹/최종평가로넘어간다.']
    (U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n','utf-8')
    write(U/'FEEDBACK_CYCLE_COMPLETE.json',dict(status='FIXED_COMPARISONS_COMPLETE_NOT_ADOPTED',shape_runs=6,motion_runs=18,
      shape_verifier=str(U/'RESULT_VERIFIED.json'),motion_verifier=str(U/'shape_recovery/MOTION_SAVED_RESULT_VERIFIED.json'),
      whole_project_complete=False,web_changed=False,fresh_final_evaluation_opened=False,
      avatar_followup=('Proportional avatar examples completed; teaching correctness remains unverified' if (O/'avatar_motion_pilot_20261113/proportion_examples/RESULT_KO.txt').exists() else 'Proportional avatar examples requested by user, ongoing separately'),report=str(U/'RESULT_KO.txt')))
    print('Completed comparison report and protected model hashes verified.')

if __name__=='__main__':main()
