from pathlib import Path
import sys,json
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
assert json.loads((U/'FINAL.json').read_text(encoding='utf8'))['status']=='LIVE_PARTIAL_ORDER_DEMO_VERIFIED'
sync_flow.U=U
sync_flow.sync('부분 순서 실제 데모8809 연결 완료·동작 충분성 감독 준비',
    'AA 새8809 실제demo 완료:4표현 저장16조건 UI/재생/원시간검사, 새공개0739 job dff00320fc0b497db638b0408244db20 1회전체분석8.7741s·버튼9.5511s·GPU.0·순서head0.0320s. 별도373검산PASS/기존47필드·123시간축·181pin보존. 준비시간미측정0표기만수정/초기freeze·보고보존/완료job UI-only재확인. 새정확도·속도향상주장없음. 9등록FIT/3DEV에서배운4표현신경망실행가능하나부분순서≠전체완료·누락없음·반복정오. C/T3F·8804/8808유지. AB 동작충분성감독자료감사 시작,goalactive.',
    'AB word100_action_completion_supervision_20261102에서 predicate_contract_review의 SUPERVISION_INVENTORY_REVIEW 확인후 기존FIT01~09 직접과정근거와sparse/weak를분리. 주요동작 시작·진행·도착의충분한수행 vs 잔여일부관측을감독할최소자료/고정비교를준비하고타당할때만새학습. 물리cut=언어오류/coverage=완료/미관측=부재금지,정상속도·손내림·겹침·중간이동보존. AA서버29836/worker40940유지(DEMO_LAUNCH/실제상태확인),새POST·완료Y/Z/AA추론재실행금지. 속도병목MP8.1687/RGB5.5262병렬,순서32ms최적화우선아님. 전체목표미완료/사용자·멘토·H14/15·ID16미사용.',
    'LIVE_PARTIAL_ORDER_DEMO_VERIFIED',
    dict(server_pid=29836,worker_pid=40940,url='http://127.0.0.1:8809/',training=None,next_review_agent='/root/predicate_contract_review'))
print('Synced AA completion and AB next task')
