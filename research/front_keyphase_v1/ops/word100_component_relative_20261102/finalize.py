from pathlib import Path
import sys,json
R=Path(__file__).resolve().parent;O=R.parent
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
read=sync_flow.read
assert read(R/'SAVED_INDEPENDENT_VERIFIED.json')['passed']
assert read(R/'COMPARISON.json')['status']=='RELATIVE_GATES_NOT_MET'
assert len([v for v in read(R/'COMPARISON.json')['checks'].values() if not v])==5
assert not read(O/'word100_component_learning_20261102/SAVED_VERIFIED.json')['verification_failures']
sync_flow.U=R
sync_flow.sync('설명 행동 연결부 5학습·검산 완료, 기존 RGB 관측 재사용 점검',
 'U 새모양1+시간2/R 상대움직임2 각80epoch 완료. 비스듬한 손축 DEV3영상19/19, 팔꿈치하강 양성2/3→3/3. 정상 팔쓸기는 U3/6·R2/6, DEV2영상3확신오류 남아 R5개gate미달/미채택. U601행35gate 및 R542별도검산 불일치0. 관절전용 연결부의 반복학습 중지하고 기존 동일정의 RGB motion 관측을 새 S 원시간창에서 비교하도록 scope_test 준비. 전체goal미완료/기존16출력·C/T3F·웹 유지.',
 'word100_component_rgb_context_20261102/SCOPE_FROZEN.json과 DESIGN_REVIEW 확인후 scope_test.py run: FIT9/DEV3 양성source sequence12창, 정확히같은 기존값1재사용/새창11만동결RGB추론. 옛완료결과재추론재채점금지. FIRST_FAILURE는unknown12도포함한24행을positive12로가정한추론전준비오류이며mask/y로원계약수정. 결과검산후 원시간창/기존지식상실 여부 판단. 필요확인전 새RGB추출/학습금지. 모든구간스캔최적선택금지/H14·15·개인·멘토금지. 전체goal active.',
 'COMPONENT_LEARNING_VERIFIED_RGB_SCOPE_PREPARED')
print('Result and root state synchronized; goal remains active')
