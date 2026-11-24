from pathlib import Path
import json,sys
T=Path(__file__).resolve().parent;O=T.parent
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
read=sync_flow.read
verify=read(T/'SCOPE_SAVED_VERIFIED.json')
assert verify['status']=='PASS_SAVED_NATIVE_SCOPE_DIAGNOSTIC' and all(verify['checks'].values())
assert read(T/'SOURCE_LINKED_EVIDENCE_v2.json')['records']==12
assert read(T/'LINK_REVIEW.json')['status']=='PASS_LINK_V2'
assert read(T/'AUTOMATIC_SCOPE_REVIEW.json')['status']=='PASS_SPARSE_WITNESS_COVERAGE_AUDIT'
a=read(T/'AUTOMATIC_SCOPE_AUDIT.json');assert a['total_proposals']==9 and a['all_witnesses_included']==4
sync_flow.U=T
sync_flow.sync('기존 RGB 관측 재사용·설명 연결 및 자동 구간 근거 감사',
 'U3/R2 각80epoch 및별도검산 완료:새비스듬축19/19,팔꿈치2/3→3/3이나새관절연결정상팔쓸기오류남아미채택. T에서동일정의기존RGB모델로새S양성창12(FIT9/DEV3) 확인:11새추론+1저장재사용,RGB9/9·3/3 vs관절9/9·1/3. 기존ID13다른창50–71실패보존. source_link_v2로출처/원시간/미확인항목/영상ID근거연결. 기존DTW의9자동창중4만모든sparse검토시점포함;정렬정확도나교육오류율아님. 전체goal미완료/운영웹미적용.',
 '동일정의기존16물리관측과T/source_link.py 보존후 설명 행동의 자동시간범위·완료여부 연결을보완. 먼저FIT01–09에서기존learnedprototype경로와직접검토sparsewitness의앞뒤관계감사,DEValready9결과로창선택튜닝금지. 관측존재≠과정완료/경계1프레임차≠오류로구분. 실패한geometry반복epochs·새RGB재학습금지(필요근거먼저). 신규정당대조가필요하면실제RGB변환후context재추출. 원정답·C/T3F·웹·H14/15·사용자/멘토보존. 완료U/R/T학습/추론/검산재실행금지. 현재훈련프로세스없음;별도데모자동연결·전체대기시간평가미완료.',
 'EXISTING_RGB_REUSE_VERIFIED_SOURCE_TIME_SCOPE_REMAINS')
print('Progress report and current state synchronized')
