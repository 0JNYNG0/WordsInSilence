from pathlib import Path
import json, sys
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'))
import sync_flow
sync_flow.U=U
read=lambda n:json.loads((U/n).read_text(encoding='utf-8-sig'))
labels=read('PROCESS_LABELS_VERIFIED.json');binding=read('PROCESS_INPUT_BINDING_VERIFIED.json')
assert labels['queries']==binding['queries']==45 and binding['direct_review_values_preserved']
prepared=(U/'PARTIAL_VIEWS_MANIFEST.json').exists()
views=read('PARTIAL_VIEWS_MANIFEST.json') if prepared else None
summary=(
 '전체 목표는 계속 진행 중이다. 현재 4표현 실제 학습 데모8809가 연결되어 있고, 전체 주요 동작 수행을 판단하는 근거를 보완한다. '
 'AB 공개 FIT36영상/45과정/원프레임1354 직접 검토와 출처·시간 검산 완료. 관계710표본·움직임163구간을 보존했다. '
 'REVIEW1567의 records/rows 형식 차이는 원본 변경 없는 COMPAT으로 해결하고 최초 실패를 남겼다. '
 'N의 기존 original36 전체 화면 캐시를 재사용할 수 있어 신규 같은 원본 추출은 생략. 새 주석→입력 시간·인덱스2462연결/85핀을 별도 알고리즘으로 검산했다. '
 'ID08 목·어깨 변이와 미확인 값 유지, 예측값을 새 정답으로 사용하지 않았고 interval 정답을 프레임 전체로 복제하지 않았다. '
 '이 단계의 새 학습·새 모델 추론0, 정확도 향상 주장 없음. 기존 C/T3F·8804/8808/8809 보존. '
)
if prepared:
    summary+=f"부분동작 비교 자료 {len(views['views'])}개 view({views['unique_pixel_time_views']}개 고유 pixel/time 구성)와 {views['pages']}개 시트 준비 완료. 모두 라벨 미확정이며 실제 편집 영상·속도 효과 검증은 아직 아니다. "
    active=dict(training=None,feature_extraction=None,preparation='complete',direct_control_review='WORD2523_primary27_in_progress',review_agent='/root/egypt_source_binding_review',demo_port=8809)
    status='PROCESS_TARGETS_BOUND_PARTIAL_VIEWS_PREPARED'
else:
    summary+='부분 비교 계약 고정:44과정×3편집 최대132개와 정상속도/여백trim/관측누락40개. prepare_partial_control_views.py 코드 검토·실행을 이어간다. '
    active=dict(training=None,preparation='code_review_before_execution',review_agent='/root/egypt_source_binding_review',demo_port=8809)
    status='PROCESS_TARGETS_BOUND_PARTIAL_VIEWS_PREPARING'
nxt=(
 'AB PARTIAL_VIEWS_PROGRESS/INTENT/MANIFEST의 실제 상태부터 확인하여 중복 준비하지 않는다. 준비 미완료면 code review와 실제 세션을 확인하고 이어받는다. '
 'egypt_source_binding_review가 WORD2523 primary27 view의 실제직접검토를 이어가고 있으므로 REVIEW_PARTIAL_2523/agent 실제상태부터 확인하고 중복하지 않는다. '
 '나머지 완료된 부분동작 view는 원본·실제 남은 앞/진행/뒤 근거를 직접 열람해 지지/부분/미확인으로 기록한다. '
 '편집종류=정답, 시작끝 동시존재=완료, 선택한 occurrence 삭제=전체 누락으로 바꾸지 않는다. 반복·통과·몸 흔들림·목/어깨 정상변이와 blackout 미확인 유지. '
 '실제 모델용 편집 입력은 원영상 픽셀에서 재추출하며 깨끗한 원본 context 특징을 잘라 복사하지 않는다. '
 '신규 process 판정부 비교 전 같은 사람 그룹 분리/감독 범위/후보 예산/통과 기준을 동결한다. 기존 readout·shape·motion·partial order는 보존. '
 '새 학습과 추가 이득은 아직 미확인. 구 비교·AA 완료public job 재추론/재채점·사용자/멘토/H14/15/ID16 접근 금지. '
 '준비 숫자를 성능 향상으로 보고하지 말고, 모델 후보 추가 대신 이 구별 학습에 집중한다. 전체 목표 active. '
 '현황 보고는 AB/PROGRESS_KO.txt. HTML 추가 없음.'
)
if (U/'PARTIAL_VIEWS_VERIFIED.json').exists():
    verification=read('PARTIAL_VIEWS_VERIFIED.json')
    summary+='부분 비교 자료의 별도 저장값 검산 파일 PARTIAL_VIEWS_VERIFIED.json도 확인했다. 직접 의미 라벨 검토와 실제 영상 모델 평가는 별도이며 아직 미완료다. '
sync_flow.sync('부분 수행 비교 자료 준비 완료·실제 관측 검토',summary,nxt,status,active)
print(status)
