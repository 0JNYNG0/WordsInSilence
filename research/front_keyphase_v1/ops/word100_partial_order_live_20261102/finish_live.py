"""Record completed live integration without claiming educational accuracy."""
from pathlib import Path
import json,hashlib,sys
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'FINAL.json').exists()
    b=read(U/'LIVE_ORDER_BROWSER_VERIFIED.json');r=read(U/'LIVE_ORDER_UI_RECHECK.json')
    v=read(U/'LIVE_RESULT_VERIFIED.json')
    assert b['status']=='PASS' and len(b['saved_cases'])==16
    assert str(v['status']).startswith('PASS'),v['status']
    assert r['completed_result_exact'] and r['absent_load_not_zero']
    assert r['current_attempt_fresh_posts']==r['new_model_inference']==0
    assert b['actual_fresh_public_posts_across_attempts']==1 and b['actual_upload_posts']==b['camera_calls']==0
    f=read(U/'EXECUTION_FROZEN.json');assert all(sha(p)==h for p,h in f['pins'].items())
    fresh=b['fresh'];assert fresh['job_id']=='dff00320fc0b497db638b0408244db20'
    report='''현재 목표와 진행 상태 — 2026-11-02

전체 목표
여러 사람이 수행한 공개 한국수어와 출처 설명으로 미리 학습한 신경망이, 목표 단어와 새 영상만 받아 중요한 손모양·신체 관계·움직임·순서를 확인하고 해당 설명과 영상 시간대로 피드백한다.
참고 영상 9편을 촬영 때마다 각각 대조하는 방식은 새 데모에서 사용하지 않는다. 모델이 배운 기준 흐름과 DTW 한 번으로 정렬하고 이 경로를 재사용한다.

현재 위치
소규모 실제 실행 데모는 만들어졌다. 검증된 일부 특징과 앞뒤 관계는 볼 수 있지만, 단어 전체를 충분히 수행했는지·필수 반복이나 짧은 동작을 빼먹었는지까지 믿고 교정하는 완성품은 아니다.
피드백 학습 범위는 순종·잘하다·조상·이집트 4표현. 기본 학습 공개 등록 ID01~09 36영상, 개발 ID11~13 12영상에서 출발했고 기존 변형 자료도 사용했다. 등록 ID를 외형만으로 서로 다른 사람이라고 확정하지 않는다. 개발 자료는 반복 노출되어 독립 최종 시험이 아니다.
전체 목표의 완료율을 객관적으로 계산할 기준이 없으므로 임의의 퍼센트를 붙이지 않는다. 4표현의 동작 확인과 모든 수어에 대한 신뢰성은 다른 문제다.

이미 한 일
1. 실제 지도학습과 SignRep 마지막 투영층 일부 가중치 미세조정을 했다. 언어모델이 모든 설명을 자유롭게 이해하는 구조가 아니라, 출처 설명을 구조화한 특징·단계와 연결한 신경망이다. 미세조정만의 추가 이득은 확인하지 못했다.
2. 손모양·양손 및 팔 관계·일부 움직임의 학습 판정부를 연결했다. 원래 잘 구분하던 결과를 보존하고 새로운 후보가 나빠지면 기존 결과와 나란히 남겼다.
3. 학습한 한 기준 흐름에 새 영상을 정렬하고, 출처 문구와 원래 영상의 시간대를 연결했다. 불확실한 부분과 RGB 시간 흐름에 의존한 추정은 별도로 표시한다.
4. 이번에는 확인한 두 동작의 앞뒤 관계를 새 실제 데모8809에 연결했다. 화면에서 그 근거 시간을 눌러 다시 볼 수 있다. 기존8804·8808과 C·T3-F는 보존했다.

이번에 새로 검증한 것
4표현 × 정상·주요 구간 삭제·주요 구간 역순·영상 정보 가림 = 저장된16조건의 화면과 재생을 확인했다. 기존 결과를 재추론하거나 재채점한 시험이 아니다.
새 공개 영상은 한 편만 처음부터 분석했다. 관절/RGB 관측→학습 신경망→DTW→설명·부분 순서→화면까지 실제로 연결했다.
전체 분석 {analysis:.3f}초, 버튼 클릭부터 결과 표시 {wall:.3f}초. 부분 순서 계산 자체는 {head:.3f}초. GPU.0이 RGB 특징 처리에 사용됐다.
MediaPipe 관측 {mp:.3f}초와 RGB 처리 {rgb:.3f}초는 병렬이므로 합산하면 안 된다. 오래 걸리는 주된 구간은 영상 관측이며, 추가 순서 판정부가 병목은 아니다. 한 편 한 번의 결과이므로 평균 속도나 이전보다 빨라졌다는 증거는 아니다.
측정하지 않은 모델 준비시간이 0초로 표시되던 문제를 수정했다. 완료 결과만 다시 열어 확인했으며 새 분석은 하지 않았다. 최초 코드·결과·동결은 별도로 보존했다.
별도 에이전트가 저장된 입력/출력/시간/기존 결과 보존을 검산했다. 이것은 독립 수어 전문가의 교육 정오 평가와 다르다.

아직 남은 가장 중요한 문제
1. 동작이 잠깐 보였다는 근거만으로 그 동작을 끝까지 했다고 말할 수 없다. 잘하다의 삭제 대조에도 앞뒤 일부가 남아 순서가 같게 보이는 예가 있다. 현재 화면은 이를 전체 성공으로 표시하지 않는다.
2. 모든 중요한 짧은 누락·필수 반복·방향·엄지·접촉과 허용 범위를 학습하고 검증한 상태가 아니다. 관측되지 않은 것을 틀렸다고 단정하지 않는다.
3. 새 사용자·다른 환경에 대한 최종 신뢰성 검증과 더 넓은 단어 확장은 남아 있다. 지금의 개발 점수를 실제 사용자 정답률로 부르지 않는다.
4. 8~9초 수준의 이번 분석도 더 줄일 여지가 있다. 프레임을 버리거나 중요한 짧은 동작을 없애서 속도를 줄이지 않는다.

바로 이어서 하는 일
공개 FIT 자료의 기존 주석에서 동작의 시작·진행·도착을 어디까지 실제로 확인했는지 점검한다. '일부 모습이 나타남'과 '중요 동작을 충분히 수행함'을 구분하는 감독 정보가 있는지 먼저 확인한다.
그 근거로 정상 속도·몸통 흔들림을 허용하면서도 주요 움직임 일부만 남긴 경우를 구분하는 다음 학습/비교를 설계한다. 예전 실패 모델의 epoch·임계값만 반복 조절하거나 특정 단어 예외로 숨기지 않는다. 전체 목표는 계속 진행 중이다.

현재 실제 데모: http://127.0.0.1:8809/
사용자·멘토 촬영 학습/튜닝/선택0. 이번 자동 검증의 카메라 호출·개인 촬영 접근0. 새 다운로드/외부 전송/운영 모델 교체0.
'''.format(analysis=fresh['combined_feedback_latency_seconds'],wall=fresh['wall_seconds_from_button'],head=fresh['partial_order_runtime']['head_seconds'],mp=fresh['base_latency']['mediapipe_seconds'],rgb=fresh['base_latency']['rgb_seconds'])
    (U/'RESULT_KO.txt').write_text(report,encoding='utf8')
    names=['EXECUTION_FROZEN.json','EXECUTION_FROZEN_INITIAL.json','LIVE_RUNTIME_CODE_REVIEW.json','SAVED_CASES_REVIEW.json','LIVE_ORDER_BROWSER_VERIFIED.json','UI_DISPLAY_AMENDMENT.json','LIVE_ORDER_UI_RECHECK.json','LIVE_RESULT_VERIFIED.json']
    out=dict(status='LIVE_PARTIAL_ORDER_DEMO_VERIFIED',url='http://127.0.0.1:8809/',saved_cases=16,
        new_public_jobs=1,fresh_job=fresh['job_id'],full_analysis_seconds=fresh['combined_feedback_latency_seconds'],
        browser_wait_seconds=fresh['wall_seconds_from_button'],partial_head_seconds=fresh['partial_order_runtime']['head_seconds'],
        new_training=0,old_models_changed=False,old_web_default_changed=False,private_training_tuning_selection=False,
        accuracy_gain_claim=False,total_latency_gain_claim=False,whole_goal_complete=False,
        implementation='Codex root + predicate_contract_review',saved_verifier='Codex retention_direction_review',
        root_visual_review='LIVE_ORDER_RESULT_UI_RECHECK.png: readable source/time/order evidence; no absent load-time number; partial extent caveat visible',
        evidence={str(U/n):sha(U/n) for n in names},report=str(U/'RESULT_KO.txt'))
    (U/'FINAL.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Live integration complete; overall goal remains active')
if __name__=='__main__':main()
