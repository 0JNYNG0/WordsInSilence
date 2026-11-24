"""Read the explicitly reported latest submission only; no model execution/tuning."""
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime
import numpy as np

ROOT = Path(r'C:\MyUnityProjects\WordsInSilence')
OPS = ROOT/'research/front_keyphase_v1/ops'
UNIT = Path(__file__).resolve().parent
BASE = OPS/'word100_multipeople_live_feedback_20261101'
JOB = BASE/'sessions/d03ff771938d4b39cef67c8d58ac86ca/245643e8758f1eb697683bdcb2369dbb'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

protected = [JOB/'request.json', JOB/'REQUEST_TIMING.json', JOB/'server_result.json',
             JOB/'adapter/COMPLETE.json', JOB/'adapter/CALIBRATION.json',
             JOB/'adapter/observations.npz', JOB/'adapter/analysis.npz',
             BASE/'adapter.py', BASE/'engine.py']
before = {str(p): sha(p) for p in protected}
request = read(JOB/'request.json')
cal = read(JOB/'adapter/CALIBRATION.json')
complete = read(JOB/'adapter/COMPLETE.json')
timing = read(JOB/'REQUEST_TIMING.json')
assert request['word'] == 'WORD1133' and request['isolated_user_submission']
assert request['training_allowed'] is False
with np.load(JOB/'adapter/observations.npz', allow_pickle=False) as z:
    xy, observed, usable, ts = [z[k].copy() for k in ('xy_px','observed','usable','timestamp_us')]
with np.load(JOB/'adapter/analysis.npz', allow_pickle=False) as z:
    final_mask = z['mask'].copy()
    predictions = z['probabilities'].copy()
    rgb = z['rgb'].copy()
n = len(ts)
inside = np.isfinite(xy).all(-1) & (xy[...,0]>=0) & (xy[...,0]<1280) & (xy[...,1]>=0) & (xy[...,1]<720)
supported = observed & usable & inside
hands = np.column_stack([supported[:,s:s+21].any(1) for s in (33,54)])
detected_ids = np.flatnonzero(hands.any(1))
span = slice(int(detected_ids[0]),int(detected_ids[-1])+1)
counts = {name: int(supported[:,idx].sum()) for name,idx in
          [('left_shoulder',11),('right_shoulder',12),('left_elbow',13),('right_elbow',14)]}
counts.update(left_hand_frames=int(hands[:,0].sum()),right_hand_frames=int(hands[:,1].sum()),
              any_hand_frames=int(hands.any(1).sum()),both_hands_frames=int(hands.all(1).sum()),
              detector_span_frames=int(span.stop-span.start),
              any_hand_in_detector_span=int(hands[span].any(1).sum()),
              both_hands_in_detector_span=int(hands[span].all(1).sum()),
              hands_first_second=int(hands[ts-ts[0]<=1_000_000].any(1).sum()))
bottom = math.floor(cal['median_shoulder_center'][1]+cal['ratio']*cal['shoulder_width'])
checks = {
    'stored_bottom_matches_arithmetic': bottom == cal['proposed_bottom'] == 865,
    'boundary_outside_source': bottom > cal['source_height'],
    'only_boundary_rejection': complete['unavailable_reasons']==['UPPER_BODY_BOUNDARY_UNAVAILABLE'],
    'length_not_rejection': n>=16 and 600_000<=int(ts[-1]-ts[0])<=20_000_000,
    'all_features_disabled_by_gate': not final_mask.any(),
    'no_finite_model_prediction': not np.isfinite(predictions).any(),
    'no_finite_rgb_feature': not np.isfinite(rgb).any(),
    'no_rgb_windows_saved': not (JOB/'adapter/rgb_windows.npz').exists(),
    'source_and_code_unmodified': before=={str(p):sha(p) for p in protected},
}
assert all(checks.values()), checks
result = dict(status='DIAGNOSED_NO_RUNTIME_CHANGE', at=datetime.now().astimezone().isoformat(),
              job=str(JOB), purpose='User-requested bug diagnosis only; excluded from training/tuning/selection',
              frames=n,duration_seconds=float(ts[-1]-ts[0])/1e6,
              rejected_request_seconds=timing['analysis_seconds'],
              predicted_lower_boundary=bottom,source_height=cal['source_height'],
              observation_counts=counts,checks=checks,pins=before,
              limitations=['Detector support is not landmark accuracy or action correctness.',
                           'Detector span is not an independently annotated main action.',
                           'Other GPU program usage is user-reported, not a measured causal finding.'],
              training_runs=0,new_inferences=0,threshold_changes=0,live_code_changes=0)
with (UNIT/'DIAGNOSIS.json').open('x',encoding='utf8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
summary = ('사용자 요청 최신 잘하다 촬영 로그만 진단. 151frame/4.989초,6.51초에 입력 거절. '
           '첫1초 손목휴식 기준 실패 후 안정 어깨로 산정한 하단865가 화면높이720밖이라는 이유로 '
           '전체69특징 사용마스크를 false처리,SignRep/판정부/DTW 미실행. 실제손미검출/영상길이실패와 다름. '
           '촬영 잘못 또는 동작 틀림 판정 아님. 사용자촬영은 버그진단만,학습/튜닝/선택/재추론0,기존웹코드변경0.')
next_action = ('새 공개자료 기반 촬영범위 호환성 시험을 먼저 설계: 추정 신체기준/실제 화면범위/특징별 관측을 분리하고, '
               '기준 하단이 화면 밖이라는 이유만으로 전체 비교를 막는 정책을 별도 후보로 검증. '
               '초기 손내림 화면밖 조건/주요동작 손누락/정상교차/짧은변화에서 관측부족을 정상으로 채우거나 비교분모 축소 금지. '
               '개인촬영으로 계수/임계값 맞추기 금지. 오류메시지는 실제 실패원인별로 구분. '
               'GPU지연은 다른프로그램 사용가능성을 함께 기록하고 통제된 부하 상태의 제한 공통시험으로 분리. '
               '기존8802/C/T3F보존. 이번은 진단이며 수정적용/신규평가/추가학습 없음.')
report = ('최신 촬영 입력 거절 진단\n\n'+summary+'\n\n'
          f'화면 안/검출 usable 기준: {json.dumps(counts,ensure_ascii=False)}\n'
          '주의: 검출점 존재는 정확한 손모양 확인이나 수어 정답의 증거가 아니다.\n'
          '원인: 공개 원본영상의 하단 범위를 정하던 offline 정책이 사용자 웹 촬영의 필수 통과조건으로 재사용됐다. '
          '어깨는 안정적으로 관측됐지만, 내려놓은 손목을 볼 수 없고 추정 하단이 화면 아래이면 전체 중단한다. '
          '이는 필요한 주요동작이 보이는지와 별개의 조건이다. 현재 경고문은 이를 영상길이/상반신 부족으로 뭉뚱그려 보여준다.\n'
          '6.51초는 거절까지의 시간이며 전체 피드백 완료 속도가 아니다. GPU 타프로그램 부하는 사용자 제보로 기록, 원인 확정 아님.\n\n'
          '다음: '+next_action+'\n검증: 저장배열·원장·코드분기 및 산술 9검사 PASS. Codex 자체 검증, 별도 사람/에이전트 검증 없음.\n')
(UNIT/'RESULT_KO.txt').write_text(report,encoding='utf8')
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'; text=p.read_text(encoding='utf-8-sig')
for marker,title,body in [
    ('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','최신 웹 촬영 입력 거절 원인 확인',summary),
    ('## 9. 다음 단일 작업','실제 화면 범위와 신체 기준의 구분',next_action)]:
    assert marker in text
    text=text.replace(marker,marker+'\n\n### 2026-11-01 '+title+'\n'+body,1)
p.write_text(text,encoding='utf8')
p=OPS/'STATE.json'; state=read(p)
state['current_stage']=result['status']; state['next_single_action']=next_action
state[UNIT.name]=dict(status=result['status'],summary=summary,result=str(UNIT/'RESULT_KO.txt'),
                     next_action=next_action,user_data_use='diagnosis_only',training=False,runtime_changed=False)
p.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+result['at']+'\n\n'+summary+'\n\n다음: '+next_action+
    '\n\n근거: '+str(UNIT/'DIAGNOSIS.json')+'\n이전7경고/속도 후보는 word100_warning_latency_20261101 참고. '
    '그 단위는 종료/8803종료. 기존8802 요청6.51초는 입력거절로 전체속도 개선증거 아님. '
    '기존 공개완료실험 재실행 금지. 이번에 개인영상 재추론/학습/후보선택/외부전송/카메라/모델다운로드/Git 없음.\n',encoding='utf8')
assert read(OPS/'STATE.json')[UNIT.name]['runtime_changed'] is False
print(json.dumps({k:result[k] for k in ['status','frames','duration_seconds','rejected_request_seconds','observation_counts','checks']},ensure_ascii=False))
