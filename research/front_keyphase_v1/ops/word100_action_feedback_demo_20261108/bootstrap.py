from task import *
def main():
 assert not (U/'FROZEN.json').exists()
 old=read(J/'FROZEN.json');protected={}
 for p in list(J.glob('*.json'))+[J/'runtime_candidate.py',J/'feedback_model.py']+list((J/'feedback_viewfix/feedback_runs').glob('*/model.pt'))+list((J/'tcn_runs').glob('*/model.pt')):
  protected[str(p)]=sha(p)
 for p in (O/'word100_integrated_test_20261104').iterdir():
  if p.is_file() and p.suffix in ['.py','.js','.html','.css']:protected[str(p)]=sha(p)
 guides=read(G/'GUIDE_SUPPORT.json')['guides'];ready=read(J/'feedback_viewfix/FEEDBACK_READY.json')
 write(U/'AUTHORIZATION.json',dict(date_kst='2026-11-08',user='계획대로 전부 진행; 주요동작별 시각비교와 짧은 교정, 일부동작 생략 후 손내림도 확인',scope=['existing public4word guide audit','new small supervised feedback heads','phase evidence before DTW and omission checks','isolated local camera-controlled demo and public replay tests'],forbidden=['private footage training/tuning/selection','automatic camera opening','old8812/C/T3F replacement','external transfer','new downloads','git']))
 write(U/'FROZEN.json',dict(status='BEFORE_NEW_TRAINING_AND_EVALUATION',words=old['feedback']['words'],labels=dict(zip(old['classes'],old['labels'])),
  fit_people=list(range(1,10)),development_people=[11,12,13],clips=48,base=dict(tcn='finetune_C_20261013',feedback='pose_transfer_42',selection='fixed first condition from prior timing experiment; not newly selected best score'),
  new_heads=dict(seeds=[42,43,44],epochs=100,lr=.001,weight_decay=.01,batch_size=6,stage_weight=1,shape_weight=.5,background_weight=.2,unknown_ignored=True,backbone_frozen=True),
  phase=dict(threshold=.65,min_evidence_ms=60,max_unknown_fraction=.4,rest_tail_ms=150,allow_repeat_count_variation=True,matching='ordered evidence before alignment; DTW is display correspondence and cannot create an absent event'),
  evaluation=dict(normal='all12DEV; no thresholds/seed selection from them',controls=['speed075','speed125','omit_last_main_then_rest','remove_middle_event','hand_gap_2','hand_gap_30'],synthetic_not_real_learner_error=True,
    gates=dict(normal_definite_omission_max=0,normal_core_confirmation_min_clips=9,controlled_omission_detection_min=.75,no_force_normal_long_missing=True)),
  demo_seed=42,port=8814,body_camera_contract='camera opened only by explicit local button; full frame; same captured frame/PTS for recording and RTMW; no second pass',
  protected=protected,source_pins={str(p):sha(p) for p in [J/'SOURCES.json',G/'INPUT_MANIFEST.json',G/'GUIDE_SUPPORT.json',G/'BLIND_STAGE_ANCHORS.json',J/'feedback_viewfix/FEEDBACK_READY.json']},
  limits=['all current DEV previously exposed','existing visual labels are Codex review, not expert educational correctness','no real learner-error gold exists in this bundle','thumb and detailed face contact not supervised','public upper-mask is not actual camera performance']))
 (U/'PLAN_KO.txt').write_text('''목표: 4단어에서 새 촬영의 주요 동작을 쉬운 문장과 나란한 영상으로 확인하는 연구 데모.
1. 기존 설명/공개영상/시간별 주석을 감사하여 핵심 동작과 선택적 유지·반복을 분리한다. 관절 특징이 없는 엄지·정밀 접촉은 미확인으로 남긴다.
2. 기존 관절 신경망을 보존하고 배경/주요단계 근거와 설명 조건의 물리 특징 차이를 작은 판정부에 지도학습한다. 공개 FIT만 학습, DEV는 고정 평가. 인위적 생략을 실제 수어 오답 정답으로 부르지 않는다.
3. 촬영 중 RTMW+녹화 재사용을 별도 로컬 웹에 연결. 촬영 종료와 의미 완성은 분리. 주요 카드2~3개, 한 번에 한 안내, 해당 손/팔·시간대 강조, 불확실성 회색. 선택적 로컬 음성만 허용.
4. 정상/속도/가림/주요 생략 및 UI·자동종료·PTS·지연을 확인. 없는 동작을 DTW 프레임 늘리기로 채우지 않는다. 정상 경고가 남으면 교정을 확정하지 않고 후보/확인어려움으로 노출한다.
기존 C·T3F/8812/원본·과거동결 보존. 사용자·멘토 촬영 학습·튜닝·선택 금지. 카메라 자동 개방·다운로드·외부전송·Git 없음. 같은 Codex의 구현/별도검산이며 독립사람검증 아님.
''',encoding='utf-8')
 write(U/'CURRENT_EXECUTION.json',dict(status='GUIDE_AUDIT_AND_IMPLEMENTATION',processes=[],training_complete=False,production_changed=False))
 event('scope_frozen',clips=48,protected=len(protected))
if __name__=='__main__':main()
