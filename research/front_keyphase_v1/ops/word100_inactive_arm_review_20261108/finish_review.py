from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib, shutil

U=Path(__file__).resolve().parent; O=U.parent; R=O.parents[2]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not (U/'RESULT.json').exists()
    ver=read(U/'VERIFICATION.json'); assert ver['status']=='PASS'
    cf=read(U/'COUNTERFACTUAL.json'); crop=read(U/'CROP_PROBE.json'); attr=read(U/'ATTRIBUTION.json')
    now=datetime.now(timezone(timedelta(hours=9))).isoformat()
    result=dict(status='PUBLIC_FRAMING_FAILURE_REPRODUCED_CAUSE_REVIEW_COMPLETE_FIX_NOT_DEPLOYED',at=now,
        model='Existing demo finetune_C_20261013 fixed; preserved original C/T3F are separate',
        implemented_by='Codex local diagnostics',verified_by=ver['verifier'],
        main_findings=[
            'In-frame RTMW coordinates enter TCN/feedback even when physical visibility/identity is unverified.',
            'Gray skeleton color is card uncertainty/default display color, not exclusion from model inputs.',
            'Current saved public full-view predictions: all four words 3/3; Sunjong does not fail universally.',
            'Public upper-body RGB crop recreated unseen-left-hand false points and Sunjong rank46.',
            'Diagnostic removal of left wrist/hand restored that Sunjong clip to rank1; hip-only removal did not.',
            'Projecting original full-view coordinates into identical crop yields rank1 for all four; new detector outputs, not coordinate framing alone, are implicated.',
            'Blind left-hand removal harms the other three two-hand examples and is not a proposed fix.',
            'Previous upper_mask test hides coordinates after full-frame detection; it does not reproduce detector behavior on actual cropped pixels.'
        ],
        scope=dict(saved_public_development_clips=12,coordinate_interventions=36,new_public_cropped_clips=4,new_rtmw_frames=505,diagnostic_attribution_forwards=16,
            actual_user_causal_analysis=False,private_scalar_status_only=True,private_pixels_or_keypoints_read=False,
            training=False,threshold_tuning=False,model_or_candidate_selection=False,production_changed=False,camera_opened_by_agent=False),
        summary=cf['summary'],crop_ranks={r['label']:r['new_tcn']['truth_rank'] for r in crop['rows']},
        limitations=[
            'Four exposed development crop clips, one framing condition; no population accuracy claim.',
            'No user recording reviewed; this is a public reproduction, not proof of every user misclassification cause.',
            'Only three Sunjong RGB frames were visually reviewed by the same Codex; geometric proxy counts are not human visibility labels.',
            'Diagnostic hand removal uses knowledge of which hand is absent; no general automatic guard has passed regression tests.',
            'Five-frame perturbation preserved top1 here, not proof that all brief jitter is harmless.',
            'Feedback shares the input path, but feedback/auto-stop effects were not quantified in this review.',
            'No new model speed measurements or deployed recognition improvement.'
        ],
        next='Validate a shared observation-identity gate on public full/cropped RGB using arm-chain, hand shape consistency and temporal evidence; distinguish actual missing hands from true crossings. Preserve raw input; no word-specific deletion or score forcing. Only then consider public training with the validated input contract.',
        verification=str(U/'VERIFICATION.json'),report=str(U/'RESULT_KO.txt'),example=str(U/'visible_example.jpg'))
    write(U/'RESULT.json',result)
    report='''순종 오인식·보이지 않는 반대 손의 관절 튐 검토

결론
사용자가 지적한 현상은 인식에 영향을 줄 수 있다. 공개 영상에서도 비슷한 문제를 재현했다.
현재는 원인을 확인한 단계이며, 인식 보완을 테스트 화면에 적용한 상태는 아니다.

무엇이 잘못 연결돼 있었나
RTMW가 화면 안에 좌표를 내놓으면, 그 손이 실제로 보이는지 또는 반대 손을 중복 추정했는지 확정하지 않아도 TCN과 피드백 입력에 들어간다.
화면의 회색은 비교 카드의 불확실/기본 표시색이다. 회색 관절을 모델 계산에서 제외했다는 뜻이 아니다.
따라서 화면 밖의 왼손을 화면 안의 오른손 주변으로 잘못 추정하면, 한 손 동작이 잘못된 양손 동작처럼 입력될 수 있다.
근거: adapter.py의 available(유한·화면 안 좌표), consumer.py의 tcn280 전달, app.js의 색상 분기.

공개 자료로 확인한 결과
1. 저장된 기존 공개 개발 결과: 순종·잘하다·조상·이집트 각각 3/3이 1위였다. 새 점수를 얻으려고 기존 추론을 다시 돌리지 않았다.
2. 공개 순종 3편의 좌표만 바꾼 진단:
   - 왼손과 왼손목을 없는 값으로 표시: 3/3이 계속 1위.
   - 왼팔·왼손을 오른팔·오른손에 영상 내내 붙임: 0/3이 1위.
   - 주요 구간 가운데 5프레임만 붙임: 3/3이 계속 1위.
   이는 오류의 영향을 분리한 인공 시험이다. 모든 짧은 오류가 무해하다는 뜻은 아니다.
3. 공개 4편을 웹캠처럼 팔꿈치 아래에서 잘라 640×480으로 만들고 RTMW로 새로 관측했다. 총505프레임을 모두 사용했다.
   원본에서는 네 편 모두 정답이 1위였지만, 새 관측에서는 순종46위·잘하다2위·조상1위·이집트1위였다.
   순종의 확인 프레임63·80·98에서, 실제 왼손은 화면 밖인데 왼손 추정선이 오른손 주변과 빈 공간으로 뻗는 것이 보였다.
4. 같은 잘린 순종 입력에서 왼손·왼손목의 잘못된 값만 진단용으로 제외하자 1위로 돌아왔다.
   골반 값만 제외하면41위였고, 원본에서 얻은 좌표를 같은 잘린 화면 좌표계로 옮긴 조건은1위였다.
   따라서 이 표본은 단순히 촬영 화면이 커지고 작아진 것보다, 잘린 RGB에서 새로 생긴 잘못된 관절 추정의 영향이 크다는 근거가 된다.
   반면 잘린 나머지 세 편의 왼손을 같은 방식으로 지우면 정답순위가4·9·11위로 나빠졌다. 왼손을 무조건 지우는 것은 해결책이 아니다.

이전 시험에서 놓친 조건
기존 상반신 시험은 전체 영상을 먼저 관절 검출한 뒤, 아래쪽 좌표를 사용하지 않는 방식이었다.
실제 웹캠처럼 영상 자체에 손이 보이지 않을 때 검출기가 어떤 좌표를 내놓는지는 그 시험으로 충분히 확인되지 않았다.
이번에는 실제 RGB 화면 범위를 바꿔 검출부터 다시 진행하여 이 차이를 확인했다.

다음 보완 방향
첫째, 단어 이름과 무관하게 어깨→팔꿈치→손목→손의 연결과 앞뒤 움직임을 함께 검사해, 실제 손과 반대 손에 잘못 붙은 추정을 구분한다.
둘째, 확인할 수 없는 관절은 원본과 분리해 불확실 상태로 전달한다. 잘못 붙은 좌표를 부드럽게 이어 붙여 정상 동작으로 만들지 않는다.
셋째, 화면 밖 대기 손·한 손 동작·실제 양손 교차·재등장·짧은 손모양 변화에서 정상 관절을 지우지 않는지 공개 자료로 함께 검사한다.
넷째, 입력 기준이 검증된 다음에만 같은 기준의 공개 학습자료로 TCN/피드백 보완 필요성을 판단한다. 규칙을 만드는 자료와 확인자료를 분리한다.
순종 전용 예외, 목표 단어 점수 올리기, 양손 동작에서 왼손 일괄 삭제는 하지 않는다.

확인 범위와 남은 한계
새 학습·운영 교체 없음. 기존 C·T3-F, 8812와 현재8814 모델·화면 코드 보존. 사용자 영상/관절 배열 열람·학습·튜닝 없음. 상태 API의 처리 스칼라만 확인했다.
실제 사용자 촬영의 모든 원인을 확정한 것은 아니다. 공개 개발 자료12편과 그중4편의 새 촬영 범위 조건을 이용한 원인 진단이다.
505프레임 중 사람이 보이는 손에 정답을 붙인 평가가 아니다. 순종3프레임은 같은 Codex가 RGB로 재검토했고, 나머지 화면 안/밖 집계는 기존 좌표에 근거한 기하학적 비교다.
저장된 결과의 산술·시간·해시를 별도 코드로148개 확인, 모두 통과. 보호 해시75개 유지. 같은 Codex의 별도 검산이며 독립 사람/한국수어 전문가의 정답 검증은 아니다.
이 결과로 인식 개선이 완료됐거나 모든 사용자에게 정확하다고 주장하지 않는다.

근거
FROZEN.json / COUNTERFACTUAL.json: 공개12편 좌표 변화 영향.
CROP_FROZEN.json / CROP_PROBE.json: 공개4편 실제 RGB 범위 변경.
ATTRIBUTION_FROZEN.json / ATTRIBUTION.json: 같은 입력의 원인 분리16조건 결과.
VERIFICATION.json: 재추론 없는 별도 검산.
visible_example.jpg: 공개 순종 프레임63. 왼쪽 실제 화면, 오른쪽 RTMW. 보라색=왼손, 초록색=오른손.
'''
    (U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
    before=U/'before'; before.mkdir(exist_ok=True)
    for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
        assert not (before/p.name).exists(); shutil.copy2(p,before/p.name)
    status='''### 2026-11-08 순종 오인식·화면 밖 반대 손 진단 — 공개 재현, 보완 미적용

word100_inactive_arm_review_20261108. 사용자 4단어 촬영 후 순종 1위 실패와 반대 손 관절 튐을 보고했다. 코드상 RTMW의 화면 안 좌표는 가시성/좌우 동일성 확인 없이 TCN·피드백에 들어가며, 회색 표시는 입력 제외가 아니다. 기존 저장 공개12DEV(4단어×3명)는 각3/3 정답1위. 순종3편 좌표 진단에서 왼손·손목 없음3/3, 오른손에 반대 팔·손 전체시간 중복0/3,5프레임 중복3/3. 새 RGB 상반신 crop4편505프레임에서 순종46위/잘하다2위/조상1위/이집트1위. 순종 crop의 잘못된 왼손·손목만 진단용 제외 시1위,골반만제외41위,원본좌표를동일crop에투영시1위. 정상 양손 세 편에 왼손 삭제는4·9·11위로 악화하므로 범용 수정으로 채택하지 않았다.

기존 upper_mask는 전체 RGB 관측 후 좌표 제한이어서, 실제 손이 영상 밖일 때의 잘못된 추정을 충분히 시험하지 못했다. 공개 순종63/80/98프레임 RGB 재검토에서 화면 밖 왼손의 추정선이 오른손과 빈 공간에 연결되는 현상을 확인. 나머지 가시성 집계는 기하 proxy이며 사람 정답 아님. 원인 진단 완료, 자동 관측 판별/인식 수정 미완. 같은 Codex 별도 저장결과 산술148검사/보호해시75PASS. 사용자 영상·좌표 읽기0(상태 스칼라만),학습/튜닝/모델선택/웹수정0. 기존 C·T3-F/8812/8814 모델 보존. 근거 RESULT_KO.txt/RESULT.json/VERIFICATION.json. 실제 사용자 오인식의 모든 원인을 확정한 것은 아님.

'''
    nxt='''### 2026-11-08 화면 밖 손의 관측·입력 구분 보완

다음 단일 작업은 어깨–팔꿈치–손목–손 및 앞뒤 시간 근거로, 보이지 않는 반대 손을 실제 움직이는 손에 잘못 붙인 입력을 구분하는 범용 후보를 공개 자료에서 검증하는 것이다. 같은 RGB 상반신 조건과 정상 전체화면, 정상 양손/교차/재등장/짧은 변화 보존을 함께 확인한다. 기존 원본을 보존하고 불확실한 부분만 별도 상태로 전달한다. 순종별 예외/왼손 일괄 삭제/평가 점수에 맞춘 규칙 선택 금지. 현재 진단에 노출된 DEV를 새 독립 평가라고 부르지 않는다. 입력 기준 통과 전 웹 반영이나 재학습 개선을 주장하지 않는다. 8814 연구 데모는 그대로, 자동 교정/누락 정오 검증은 계속 미완. 사용자·멘토 학습/튜닝/선택 금지. 이번 진단36+4+16조건 완료, 재실행 금지.

'''
    p=R/'WordsInSilence_PROJECT_CONTEXT.md'; text=p.read_text(encoding='utf-8-sig')
    h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'; h9='## 9. 다음 단일 작업\n'
    assert text.count(h8)==text.count(h9)==1
    p.write_text(text.replace(h8,h8+'\n'+status,1).replace(h9,h9+'\n'+nxt,1),encoding='utf-8')
    state=read(O/'STATE.json')
    state['current_stage']='RTMW_INACTIVE_ARM_PUBLIC_FAILURE_DIAGNOSED_FIX_PENDING'
    state['updated_date_kst']='2026-11-08'
    state['next_single_action']='공개 RGB 상반신/전체화면에서 화면 밖 손의 잘못된 좌우 연결을 구분하는 공통 입력 보완을 검증한다. 정상 양손·교차·재등장·짧은 변화 보존, 단어별 예외/왼손 일괄 삭제/사용자 자료 학습 튜닝 금지. 이번 원인 진단 완료결과 재추론 금지.'
    state={'word100_inactive_arm_review_20261108':dict(status=result['status'],report=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFICATION.json'),public_clips=12,cropped_clips=4,cropped_frames=505,model_changed=False,web_changed=False,training=False,private_pixels_or_keypoints_read=False),**state}
    state['word100_webcam_recording_fix_20261108']['followup']='User reports completing four word tests; next issue is recognition/false opposite-arm coordinates. Scalar server status done/error null observed; no inference about all camera hardware.'
    write(O/'STATE.json',state)
    (R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+status+nxt+
        '서버는 기존8814(http://127.0.0.1:8814/), 마지막 확인 PID29264. 이번에 재시작/코드변경 없음. 재개 시 상태 확인 후 중복 실행하지 않는다. 카메라는 사용자 버튼으로만.\n\n'+
        '직전 녹화기 PTS/실패전파/MJPEG 수정은 word100_webcam_recording_fix_20261108에 보존. 사용자는 이후4단어 테스트를 마쳤다고 보고했고, 상태 API는 done/error null이었다. 모든 환경의 녹화 안정성 검증 완료를 뜻하지 않는다. 현재 RTMW TCN 후보는 finetune_C_20261013이며 원본 C·T3-F와 별개다. 피드백은 관절전용 시간합성곱+GRU 기반; 런타임 SignRep RGB 분석 없음. 자동 정오/교정은 아직 검증되지 않았다.\n',encoding='utf-8')
    write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE',running=False,training=False,report=str(U/'RESULT_KO.txt')))
    check={}
    for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
        content=p.read_text(encoding='utf-8'); check[str(p)]=dict(contains_unit=U.name in content,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    assert all(r['contains_unit'] for r in check.values())
    assert read(O/'STATE.json')['current_stage']=='RTMW_INACTIVE_ARM_PUBLIC_FAILURE_DIAGNOSED_FIX_PENDING'
    write(U/'DOCUMENT_SYNC.json',dict(status='PASS',files=check))
    print(json.dumps(dict(report=str(U/'RESULT_KO.txt'),checks=ver['check_count'],doc_sync='PASS'),ensure_ascii=False))

if __name__=='__main__': main()
