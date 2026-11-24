"""Record the literature-only shortlist; never acquire or execute a model."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

U = Path(__file__).resolve().parent
ROOT = U.parents[3]
NOW = datetime.now(timezone(timedelta(hours=9))).isoformat()
MARKER = '2026-10-28 수어 직접 평가 기반 모델 후보 2개로 제한'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


report = U / 'SIGN_MODEL_SHORTLIST_KO.txt'
assert report.is_file()
summary = (
    '수어 직접 평가·공식 배포·노트북 조건을 검토해 SHuBERT를 동작 속성 연구 1순위, '
    'SignRep를 비상업 RGB 특징 비교 후보로 제한했다. SHuBERT 표9는 ASL 전체 미세조정 후 '
    '클립 속성 분류이고 SignRep 표4는 동결 영상 검색으로, 한국수어 교정 정확도가 아니다. '
    '두 모델의 CUDA 고정 예제는 Intel 실행 수정·검증이 필요하며 3초 응답은 미검증. '
    '후보 보고만 완료; 새 다운로드·설치·추론·학습 없음, 기존 4단계 전체 미완료.'
)
next_action = (
    'SIGN_MODEL_SHORTLIST_KO.txt의 SHuBERT 우선/SignRep 조건부 비교와 역할·한계를 사용자에게 보고하고 '
    '결과 확인을 기다린다. 최신 지시가 다운로드·학습을 유보했으므로 기존 Qwen/MiniCPM 계획을 실행하지 않는다. '
    '다음 승인이 있으면 정확한 가중치 구성·용량·사용 조건을 확인한 범위에서 CPU/Arc 호환성 소시험부터 '
    '시작하고 기존 공개 KSL에서 관측·단계 정확성과 전체 지연을 비교한다. 공개 초안을 독립 정답으로 '
    '간주하지 않는다. C·T3-F·운영 유지, 사용자/멘토 자료 접근·학습·튜닝·선택 금지.'
)

review = {
    'at': NOW,
    'status': 'LITERATURE_SHORTLIST_COMPLETE_USER_REVIEW_REQUIRED_NO_EXECUTION',
    'request_scope': 'Sign-specific evidence, laptop fit and 1-2 candidates; defer downloads/training until user review.',
    'report': str(report),
    'hardware_source': str(U / 'HARDWARE_CHECK.json'),
    'decision': {
        'first_research_candidate': 'SHuBERT',
        'conditional_comparator': 'SignRep',
        'adopted_production_model': None,
        'both_models_required': False,
        'criterion': 'Direct sign attribute evidence first; released RGB representation as comparator. Not a generic-video leaderboard ranking.',
    },
    'evidence': {
        'SHuBERT': {
            'paper': 'https://aclanthology.org/2025.acl-long.1397.pdf',
            'table': 'Appendix B Table 9, ASL Citizen column',
            'metrics_percent': {'handshape': 70.80, 'major_location': 90.22, 'repeated_movement': 89.93, 'path_movement': 79.42},
            'scope': 'Full fine-tuning with 16 heads; temporal average and clip-level attribute classification. Not pretrained zero-shot or frame-level KSL feedback.',
            'repository': 'https://github.com/ShesterG/SHuBERT',
            'release_state': 'Pretrained SHuBERT and DINO links public; actual artifact contents, hashes, full license and 16-attribute head availability unverified.',
            'runtime': '86M contextual encoder excludes DINO extraction; supplied scripts hardcode CUDA. Intel/CPU end-to-end unmeasured.',
        },
        'SignRep': {
            'paper': 'https://arxiv.org/html/2503.08529v1',
            'table': 'Table 4 weighted frozen dictionary retrieval',
            'recall_at_1_percent': {'ASL_Citizen': 49.95, 'WLASL2000': 29.92, 'NMFs_CSL': 63.04},
            'scope': 'No downstream training for retrieval. Not phase or corrective-feedback accuracy.',
            'release': 'https://github.com/ryanwongsa/SignRep/releases/tag/v0.0.1',
            'asset_listing': 'https://github.com/ryanwongsa/SignRep/releases/expanded_assets/v0.0.1',
            'asset': 'ckpt.pt',
            'listed_size_mb': 212,
            'listed_sha256': 'f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b',
            'hash_locally_verified': False,
            'license': 'CC BY-NC-SA 4.0; release expressly noncommercial research/education',
            'runtime': '16-frame RGB example hardcodes CUDA; CPU/Arc port, all-window latency and memory unverified.',
        },
    },
    'limitations': [
        'No public end-to-end KSL pedagogical-feedback result established for either candidate.',
        'Neither directly reads Korean textual instructions as a VLM; guide/feature/phase/DTW links remain to implement and validate.',
        'Existing public observation drafts are not independent semantic or expert truth.',
        'No 3-second laptop latency promise; no from-scratch foundation pretraining proposed.',
        'Foreign sign pretraining is not adoption of foreign sign rules as Korean sign correctness.',
    ],
    'separate_reviewers': ['Codex /root/shubert_fit_audit', 'Codex /root/sign_candidate_audit'],
    'review_type': 'Read-only primary-literature and public-code cross-review; not independent human sign-language annotation or runtime validation.',
    'download_approved': False,
    'download_started': False,
    'installations': 0,
    'inference_runs': 0,
    'training_runs': 0,
    'private_data_used': False,
    'production_changed': False,
    'next': next_action,
}
save_json(U / 'SIGN_MODEL_SHORTLIST.json', review)

plan_path = U / 'MODEL_TRIAL_PLAN.json'
plan = read_json(plan_path)
if 'superseded_vlm_trial' not in plan:
    # Keep the old concrete recipe as history, never silently apply it to non-VLMs.
    fields = ('source', 'repository_size_gb', 'license', 'alternative_considered', 'input',
              'prompt_conditions', 'model_settings', 'metrics', 'model_pass',
              'recommended_first_trial', 'efficiency_comparator', 'review')
    plan['superseded_vlm_trial'] = {key: plan.pop(key) for key in fields if key in plan}
plan.update({
    'at': NOW,
    'status': review['status'],
    'candidate': None,
    'recommended_first_trial': 'SHuBERT',
    'conditional_comparator': 'SignRep',
    'review': 'SIGN_MODEL_SHORTLIST.json',
    'reason': 'Current request prioritizes directly measured sign attributes/representations over general-video VLM scores.',
    'execution': 'No execution approved by this shortlist. CPU/Intel Arc feasibility and exact artifacts must be checked after user review.',
    'evaluation_recipe_status': 'LITERATURE_PLAN_ONLY_NOT_FROZEN_FOR_EXECUTION',
    'legacy_vlm_runner_applicable': False,
    'download_approved': False,
    'download_started': False,
    'actual_inference_runs': 0,
    'next': next_action,
})
save_json(plan_path, plan)

state_path = U.parent / 'STATE.json'
state = read_json(state_path)
state['updated_date_kst'] = NOW
state['current_stage'] = 'GUIDE_VIDEO_SIGN_SPECIFIC_SHORTLIST_USER_REVIEW'
state['next_single_action'] = next_action
unit = state['word100_guide_video_feedback_20261028']
unit.update({
    'status': 'PARTIAL_COMPLETE_SIGN_SPECIFIC_SHORTLIST_NO_ACQUISITION_APPROVAL',
    'at': NOW,
    'summary': summary,
    'next': next_action,
    'sign_model_shortlist': str(U / 'SIGN_MODEL_SHORTLIST.json'),
    'current_selection_report': str(report),
    'all_four_steps_complete': False,
})
unit['steps'][1]['remaining'] = 'SHuBERT/SignRep의 역할·문헌 근거·하드웨어 제약 검토 완료; 사용자 검토 전 다운로드·새 추론·학습하지 않음.'
save_json(state_path, state)

context_path = ROOT / 'WordsInSilence_PROJECT_CONTEXT.md'
context = context_path.read_text(encoding='utf-8-sig')
assert MARKER not in context, 'This documentation decision is already recorded; do not duplicate it.'
for heading, text in (
    ('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인', summary),
    ('## 9. 다음 단일 작업', next_action),
):
    entry = f'\n\n### {MARKER}\n{NOW}\n{text}\n근거: research/front_keyphase_v1/ops/word100_guide_video_feedback_20261028/SIGN_MODEL_SHORTLIST_KO.txt.\n'
    assert context.count(heading) == 1
    context = context.replace(heading, heading + entry, 1)
context_path.write_text(context, encoding='utf-8')

handoff = (
    '# WordsInSilence 최신 인수인계\n\n' + NOW + '\n\n'
    '목표: ' + unit['goal'] + '\n\n'
    + summary + '\n\n'
    '사양: Samsung 960XHA / Core Ultra 7 258V / RAM 32GB / Intel Arc 140V 내장 GPU / AI Boost NPU. '
    '확인한 Python의 Torch만 CPU 전용; 전체 GPU 부재로 해석하지 않는다.\n\n'
    '기존 완료: 공개 12표현36영상 연결·69관측값·24쌍 재생·소프트웨어 검산. '
    '의미 단계/교육 피드백 모델 완료나 한국수어 정확도 개선이 아니다.\n\n'
    '다음: ' + next_action + '\n\n'
    '현재 작업단위: word100_guide_video_feedback_20261028\n'
    '최신 선정 근거: SIGN_MODEL_SHORTLIST.json / SIGN_MODEL_SHORTLIST_KO.txt. '
    '기존 RESULT_KO.html은 앞선 자료 연결·재생 결과이며 새 모델 시험 결과가 아니다.\n'
    'Qwen/MiniCPM 계획은 MODEL_TRIAL_PLAN.superseded_vlm_trial에 이력으로 보존. '
    'run_local_vlm.py는 새 두 후보에 적용할 실행기가 아니다.\n'
    '별도 검토: Codex shubert_fit_audit 및 sign_candidate_audit의 문헌·코드 교차검토. '
    '전문가 정답이나 실행 검증으로 부르지 않는다. 현재 실행 중 모델 프로세스 없음.\n'
)
(ROOT / 'NEW_CHAT_HANDOFF.md').write_text(handoff, encoding='utf-8')

# Documentation consistency only. Do not execute any model/test-data pipeline.
fresh_plan = read_json(plan_path)
fresh_state = read_json(state_path)
assert fresh_plan['candidate'] is None and not fresh_plan['download_approved']
assert fresh_plan['recommended_first_trial'] == 'SHuBERT'
assert fresh_plan['conditional_comparator'] == 'SignRep'
assert not fresh_state['word100_guide_video_feedback_20261028']['all_four_steps_complete']
assert fresh_state['next_single_action'] == next_action
assert context.count(MARKER) == 2
print(json.dumps({'status': 'DOCUMENTATION_CHECKS_PASS', 'at': NOW,
                  'model_downloads': 0, 'model_inference': 0, 'training': 0,
                  'report': str(report)}, ensure_ascii=False))
