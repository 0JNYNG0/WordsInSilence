"""Record read-only hardware findings and amend the provisional model plan.

No installation, downloading, inference, or protected input access.
"""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

U = Path(__file__).resolve().parent
O = U.parent
R = U.parents[3]
NOW = datetime.now(timezone(timedelta(hours=9))).isoformat()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

hardware = {
    'at': NOW,
    'method': 'Read-only Get-CimInstance and Get-PnpDevice; local Python import probes',
    'initial_probe_failure': 'Sandbox CIM resource access unavailable; repeated same read-only hardware query with approved elevated execution; success.',
    'manufacturer': 'SAMSUNG ELECTRONICS CO., LTD.', 'model': '960XHA',
    'cpu': 'Intel(R) Core(TM) Ultra 7 258V', 'cores': 8, 'logical_processors': 8,
    'installed_memory_class_gb': 32, 'total_physical_memory_bytes': 33834082304,
    'free_physical_memory_kib_at_probe': 11914772,
    'display_adapters': [{'name': 'Intel(R) Arc(TM) 140V GPU (16GB)', 'driver': '32.0.101.8626', 'status': 'OK', 'wmi_AdapterRAM_raw_bytes': 2147479552}],
    'memory_caution': 'Integrated GPU shares system memory. Neither device-name 16GB nor WMI AdapterRAM is verified dedicated VRAM. Do not add 16GB to system RAM.',
    'npu': {'name': 'Intel(R) AI Boost', 'status': 'OK'},
    'checked_python': 'C:/Users/local-user/AppData/Local/Programs/Python/Python313/python.exe',
    'torch_version': '2.12.0+cpu', 'torch_cuda_available': False, 'torch_xpu_available': False, 'torch_xpu_count': 0,
    'packages_found': {'torch': True, 'openvino': False, 'onnxruntime': False, 'transformers': False, 'torch_directml': False},
    'scope_caution': 'Package findings apply to this interpreter only, not all environments or all previous experiments.',
    'hardware_gpu_present': True, 'actual_vlm_gpu_execution_verified': False,
    'model_inference_runs': 0, 'installation_performed': False,
}
save(U/'HARDWARE_CHECK.json', hardware)

review = {
    'at': NOW,
    'status': 'HARDWARE_VERIFIED_SHORTLIST_PROVISIONAL_NO_ACQUISITION_APPROVAL',
    'original_proposal': 'Qwen/Qwen3-VL-2B-Instruct',
    'original_rationale': ['Video+text input and timestamp-aware architecture fit observation-to-guide task', 'Compact open weights and local execution path'],
    'original_gap': 'Insufficient recent-alternative comparison and actual Intel hardware check; no KSL or end-to-end latency evidence.',
    'recommended_first_trial': 'Qwen/Qwen3.5-2B',
    'efficiency_comparator': 'openbmb/MiniCPM-V-4.7-1B',
    'max_initial_candidates': 2,
    'original_model_role': 'Documented original proposal, no automatic third download',
    'qwen3_5': {
        'source': 'https://huggingface.co/Qwen/Qwen3.5-2B',
        'file_index': 'https://huggingface.co/Qwen/Qwen3.5-2B/tree/main',
        'license': 'Apache-2.0', 'repository_size_gb_at_review': 4.57,
        'vendor_video_mme_without_subtitles': {'Qwen3-VL-2B': 62.1, 'Qwen3.5-2B_thinking': 69.0},
        'benchmark_cautions': ['Vendor general-video benchmark, not our KSL results', 'New-model number is thinking; non-thinking video number absent', 'Not all metrics improve; MVBench 64.5 to 64.9, MMVU 48.9 to 48.6', 'No valid transfer of these numbers to hand precision or 3-second response'],
    },
    'minicpm': {
        'source': 'https://huggingface.co/openbmb/MiniCPM-V-4.7-1B',
        'license': 'Apache-2.0',
        'rationale': 'Early visual token compression and local deployment emphasis make a useful efficiency challenger.',
        'cautions': ['Name 1B is not verified total footprint; model page safetensors summary says 2B', '4x/16x visual compression must be checked for lost finger detail', 'Official comparison to Qwen3.5-0.8B does not prove superiority to Qwen3.5-2B', 'Exact chosen weight artifact size must be disclosed before acquisition'],
    },
    'other_reviewed': {
        'Gemma4_E2B': {'source': 'https://ai.google.dev/gemma/docs/core/model_card_4', 'decision': 'Reserve, not rejected as inaccurate. 2.3B effective versus 5.1B with embeddings means names do not imply equivalent memory; limit first experiment to two contrasting candidates.'},
        'SmolVLM2_500M': {'source': 'https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct', 'decision': 'Small English-scoped prior alternative alone did not establish optimality of original Qwen.'},
    },
    'runtime': {
        'first_deployment_route': 'Isolated OpenVINO GenAI CPU vs Intel GPU inference; compatibility smoke check before data run',
        'source': 'https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/',
        'listed_vlm_architectures': ['Qwen3-VL', 'Qwen3.5', 'MiniCPM-V-4.7'],
        'cautions': ['Support listing is not a successful test on this notebook', 'Qwen3.5 export currently requires transformers==5.2; MiniCPM4.7 requires >=5.18.0, use separate environments', 'NPU is not presumed to support the complete VLM pipeline', 'No driver change, memory override or installation now'],
    },
    'evaluation_order': [
        'Confirm exact artifact source, size, license and isolated library scope before acquisition.',
        'Export/compile smoke test and numerical or output agreement check; keep C/T3F environment untouched.',
        'Same prepared public KSL clips: side, shape, direction, relation, order and timestamp support; word/guide-blind first.',
        'Wrong-guide, reverse-order and missing-observation controls: never invent observed actions to match a guide.',
        'Compare CPU vs GPU with equivalent inputs, precision and output budget; record preprocessing+inference+postprocessing p50/p95, cold-start separately, and peak shared memory.',
        'Only after observation reliability: guide/phase matching, validated targets, person-separated learning, and actual feedback evaluation.'
    ],
    'limits': ['No KSL model superiority demonstrated', 'No independent human/expert truth created', '16 samples are not complete dense temporal evidence', 'No 3-second latency guarantee', 'Pretrained inference feasibility is not feasibility of full VLM training on this laptop'],
    'download_approved': False, 'download_started': False, 'training_runs': 0, 'vlm_inference_runs': 0,
    'protected': ['C', 'T3-F', 'operating web/DTW', 'user and mentor recordings'],
    'separate_review': {
        'reviewer': 'Codex subagent /root/vlm_hardware_review',
        'scope': 'Official-source and reasoning review only, not hardware execution or independent KSL ground truth',
        'conclusion': 'Bounded two-candidate recommendation reasonable; no KSL optimum or GPU latency proven.',
        'checks': ['Thinking benchmark cannot support non-thinking accuracy claim', 'CPU-only installed Torch does not mean missing GPU', 'MiniCPM 1B name does not imply half total footprint'],
    },
}
save(U/'MODEL_SELECTION_REVIEW.json', review)

report = '''노트북 확인과 영상언어 모델 선택 재검토

확인한 사양
Samsung 960XHA / Intel Core Ultra 7 258V, 8코어 8스레드 / RAM 32GB / Intel Arc 140V 내장 GPU / Intel AI Boost NPU.
GPU와 NPU가 실제 장치로 확인됐다. CPU만 가능한 기계가 아니다. 검사 시 여유 RAM은 약11.36GiB였으며 실행 중 프로그램에 따라 달라진다. GPU 이름의16GB는 독립16GB VRAM을 확보했다는 뜻이 아니다. 시스템 RAM을 공유한다.
확인한 Python313에는 torch2.12.0+cpu가 설치되어 cuda/xpu 모두False였다. 이 환경은 GPU를 쓰도록 준비되지 않은 상태다. 다른 모든 환경/과거작업이 CPU였다는 의미는 아니다.

왜 원래 Qwen3-VL-2B를 제안했나
영상과 설명을 함께 입력하고, 사건의 시간과 공간관계를 다루며, 작은 공개 가중치로 로컬 시험이 가능한 모델이라 목표에 맞는 후보였다. 하지만 최신 대안과 노트북 가속을 충분히 비교하지 않았으므로 최적 모델로 추천할 근거는 부족했다.

수정한 선택
첫 정확도 검토 후보 Qwen3.5-2B, 처리효율 대조 후보 MiniCPM-V4.7-1B로 최대2개만 검토한다. 최종 채택이 아니다. 원제안Qwen3-VL도 자동으로 추가 다운로드하지 않는다.
Qwen3.5 공식 일반영상 평가에서 자막없는VideoMME는 원Qwen3VL2B62.1, Qwen3.5-2B69.0이다. 후자는thinking조건이며non-thinking영상점수는없다. 다른항목은개선폭이작거나하락한다. 이 숫자는 한국수어 정확도도, 우리 노트북의 속도도 아니다.
MiniCPM4.7은 영상정보를 일찍 압축해 연산을 줄이는 설계와 로컬 실행 지원 때문에 속도 대조 후보로 적합하다. 손가락 세부정보까지 잘 유지하는지는 별도 확인한다. 이름1B만으로 전체메모리가절반이라고하지않는다.
Gemma4E2B도검토했지만유효2.3B와임베딩포함5.1B가다르므로단순동급으로보지않았다. 성능열등을확정한것은아니며최초비교의범위를2개로제한한다.

노트북에 맞는 시험
Intel용OpenVINO의공식VLM목록에세후보계열이있다. 격리환경에서실제GPU장치인식/모델변환/실행이되는지먼저확인해야한다. CPU와GPU는같은입력·비트정밀도·출력예산으로비교하며NPU는자동사용하지않는다.
기존공개KSL8영상으로어느손/모양/위치관계/변화순서를영상대로말하는지본다. 가이드가잘못되어도그말에끌려없는행동을만들지않는지,순서를뒤집었을때구별하는지검토한다. 영상전처리부터결과까지의시간과공유메모리도측정하고최초로딩은따로기록한다. 일반벤치마크,코드통과,AI초안합의만으로교육정확도를인정하지않는다.
이번에는하드웨어조회와공식문서검토만했다. 다운로드/설치/새학습/모델추론/운영변경없음. 사용자응답은선정근거와사양반영요청이며신규가중치다운로드승인으로취급하지않는다.

근거
https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
https://huggingface.co/Qwen/Qwen3.5-2B
https://huggingface.co/Qwen/Qwen3.5-2B/tree/main
https://huggingface.co/openbmb/MiniCPM-V-4.7-1B
https://ai.google.dev/gemma/docs/core/model_card_4
https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/
https://www.intel.com/content/www/us/en/support/articles/000041253/graphics.html
'''
(U/'MODEL_SELECTION_REVIEW_KO.txt').write_text(report, encoding='utf-8')

plan = read(U/'MODEL_TRIAL_PLAN.json')
archive = U/'MODEL_TRIAL_PLAN.original_proposal.json'
if not archive.exists():
    archive.write_bytes((U/'MODEL_TRIAL_PLAN.json').read_bytes())
plan.update(status='SELECTION_REVIEWED_AWAITING_SPECIFIC_ACQUISITION_CONFIRMATION',
            original_proposal_candidate=plan['candidate'], candidate=None,
            recommended_first_trial=review['recommended_first_trial'],
            efficiency_comparator=review['efficiency_comparator'],
            execution='CPU vs Intel Arc 140V through a compatible isolated runtime; actual GPU execution not yet verified.',
            review='MODEL_SELECTION_REVIEW.json', hardware='HARDWARE_CHECK.json',
            download_approved=False, download_started=False)
plan['repository_size_gb'] = None
plan['reason'] = 'Original proposal not validated as optimal. See reviewed shortlist and hardware constraints.'
save(U/'MODEL_TRIAL_PLAN.json', plan)

msg = '실제 노트북은 Samsung960XHA/CoreUltra7 258V/RAM32GB/Arc140V 내장GPU/AI Boost NPU다. 확인한Python은CPU전용Torch이므로GPU사용불가가하드웨어부재를뜻하지않는다. 기존Qwen3VL2B제안의최신후보비교부족을정정하고Qwen3.5-2B와MiniCPM-V4.7-1B의제한비교를권고했다. 최적/한국수어정확도/3초응답은미검증. 새다운로드승인없음·설치/학습/추론0·운영/개인자료보존. 이전4단계전체미완료상태유지.'
nxt = 'MODEL_SELECTION_REVIEW와HARDWARE_CHECK를기준으로선정근거를사용자에게보고한다. 원Qwen3VL다운로드대기지시를자동실행하지않는다. 정확한새가중치/격리런타임확보승인을받은뒤IntelGPU호환성소규모시험→동일공개KSL관찰정확성/허위근거검토→CPU/GPU전체지연비교. C/T3F보존,개인자료사용금지;현재다운로드·학습·추론없음.'
for heading, title, body in [
    ('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인', '2026-10-28 실제 노트북 사양 확인과 영상언어 후보 재검토', msg),
    ('## 9. 다음 단일 작업', '2026-10-28 노트북 가속 조건을 반영한 후보 선정 보고', nxt),
]:
    p = R/'WordsInSilence_PROJECT_CONTEXT.md'
    txt = p.read_text(encoding='utf-8-sig')
    if '### '+title not in txt:
        assert heading in txt
        txt = txt.replace(heading, heading+'\n\n### '+title+'\n'+NOW+'\n'+body+'\n근거: research/front_keyphase_v1/ops/'+U.name+'/MODEL_SELECTION_REVIEW_KO.txt, HARDWARE_CHECK.json.\n', 1)
        p.write_text(txt, encoding='utf-8')

s = read(O/'STATE.json')
s.update(current_stage='GUIDE_VIDEO_HARDWARE_VERIFIED_MODEL_SELECTION_REVIEWED', updated_at=NOW, updated_date_kst=NOW, next_single_action=nxt)
s[U.name].update(status='PARTIAL_COMPLETE_MODEL_SELECTION_REVIEWED_NO_ACQUISITION_APPROVAL', at=NOW, summary=msg, next=nxt,
                 model_selection_review=str(U/'MODEL_SELECTION_REVIEW.json'), hardware_check=str(U/'HARDWARE_CHECK.json'))
s[U.name]['steps'][1]['remaining'] = '노트북GPU조건과최신대안재검토완료;새모델확보승인/실제관찰시험대기. 추론0회.'
save(O/'STATE.json', s)
goal = read(U/'AUTHORIZATION.json')['goal_ko']
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+NOW+'\n\n목표: '+goal+'\n\n'+msg+'\n\n기존완료: 공개12표현36영상연결·69관측값·24쌍재생·소프트웨어검산. 교육의미모델완료아님.\n\n다음: '+nxt+'\n\n작업단위: '+U.name+'\nMODEL_TRIAL_PLAN.original_proposal.json은이력이며현재후보승인이아니다. 현재실행프로세스없음.\n', encoding='utf-8')
save(U/'CURRENT_EXECUTION.json', {'status':'NO_ACTIVE_PROCESS_SELECTION_REVIEWED_NO_ACQUISITION_APPROVAL', 'at':NOW, 'training_active':False, 'download_started':False, 'all_four_steps_complete':False})
for f in ['HARDWARE_CHECK.json','MODEL_SELECTION_REVIEW.json','MODEL_TRIAL_PLAN.json','CURRENT_EXECUTION.json']:
    read(U/f)
assert read(O/'STATE.json')['next_single_action'] == nxt
assert not (U/'DOWNLOAD_AUTHORIZATION.json').exists()
print('Read-only hardware findings and revised provisional selection recorded. No model execution or acquisition.')
