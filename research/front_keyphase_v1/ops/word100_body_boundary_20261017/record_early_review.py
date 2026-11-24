"""Record the completed, bounded review; this never opens the training gate."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

U = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
numeric = read(U / 'BOUNDARY_RECORDS_INDEPENDENT.json')
assert numeric['prepared_records'] == 30
assert numeric['matched_feature_values'] == 6189400
sources = ['FROZEN_PROTOCOL.json', 'design_data_COEFFICIENT.json',
           'BOUNDARY_FIT_INDEPENDENT.json', 'BOUNDARY_PREFLIGHT_INDEPENDENT.json',
           'BOUNDARY_RECORDS_INDEPENDENT.json', 'EXECUTION_CONTRACT_INDEPENDENT.json',
           'evidence_VISUAL_REVIEW.json', 'evidence_visual_triage_REVIEW.json',
           'training_reference_visual_review_RESULT.json']
review = {
    'status': 'EARLY_REVIEW_COMPLETE_FULL_ACTUAL_INPUTS_PENDING',
    'at_utc': datetime.now(timezone.utc).isoformat(),
    'training_allowed': False,
    'root_review': 'Common boundary coefficient and source evidence reviewed; root additionally viewed fallback 얌전하다/장애인복지관 sheets and 북구 REAL06 low-transition detail sheets.',
    'numeric_early_clips': 30,
    'early_new_detector_conditions': 18,
    'frozen_rule': 'Preserve valid stable-rest boundary; otherwise fixed TRAIN-median shoulder-relative lower line. No label/core/model-score input to boundary.',
    'motion_review_scope': {
        'fixed_train_references': 100,
        'fixed_reference_unique_sample_frames': 459,
        'old_failed_train_clips': 10,
        'additional_priority_train_clips': 5,
        'scope_overlap': 'These sets may overlap; counts must not be summed as unique videos.'
    },
    'observed': [
        'No clearly newly cropped raised action hand found in inspected samples.',
        'Most longest no-hand intervals in priority scenes are lowered side hands near annotation endpoints.',
        'Visible hands can be missing in pre-existing full-frame observations; this is not proof that ROI removed a visible point.',
        '북구 REAL06 low target hand position remains21/21 at frames76-84, then lowering frames86-91 is partially outside the boundary. Root independently viewed both detailed sheets.'
    ],
    'interpretation': '북구 later lowering looks like a return to rest after a held low pose. This visual interpretation is not expert linguistic adjudication.',
    'limitations': [
        'Sparse source review does not certify every frame or every sign meaning.',
        'Legacy-observation geometry must be checked against the still-running new full/cropped detector inputs.',
        'Known detector misses remain; adding realistic missingness is a candidate training condition, not a detector accuracy claim.',
        'No classifier comparison, new-user generalization or performance improvement established.'
    ],
    'remaining_before_training': [
        'All1199 committed input records with every required TRAIN view available.',
        'Independent full channel/mask/PTS/source/fresh-sensor-state verification.',
        'Review any new material actual-observation loss beyond existing triage and preserve first failure.',
        'Evidence-backed independent readiness and training context freeze.'
    ],
    'evidence_sha256': {str((U / p).resolve()): sha(U / p) for p in sources},
    'training_runs': 0, 'production_changes': 0, 'user_data_used': 0
}
with (U / 'EARLY_REVIEW_RESULT.json').open('x', encoding='utf-8') as f:
    json.dump(review, f, ensure_ascii=False, indent=2)
print(review['status'])
