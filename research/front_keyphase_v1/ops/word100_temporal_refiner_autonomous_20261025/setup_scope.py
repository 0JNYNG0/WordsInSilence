from pathlib import Path
from datetime import datetime
import json, hashlib

U=Path(__file__).resolve().parent
ROOT=U.parents[3]
OLD=U.parent/'word100_temporal_refiner_pilot_20261025'
def write(name,obj):
    (U/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now().astimezone().isoformat()
write('AUTHORIZATION.json',dict(at=now,source='2026-10-25 latest direct user request',
    scope='Autonomously repair validation issues, execute bounded public refiner learning and B0-B3 comparison, assess fixed C/T3-F and DTW downstream effects; report final results.',
    routine_checkpoint_approval_required=False,training_authorized=True,production_replacement=False,
    forbidden=['user/mentor recordings for training, tuning, selection or access','changing frozen prior results','Git changes','new external transfer','claiming proxy labels are anatomical ground truth'],
    major_stop='Unresolvable evidence insufficiency for an intended claim or material change of research direction; failed candidates do not require abandoning remaining valid evaluation.'))
write('RESEARCH_CONTRACT.json',dict(at=now,
    claim_levels=['synthetic reconstruction of quality-weighted MediaPipe proxy coordinates','image-only coarse visible wrist localization on frozen diagnostic examples','downstream fixed-model compatibility and DTW evidence accounting'],
    not_supported=['all-sign-language generalization','hidden-finger anatomical correctness','new-user accuracy from people already seen by classifiers'],
    original_hold_preserved=str(OLD/'VISIBLE_READINESS.json'),
    remedy='Robust pseudo-target loss plus separate quality weights, raw unweighted metrics and image audit. No data point is declared true merely by agreement; normal coordinates stay unchanged.',
    training_plan=dict(seeds=[17,29,43],epochs=80,train_people=[1,2,3,4,5,6],dev_people=[7],test_people=[8,9]),
    acceptance=dict(observed_coordinate_changes=0,user_data_access=0,target_word_model_input=False,test_based_tuning=False,
                    normal_crossing_preservation='exact unchanged available points; estimated gap coordinates never promoted to observed',
                    learned_candidate_adoption='requires reconstruction benefit, no material control regression, and downstream compatibility; negative results remain reportable'),
    optional_handshape_supervision='Only if trustworthy independent handshape annotations exist; no invented labels or compelled expansion.'))
write('RGB_FROZEN.json',dict(status='FROZEN',at=now,script_sha256=sha(U/'rgb_features.py'),
    bundle_manifest_sha256=sha(U.parent/'word100_temporal_public_bundle_20261025/MANIFEST.json'),
    purpose='Frozen feature computation only; no trained model, no ground-truth adequacy claim',
    rules='Fixed full-source RGB, body wrist/shoulder crop; no hand-target or Gloss crop information',
    batch_size=32,threads=4))
write('FEATURE_READINESS.json',dict(status='PASS_FEATURE_PREPARATION_ONLY',ready_for_feature_extraction=True,
    training_allowed=False,reason='Read-only frozen encoding has no pseudo-label loss. Training separately gated after reliability and independent review.',
    old_visible_hold_overridden=False,authorization=str(U/'AUTHORIZATION.json')))
write('PROGRESS.json',dict(at=now,stage='RELIABILITY_AND_DOWNSTREAM_PREPARATION',training_runs_completed=0,
    actors=dict(root='orchestration/training/features',temporal_refiner_review='teacher reliability',temporal_bundle_inventory='independent scientific/code review',temporal_bundle_split='downstream comparison'),
    no_user_data=True,production_changed=False))
print(json.dumps({'created_scope':str(U),'at':now}))
