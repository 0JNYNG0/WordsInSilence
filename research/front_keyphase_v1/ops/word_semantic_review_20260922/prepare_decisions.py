"""Create traceable research decisions/manifests; no model or signal processing."""
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

P = Path('C:/MyUnityProjects/WordsInSilence')
L = P / 'research/front_keyphase_v1'
H = Path(__file__).resolve().parent
N = L / 'ops/word_native_readiness_20260922'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(name, value):
    with (H / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

now = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
prepared = read(N / 'PREPARATION.json')['rows']
audit_path = Path('C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/TCN_일반화감사_20260918/word_input_audit.json')
source_audit = read(audit_path)
source_rows = {r['clip_id']: r for r in source_audit['source_rows']}
dictionary = read(H / 'DICTIONARY_SOURCE.json')
meta_path = L / 'ops/context_training_pair_20260921/feature_metadata.json'
meta = read(meta_path)
previous_policy = read(L / 'ops/partial_encoder_pair_20260922/comparison_policy.json')
previous_training = read(L / 'runs/partial_encoder_pair_v1/training.json')
target_names = dict(JOTA='좋다', MOREUDA='모르다', NA='나', ALDA='알다', JOESONGHADA='죄송하다', JADA='자다', SILTA='싫다', GAMSAHAMNIDA='감사합니다', NEO='너')
dictionary_rows = {'JOTA':[13546], 'MOREUDA':[4280], 'NA':[2230], 'ALDA':[7709], 'JOESONGHADA':[13548], 'JADA':[9399,9560], 'SILTA':[7423,7424], 'GAMSAHAMNIDA':[399,400], 'NEO':[2543]}
holds = {
    '1290':'감사 has attested gratitude/audit senses. This source lacks a sense identifier; do not select gratitude from spelling.',
    '1544':'Existing REAL01 family review recorded a materially different sleep form and left equivalence unresolved. No new visual/lexical family evidence resolves it.',
    '2485':'Existing REAL01 family review recorded face versus chest location and left equivalence unresolved. No new family evidence resolves it.'
}
rows = []
for r in prepared:
    s = source_rows[r['clip_id']]
    assert r['signer_id'] == s['signer_id'] == 2
    assert r['raw_label'] == s['raw_label']
    target, = r['target_candidates_not_approved']
    approved = r['word_number'] not in holds
    row = {
        'clip_id':r['clip_id'], 'word_number':r['word_number'],
        'raw_label':r['raw_label'], 'source_family_id':s['source_family_id'],
        'source_family_is_linguistically_verified_variant':False,
        'label_member':s['label_member'], 'label_member_sha256':s['label_member_sha256'],
        'signer_id':2, 'split_role':'train', 'view':'F',
        'candidate_target_id':target, 'candidate_target_name':target_names[target],
        'semantic_target_id':target if approved else None,
        'training_approved':approved,
        'approval_scope':'One source-label-supervised classification comparison only' if approved else None,
        'decision':'ACCEPT_SOURCE_LABEL_FOR_CLASSIFICATION' if approved else 'HOLD_SPECIFIC_UNRESOLVED_MAPPING',
        'basis':('Official AIHub morph annotation is the supplied headword label; NIKL lexical/form table supports the target sense. This is an explicit source-annotation assumption, not independent validation of the performed sign.' if approved else holds[r['word_number']]),
        'dictionary_row_ids':dictionary_rows[target],
        'dictionary_word_number_crosswalk':'NOT_ESTABLISHED; NIKL form names/numbers are not AIHub WORD identifiers',
        'performed_sign_semantics_independently_verified':False,
        'same_form_as_existing_reference_verified':False,
        'feedback_reference_binding':None,
        'npz':r['npz'], 'npz_sha256':r['npz_sha256'],
        'full_frames':r['frames'], 'annotation_frames':r['annotation_frames'],
        'window_keys':['x_annotation','x_full'] if approved else [],
    }
    if target == 'JOESONGHADA':
        row['lexical_interpretation'] = '죄송 is interpreted as the apology headword corresponding to 죄송하다. Prior SEN0356 user approval is NOT transferred to this WORD performance.'
    if target == 'JADA' and approved:
        row['lexical_interpretation'] = '자다/잠자다 are interpreted as sleep under source annotation. Neither 자다1 nor 자다2 in other source rows is treated as an exact dictionary form ID.'
    rows.append(row)
accepted = [r for r in rows if r['training_approved']]
assert len(rows)==16 and len(accepted)==13
assert len({r['semantic_target_id'] for r in accepted})==8
write('SEMANTIC_DECISIONS.json', {
    'at_kst':now, 'status':'SOURCE_LABEL_REVIEW_COMPLETE_WITH_EXPLICIT_LIMITS',
    'policy':'Trust supplied headword supervision where the intended lexical class is supported; keep known ambiguity/unresolved counterevidence on hold. Do not claim that corpus labels or dictionary tables prove gesture correctness.',
    'decision_authority':'Codex research judgment under user-authorized autonomous data preparation; not user or expert confirmation',
    'sources':{
        'aihub_annotation_definition':'https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103',
        'dictionary_help':'https://sldict.korean.go.kr/signData/help/help.html',
        'dictionary_pdf':dictionary,
        'existing_source_audit':{'path':str(audit_path),'sha256':sha(audit_path)},
        'previous_form_review':str(L/'evidence/word_media_review_20260918.json')},
    'counts':{'candidates':16,'accepted_for_limited_comparison':13,'held':3,'accepted_target_classes':8,'new_people':0},
    'rows':rows, 'training_runs':0, 'model_inference':0,
    'user_and_reserved_arrays_read':0,
})

existing_context = meta['pair_context']
extra = []
for r in accepted:
    for key in r['window_keys']:
        extra.append({'clip_id':r['clip_id'], 'source_family_id':r['source_family_id'],
                      'label':r['candidate_target_name'], 'signer_id':2,
                      'path':r['npz'], 'npz_sha256':r['npz_sha256'], 'array_key':key})
schedule_path = Path(previous_policy['paths']['schedule'])
with np.load(schedule_path, allow_pickle=False) as z:
    bi, ci = z['base_indices'].copy(), z['pair_indices'].copy()
assert bi.shape==ci.shape==(660,16) and len(existing_context)==224 and len(extra)==26
rng = np.random.default_rng(20260922)
replace = rng.random(ci.shape) < (26 / 250)
new_draws = rng.integers(0,26,size=ci.shape)
ti = np.where(replace,224+new_draws,ci)
with (H/'comparison_schedule.npz').open('xb') as f:
    np.savez_compressed(f,base_indices=bi,control_context_indices=ci,treatment_context_indices=ti,replaced_context_slots=replace)
dev_counts = {v:sum(r['label']==v for r in meta['annotation_dev']) for v in target_names.values()}
plan = {
    'at_kst':now, 'status':'COMPARISON_CONTRACT_READY_NOT_EXECUTED',
    'question':'Under fixed compute and TCN, does adding thirteen WORD recordings improve available public development recognition while preserving existing behavior?',
    'mapping_manifest_sha256':sha(H/'SEMANTIC_DECISIONS.json'),
    'parent':previous_policy['parent'],
    'control':{'reuse':previous_training['models']['control'], 'retrain':False,
               'evaluation':str(L/'runs/partial_encoder_pair_v1/evaluation.json')},
    'trainable':'head.* only; existing encoder and all buffers fixed',
    'settings':previous_policy['settings'],
    'loss':previous_policy['loss'],
    'base':{'metadata':str(meta_path),'metadata_sha256':sha(meta_path),'rows':2122,'unchanged':True},
    'context':{'control_rows':224,'treatment_rows':250,'new_recordings':13,'new_windows':26,
               'existing_first224_order_preserved':True,'added_rows':extra},
    'schedule':{'path':str(H/'comparison_schedule.npz'),'sha256':sha(H/'comparison_schedule.npz'),
                'old_schedule_sha256':sha(schedule_path),'seed':20260922,'shape':[660,16],
                'replacement_probability':26/250,'actual_new_window_draws':int(replace.sum()),
                'policy':'Preserve both schedules on all nonreplacement slots and all base slots. Uniform draw among26 added windows on fixed replacement slots. Same660x32 compute; no additional epoch budget.'},
    'comparability_limit':'This tests this fixed-budget data-mixture policy; it is not an equal-exposure test preserving every old context draw. Two windows from one recording are not independent samples.',
    'preflight_remaining':['Reuse approved Claude head-only routine if callable without changing model logic; otherwise Claude writes only necessary generic adapter.',
                           'Compute frozen-encoder features/prefix only for26 new windows and teacher scores as required; reuse all existing caches.',
                           'Verify class order93, source hashes, train-only assignment and exact old-control contract before one treatment run.'],
    'evaluation':{'annotation_dev':699,'whole_dev':74,'development_signers':[11,12,13],
                  'target_counts_in_annotation_dev':dev_counts,
                  'public_reference_and_mp':'Reuse existing32 reference and14 MP conditions; these are not fresh-person tests.',
                  'missing_target_recall':['너','싫다','자다'],
                  'selection':'Use only public development and reference conditions. Existing user recordings remain evaluation-only and are not used for candidate selection in this unit.',
                  'candidate_retention':'Require nondecreasing total and macro recall and per-signer correct counts on annotation_dev, and no loss on existing whole/public/MP conditions. Report individual gains/regressions. A pass without a supported independent-development gain is nonregression, not demonstrated improvement.',
                  'new_window_accuracy':'Training sanity only, not generalization.'},
    'fresh_people':{'timing':'Deferred by explicit user request; not a prerequisite for this comparison',
                    'interface':'Reuse existing test screen at http://127.0.0.1:8765/ when user has participants',
                    'live_server_checked_this_unit':False,'new_interface_required':False,
                    'later_protocol':'Freeze candidate and test protocol before new participant recordings; keep participant and recording IDs separated from training. Do not train/select on these observations before reporting the first frozen result.'},
    'boundaries':previous_policy['boundaries'],
    'stopping':'One fixed treatment. No seed/coefficient/step/threshold search after results. No automatic default model replacement.',
    'new_archives':0,'new_training_or_model_inference_this_unit':0,
}
write('COMPARISON_PLAN.json',plan)
write('USER_DIRECTION.json', {'at_kst':now,
    'verbatim':'그럼 다음 작업은 뭐야 그것까지 일단 진행해줘. 그리고 다른 새로운 사람에 대한 테스트 확인은 사전에 미리 테스트 환경을 만들어놨었잖아, 거기서 나중에 시간될 때 새로운 사람들한테 부탁해서 진행해보는 정도로 하고, 그 전에는 최대한 성능 높일 수 있게 목표대로 진행하자.',
    'effect':'Defer fresh-person confirmation to existing interface. Continue valid public-data improvement without treating missing fresh participants as a development stop condition.',
    'not_implied':['No semantic approval of new WORD gestures','No training on user recordings','No additional ZIP permission','No claim of measured improvement','No default model promotion']})
print(json.dumps({'accepted':len(accepted),'held':len(holds),'targets':len({r['semantic_target_id'] for r in accepted}),'new_window_draws':int(replace.sum()),'public_dev_counts':dev_counts,'training_runs':0},ensure_ascii=False))

if __name__ == '__main__':
    pass
