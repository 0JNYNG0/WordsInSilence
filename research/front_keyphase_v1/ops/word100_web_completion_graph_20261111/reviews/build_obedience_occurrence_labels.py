"""Serialize the completed RGB-only visual review. No model inference or training."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
INPUT = HERE / 'obedience_occurrences' / 'INPUTS.json'
OUT = HERE / 'obedience_occurrence_labels.json'
sources = json.loads(INPUT.read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def sample(frames, phase, observation, shape=None):
    return {'frames': frames, 'phase': phase, 'observation_ko': observation,
            'shape_evidence': shape, 'source': 'direct_RGB_contact_sheet_review'}

def episode(key, role, guide, start, end, samples, movement_end=None):
    d = {'local_occurrence_id': key, 'role': role, 'guide_clause': guide,
         'start_boundary_uncertainty_frames': start,
         'end_boundary_uncertainty_frames': end,
         'direct_observations': samples}
    if movement_end is not None:
        d['movement_end_boundary_uncertainty_frames'] = movement_end
    return d

reviews = {
1: [
episode('entry_1', 'preparation', None, [36,38], [46,48], [
    sample([36], 'rest_before_entry', '오른손이 내려가 있다.'),
    sample([38,40,42,44,46], 'entry', '오른손을 몸 아래에서 얼굴 앞쪽으로 올린다. 들어 올리는 중에도 편 손 모양이 나타나므로 이 모양만으로 주요 제시 행동을 세면 안 된다.')]),
episode('main_1', 'main', 'open_front', [46,48], [60,62], [
    sample([48,50,52,54,56,58,60], 'presented_hold', '오른손의 펴진 손가락들이 얼굴 앞에 놓인 상태를 유지한다. 코·입 앞 범위로 보이나 정확한 접촉 부위는 판정하지 않는다.', 'extended_fingers_together')]),
episode('main_2', 'main', 'close_chest', [60,62], [72,74], [
    sample([62,64], 'closing_and_lowering', '얼굴 앞에서 손을 내리면서 손가락을 접는 과정이다. 시작부터 닫힌 손으로 강제 라벨링하지 않는다.', 'transition_extended_to_closed'),
    sample([66,68], 'closed_hand_lowering', '닫힌 손이 가슴 높이 쪽으로 내려온다.', 'closed'),
    sample([70,72], 'terminal_posture', '닫힌 손이 가슴 높이에 있고 전완이 비교적 가로 방향이다. 새 행동이 아니라 같은 내림 행동의 끝이다.', 'closed')], [68,70]),
episode('main_3', 'main', 'raise_forearm', [72,74], [92,94], [
    sample([74,76,78], 'forearm_reorientation', '닫힌 손을 유지하면서 전완이 가로에 가까운 상태에서 더 세워진 상태로 변한다.', 'closed'),
    sample([80,82,84,86,88,90,92], 'terminal_hold', '팔뚝을 세운 닫힌 손 상태를 유지한다. 이 유지 프레임들은 새 반복이나 새 완료가 아니다.', 'closed')], [78,80]),
episode('return_1', 'return', None, [92,94], [102,104], [
    sample([94,96,98], 'lowering_return', '동작 후 손을 내린다. 주먹 모양이 아직 남아 있어도 주요 동작의 재실행으로 세지 않는다.', 'closed'),
    sample([100,102], 'relaxing_return', '손을 낮추면서 손가락을 풀고 대기 자세로 돌아간다.', 'relaxing'),
    sample([104], 'rest_after_return', '오른손이 옆으로 내려간 대기 자세다.')])],
4: [
episode('entry_1', 'preparation', None, [54,58], [66,70], [
    sample([54], 'rest_before_entry', '오른손이 내려가 있다.'),
    sample([56,58,60,62,64,66,68], 'entry', '오른손을 올려 얼굴 앞으로 가져온다. 56은 작은 시작 변화, 58부터 들어 올림이 더 분명하다.')]),
episode('main_1', 'main', 'open_front', [66,70], [86,88], [
    sample([70,72,74,76,78,80,82,84,86], 'presented_hold', '오른손의 펴진 손가락들을 얼굴 앞에 제시하여 유지한다. 정확한 얼굴 접촉은 판정하지 않는다.', 'extended_fingers_together')]),
episode('main_2', 'main', 'close_chest', [86,88], [102,104], [
    sample([88,90], 'closing_and_lowering', '얼굴 앞의 손이 아래로 이동하며 손가락이 접힌다.', 'transition_extended_to_closed'),
    sample([92,94,96], 'closed_hand_lowering', '닫힌 손이 가슴 높이로 이동한다.', 'closed'),
    sample([98,100,102], 'terminal_posture', '가슴 높이 닫힌 손과 가로에 가까운 전완 자세. 같은 내림 행동의 끝 자세이다.', 'closed')], [96,100]),
episode('main_3', 'main', 'raise_forearm', [102,104], [130,134], [
    sample([104,106,108,110], 'forearm_reorientation', '주먹을 유지하며 전완을 더 세운 방향으로 바꾼다.', 'closed'),
    sample([112,114,116,118,120,122,124,126,128,130], 'terminal_hold', '세워진 팔뚝과 닫힌 손을 유지한다. 다른 두 기록보다 유지 시간이 길지만 별도의 반복으로 보이지 않는다.', 'closed')], [110,112]),
episode('return_1', 'return', None, [130,134], [140,144], [
    sample([132,134,136,138,140], 'lowering_and_relaxing_return', '132에서 작은 변화, 134부터 분명히 손을 내리며 손을 푼다. 140에서는 복귀 중이다.'),
    sample([144], 'rest_after_return', '오른손이 내려간 대기 자세다. 142는 이 검토에서 직접 보지 않았다.')])],
7: [
episode('entry_1', 'preparation', None, [24,26], [36,38], [
    sample([24], 'rest_before_entry', '오른손이 내려가 있다.'),
    sample([26,28,30,32,34,36], 'entry', '오른손을 몸 아래에서 얼굴 앞까지 올린다. 36은 얼굴 앞 도착 과정이다.')]),
episode('main_1', 'main', 'open_front', [36,38], [50,54], [
    sample([38,40,42,44,46,48,50], 'presented_hold', '오른손의 펴진 손가락들이 얼굴 앞에 놓인 상태다.', 'extended_fingers_together')]),
episode('main_2', 'main', 'close_chest', [50,54], [64,66], [
    sample([52,54], 'closing_and_lowering', '52는 입 앞에서 이동 시작, 54는 턱 근처에서 접히는 과정이다.', 'transition_extended_to_closed'),
    sample([56,58,60], 'closed_hand_lowering', '접힌 손이 가슴 높이 쪽으로 내려온다.', 'closed'),
    sample([62,64], 'terminal_posture', '닫힌 손이 가슴 높이, 전완이 가로에 가까운 끝 자세이다.', 'closed')], [60,64]),
episode('main_3', 'main', 'raise_forearm', [64,66], [78,82], [
    sample([66,68,70], 'forearm_reorientation', '닫힌 손을 유지하면서 전완을 더 세운 방향으로 바꾼다.', 'closed'),
    sample([72,74,76,78], 'terminal_hold', '팔뚝을 세운 닫힌 손 자세를 유지한다. 같은 행동의 끝을 유지하는 구간이다.', 'closed')], [70,72]),
episode('return_1', 'return', None, [78,82], [90,92], [
    sample([80,82,84], 'lowering_return', '80은 작은 하강, 82부터 분명한 내림. 아직 주먹이 남아 있어도 주요 동작이 아니다.', 'closed'),
    sample([86,88,90], 'relaxing_return', '손을 내리면서 풀고 몸 옆으로 돌아간다.', 'relaxing'),
    sample([92], 'rest_after_return', '오른손이 내려간 대기 자세다.')])]
}

clips = []
checks = []
for src in sources:
    source_hash = sha(src['source'])
    assert source_hash == src['source_sha256'], src['clip_id']
    viewed = sorted(set(f for p in src['review_images'] for f in p['frames']))
    rec = {**src, 'split': 'FIT', 'word_id': 'WORD0739',
           'recorded_performer_id': src['person'], 'fps': 30,
           'frame_numbering': 'zero_based_source_video', 'source_hash_match': True,
           'reviewed_frame_ids': viewed, 'reviewer': 'Codex_visual_review',
           'occurrences': reviews[src['person']],
           'left_arm': 'Visible left arm remains lowered; not an additional required action.',
           'guide_consistency': 'Broad observed flow agrees with the three existing guide clauses; exact contact, angles, thumb configuration and educational correctness are not established.'}
    for panel in rec['review_images']:
        panel['sha256'] = sha(ROOT / panel['path'])
    bands = []
    for ep in rec['occurrences']:
        ep['occurrence_id'] = src['clip_id'] + ':' + ep['local_occurrence_id']
        ep['hand'] = 'right'
        for k, v in ep.items():
            if k.endswith('_uncertainty_frames'):
                assert 0 <= v[0] <= v[1] < src['frames'], (k, v)
                assert set(v) <= set(viewed), (k, v)
                bands.append(v)
        for obs in ep['direct_observations']:
            assert set(obs['frames']) <= set(viewed), obs
    uncertain = set(f for a,b in bands for f in range(a,b+1))
    for ep in rec['occurrences']:
        for obs in ep['direct_observations']:
            obs['role_supervision_frames_outside_boundary_bands'] = [f for f in obs['frames'] if f not in uncertain]
            obs['role_supervision_deferred_boundary_frames'] = [f for f in obs['frames'] if f in uncertain]
    rec['role_boundary_exclusion_frames'] = sorted(uncertain)
    rec['direct_observation_count'] = sum(len(x['frames']) for e in rec['occurrences'] for x in e['direct_observations'])
    clips.append(rec)
    checks.append({'clip_id': src['clip_id'], 'source_hash_match': True,
                   'all_direct_frames_in_viewed_panels': True,
                   'all_uncertainty_endpoints_directly_viewed': True,
                   'reviewed_frame_count': len(viewed),
                   'direct_observation_count': rec['direct_observation_count']})

result = {
 'task': 'G1_public_FIT_obedience_occurrence_RGB_review',
 'status': 'RGB_OBSERVATIONS_WITH_BOUNDARY_UNCERTAINTY_READY_FOR_INDEPENDENT_REVIEW',
 'review_scope': {'public_recorded_performers': [1,4,7], 'word': 'WORD0739',
                 'clips': 3, 'panels_viewed': 9, 'dev_access': 0,
                 'user_or_mentor_data_access': 0, 'model_inference': 0,
                 'training': 0, 'existing_model_or_document_changes': 0},
 'reviewer_disclosure': 'Codex directly inspected sampled RGB contact sheets. This is not independent human annotator agreement or Korean Sign Language instructor validation. Model outputs and DEV labels were not used for these annotations.',
 'sampling_policy': 'Overview every 8 frames and focused every 2 frames, as explicitly listed per panel. Not every original frame was inspected. No unviewed frame is promoted to a direct label.',
 'boundary_policy': 'Inclusive [lo,hi] source frame bands locate uncertain onset/offset between directly viewed states. They are not exact dense boundaries. Frames in any boundary band are excluded from strong role supervision; direct visual descriptions remain available there. Any later interpolation must be separately marked weak/derived.',
 'main_occurrence_policy': 'An action includes its terminal hold; reaching the endpoint and maintaining it do not create two completed actions. Main2 and main3 share closed-hand appearance but differ in relative path and forearm reorientation. Main1 is a deliberate held presentation and need not contain continuing wrist motion.',
 'common_guides': [
   {'clause': 'open_front', 'text_ko': '오른손을 펴서 얼굴 앞에 세워 보세요.',
    'evidence': 'Extended fingers held in front of face.', 'not_established': ['exact face contact','exact palm orientation','every digit detail']},
   {'clause': 'close_chest', 'text_ko': '손을 닫으면서 가슴 쪽으로 내려 보세요.',
    'evidence': 'Open-to-closed hand transition with face-to-chest lowering.', 'shape_scope': 'transition_then_closed_late'},
   {'clause': 'raise_forearm', 'text_ko': '닫힌 손을 유지하며 팔뚝을 세워 보세요.',
    'evidence': 'Closed hand maintained while forearm changes from more horizontal to more upright; terminal posture held.', 'not_established': ['exact 90-degree requirement']}],
 'confusable_nonmain_evidence': [
   'Preparation also contains extended fingers and lifting; open shape alone does not mean the face presentation has been completed.',
   'Return may keep a closed hand while moving down; closed shape/downward path alone does not identify the main face-to-chest action.',
   'A long terminal hold is not a repetition. Count occurrence, not separately selected peak/arrival spans.',
   'Same final pose does not establish how the hand arrived there. Direction needs ordered observations.',
   'No-expression scope: visible head/blink changes do not create expression feedback labels.'],
 'reversal_semantics': [
   {'descriptor':'static_handshape_or_body_relative_posture', 'reverse_effect':'Intrinsic single-frame shape/posture remains the same; temporal role does not follow automatically.'},
   {'descriptor':'open_front_hold', 'reverse_effect':'The visible held open posture remains, but approach/departure and ordering change. Do not use source occurrence identity as semantic completion truth.'},
   {'descriptor':'close_chest', 'reverse_effect':'Becomes closed-to-open with chest-to-face upward travel; it does not retain the original closing/lowering completion label.'},
   {'descriptor':'raise_forearm', 'reverse_effect':'Keeps a closed hand but changes upright-to-horizontal, the opposite reorientation. The original raising/standing action label is not invariant.'},
   {'descriptor':'entry_or_return', 'reverse_effect':'Roles swap in a complete temporal reversal; the same pose is insufficient to keep the role.'},
   {'descriptor':'occurrence_source_id', 'reverse_effect':'Retain only as provenance. Reversed semantic labels require transformed descriptor review, not copying y[source_indices].'}],
 'clips': clips,
 'validation': {'type':'same_Codex_structural_and_provenance_check_not_independent_visual_confirmation',
                'checks':checks, 'pass':True},
 'limits': ['Three FIT demonstrations of one guide form do not establish all acceptable variants or any final-test accuracy.',
            '2D RGB views do not prove physical contact, exact depth or joint angles.',
            'Direct observations describe visible behavior and guide correspondence, not certified educational correctness.',
            'No synthetic replay was visually reviewed here. Reverse semantics are descriptor-based cautions, not evaluated model results.',
            'Record performer IDs are metadata IDs, not appearance-based identity judgments.']}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'output':str(OUT), 'clips':len(clips), 'checks':checks}, ensure_ascii=False))
