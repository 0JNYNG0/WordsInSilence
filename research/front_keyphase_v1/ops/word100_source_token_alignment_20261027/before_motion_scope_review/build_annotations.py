"""Package root's sampled-RGB observations; no detector, inference or training.

The manual stage table is evidence at listed frames, NOT dense interval truth.
Source descriptions never fill unknown image labels.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

U = Path(__file__).resolve().parent

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name, obj):
    (U/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

# Ordinals are from the twelve personally viewed 16-image sheets. Anatomical
# side is a visual attribution in the non-mirrored source, not detector truth.
STAGES = {
 'V01': [('얼굴 옆에 검지', [3,4,5], 'face_index'), ('앞으로 향한 검지', [9,10,11,12], 'outward_index'), ('손을 내림', [14,15,16], 'lowering')],
 'V02': [('얼굴 옆에 검지', [4,5], 'face_index'), ('앞으로 향한 검지', [10,11,12,13], 'outward_index'), ('손을 내림', [15,16], 'lowering')],
 'V03': [('얼굴 옆에 검지', [3,4,5,6], 'face_index'), ('앞으로 향한 검지', [9,10,11,12,13], 'outward_index'), ('손을 내림', [15,16], 'lowering')],
 'V04': [('두 손 접근', [2,3,4], 'hands_approach'), ('검지와 반대 손 가까이', [5,6,7,8,9,10,11,12,13], 'index_other_hand'), ('손을 내림', [14,15,16], 'lowering')],
 'V05': [('두 손 접근', [4,5,6], 'hands_approach'), ('검지와 반대 손 가까이', [6,7,8,9,10,11], 'index_other_hand'), ('손을 내림', [12,13,14,15,16], 'lowering')],
 'V06': [('두 손 접근', [4,5,6], 'hands_approach'), ('검지와 반대 손 가까이', [7,8,9,10,11], 'index_other_hand'), ('손을 내림', [13,14,15,16], 'lowering')],
 'V07': [('펼친 손 위아래 배치', [4,5], 'two_open_arranged'), ('위 손이 아래 손에 접근·떨어짐', [5,6,7,8], 'open_approach_cycle'), ('두 손 가까이 유지', [9,10,11,12], 'two_open_arranged'), ('손을 내림', [13,14,15,16], 'lowering')],
 'V08': [('펼친 손 위아래 배치', [4,5], 'two_open_arranged'), ('위 손이 아래 손에 접근·떨어짐', [5,6,7,8], 'open_approach_cycle'), ('두 손 가까이 유지', [9,10,11,12], 'two_open_arranged'), ('손을 내림', [14,15,16], 'lowering')],
 'V09': [('펼친 손 위아래 배치', [5,6], 'two_open_arranged'), ('위 손이 아래 손에 접근·떨어짐', [6,7,8,9], 'open_approach_cycle'), ('두 손 가까이 유지', [10,11,12,13], 'two_open_arranged'), ('손을 내림', [14,15,16], 'lowering')],
 'V10': [('손가락을 모으거나 굽힘', [5,7], 'head_grouped'), ('손가락을 펼침', [6,8,9,10,11], 'head_open'), ('손을 내림', [12,13,14,15,16], 'lowering')],
 'V11': [('손가락을 모으거나 굽힘', [5,7,8,9,10], 'head_grouped'), ('손가락을 펼침', [11,12,13,14,15,16], 'head_open')],
 'V12': [('손가락을 모으거나 굽힘', [6,7,8], 'head_grouped'), ('손가락을 펼침', [5,9,10,11,12,13], 'head_open'), ('손을 내림', [14,15,16], 'lowering')],
}

FEATURES = {
 'face_index': [('R','handshape','index_extended_coarse'), ('R','body_relation','face_side_eye_cheek_neighborhood')],
 'outward_index': [('R','handshape','index_extended_coarse'), ('R','body_relation','away_from_face_in_front')],
 'hands_approach': [('both','inter_hand_motion','approach')],
 'index_other_hand': [('R','handshape','index_extended_coarse'), ('L','handshape','open_upright_coarse'), ('both','inter_hand_relation','right_index_near_left_hand'), ('both','body_relation','torso_front')],
 'two_open_arranged': [('R','handshape','open_fingers_together_coarse'), ('L','handshape','open_coarse'), ('both','inter_hand_relation','right_above_left'), ('both','body_relation','torso_front')],
 'open_approach_cycle': [('both','inter_hand_motion','upper_hand_approach_and_separate'), ('both','handshape','two_open_hands_coarse')],
 'head_grouped': [('R','handshape','grouped_or_curled_coarse'), ('R','body_relation','upper_face_forehead_front')],
 'head_open': [('R','handshape','open_spread_coarse'), ('R','body_relation','upper_face_forehead_front')],
 'lowering': [('active_hands','movement','lowering_toward_rest')],
}

LINKS = [
 dict(word_id='WORD0949',label='멀다',aliases=['V01','V02','V03'],sense='AI Hub 표제어는 멀다. 거리·다른 뜻 또는 표현형의 확정은 보류.',state='FORM_LINK_UNRESOLVED',status_ko='현재 사전 형태와 다름',source_ids=['NIKL_legacy:8171'],rejected_ids=['NIKL_legacy:8171'],
      summary='세 촬영본에서 한 손 검지가 얼굴 옆에 왔다가 앞으로 향한다. 확보한 사전 설명은 두 손으로 시작해 현재 표본과 맞지 않는다.',unknown=['올바른 뜻·표현형의 공식 연결','정확한 얼굴 접촉점','짧은 미표본 구간'],use='영상에서 직접 본 특징만 보관. 사전의 두 손 시작 규칙을 가져오지 않음.'),
 dict(word_id='WORD1139',label='믿다',aliases=['V04','V05','V06'],sense='믿음·신뢰 의미의 사전 항목과 큰 형태가 잠정적으로 부합.',state='COARSE_FORM_PROVISIONAL',status_ko='큰 특징 잠정 연결',source_ids=['NIKL_legacy:6700'],rejected_ids=[],
      summary='한 손의 편 검지가 반대편 펼친 손에 접근하고 몸 앞에서 가까운 관계를 유지한다.',unknown=['검지 어느 면이 실제 닿는지','가려진 손가락 각도','언어적으로 같은 변이인지 최종 확인'],use='검지·펼친 손·손 사이 접근만 표본 특징으로 연결.'),
 dict(word_id='WORD1290',label='감사',aliases=['V07','V08','V09'],sense='검사·감찰 의미의 동음이의 항목은 맞지 않음. 고마움 의미의 항목을 잠정 연결.',state='COARSE_FORM_FINE_ORIENTATION_UNRESOLVED',status_ko='뜻 구분·세부 방향 보류',source_ids=['NIKL_legacy:2372','NIKL_legacy:12468'],rejected_ids=['NIKL_legacy:12468'],
      summary='두 펼친 손을 위아래로 두고 위 손이 아래 손에 접근한다. 아래 손이 손바닥을 위로 향한 듯 보이는 장면이 있어 사전의 손등 방향을 그대로 정답으로 쓰지 않는다.',unknown=['아래 손바닥/손등 면 방향의 정밀 확인','실제 접촉','정확히 두 번인지','허용되는 높이·횟수 변화'],use='두 펼친 손과 접근 관계만 연결. 면 방향·정확 횟수·접촉은 보류.'),
 dict(word_id='WORD2551',label='별',aliases=['V10','V11','V12'],sense='별 표제어의 한 손 오므림·펼침과 부분 연결. 별1/별2 세부 항목은 확정 못함.',state='PARTIAL_FORM_VARIANT_UNRESOLVED',status_ko='일부 특징만 연결',source_ids=['NIKL_legacy:6163','NIKL_nuri:1605','NIKL_nuri:1654'],rejected_ids=[],
      summary='얼굴·이마 앞의 한 손에서 모음·굽힘과 펼침이 교대로 보인다. 정확한 가로 이동이나 높이의 허용 범위는 아직 정답이 없다.',unknown=['별1/별2 변이 ID','사전의 가로 이동 전체 경로','이마 앞·머리 위 높이의 허용 범위','정확 반복 횟수','V11 종료는 표본 범위 밖'],use='보이는 모음·펼침만 시점별로 기록. 머리 위 높이를 강제하지 않음.'),
]

def main():
    index=read(U/'visual_inputs/INDEX.json')
    rows={r['alias']:r for r in index['rows']}
    annotations=[]
    for alias, stages in STAGES.items():
        row=rows[alias]
        for n,(name,ords,kind) in enumerate(stages,1):
            selected=[row['frames'][o-1] for o in ords]
            facts=[dict(hand=h,feature=f,value=v,truth_status='IMAGE_SUPPORTED_DRAFT',must_be_correct_for_sign=None) for h,f,v in FEATURES[kind]]
            annotations.append(dict(annotation_id=f'{alias}-S{n:02}',alias=alias,clip_id=row['clip_id'],word_id=row['word_id'],recorded_person_id=row['recorded_person_id'],
                label_ko=name,kind=kind,annotation_group_index=n,
                sampled_time_extent_seconds=[selected[0]['pts_us']/1e6,selected[-1]['pts_us']/1e6],
                dense_interval_truth=False,chronology_note='List is feature grouping, not linguistic phase order. Sort actual evidence PTS; groups may interleave/overlap.',
                evidence=[dict(ordinal=f['ordinal'],frame_id=f['frame_id'],pts_us=f['pts_us'],file=str(Path(f['file']).relative_to(U)).replace('\\','/'),rgb_sha256=f['rgb_sha256']) for f in selected],
                observed_features=facts,unknown=dict(exact_joint_angles=None,hidden_fingers=None,physical_contact=None,exact_repeat_count=None,semantic_acceptability=None),
                provenance=dict(annotation_method='Root manual inspection of actual cached RGB samples; no keypoint/source-text-generated truth',review='Separate Codex image-first review; not independent human or expert truth',side_basis='Performer side from visible front-view arm connection; not model handedness ground truth'),
                training_eligible=False))
    payload=dict(schema_version=1,status='SAMPLED_VISIBLE_FEATURE_DRAFTS',created_at=datetime.now(timezone.utc).isoformat(),
        scope=dict(expressions=4,clips=12,recorded_people=3,raw_rgb_frames=192,frames_per_clip=16,subset='Existing public TRAIN only; not generalization evaluation'),
        selection=read(U/'AUTHORIZATION.json')['selection'],annotations=annotations,
        excluded=['user/mentor media','facial expression','forced hidden values','correct/wrong performance labels','continuous labels between sampled frames'],
        field_policy='Unknown is null, never zero/error/correct. Per-frame evidence is exact; time extent is only first/last sampled support.',
        source_index_sha256=sha(U/'visual_inputs/INDEX.json'),review_observation_sha256=sha(U/'review/OBSERVATIONS_STAGE1.json'))
    save('TOKEN_OBSERVATIONS.json',payload)
    save('SOURCE_FORM_LINKS.json',dict(links=LINKS,metadata_is_not_form_truth=True,full_variant_confirmed_count=0,
        source_profiles_sha256=sha(U/'SOURCE_PROFILES.json'),additional_search='source_addendum/RESULT.json',review='review/SOURCE_TOKEN_DECISIONS.json'))
    used=set((a['alias'],e['frame_id']) for a in annotations for e in a['evidence'])
    save('ANNOTATION_READINESS.json',dict(
        status='VISIBLE_FEATURE_PREPARATION_DONE_LEARNING_LABELS_NOT_YET_VALIDATED',training_authorized_conditionally=True,training_runs=0,
        image_review=dict(raw_samples=192,annotated_unique_frames=len(used),groups=len(annotations),additional_root_full_frames=5,review_clips=12),
        available=['Exact source/frame/PTS/RGB hash references','Coarse visible hand shape and body/hand relation draft','Source-word homonym exclusion and source-form mismatch protection'],
        missing=['Independent validation of coarse visual targets','More expression groups sharing each feature, including contrast examples','Frozen held-person and held-expression feature evaluation','Reliable tolerance and real error labels for educational correctness'],
        why_not_train='Twelve clips from four expressions can seed labels but cannot establish general feature accuracy. Learning word identity as a shortcut would not demonstrate handshape understanding.',
        next_single_work='Use these sampled targets to define shared visual feature classes; expand an independently checked public set across different expressions for each feature and freeze person/expression-separated evaluation. Treat tolerance and real errors separately.',
        no_repeat_permission_needed_for_authorized_preparation=True))
    save('LEARNING_CONTRACT.json',dict(
        purpose='Learn what is visually present first; assess sign-specific importance only when supported.',
        observation_model_inputs=['RGB','joint coordinates with observation masks','time context'],
        excluded_observation_inputs=['target word ID','target gloss','dictionary expected answer','user identity'],
        target_candidates=['visible index extension','open versus grouped coarse hand configuration','hand-to-hand relation','hand-to-face/body relation','order of observed changes'],
        labels='Only independently checked visible targets; nulls excluded from supervised loss. Do not train on all frames by filling between sampled evidence.',
        split='Separate recorded people, and separate expressions for a feature-generalization test; audit duplicate video hashes.',
        baseline='Existing extracted features and fixed C/T3-F remain preserved; compare feature recall, false alarms, coverage, latency, then separate TCN/DTW downstream effects.',
        correctness='Dictionary form match does not establish allowed deviation, mandatory features or correction rules. Need separately supported tolerances and true errors.',
        current_training_eligible=False))
    save('RESULT.json',dict(status='PREPARATION_COMPLETE',counts=dict(clips=12,expressions=4,recorded_people=3,rgb_samples=192,annotation_groups=len(annotations),annotated_unique_frames=len(used),
        coarse_links=3,full_variant_verified=0,training_runs=0,correctness_labels=0),
        observation_feature_counts=dict(Counter(f['feature'] for a in annotations for f in a['observed_features'])),
        form_outcomes={x['word_id']:x['state'] for x in LINKS},
        preserved=['C','T3-F','operating web/DTW'],user_mentor_accessed=False,
        limitations=['Same-AI manual labels, not independent expert truth','Sparse frames, not dense phases','Four expression groups are not all100 or all KSL','No new accuracy result']))
    print(json.dumps(read(U/'RESULT.json'),ensure_ascii=False))

if __name__=='__main__': main()
