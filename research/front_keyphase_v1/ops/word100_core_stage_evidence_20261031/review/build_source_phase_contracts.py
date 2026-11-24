"""Serialize a cached-source / directly inspected RGB audit. No model execution."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import numpy as np

U = Path(__file__).resolve().parent.parent
OPS = U.parent
OLD = OPS / 'word100_semantic_episode_pilot_20261031'
pins = {}


def pin(p):
    p = Path(p)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    pins[str(p.resolve())] = h
    return dict(path=str(p.resolve()), sha256=h)


def read(p):
    pin(p)
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


auth = read(U / 'AUTHORIZATION.json')
cards_path = OPS / 'word100_guide_video_feedback_20261028/GUIDE_CARDS.json'
cards = read(cards_path)['cards']
source_path = OPS / 'word100_guide_grounded_feedback_20261031/source_audit/SOURCE_CONTRACTS.json'
source = read(source_path)
profiles_path = OPS / 'word100_source_token_alignment_20261027/SOURCE_PROFILES.json'
profiles = read(profiles_path)['profiles']
prior_usage = pin(OLD / 'review/SOURCE_USAGE_AND_RANGE.json')
frozen = read(OLD / 'FROZEN.json')
prepared = read(OLD / 'PREPARED.json')
code_pin = pin(OLD / 'episode.py')
experiment_pin = pin(OLD / 'experiment.py')
rows = {r['alias']: r for r in auth['rows']}
viewed = ['P033', 'P036', 'P045', 'P048', 'P063', 'P066', 'P069', 'P072']
individual = {'P036': [68, 89, 96, 107], 'P048': [79, 97], 'P066': [68, 75], 'P072': [79, 94]}
manifests = {}
identity_checks = []
clip_identity = []


def ck(name, ok):
    identity_checks.append(dict(name=name, status='PASS' if bool(ok) else 'FAIL'))


for alias, r in rows.items():
    mpath = OLD / 'visuals' / alias / 'MANIFEST.json'
    m = read(mpath)
    manifests[alias] = m
    inp = np.load(r['input'], allow_pickle=False)
    frame_lookup = {int(f): i for i, f in enumerate(inp['frame_id'])}
    ck(alias + '/source_sha_auth_manifest', r['source_sha256'] == m['source_sha256'])
    ck(alias + '/alias', alias == m['alias'])
    ck(alias + '/frame_count', r['frames'] == m['frames'] == len(inp['frame_id']))
    ck(alias + '/input_sha', pin(r['input'])['sha256'] == r['input_sha256'])
    for s in m['selected']:
        ck(f"{alias}/frame{s['frame_id']}/PTS", s['timestamp_us'] == int(inp['timestamp_us'][frame_lookup[s['frame_id']]]))
        ck(f"{alias}/frame{s['frame_id']}/file", Path(s['image']).is_file())
    clip_identity.append(dict(alias=alias, word_id=r['word_id'], clip_id=r['clip_id'], recorded_id=r['person'], role=r['role'],
                              source_sha256=r['source_sha256'], frames=r['frames'], manifest=pin(mpath),
                              input=pin(r['input']), directly_viewed_this_audit=alias in viewed,
                              visual_scope='48 sampled frames in RGB sheet; extra individual frames if listed' if alias in viewed else 'identity/time metadata only; do not infer visual form',
                              individual_full_resolution_frames=individual.get(alias, []),
                              source_video_bytes_rehashed_this_audit=False))
    if alias in viewed:
        pin(OLD / 'visuals' / alias / 'sequence_full.jpg')


def evidence(alias, frames, visible, purpose, limit):
    samples = {s['frame_id']: s for s in manifests[alias]['selected']}
    anchors = []
    for f in frames:
        s = samples[f]
        anchors.append(dict(frame_id=f, timestamp_us=s['timestamp_us'], seconds=s['timestamp_us']/1e6,
                            image=pin(s['image']), viewed_as='individual_full_resolution' if f in individual.get(alias, []) else 'sample_in_full_sequence_sheet'))
    return dict(alias=alias, anchors=anchors, visible_description=visible, relation_to_source=purpose,
                limitations=limit, linguistic_truth=False, exact_phase_boundaries=False,
                evidence_extent_only=True)


def candidate(word, entry):
    card_index = next(i for i,c in enumerate(cards) if c['word_id'] == word)
    cs = cards[card_index]['candidates']
    ix = next(i for i,c in enumerate(cs) if c['entry_id'] == entry)
    c = cs[ix]
    matching = [p for pp in profiles if pp['word_id'] == word for p in pp['candidate_entries'] if str(p['entry_id']) == entry.split(':')[-1]]
    raw_evidence = None
    if c.get('description') is not None:
        raw_path = OPS / 'word100_linguistic_feedback_pipeline_20261027/source_audit/public_raw' / f"legacy_detail_{entry.split(':')[-1]}.html"
        raw = raw_path.read_text(encoding='utf-8-sig')
        raw_evidence = dict(**pin(raw_path), description_character_offset=raw.index(c['description']))
        ck(word + '/' + entry + '/cached_raw_exact_quote', '<dd>'+c['description']+'</dd>' in raw)
        ck(word + '/' + entry + '/cached_raw_original_hash', raw_evidence['sha256'] in {claim['source']['content_sha256'] for claim in c.get('claims', [])})
        if matching:
            ck(word + '/' + entry + '/cached_profile_quote', any(p.get('description') == c['description'] for p in matching))
    return dict(entry_id=entry, source_url=c['url'], cached_description=c.get('description'),
                description_sha256=hashlib.sha256(c['description'].encode()).hexdigest() if c.get('description') else None,
                source=dict(**pin(cards_path), json_pointer=f'/cards/{card_index}/candidates/{ix}'),
                original_content_sha256s=sorted({claim['source']['content_sha256'] for claim in c.get('claims', [])}),
                cached_raw_page=raw_evidence, refreshed_online_this_audit=False, authoritative_meaning_rule=False)


def clause(id_, c, span, feature, status, evidence_, rationale, restrictions, temporal=False):
    text = c['cached_description']
    at = text.index(span)
    return dict(id=id_, entry_id=c['entry_id'], source_url=c['source_url'], source_span=span,
                source_character_span=dict(start=at,end=at+len(span),unit='Unicode codepoints; end exclusive'),
                source=c['source'], expected_observation=feature, visible_match_status=status,
                evidence=evidence_, rationale=rationale, mapping_stage_restrictions=restrictions,
                source_temporal_relation_explicit=temporal, training_label_from_text=False,
                requiredness=None, tolerance=None, linguistic_verdict=None)


s = candidate('WORD0739', 'NIKL_legacy:12209')
obedience_unexplained = [
    evidence('P036',[57,64,68,85,89], 'Open right hand moves from low position to the mouth/nose region, remains there in several sampled frames, then changes configuration.',
             'This visible earlier action is not described by the cached fist/side-arm sentence.',
             'Do not call all of it preparation, assign it a lexical component name, or delete it from expected scope. Compound/expanded-form interpretation remains unverified.'),
    evidence('P033',[38,43,46,58,61,63], 'The ID1 fit demonstration also has an open-hand face-region action before the closed-hand arm action.',
             'Cross-record visible sequence agreement; not evidence of its semantic identity.', 'Only sampled RGB evidence; no obligatory lexical stage claim.'),
]
obedience_late = [
    evidence('P036',[89,96,99,103,107,110,131], 'Right hand changes to a closed configuration near the chest; right elbow is extended laterally at96/99/103 and is lower at107/110 while the fist is nearer the shoulder.',
             'Candidate later fist/side-arm component of12209.', 'Wrist vertical direction is not elbow/upper-arm vertical direction. Exact onset/offset and full fist anatomy remain unverified.'),
    evidence('P033',[63,66,68,71,76,78,89], 'The later closed-hand right elbow is held out to the side and subsequently drops as the forearm becomes more upright.',
             'Candidate same later arm component in ID1.', 'Not the earlier open-hand rise; no cross-person boundary copied.'),
]
ex1 = dict(word_id='WORD0739', label='순종', source_candidates=[s],
           form_assessment='PARTIAL_LATER_COMPONENT_WITH_UNEXPLAINED_EARLIER_ACTION',
           full_source_form_verified=False, compound_stage_identity='possible but unverified; no semantic name assigned',
           visible_source_uncovered_actions=obedience_unexplained,
           clauses=[
               clause('obedience_closed_hand',s,'주먹을 쥔 오른팔','right closed-hand appearance plus anatomical right arm', 'visible_partial_match',obedience_late,
                      'Closed-hand appearance occurs in the later component. Initial face-region hand is visibly open.',
                      ['Never infer fist from open_four=0 or index_only=0.', 'Do not impose a fist on the whole clip, initial approach or face-region action.']),
               clause('obedience_lateral_arm_then_lower',s,'옆으로 들었다가 내린다','right elbow/upper-arm lateral elevation followed by lowering', 'visible_partial_match',obedience_late,
                      'The later elbow/arm posture sequence is compatible with the source fragment. A right-wrist up/down output alone does not identify it.',
                      ['Initial right-hand up cannot be attached to this clause.', 'Need closed-hand state, lateral elbow/body relation and their temporal co-occurrence.', 'Keep later final arm return distinct from this candidate action; text alone does not resolve exact border.'],True),
           ],
           stage_hypotheses=[
               dict(alias='P036', observed_samples=[53,57,64,68,85,89,96,103,110,131,135,146],
                    organization='rest → open-hand face-region action → configuration/arm transition → later closed-hand arm component → held endpoint → return',
                    boundary_brackets_are_not_labels=True, prep_core_split='Unresolved for the initial face-region action; retain as source-uncovered action, not background.'),
               dict(alias='P033', observed_samples=[35,38,46,61,63,68,76,89,94,104],
                    organization='same visible order at different times; each frame identity remains clip-local', boundary_brackets_are_not_labels=True),
           ],
           unsupported=['Meaning of the initial face-region action', 'Whole lexical form equivalence', 'Required elbow height/angle or fist details', 'Complete obligatory phase list and tolerance'])

w = candidate('WORD1133','NIKL_legacy:9543')
wa = candidate('WORD1133','NIKL_legacy:41')
well_evidence = [
    evidence('P048',[70,73,79,88,91,97,103,106], 'Left arm is presented forward; flat right hand lies over/very near the left arm and progresses toward its distal region in the sampled sequence.',
             'Visual partial agreement with palm/left-arm brushing9543.', 'Depth/contact pressure and every-frame palm orientation are not certified. Source says left arm surface; a forearm-axis diagnostic covers only its supported portion.'),
    evidence('P045',[46,49,54,57,63,76,79,82], 'Flat right hand moves along the presented left arm toward the wrist region, followed later by lowering both arms.',
             'ID1 corroborates the arm-relative candidate.', 'Sampled view does not prove exact contact duration, pressure, or linguistic correctness.'),
]
ex2 = dict(word_id='WORD1133',label='잘하다',source_candidates=[w,wa]+[candidate('WORD1133',e) for e in ['NIKL_nuri:297','NIKL_nuri:1584','NIKL_nuri:295']],
           form_assessment='9543_ARM_RELATION_PARTIALLY_MATCHED_41_DIFFERENT_FORM',full_source_form_verified=False,
           clauses=[
               clause('well_palm_left_arm',w,'오른 손바닥으로 왼 팔등을','flat right palm appearance with left-arm spatial relation','visible_partial_match',well_evidence,
                      'Anatomical arm connections and flat right-hand appearance are visible. Mere hand overlap does not establish exact contact.',
                      ['Evaluate only while the left arm and right hand are both observed.', 'Do not substitute R_motion_down for palm/arm relation.']),
               clause('well_brush',w,'스쳐 내린다','right hand progression along the left-arm region','visible_partial_match',well_evidence,
                      'The sampled arm-relative path supports a brushing candidate. Image y alone is ambiguous under different left-arm pose.',
                      ['Initial two-arm setup and final lowering to the sides are separate candidates, not automatically the described stroke.', 'A shared left elbow→wrist axis may be a diagnostic where valid; it is not an exact-contact detector.'],True),
               clause('well_alternative_thumb',wa,'오른 주먹의 5지를 펴서 세워 전후로 두 번 움직인다','thumb-up right fist with forward/back repetitions','incompatible_with_selected_observed_component',[],
                      'The inspected clips show the two-arm brushing form, not this one-handed thumb-motion description.',
                      ['Do not merge this candidate with9543.', 'Do not require two repetitions from41 in the brushing-form clip.'],True),
           ],
           stage_hypotheses=[dict(alias='P048', observed_samples=[58,61,67,70,73,79,91,97,103,106,113,119],
                                  organization='rest/setup → right palm over left arm → arm-relative progression/endpoint → both-arm return',
                                  boundary_brackets_are_not_labels=True),
                             dict(alias='P045', observed_samples=[38,41,46,49,57,76,82,87,93], organization='same broad arm-relation sequence; timing stays clip-specific',boundary_brackets_are_not_labels=True)],
           unsupported=['Exact touch/pressure', 'Universal arm-axis or speed tolerance', 'Semantic choice among dictionary senses', 'Nuri candidates have no cached verbal description and cannot supply invented rules'])

a = candidate('WORD1567','NIKL_legacy:6521')
ancestor_evidence = [
    evidence('P066',[60,64,68,75,85,89,96,103,117], 'Both hands have visible raised thumbs with other fingers curled at inspected68/75. Right hand is higher at68, left at75, right at85/89 and left at96.',
             'Visible partial agreement with bilateral thumb-up alternating rises.', 'Crossing/overlap needs arm connection tracking. These samples do not establish exact required count or boundaries.'),
    evidence('P063',[54,57,60,66,69,75,78,96], 'The two compact thumb-up hand shapes alternate their high/low positions, then retain a final elevated configuration.',
             'ID1 supports the same broad two-hand relation.', 'Do not assign source meaning to every initial rise or treat final hold as proof of a trained stable class.'),
]
ex3 = dict(word_id='WORD1567',label='조상',source_candidates=[a],form_assessment='BILATERAL_THUMB_ALTERNATION_PARTIALLY_MATCHED',full_source_form_verified=False,
           clauses=[
               clause('ancestor_thumbs',a,'5지를 펴서 세운 두 주먹','both thumb-up compact hand configurations','visible_partial_match',ancestor_evidence,
                      'Cached official terminology maps5지 to thumb. Full-resolution68/75 visibly support thumbs rather than all-five-fingers-open.',
                      ['5지 is thumb; not five open fingers.', 'Current small motion head has no thumb/fist target.', 'Do not infer exact finger state from wrist motion.']),
               clause('ancestor_alternation',a,'번갈아 위로 올린다','alternating right/left upward movements with tracked anatomical side','visible_partial_match',ancestor_evidence,
                      'The high hand changes sides in the directly viewed sequence. This needs bilateral tracks and temporal relation.',
                      ['One R_up event cannot stand for both-hands alternation.', 'This source gives no numeric repeat requirement.', 'Preserve the final held configuration and return separately; other is not semantic hold.'],True),
           ],
           stage_hypotheses=[dict(alias='P066',observed_samples=[50,53,57,60,68,75,85,96,117,121,128,135,142],
                                  organization='rest/entry → multiple alternating rises → final elevated endpoint → return',boundary_brackets_are_not_labels=True,
                                  prep_core_split='First hand rise may already belong to the alternating sequence; do not force it to preparation.'),
                             dict(alias='P063',observed_samples=[45,48,51,54,60,69,75,96,102,108,114],organization='same broad alternating sequence, not a singleup event',boundary_brackets_are_not_labels=True)],
           unsupported=['Exact mandatory count', 'Correctness of every overlap/side identity without tracking support', 'Linguistic timing/height tolerance', 'A right-only learned class as complete source match'])

e1 = candidate('WORD2523','NIKL_legacy:23905')
e2 = candidate('WORD2523','NIKL_legacy:5990')
egypt_evidence = [
    evidence('P072',[73,76,79,85,88,91,94,103,106], 'Both hands form opposed sloping surfaces with near-convergent fingertips near the face, then separate/lower toward shoulder height. Later fingers are viewed mainly edge-on.',
             'Visual partial agreement with shared oblique pose→lowering fragment of23905/5990.', 'Exact all-fingertip contact and subsequent per-hand gathering cannot be certified from the edge-on final configuration.'),
    evidence('P069',[29,32,36,39,41,43,50,55], 'Similar high opposed-hand configuration changes to lower separated hands, then maintains that configuration before return.',
             'ID1 supports the common pose/lowering fragment.', 'Gathering and touch remain unverified; shared clauses count once, not as two independent source successes.'),
]
ex4 = dict(word_id='WORD2523',label='이집트',source_candidates=[e1,e2],form_assessment='SHARED_POSE_LOWERING_FRAGMENT_MATCHED_SUFFIX_UNRESOLVED',full_source_form_verified=False,
           clauses=[
               clause('egypt_tip_meeting',e1,'두 손의 손끝을 맞대어','near-convergent two-hand fingertips','visible_proximity_only',egypt_evidence,
                      'Near meeting and oblique outline are visible; every fingertip actually touching is not independently shown.',
                      ['Near overlap is not exact contact.', 'Do not make an automatic contact requirement.']),
               clause('egypt_oblique_pose',e1,'좌우로 비스듬히 세웠다가','opposed oblique hand posture preceding lowering','visible_partial_match',egypt_evidence,
                      'The high sloping pose is visible before the lower separated configuration.',
                      ['세웠다가 is a posture/order description; do not translate it into R_motion_up.', 'Initial transit up to the pose is not by itself the described core.'],True),
               clause('egypt_lower_shared',e2,'내린다','both-hands lowering after the opposed posture','visible_partial_match',egypt_evidence,
                      'The supported fragment is common to both candidates, despite their different suffixes.',
                      ['R_down alone is only one hand component.', 'Final return to the sides is distinct from the earlier lowering/shape change candidate.', 'Do not double count23905/5990.'],True),
               clause('egypt_gather_suffix',e1,'두 손 각각 손끝을 모은다','per-hand fingertip gathering after lowering','unverified_edge_on',egypt_evidence,
                      'The terminal hand shape is edge-on in the inspected dev frame94; small motion and10-feature outputs do not identify gathering.',
                      ['open_four=0 is not fingertip gathering.', '5990 does not contain this suffix; do not require it for every source candidate.', 'Missing/edge-on shape remains unknown.'],True),
           ],
           stage_hypotheses=[dict(alias='P072',observed_samples=[64,67,70,73,76,88,91,94,106,110,116,122],
                                  organization='rest/approach → opposed high posture → lowering/separating/shape transition → endpoint → return',boundary_brackets_are_not_labels=True),
                             dict(alias='P069',observed_samples=[23,25,29,32,39,41,43,55,59,64,68],organization='same broad pose→lowering sequence with clip-local timing',boundary_brackets_are_not_labels=True)],
           unsupported=['All-tip exact touch', 'Fingertip gathering suffix verification', 'Single mandatory source form', 'Numerical shape/timing tolerance'])

pipeline = dict(
    scope='Facts about the completed semantic_episode_pilot experiment; not a claim about every historical project model.',
    evidence=[code_pin,experiment_pin,pin(OLD/'FROZEN.json'),pin(OLD/'PREPARED.json')],
    input='Saved original-frame69geometry/masks. The small candidate selects8 right-side coordinate channels: actual hand wristxy,palm centerxy,body pose wristxy,elbowxy.',
    encoding=dict(relative_positions=8,adjacent_deltas=8,observed_masks=8,valid_delta_masks=8,window_displacement=8,normalized_time=1,total=41),
    architecture=frozen['model'],parameters=frozen['parameters'],seeds=frozen['seeds'],epochs=frozen['epochs'],
    targets=['up','down','other'],target_meaning='image-plane right-hand physical motion; other includes more than stillness; unknown is an evidence gate, notother',
    training_data=dict(fit_videos=12,development_videos=4,fit_draft_events=42,eligible_fit_events=31,development_draft_events=16,
                       label_origin='Codex RGB physical event drafts; not expert semantic/correctness labels'),
    direct_RGB_network_input=False,text_or_dictionary_embedding_input=False,word_id_as_model_input=False,
    finger_angles_as_small_head_input=False,source_description_used_as_training_target=False,
    RGB_role='Human-readable images inspected by Codex to draft/verify physical labels and phase hypotheses; not RGB pixels fed to this small head.',
    source_role='Cached description clauses linked after observation; they do not create missing coordinates, force observer predictions or provide video-time training targets.',
    handshape_role='Existing shape outputs remain separate. New small motion head does not learn fist,thumb-up,gathering,contact,preparation/core/return or meaning.',
    reference_role='Whole-reference savedDTW comparisons and fit3LOPO observed-only range are a downstream empirical comparison, not dictionary-trained semantic stages.',
    current_stage='This new unit is preparing supported phase/shape/body-relation evidence before any new classifier training; no such training is performed by this audit.',
    production='Old C/T3F and production defaults remain unchanged.')

report=dict(schema='wis_source_phase_contracts_v1',created_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
            author='Codex /root/guide_stage_map; same-model visual/source audit, not a human linguistic adjudicator',
            status='BOUNDED_SOURCE_PHASE_CANDIDATES_REVIEWED',
            activities=dict(new_training=0,new_inference=0,new_downloads=0,web_calls=0,private_data_access=0,source_edits=0),
            review_selection='All4development clips plus fixedID1fit clip per expression, chosen by role/record ID, not prior label or score success.',
            exposure='Reviewer previously audited old experiment scores. Those scores/predictions were not read to choose anchors or phase matches in this audit. Source text was intentionally available for this source audit; this is not a blinded RGB annotation pass.',
            clip_identity=clip_identity,
            terminology=dict(cached_verified_mapping=source['terminology_audit']['finger_numbering'],refreshed_online=False,
                             restriction='Use the specific recorded official numbered-finger context, not generic anatomical numbering.'),
            expressions=[ex1,ex2,ex3,ex4],
            common_mapping_restrictions=[
                'Sample timestamps come from existing per-clip manifests/nativePTS and inspected RGB; descriptions do not create time labels.',
                'Every listed extent is a set of inspected anchors, not a dense interval or exact phase boundary ground truth.',
                'Preparation/core/return are provisional organization for supported visual sequences. An unexplained visible action cannot simply be relabeledpreparation or discarded.',
                'A source clause may match only part of a multi-action video. Full form equivalence/mandatory semantics remain unresolved unless independently established.',
                'Link motion only when hand configuration, anatomical side, body relation and temporal context agree. Image-plane wristup/down alone is insufficient.',
                'Fist,thumb-up,gathering and exact touch are not current small-head outputs; use RGB draft evidence or explicit supported geometry diagnostics without pretending they are learned semantic labels.',
                'Normal anatomical arm crossing retains side identity via arm connections; do not swap left/right to make a clause match.',
                'No guessed labels for occluded/edge-on hands. Original expected scope and unknown support are preserved.',
                'Source matches remain descriptive research candidates, not correctness, requiredness or acceptance thresholds.',
            ],
            safe_next_evidence=[
                'Keep the four expressions and existing fit/dev split. Cross-check current independent phase drafts against these clip-specific anchors; do not propagate dev timing into fit targets.',
                'For순종, explicitly retain initial open-hand face action as source-uncovered; test later rightelbow/body trajectory with concurrent closed-hand evidence before connecting12209.',
                'For잘하다, prioritize right-hand position along a supported left-arm axis plus visible flat-hand configuration; report coverage and avoid exact-contact claims.',
                'For조상, require both side-tracked hands and temporalalternation with thumb evidence. For이집트, comparepose→lowering and leave23905-onlygathering unknown.',
                'Ask expert linguistic confirmation later only for unresolved form/compound identity and acceptance boundaries; bounded physical evidence work can continue now.',
            ],
            pipeline_factcheck=pipeline,prior_source_usage_audit=prior_usage,
            verification=dict(total=len(identity_checks),passed=sum(c['status']=='PASS' for c in identity_checks),checks=identity_checks),
            evidence_files=[dict(path=p,sha256=h) for p,h in sorted(pins.items())],
            limitations=['Direct current RGB coverage8/16, including all4dev; other8 only identity/time metadata checked here and remain for separate fit RGB reviewer.',
                         'Viewed sheets contain48sampled frames each, not every frame; no exact boundary or every-frame motion claim.',
                         'Six described candidates were checked against their exact cached officialHTML and original content hash. SOURCE_PROFILES contains four other expressions and is not source evidence for these four; no online refresh.',
                         'Physical/source fit is not a semantic correctness or permissible-variation truth set.'])
assert all(c['status']=='PASS' for c in identity_checks)
for p,h in pins.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
out=U/'SOURCE_PHASE_CONTRACTS.json'
with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
memo='''현재 실제 학습과 이번 설명 연결 작업

완료한 작은 모델은 수어 설명 문장을 배운 모델이 아니다. 공개 영상에서 이미 얻은 손목·손바닥 중심·팔꿈치 등의 좌표와 관측 여부를 시간순으로 받아, 오른손이 화면에서 올라가는지·내려가는지·그 밖의 움직임인지를 배웠다. 기존69기하값 중8좌표를 사용하며 상대위치/변화/관측마스크/전체변위/시간으로41입력값을 만든다. 손가락 각도나 RGB 픽셀, 사전 문장, 단어 ID는 이 작은 모델 입력이 아니다.

RGB 영상은 Codex가 물리 동작의 정답 초안을 만들고 확인하는 데 썼다. 12fit 영상의42초안 구간 중 관측이 충분한31개로 학습했고, 4dev 영상16구간으로 따로 확인했다. 이 초안은 전문가의 수어 의미·정오 정답이 아니다. 작은 모델의 기타(other)는 준비·정지·핵심·마무리를 구분하는 출력도 아니다.

사전 설명은 관측이 끝난 뒤 붙이는 연결 근거다. 관측을 설명에 맞게 고치거나, 설명을 보고 영상 시간을 정답으로 복사하지 않는다. 손모양 판단과 기존 C/T3F는 별도이며 유지된다. 준비/핵심/마무리 및 손모양·팔관계를 학습한 새 단계 모델은 이번 감사에서 만들거나 실행하지 않았다.

이번 실제 RGB 대조에서 확인한 연결
- 순종: 처음 펼친 오른손을 얼굴 앞으로 올리는 동작이 먼저 보인다. 이를 곧바로 사전12209의 주먹 든 오른팔 동작으로 연결하면 안 된다. 뒤쪽의 주먹·옆으로 든 팔꿈치·팔꿈치 내림이 별도 부분 후보다. 앞선 얼굴 동작을 단순 준비로 지우거나 그 의미를 추측하지 않았다.
- 잘하다: 직접 본 영상은 오른 손바닥이 왼팔 위를 따라 움직이는9543 후보와 부분적으로 맞는다. 같은 이름의 엄지 전후 두 번 움직임41 후보를 합치지 않는다. 화면상 아래로 간다는 값보다 두 팔의 상대 위치가 필요하다.
- 조상: 양손 엄지를 세운 모양과 번갈아 높아지는 모습이 보인다. 오른손 상승 하나로 양손 교대를 확인할 수 없다. 여기서5지는 엄지라는 기존 공식 문맥을 따른다.
- 이집트: 높은 위치에서 두 손이 비스듬히 마주 선 모습 뒤에 낮아지고 벌어지는 모습이 보인다. 세웠다는 설명을 상승 동작 정답으로 바꾸지 않는다. 손끝 실제 접촉과23905에만 있는 손끝 모음은 확정하지 않았다.

공개16개의 영상 ID·출처해시 대응·프레임/시간 기록을 확인했다. 이번 직접 RGB 재검토는 dev4 전부와 표현별ID1fit4, 총8개다. 48장씩의 시트와 dev 상세10장을 보았으며, 나머지8개에 본 적 없는 모양을 채우지 않았다. 시간은 본 프레임의 근거이며 정확한 단계 경계 정답은 아니다. SOURCE_PHASE_CONTRACTS.json에 출처 페이지·문장·프레임·해시와 미확인 항목을 분리했다.
'''
with (U/'TRAINING_PIPELINE_FACTCHECK_KO.txt').open('x',encoding='utf8') as f:f.write(memo)
print(json.dumps(dict(output=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),checks=len(identity_checks),passed=len(identity_checks),expressions=4,clauses=sum(len(e['clauses']) for e in report['expressions']),viewed_clips=8),ensure_ascii=False))
