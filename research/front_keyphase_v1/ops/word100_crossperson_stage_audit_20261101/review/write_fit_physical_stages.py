"""Serialize manually reviewed RGB observations; no prediction/model/search inputs."""
from pathlib import Path
import json, hashlib

U=Path(__file__).resolve().parents[1]
S=U.parent/'word100_core_stage_evidence_20261031'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('x',encoding='utf8') as f: json.dump(d,f,ensure_ascii=False,indent=2)
phase={c['alias']:c for c in read(S/'PHASE_DATA.json')['clips']}
contacts=read(U/'evidence/FIT_CONTACT_MANIFEST.json')['clips']
dense=read(U/'evidence/FIT_DENSE_MANIFEST.json')['sheets']

# These intervals were manually entered after actually viewing all listed RGB sheets.
# (short kind code, certain interior, broad-kind motion subphase, uncertain outer)
K={
 'face':('right_open_hand_near_lower_face','오른손을 펼쳐 아래 얼굴 앞에 가까이 둠','R flat/open grouped fingers; L lowered','R hand near mouth/lower nose; exact touch unknown'),
 'down':('right_closing_hand_descends_to_lateral_chest','오른손을 닫아 가며 오른쪽 가슴 옆으로 내림','R open-to-flexed transition; L lowered','R forearm moves toward lateral upper chest, becoming more horizontal'),
 'side':('right_closed_hand_reorients_toward_shoulder','닫힌 오른손을 어깨 쪽으로 재지향한 뒤 유지','R flexed/closed hand; L lowered','R forearm turns toward upright, hand beside shoulder; movement/hold separated'),
 'sweep':('right_flat_hand_moves_along_left_forearm','편 오른손이 왼전완을 따라 원위 쪽으로 이동','R flat/open; L exact fingers unknown','L forearm across chest; R moves from elbow-side toward wrist-side; contact not proven'),
 'distal':('right_flat_hand_holds_over_left_distal_forearm','편 오른손을 왼쪽 원위 전완 위에 유지','R flat/open; L exact fingers unknown','R over L distal forearm/wrist; relation held despite small settling'),
 'riseR':('right_thumb_up_hand_rises_above_left','엄지를 든 오른손이 왼손보다 위로 상승','Both nonthumb fingers flexed, thumbs up','R rises across/above L; crossing may obscure precise support'),
 'riseL':('left_thumb_up_hand_rises_above_right','엄지를 든 왼손이 오른손보다 위로 상승','Both nonthumb fingers flexed, thumbs up','L rises across/above R; crossing may obscure precise support'),
 'highL':('left_thumb_up_hand_high_hold','엄지를 든 왼손을 오른손 위쪽 높은 위치에 유지','Both nonthumb fingers flexed, thumbs up','L high near/above face, R lower near upper chest'),
 'converge':('bilateral_open_hands_converge_near_face','양 편손을 얼굴 앞에서 모은 뒤 가까이 유지','Both open/flat, grouped extended fingers','Hands approach centrally, triangular gap visible; exact fingertip contact unknown'),
 'apart':('bilateral_open_hands_separate_outward_down','양 편손을 바깥 아래쪽으로 벌림','Both open/flat, edge-on during turn','Bilateral separation and slight downward movement near face/upper chest'),
 'apartHold':('bilateral_open_hands_hold_apart_upper_chest','양 편손을 상부 가슴 양쪽에 벌린 채 유지','Both open/flat; precise palm normal unknown','Hands apart around shoulder width and upper-chest/shoulder height')
}
# Each list corresponds to the existing source core, never an invented denominator.
D={
'P033':[
 ('face',1,[('face',[48,58],'hold',[43,61])]),
 ('down',1,[('down',[63,68],'closing_and_descending',[58,73])]),
 ('side',1,[('side',[73,78],'reorientation',[71,80]),('side',[81,89],'hold',[79,94])])],
'P034':[
 ('face',1,[('face',[42,60],'hold',[38,63])]),
 ('down',1,[('down',[64,71],'closing_and_descending',[60,77])]),
 ('side',1,[('side',[77,83],'reorientation',[74,85]),('side',[86,96],'hold',[84,99])])],
'P035':[
 ('face',1,[('face',[54,67],'hold',[48,70])]),
 ('down',1,[('down',[71,80],'closing_and_descending',[67,86])]),
 ('side',1,[('side',[86,90],'reorientation',[82,92]),('side',[93,101],'hold',[91,105])])],
'P045':[
 ('sweep',1,[('sweep',[50,59],'along_forearm',[44,61]),('distal',[62,63],'hold',[60,65])]),
 ('distal',1,[('distal',[65,79],'hold',[63,82])])],
'P046':[
 ('sweep',1,[('sweep',[50,60],'along_forearm',[43,64])]),
 ('distal',1,[('distal',[67,75],'hold',[64,77])])],
'P047':[
 ('sweep',1,[('sweep',[54,62],'along_forearm',[48,65])]),
 ('distal',1,[('distal',[68,80],'hold',[65,83])])],
'P063':[
 ('riseR',1,[('riseR',[51,54],'rise',[48,57])]),
 ('riseL',1,[('riseL',[58,60],'rise',[54,63])]),
 ('riseR',2,[('riseR',[66,68],'rise',[60,72])]),
 ('riseL',2,[('riseL',[74,76],'rise',[69,81])]),
 ('highL',2,[('highL',[81,96],'hold',[78,99])])],
'P064':[
 ('riseR',1,[('riseR',[43,47],'rise',[40,51])]),
 ('riseL',1,[('riseL',[52,55],'rise',[48,59])]),
 ('riseR',2,[('riseR',[62,66],'rise',[56,70])]),
 ('riseL',2,[('riseL',[72,75],'rise',[67,77]),('highL',[78,78],'hold',[76,80])]),
 ('highL',2,[('highL',[80,99],'hold',[78,102])])],
'P065':[
 ('riseR',1,[('riseR',[38,42],'rise',[35,46])]),
 ('riseL',1,[('riseL',[47,50],'rise',[43,53])]),
 ('riseR',2,[('riseR',[56,60],'rise',[51,63])]),
 ('riseL',2,[('riseL',[65,68],'rise',[61,70]),('highL',[70,71],'hold',[69,73])]),
 ('highL',2,[('highL',[73,81],'hold',[71,84])])],
'P069':[
 ('converge',1,[('converge',[29,32],'converging',[27,33]),('converge',[34,38],'hold',[33,41])]),
 ('apart',1,[('apart',[41,43],'separating_outward_down',[39,45])]),
 ('apartHold',1,[('apartHold',[45,55],'hold',[43,57])])],
'P070':[
 ('converge',1,[('converge',[47,49],'converging',[44,50]),('converge',[50,54],'hold',[49,56])]),
 ('apart',1,[('apart',[57,59],'separating_outward_down',[54,61])]),
 ('apartHold',1,[('apartHold',[61,80],'hold',[59,82])])],
'P071':[
 ('converge',1,[('converge',[55,58],'converging',[53,59]),('converge',[60,66],'hold',[58,69])]),
 ('apart',1,[('apart',[69,74],'separating_outward_down',[66,77])]),
 ('apartHold',1,[('apartHold',[77,84],'hold',[74,87])])]
}
clips=[]; images=[]; totalcores=0
for contact in contacts:
    alias=contact['alias']; metadata=read(S/'media'/f'{alias}.json')
    originals=[s for s in phase[alias]['segments'] if s['role']=='core_candidate']
    assert len(originals)==len(D[alias])
    ds=[x for x in dense if x['alias']==alias]
    denseframes=sorted({f for x in ds for f in x['frames']})
    views=[dict(path=str(U/'evidence'/alias/'whole_contact.jpg'),frames=contact['whole_contact_frames'],view_type='whole_clip_24_samples'),
           dict(path=str(U/'evidence'/alias/'core_contact.jpg'),frames=sorted({f for row in contact['core_contact_rows'] for f in row['frames']}),view_type='source_core_samples')]
    views.extend(dict(path=x['path'],frames=x['frames'],view_type='continuous_native_core_bracket_frames') for x in ds)
    if alias=='P033': views.append(dict(path=str(U/'evidence/P033/frame0051.jpg'),frames=[51],view_type='native_full_frame'))
    for v in views: v['actually_viewed_via_view_image']=True;v['sha256']=sha(Path(v['path']))
    images.extend(views)
    segs=[]
    for orig,(broad,occ,rawparts) in zip(originals,D[alias]):
        totalcores+=1; lo,hi=orig['core'];parts=[];seen=set()
        for idx,(code,interior,subphase,outer) in enumerate(rawparts):
            a,b=interior; assert lo<=a<=b<=hi
            assert orig['start_bracket'][0]<=outer[0]<=a<=b<=outer[1]<=orig['end_bracket'][1]
            frames=set(range(a,b+1));assert not (seen&frames);seen|=frames
            parts.append(dict(part_id=f"{orig['segment_id']}_part{idx+1}",core=interior,certain_interior=interior,outer=outer,
                kind_key=K[code][0],proposed_kind_key=K[code][0],motion_subphase=subphase,
                description_ko=K[code][1],evidence_frames=list(range(a,b+1)),
                evidence_pts_seconds=[metadata['source_pts_seconds'][f] for f in range(a,b+1)],
                support='direct_RGB_continuous_native_frames_in_contact_sheet'))
        center=(lo+hi)//2
        cp=next((p for p in parts if p['core'][0]<=center<=p['core'][1]),None)
        unknown=[f for f in range(lo,hi+1) if f not in seen]
        assert set(range(lo,hi+1))<=set(denseframes)
        perframe=[dict(frame=f,kind_key=next((p['kind_key'] for p in parts if p['core'][0]<=f<=p['core'][1]),None),
            motion_subphase=next((p['motion_subphase'] for p in parts if p['core'][0]<=f<=p['core'][1]),None)) for f in range(lo,hi+1)]
        segs.append(dict(segment_id=orig['segment_id'],source_role='core_candidate',core=orig['core'],
            start_bracket=orig['start_bracket'],end_bracket=orig['end_bracket'],
            proposed_kind_key=K[broad][0],kind_key=K[broad][0],description_ko=K[broad][1],
            occurrence_index_candidates=[occ],cycle_index_candidates=([occ] if broad in ['riseR','riseL','highL'] else None),
            occurrence_note='Visible within-clip occurrence candidate only; no required count or cross-person one-to-one identity.',
            source_center_frame=center,source_center_kind_key=cp['kind_key'] if cp else None,
            source_center_motion_subphase=cp['motion_subphase'] if cp else None,
            parts=parts,framewise_physical_draft=perframe,unknown_core_frames=unknown,
            visible_features=dict(handshape=K[broad][2],body_relation=K[broad][3],exact_finger_joint_angles=None,palm_normal_3d=None,touch_pressure=None),
            evidence_frames=list(range(lo,hi+1)),
            unknown_note_ko='parts의 확실 내부 밖은 경계·전환·해상도상 미확정이다. 손가림이나 미확정을 동작 오류로 바꾸지 않는다.',
            linguistic_necessity=None,sign_correctness=None))
    unique=sorted({f for v in views for f in v['frames']})
    clips.append(dict(alias=alias,role='fit',native_frame_count=metadata['frames'],
        source_webm=str(S/'media'/f'{alias}.webm'),source_webm_sha256=sha(S/'media'/f'{alias}.webm'),
        source_timing_metadata=str(S/'media'/f'{alias}.json'),source_timing_sha256=sha(S/'media'/f'{alias}.json'),
        inspected_images=views,reviewed_native_frame_ids=unique,continuous_core_bracket_frames=denseframes,
        whole_clip_dense_reviewed=False,all_original_core_native_frames_reviewed=True,segments=segs))
assert totalcores==39 and len(clips)==12 and len(dense)==30
out=dict(schema_version=1,status='RGB_PHYSICAL_STAGE_DRAFT_FOR_PREFREEZE_REVIEW',reviewer='same Codex subagent B',
    frame_convention='zero-based original native indices; inclusive endpoints; PTS copied from source media JSON',
    scope=dict(fit_clips=12,original_source_cores=39,continuous_core_bracket_native_frames=sum(len(c['continuous_core_bracket_frames']) for c in clips),
        dense_core_sheets=30,whole_clip_contact_sheets=12,whole_clip_samples_per_clip=24,core_contact_sheets=12,native_detail_images=1,
        total_actual_image_views=len(images),unique_viewed_clip_frames=sum(len(c['reviewed_native_frame_ids']) for c in clips),
        whole_clip_dense=False,person_pair_predictions_opened_for_annotation=False,new_model_inference=False),
    source_pins={str(p):sha(p) for p in [S/'PHASE_DATA.json',U/'evidence/FIT_CONTACT_MANIFEST.json',U/'evidence/FIT_DENSE_MANIFEST.json']},
    interpretation_policy=[
        'Original39 source core IDs/brackets retained; parts add physical resolution without changing denominator.',
        'Only parts.core/certain_interior supports framewise kind. Outer intervals are uncertain extent, not positive labels; unassigned frames are null.',
        'Broad kind and motion_subphase are separate. Reorientation then hold may share broad kind; a core center can be hold even when original broad event includes motion.',
        'Same physical kind may repeat. R-rise first/second and L-rise first/second share kinds; no one-to-one repetition mapping is forced.',
        'Kind keys describe visible hands/motion/body relation, not word IDs, sign meaning, obligatory stage or correct/incorrect signing.'
    ],
    limitations=[
        'Same Codex RGB draft, not independent human or Deaf teacher truth; previous project/development results were already known from conversation, so not strictly blinded.',
        'Current person-pair predictions and matching-result paths were not opened for this annotation. Source PHASE_DATA was used for IDs/brackets; one schema inspection after RGB review exposed P033 label/preparation metadata, not used to infer annotations.',
        'All 552 native frames within the selected core/bracket windows were viewed in 30 continuous sheets; full-video coverage outside them is 24 samples per clip, not all native frames.',
        'Display WebM is compressed 960x540; most inspection tiles crop body region and resize to 300x273. Fine fingers, depth, exact contact and palm normals remain unknown when unsupported.',
        'Intervals are conservative physical drafts; viewing every frame of core windows does not make them linguistic dense ground truth.',
        'Common vocabulary was discussed with DEV reviewer after independent observations. This vocabulary coordination is not independent agreement evidence; DEV schema was read only after FIT RGB observation.'
    ],kind_catalog=[dict(kind_key=v[0],description_ko=v[1],handshape=v[2],body_relation=v[3]) for v in K.values()],clips=clips)
save(U/'review/FIT_PHYSICAL_STAGES.json',out)
save(U/'review/FIT_IMAGES_VIEWED.json',dict(status='ACTUAL_VIEW_LOG',images=images,scope=out['scope'],generation_manifest_viewed_false_is_preinspection_only=True))
memo='''FIT RGB 물리 단계 검토 초안\n\n12개 공개 참고 영상, 기존 39개 source core의 ID·core·경계 bracket을 보존했다. 기존 개인쌍 예측은 열지 않았다. 12개 전체시점 표본 시트(각24프레임),12개 core 표본 시트, core/bracket 구간 연속30시트 552프레임과 P033 원본 상세1장을 실제 보았다. 전체 영상의 모든 native 프레임을 본 것은 아니다.\n\nP033/034/035의 s04는 어깨쪽 재지향과 뒤의 유지가 같은 원 core에 섞였다. 각각 재지향73–78/77–83/86–90, 유지81–89/86–96/93–101로 확실 내부를 나누었다. 원 core 중심81/86/93은 유지에 속한다. 같은 broad kind에 motion_subphase를 별도 기록했다.\n\n전완을 따라 이동한 뒤 관계를 유지하는 구간, 양손을 얼굴 앞에서 모은 뒤 유지하는 구간도 parts로 나눴다. 애매한 전환은 null이다. 엄지를 든 오른손 상승의 두 회차, 왼손 상승의 두 회차는 각각 같은 종류로 기록했다. 회차가 다르다고 다른 동작 종류로 강제하지 않으며, 반복 횟수의 언어적 필수성도 판정하지 않는다.\n\n손모양은 보이는 거친 형태만 기록했다. 정확한 손가락 각도·손바닥 3D 법선·접촉 압력은 미확정이다. 가림/경계/미확정은 오류가 아니다. parts의 outer는 불확실 범위이고 확실 라벨은 core 내부뿐이다. 원 core의 남은 프레임을 자동 보간하지 않았다.\n\n이것은 같은 Codex의 물리 관측 초안이며 독립 사람 수어 정답이나 인식 정확도가 아니다. 이전 개발 결과를 대화에서 알고 있어 엄격한 맹검도 아니다. 두 검토자 사이의 어휘 조율은 독립 합의 증거로 세지 않는다. 원본·모델·과거 결과는 변경하지 않았다.\n'''
(U/'review/FIT_PHYSICAL_STAGES_KO.txt').write_text(memo,encoding='utf8')
print(json.dumps(dict(status=out['status'],scope=out['scope'],original_cores=totalcores,source_centers_unknown=sum(s['source_center_kind_key'] is None for c in clips for s in c['segments']),sha256=sha(U/'review/FIT_PHYSICAL_STAGES.json')),ensure_ascii=False))
