"""Existing-public-FIT annotation contract inventory, not new RGB annotation."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
S=Path(__file__).resolve().parent
G=S.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    dest=S/'SHAPE_TRANSITION_EVIDENCE_REVIEW.json';assert not dest.exists()
    oldp=G/'reviews/obedience_occurrence_labels.json';old=read(oldp)
    sourcepaths=[G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',
                 G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json']
    rr={r['person']:r for p in sourcepaths for r in read(p)['rows'] if r['word_id']=='WORD0739' and r['person'] in [2,3,4,5,6,7]}
    assert set(rr)=={2,3,4,5,6,7}
    existing=[]
    for c in old['clips']:
        if c['recorded_performer_id'] not in [4,7]:continue
        for o in c['occurrences']:
            if o['guide_clause'] not in ['open_front','close_chest','raise_forearm']:continue
            existing.append(dict(clip_id=c['clip_id'],person=c['recorded_performer_id'],
                source_sha256=c['source_sha256'],clause=o['guide_clause'],
                direct_observations=[{k:r[k] for k in ['frames','phase','shape_evidence',
                    'role_supervision_frames_outside_boundary_bands','role_supervision_deferred_boundary_frames']}
                    for r in o['direct_observations']]))
    proposed=[]
    for person,r in sorted(rr.items()):
        families={g['family']:g for g in r['groups']}
        a=max(0,int(families[2]['onset'][0])-4)
        b=min(r['frames']-1,int(families[3]['offset'][1])+4)
        proposed.append(dict(clip_id=r['clip_id'],person=person,source=r['source'],source_sha256=r['source_sha256'],
            proposed_inspection_range=[a,b],frames=b-a+1,
            rule='source family2 outer onset minus4 through family3 outer offset plus4, clipped to source length; fixed from existing labels before new images',
            existing_motion_events=[e for e in r['events'] if e['family'] in [2,3]],
            new_images_created=False,new_handshape_labels_created=False))
    report=dict(at=datetime.now(timezone.utc).isoformat(),reviewer='/root/absence_contract_review_1113',
        task='Read-only existing evidence inventory for public FIT0739 REAL02..07',
        existing_detailed_contract_path=str(oldp),existing_detailed_contract_sha256=sha(oldp),
        sampling_policy=old['sampling_policy'],boundary_policy=old['boundary_policy'],existing_shape_annotations=existing,
        finding='Existing direct RGB sampled states explicitly separate closing-and-lowering from already-closed lowering and closed-hand forearm raising for REAL04/07. The current SOURCE_EPISODES files structure movement bands/witnesses for all six, but do not give a separate per-frame closing-vs-already-closed hand-state contract for02/03/05/06.',
        important_limit='REAL07 closing52/54 are both in role-boundary exclusion bands; REAL04 closing88/90 leaves only90 outside those bands. Do not equate existing family2 extent labels with dense hand-closing supervision.',
        proposed_short_contract=dict(status='PROPOSAL_ONLY_ROOT_TO_FREEZE_BEFORE_NEW_RGB_REVIEW',
            sources=proposed,total_proposed_frames=sum(x['frames'] for x in proposed),
            observation_labels=['visible_extended','visible_closed','visible_flexing_transition','uncertain_or_self_occluded'],
            sequence_labels=['directly_supported_extended_to_closed_change','closed_maintained_across_viewed_interval','endpoints_differ_transition_unobserved','insufficient_evidence'],
            body_relation_labels='Record lowering/forearm-raising relation separately from finger-state change; never infer finger closure from wrist motion alone.',
            frame_policy='Inspect every original frame in fixed range with face/forearm context and native hand crop. Mark actual inspected frame IDs and hashes. No missing frame becomes direct truth by interpolation.',
            already_closed_policy='If first inspected state is already closed, say already closed at first observed frame; do not infer absence of earlier closing outside the inspected range.',
            uncertainty_policy='Hand rotation/perspective/self-occlusion may hide fingers; endpoint state differences do not create exact intervening closure labels.',
            scope='Generic shape-change vs shape-maintenance concept. First6 FIT sources are evidence collection only, not a word-specific exception. No required finger angles, semantic completion, educational correctness or final-test claim.',
            no_training_or_evaluation_yet=True),
        reviewed_files={str(p):sha(p) for p in sourcepaths+[oldp,G/'reviews/build_obedience_occurrence_labels.py',
            G/'local_motion_quality/AUX_WITNESS_REVIEW.json',G/'local_motion_quality/build_aux_witness_review.py',G/'CAPABILITY_CONTRACT.json']},
        scope_disclosure='One rg search unexpectedly returned DEV11/13 clip-ID/card string references from an existing CONTRACT_AUDIT file. No raw DEV video/keypoints/scores were opened, no such information used in this contract; that file was then excluded.',
        source_video_opened=False,new_frame_extraction=False,new_visual_annotation=False,
        model_inference=False,training=False,private_or_mentor_input_access=False)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['손모양 전환 기존 근거 검토(새 주석 아님)','',
        '기존 계약이 있다. reviews/obedience_occurrence_labels.json은 공개 REAL04/07에서 손가락을 접는 과정, 이미 닫힌 손의 내림, 닫힌 손을 유지한 팔뚝 세우기를 구분한다. 2프레임 간격 표본이며 원본 전프레임 정답은 아니다.',
        'REAL04: 열린 손70~86 표본 → 접힘 과정88/90 → 닫힌 손92/94/96 → 닫힌 손을 유지한 전완 세움104/106/108/110.',
        'REAL07: 열린 손38~50 표본 → 접힘 과정52/54 → 닫힌 손56/58/60 → 닫힌 손 유지 전완 세움66/68/70.',
        '중요: REAL07의 접힘 표본52/54는 기존 역할 경계 보류에 모두 걸리고, REAL04의88/90 중 강한 역할 학습에 남는 것은90뿐이다. 현재 움직임 구간 라벨만으로 접힘 변화를 충분히 가르쳤다고 볼 수 없다.',
        '현재 SOURCE_EPISODES에는02~07 모두 팔 움직임의 시작·끝·진행 표본이 있지만,02/03/05/06의 접힘 과정 대 이미 닫힌 상태를 별도 프레임 필드로 검증한 계약은 이번 검토 자료에서 찾지 못했다.',
        '다음 최소 계약: 기존 family2 시작4프레임 전부터 family3 끝4프레임 후까지 공개6원본 전프레임을 실제RGB로 확인한다. 열린/닫힌/접히는중/불확실을 기록하고, 손목 내림과 손가락 접힘은 별개로 다룬다. 가려진 사이는 강제 정답으로 메우지 않는다. 새 이미지나 정답 생성은 아직 하지 않았다.','']
    lines += [f"ID{x['person']:02}: 제안 확인범위{x['proposed_inspection_range']} ({x['frames']}프레임)" for x in proposed]
    (S/'SHAPE_TRANSITION_EVIDENCE_REVIEW_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(existing_shape_contract_people=[4,7],proposed_frames=report['proposed_short_contract']['total_proposed_frames'],new_rgb_annotation=False)))

if __name__=='__main__':main()
