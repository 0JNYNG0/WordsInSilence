"""Record actual RGB review without changing the original annotation drafts."""
from pathlib import Path
import hashlib,json,datetime
U=Path(__file__).resolve().parent; H=U.parent; G=H.parent
def read(p): return json.loads(p.read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with p.open('x',encoding='utf-8') as f: json.dump(d,f,ensure_ascii=False,indent=2)
base=G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json'
added=G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json'
prior=H/'observed_action_anchor_design/CORRECTION_BOUNDARY_HOLD.json'
assert read(base)['training_allowed'] is False and read(added)['training_allowed'] is False
notes={
('WORD0739',3):'얼굴 앞 손을 닫으며 내린 뒤 팔뚝을 세운다. 94/96/102의 세운 자세 유지가 dense와 overview에 보이고 108에는 마무리 하강한다.',
('WORD0739',7):'손을 닫으며 내려 62 부근에 도달하고 66~70 팔뚝을 세운다. 74/76/78 유지. 72는 끝 경계와 겹쳐 학습 음성에서 제외한다.',
('WORD1133',3):'오른손이 반대팔 위쪽에서 손목 쪽으로 이동한다. 68~78 손과 반대팔 관계는 유지되고 이후 84에서 내려간다. 몸의 작은 흔들림은 새 쓸기 동작으로 보지 않는다.',
('WORD1133',7):'54~64 반대팔을 따라 이동, 66 끝 부근 도달 후 68/70/72 관계 유지. 78 이후 마무리로 내려간다.',
('WORD1567',3):'양손 상대 높이가 번갈아 바뀐 뒤 72/74/76/78 마지막 자세 유지. 관측된 반복과 유지일 뿐 언어적으로 필요한 반복 횟수 정답은 아니다.',
('WORD1567',7):'두 손의 높이 교대 후 64~66 끝 부근, 68/70/72 마지막 배치 유지. 78 이후 손을 내린다.',
('WORD2523',3):'중앙에서 모양을 만든 양손이 옆·아래로 벌어지고 80/82/84 위치관계 유지. 78은 끝 경계에 겹쳐 제외. 손가락 모으기 완료 여부는 이 정답의 범위 밖이다.',
('WORD2523',7):'44 부근 중앙의 양손에서 48~52 옆·아래로 벌어진 뒤 56/58/60 위치관계 유지. 66 이후 마무리. 손끝 상세 모양의 정답을 추가하지 않는다.'}
rows=[]; image_pins={}; source_pins={}
for r in read(added)['rows']:
    assert r['person'] in [3,7] and r['role']=='fit'
    assert sha(r['source'])==r['source_sha256'];source_pins[r['source']]=r['source_sha256']
    assert len(r['sheets'])==2
    for s in r['sheets']:
        assert sha(s['path'])==s['sha256'];image_pins[s['path']]=s['sha256']
    excluded=[f for f in r['hold_sample_frames'] if any(a<=f<=b for e in r['events'] for a,b in [e['onset'],e['offset']])]
    rows.append(dict(clip_id=r['clip_id'],person=r['person'],note=notes[r['word_id'],r['person']],
        reviewed_sheets=r['sheets'],sampled_hold_frames=r['hold_sample_frames'],excluded_boundary_hold_frames=excluded,
        eligible_hold_frames=[f for f in r['hold_sample_frames'] if f not in excluded],
        reviewer='root Codex actual RGB dense and overview inspection; no independent human/KSL expert'))
assert len(rows)==8 and len(image_pins)==16
review=U/'ADDITIONAL_RGB_REVIEW.json'
write(review,dict(at=datetime.datetime.now().astimezone().isoformat(),status='ACTUAL_EIGHT_SOURCE_RGB_REVIEW_COMPLETE',rows=rows,
    images=image_pins,sources=source_pins,source_annotations_sha256=sha(added),
    scope='Sparse body progress and maintained body pose. Boundary uncertainty remains masked. No hand articulation, correctness, absence or required completion labels.',
    annotation_flags_unchanged=True,no_new_video_or_pose_extraction=True))
base_rows=[r for r in read(base)['rows'] if r['person'] in [2,4,5,6]]
previous=read(prior)['rows'];assert {r['clip_id'] for r in previous}=={r['clip_id'] for r in base_rows}
ids=sorted([r['clip_id'] for r in base_rows]+[r['clip_id'] for r in rows]);assert len(ids)==len(set(ids))==24
write(U/'SOURCE_ADMISSION.json',dict(at=datetime.datetime.now().astimezone().isoformat(),status='QUALIFIED_FOR_SPARSE_BODY_PROGRESS',
    training_allowed=True,scope='sparse_body_progress_only',source_annotations={str(base):sha(base),str(added):sha(added)},
    approved_clip_ids=ids,original_flags_preserved=True,review_evidence={str(prior):sha(prior),str(review):sha(review)},
    admission_basis='User-authorized public local annotation review, followed by root actual RGB inspection and separate code/array provenance review.',
    base_review='Prior saved actual RGB review of 16 FIT clips retained; additional8 directly reviewed in this resumed unit.',
    excluded='All uncertain boundary negative points; unseen dense frames; user/mentor/final8; handshape completion; educational correctness.',
    limitation='Same Codex visual assessment, not independent human or KSL expert annotation. Limited body-motion continuation only. Does not retroactively authorize any earlier experiment.',
    first_failure_preserved=True))
print('Reviewed8 and admitted24 sparse body-only FIT sources; original false flags retained.')
