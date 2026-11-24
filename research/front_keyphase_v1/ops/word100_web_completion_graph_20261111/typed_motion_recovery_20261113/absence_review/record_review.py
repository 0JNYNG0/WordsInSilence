"""Record the agent's completed direct RGB review, not an inferred model label."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone

P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))

# These annotations were authored after viewing all eight contact sheets with
# view_image. Full-resolution endpoints were additionally viewed for two rows.
DECISIONS={
 'NIA_SL_WORD0739_REAL02_F':([84,96],[],
   '오른팔을 세우고 오른주먹을 어깨 부근에 둔 자세가 유지된다. 약한 몸통·손 위치 변화가 있지만 새로 팔을 세우는 진행은 보이지 않는다.'),
 'NIA_SL_WORD0739_REAL03_F':([94,100],[101,102],
   '94~100은 세운 오른팔과 주먹 자세를 유지한다. 101~102는 주먹·전완이 조금 아래로 이완되기 시작해 순수 유지 정답에서는 제외한다. 이 작은 변화를 주요동작 또는 교육적 오류로 판단하지 않는다.'),
 'NIA_SL_WORD1133_REAL02_F':([64,72],[],
   '오른손이 반대팔의 손목 가까이에 놓인 관계가 유지된다. 양팔의 작은 자세 안정화와 별개로 팔을 따라 쓸어가는 추가 진행은 확인되지 않는다.'),
 'NIA_SL_WORD1133_REAL03_F':([68,78],[],
   '오른손과 반대팔 손목 부근의 관계를 유지한다. 몸통·팔의 작은 이동은 있으나 팔을 따라 쓸어가는 추가 진행은 확인되지 않는다.'),
 'NIA_SL_WORD1567_REAL02_F':([80,100],[],
   '양손의 위아래 배치와 엄지를 세운 모습이 유지된다. 손목 높이의 미세한 이완을 새 교대·회전 동작으로 분류하지 않는다. 이 구간 안에서 추가 교대 진행은 보이지 않는다.'),
 'NIA_SL_WORD1567_REAL03_F':([72,78],[],
   '양손의 위아래 배치를 유지한다. 두 팔이 조금 안정화되지만 어느 손이 위에 있는지 바뀌는 추가 교대 진행은 보이지 않는다.'),
 'NIA_SL_WORD2523_REAL02_F':([64,80],[],
   '좌우로 떨어진 양손의 배치를 유지한다. 손·팔의 작은 하강 이완이 있지만 양손 간격을 더 벌리는 진행은 확인되지 않는다. 손가락 접힘의 정오·완료는 이 주석의 대상이 아니다.'),
 'NIA_SL_WORD2523_REAL03_F':([78,82],[83,84],
   '78~82는 좌우로 벌어진 양손 배치를 유지한다. 83~84는 손의 기울기·이완 변화로 다음 복귀와의 경계가 불확실하여 순수 유지 정답에서는 제외한다. 손모양 오답이나 새 주요동작으로 단정하지 않는다.'),
}

def main():
 assert not(P/'REVIEW.json').exists(),'Preserve completed annotation'
 prepared=read(P/'PREPARED.json');rows=[]
 for r in prepared['rows']:
  cid=r['clip_id'];hold,unknown,note=DECISIONS[cid];hf=list(range(hold[0],hold[1]+1))
  assert sorted(hf+unknown)==r['frames'] and not(set(hf)&set(unknown))
  assert sha(r['source'])==r['source_sha256'] and sha(r['source_annotation'])==r['source_annotation_sha256']
  assert sha(r['contact_sheet'])==r['contact_sheet_sha256']
  for f in r['frame_records']:assert sha(f['path'])==f['sha256']
  rows.append(dict(**r,review=dict(reviewer='/root/absence_contract_review_1113 (Codex agent)',method='Direct visual inspection using view_image of every cell of all-frame source-RGB contact sheet; four full-resolution endpoint frames additionally inspected.',all_frames_inspected=True,inspected_frames=r['frames'],hold_span=hold,hold_frames=hf,uncertain_frames=unknown,progression_frames=[],target_family=r['family'],state='terminal_hold_of_queried_family',observed_note_ko=note,labels_scope='Only whether the queried movement continues within this local reviewed interval; not correctness of the sign or proof of its previous completion.',semantic_completion=None,educational_correctness=None,mandatory_repetitions=None,independent_human_or_ksl_expert=False,extra_full_frame_inspection=([100,102] if cid=='NIA_SL_WORD0739_REAL03_F' else [82,84] if cid=='NIA_SL_WORD2523_REAL03_F' else []))))
 report=dict(at=datetime.now(timezone.utc).isoformat(),status='AGENT_RGB_REVIEW_COMPLETE_PENDING_ROOT_VERIFICATION',prepared_path=str(P/'PREPARED.json'),prepared_sha256=sha(P/'PREPARED.json'),script_sha256=sha(__file__),rows=rows,summary=dict(clips=8,persons=[2,3],source_frames_inspected=94,hold_frames=90,uncertain_frames=4,target_families=[3,4,5,7],family2_hold_evidence_not_provided=True),policy=[
  'Existing public FIT clips only; original source and old annotations remain untouched.',
  'No frame between range endpoints is skipped. This is dense review only within these small local ranges, not whole-video frame truth.',
  'Terminal hold allows small bodily settling and is relative to the queried movement; not a claim that all joints are static.',
  'Do not turn uncertainty into movement absence or an educational error.',
  'A null-candidate bag may be labelled only when its entire search domain lies within the approved hold span and uses the SAME target family.',
  'Do not label the complete original video as an absent/wrong sign: it contains the preceding movement.',
  'Absence of progression inside terminal hold does not establish that a required action was omitted or completed.',
  'This is agent visual annotation, not independent human/KSL gold truth. Root verification pending.',
  'No model predictions, detector outputs, metrics, or development thresholds were used to label these ranges.',
  'No training, new learned inference, private/mentor/final evaluation input access occurred.'
 ])
 (P/'REVIEW.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 text=['공개 FIT 마지막 유지 구간 연속 RGB 검토','',
 '기존 4표현×공개 수행자 ID02/03의 8원본에서, 미리 지정된 유지 표본의 최소~최대 범위94프레임을 하나도 빠뜨리지 않고 실제 이미지로 확인했다.',
 '90프레임은 해당 움직임 이후의 유지로, 순종03의101~102 및 이집트03의83~84 총4프레임은 이완·복귀 경계 불확실로 남겼다. 움직임이 조금 보인다는 이유로 오답이나 새로운 주요동작으로 판단하지 않았다.',
 '각 승인 구간만 잘라 해당 family의 진행이 없는 국소 후보 묶음 학습에 사용할 수 있다. 영상 전체에서 행동이 생략됐다는 정답이 아니며, 다른 family로 자동 확장하지 않는다. family2 유지 정답은 이번 묶음에 없다.',
 '행동 완료·반복 횟수·손모양의 교육적 정오는 모두 판정하지 않았다. 원본과 기존 주석을 수정하지 않았다.',
 'Codex 에이전트의 실제 RGB 확인이며 독립 사람·수어 전문가의 검증 정답은 아니다. root 별도검수 대기. 학습·평가·모델추론 실행 없음.','']
 for r in rows:text.append(f"{r['clip_id']} family{r['family']}: 유지{r['review']['hold_span']}, 보류{r['review']['uncertain_frames']}. {r['review']['observed_note_ko']}")
 (P/'RESULT_KO.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
 print(json.dumps(report['summary']))

if __name__=='__main__':main()
