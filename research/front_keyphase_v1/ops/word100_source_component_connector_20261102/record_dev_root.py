"""Sparse direct RGB review of the six public development sheets, not predictions."""
from pathlib import Path
import hashlib,json

U=Path(__file__).resolve().parent; O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

OBS={
 ('WORD0739',11):dict(open=62,closed=[77,97],sequence=[81,84,87,93,99]),
 ('WORD0739',12):dict(open=55,closed=[69,93],sequence=[75,77,79,82,88]),
 ('WORD0739',13):dict(open=71,closed=[90,108],sequence=[93,96,100,103,108]),
 ('WORD1133',11):dict(sequence=[50,54,58,62,65],relation=[54,62,71],endpoint=[69,73,75]),
 ('WORD1133',12):dict(sequence=[62,66,70,75,79],relation=[66,75,83],endpoint=[79,83,85]),
 ('WORD1133',13):dict(sequence=[53,57,63,68,73],relation=[57,68,80],endpoint=[78,82,85]),
}
def claim(clause,target,frames,kind,support,note,unsupported):
 return dict(clause_id=clause,target=target,witness_frames=frames,kind=kind,support=support,
  note=note,unsupported=unsupported,whole_clause_truth=None,learner_error=None,
  boundary='Sparse direct witnesses only; no exact event onset or end')

def main():
 manifest=O/'word100_action_episode_supervision_20261102/DEV_REVIEW_MANIFEST.json'
 rows=[]
 for r in read(manifest)['rows']:
  key=(r['word_id'],r['person'])
  if key not in OBS:continue
  assert sha(r['sheet'])==r['sheet_sha256']
  assert sha(r['source'])==r['source_sha256']
  p=OBS[key]
  if key[0]=='WORD0739':
   claims=[
    claim('obedience_closed_hand','right_non_thumb_closed',p['closed'],'point','supported',
     'Right long fingers form a closed-looking mass at the two witnessed points.',
     ['precise thumb position','all intervening frames']),
    claim('obedience_closed_hand','right_non_thumb_closed',[p['open']],'point','visibly_contrary',
     'Earlier face presentation has extended right long fingers. This is a normal preceding action.',
     ['Not a whole-sign error label']),
    claim('obedience_lateral_arm_then_lower','right_elbow_lowers_hand_remains_raised',p['sequence'],'ordered_witness_sequence','supported',
     'Elbow lowers from the lateral raised position while the hand stays raised; forearm becomes more upright.',
     ['Not the earlier wrist descent from face','exact boundary','required endpoint pause'])]
  else:
   claims=[
    claim('well_palm_left_arm','right_hand_at_left_arm_projection',p['relation'],'point','supported',
     'Right hand lies on/near the left forearm in the front projection.',
     ['physical contact pressure','exact palm orientation']),
    claim('well_brush','right_hand_traverses_left_arm_toward_wrist',p['sequence'],'ordered_witness_sequence','supported',
     'Right hand advances along the projected left arm toward the left wrist.',
     ['exact movement boundary','mandatory pause','contact pressure']),
    claim('well_brush','right_hand_traverses_left_arm_toward_wrist',p['endpoint'],'ordered_witness_sequence','unknown',
     'Later wrist-region placement alone does not establish ongoing sweep at every sampled point.',
     ['No negative claim about the preceding sweep','No mandatory hold is inferred'])]
  for c in claims:assert set(c['witness_frames'])<=set(r['indices'])
  rows.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],role='development',
   source=r['source'],source_sha256=r['source_sha256'],
   reviewed_sheet=dict(path=r['sheet'],sha256=r['sheet_sha256']),
   reviewed_frames=r['indices'],claims=claims,
   reviewer='root Codex direct chronological RGB sheet review',predictions_shown=False,
   historical_results_known=True,independent_human_or_expert_gold=False))
 assert len(rows)==6
 out=dict(rows=rows,manifest=str(manifest),manifest_sha256=sha(manifest),
  clips=len(rows),sampled_image_cells=sum(len(r['reviewed_frames']) for r in rows),
  no_new_candidate_predictions_seen=True,development_repeatedly_exposed=True,
  existing_labels_unchanged=True,whole_goal_complete=False)
 with (U/'DEV_BINDINGS_ROOT.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps({k:v for k,v in out.items() if k not in ['rows']},ensure_ascii=False))
if __name__=='__main__':main()
