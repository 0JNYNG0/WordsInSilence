"""Root's actual chronological-sheet review; sparse witnesses, not dense truth."""
from compile_supervision import U,O,read,sha,save

# Selected only after directly viewing all18 named sheets. Numbers denote visible
# cells, not interpolation through the unshown frames.
OBEDIENCE={
 1:dict(open=55,closed=[68,81],sequence=[67,70,75,81,89]),
 2:dict(open=57,closed=[74,91],sequence=[69,76,79,91]),
 3:dict(open=62,closed=[78,94],sequence=[74,82,88,94]),
 4:dict(open=84,closed=[98,115],sequence=[98,103,108,115]),
 5:dict(open=50,closed=[65,81],sequence=[65,69,75,81]),
 6:dict(open=63,closed=[81,99],sequence=[81,86,92,99]),
 7:dict(open=47,closed=[61,74],sequence=[61,63,66,69,74]),
 8:dict(open=57,closed=[72,87],sequence=[69,72,78,82,87]),
 9:dict(open=48,closed=[64,81],sequence=[60,64,69,75,81])}
WELL={
 1:dict(sequence=[49,54,58,64],relation=[54,64,73],endpoint=[67,73,77]),
 2:dict(sequence=[47,53,59,64],relation=[53,64,72],endpoint=[66,72,76]),
 3:dict(sequence=[50,57,63,68],relation=[57,68,78],endpoint=[69,74,78]),
 4:dict(sequence=[74,80,86,93],relation=[80,93,102],endpoint=[95,99,103]),
 5:dict(sequence=[44,48,55,62,66],relation=[55,62,68],endpoint=[64,68,70]),
 6:dict(sequence=[40,44,48,54,57],relation=[48,54,60],endpoint=[55,59,62]),
 7:dict(sequence=[44,49,56,63,69],relation=[49,60,69],endpoint=[63,69,72]),
 8:dict(sequence=[69,74,82,89,94],relation=[79,87,94],endpoint=[91,97,99]),
 9:dict(sequence=[43,47,53,58,63],relation=[53,58,63],endpoint=[60,64,68])}

def claim(cid,component,frames,kind='ordered_witness_sequence',support='supported',unsupported=None):
 return dict(clause_id=cid,supported_component=component,witness_frames=frames,kind=kind,support=support,
             unsupported=unsupported or [],boundary='uncertain, no dense propagation',whole_clause_truth=None)

def main():
 m=read(O/'word100_action_episode_supervision_20261102/REVIEW_MANIFEST.json')
 rows=[]
 for r in m['rows']:
  word=r['word_id'];pid=r['person']
  if word not in ['WORD0739','WORD1133']:continue
  assert sha(r['sheet'])==r['sheet_sha256']
  if word=='WORD0739':
   p=OBEDIENCE[pid]
   claims=[claim('obedience_closed_hand','Right non-thumb fingers visibly curled into a closed configuration',p['closed'],'point',
              unsupported=['precise thumb placement','all intervening frames']),
           claim('obedience_closed_hand','At this earlier face-presentation point the right long fingers are extended, not closed',[p['open']],'point','visibly_contrary',
              ['This is a normal earlier action, not an error in the whole sign']),
           claim('obedience_lateral_arm_then_lower','Right elbow moves from lateral higher position toward lower position while the closed-looking hand stays near upper torso; forearm becomes more upright',p['sequence'],
              unsupported=['exact onset/end','semantic necessity of the endpoint hold','full source-variant equivalence'])]
   note='Face-level open hand precedes closing/reorientation. The source lowering is supported by elbow/forearm relation, not equated to earlier right-wrist face-to-torso descent.'
  else:
   p=WELL[pid]
   claims=[claim('well_palm_left_arm','The right hand lies in projection on/near the left forearm',p['relation'],'point',
              unsupported=['physical contact/pressure','exact palm surface orientation','all finger positions']),
           claim('well_brush','Right hand progresses along the projected left arm toward the left-wrist region',p['sequence'],
              unsupported=['touch/pressure','exact event boundary','mandatory pause at wrist']),
           claim('well_brush','Right hand remains near the left-wrist region in later sampled images; those endpoint points alone do not establish an ongoing sweep',p['endpoint'],'ordered_witness_sequence','unknown',
              ['Ongoing directed sweep at every endpoint frame','whether pause is linguistically required'])]
   note='Sweep evidence and later wrist-region relation are recorded separately; same source phrase does not create a required second event. Sparse samples do not prove precise motion cessation.'
  available={f['frame']:f for f in r['frames']}
  for c in claims:
   assert set(c['witness_frames'])<=set(available)
   c['witness_images']=[dict(frame=f,path=available[f]['path'],sha256=available[f]['sha256']) for f in c['witness_frames']]
  rows.append(dict(clip_id=r['clip_id'],word_id=word,person=pid,role='fit',
   reviewed_sheet=dict(path=r['sheet'],sha256=r['sheet_sha256']),reviewed_frames=sorted(available),
   claims=claims,note=note,reviewer='root Codex direct RGB sheet review',predictions_shown=False,
   historical_results_known=True,independent_human_or_expert_gold=False))
 assert len(rows)==18
 save('DIRECT_BINDINGS_ROOT.json',dict(rows=rows,scope='Sparse source-component observations only; source phrase entailment and educational correctness remain distinct',
  clips=18,new_dense_targets=0,existing_labels_unchanged=True,whole_goal_complete=False))
 print('Recorded18FITdirectsource-componentreviews')
if __name__=='__main__':main()
