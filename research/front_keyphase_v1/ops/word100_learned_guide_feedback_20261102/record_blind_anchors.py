"""Record actually viewed public frame-sheet observations before stage scoring."""
from prepare import U,read,save,sha
def main():
 # Exact original-frame samples; gaps are NOT densely annotated.
 definitions={
 'WORD0739':{
  11:[[54,60,66],[75,78,81],[90,96,102]],
  12:[[45,51,57],[69,72,75],[84,90,96]],
  13:[[63,69,75],[87,90,93],[102,108,114]]},
 'WORD1133':{
  11:[[54,57,60],[66,69,72]],
  12:[[63,66,69],[78,81,84]],
  13:[[57,60,63],[75,78,81]]},
 'WORD1567':{
  11:[[48,51],[57,60],[69,72],[78],[84,90,96]],
  12:[[48,51],[60],[66,69],[78],[84,87,90]],
  13:[[36,39],[48,51],[60,63],[72],[78,84,87]]},
 'WORD2523':{
  11:[[48,51,54],[],[66,69,72]],
  12:[[48,51,54],[],[66,69,72]],
  13:[[57,60,63],[],[72,75,78]]}}
 contacts={r['clip_id']:r for r in read(U/'REVIEW_CONTACTS.json')['rows']}
 support=read(U/'GUIDE_SUPPORT.json');lookup={g['canonical_stage_id']:g for g in support['guides']}
 rows=[]
 for word,people in definitions.items():
  for person,stages in people.items():
   cid=f'NIA_SL_{word}_REAL{person:02}_F';contact=contacts[cid]
   for j,frames in enumerate(stages,1):
    sid=f'{word}:stage{j:02}'
    equivalence=[sid]
    if word=='WORD1567':
     equivalence=[f'{word}:stage{k:02}' for k in ([1,3] if j in [1,3] else [2,4,5])]
    for frame in frames:
     assert frame in contact['frame_ids']
     rows.append(dict(clip_id=cid,word_id=word,person=person,frame_id=frame,stage_id=sid,
      allowed_same_physical_configuration=equivalence,description=lookup[sid]['description'],
      provenance='Codex direct inspection of public contact sheet, no new stage predictions viewed',
      sheet=contact['path'],sheet_sha256=sha(__import__('pathlib').Path(contact['path'])),
      independent_human_gold=False,educational_correctness=False))
 save('BLIND_STAGE_ANCHORS.json',dict(status='PUBLIC_PHYSICAL_STAGE_OBSERVATIONS_RECORDED_BEFORE_STAGE_SCORING',rows=rows,
  reviewed_clips=12,sampling_stride_original_frames=3,labels_between_samples=False,
  observer='same Codex assistant; not KSL teacher',model_stage_predictions_used=False,
  limitations=['Existing development records, not fresh independent evaluation.',
   'Only clearly visible physical configurations/movements sampled. No exact linguistic correctness, contact or repetition requirement.',
   'Egypt short separation stage has no confident sampled anchors; not scored as absent or successful.',
   'Ancestor repeated right-high and left-high configurations additionally reported as equivalence groups; exact occurrence and configuration scores kept separate.'],
  uncertain_short_transition=[dict(clip_id=f'NIA_SL_WORD2523_REAL{p:02}_F',candidate_frame=f,reason='3-frame sampling cannot establish exact transition boundaries') for p,f in [(11,60),(12,60),(13,66)]]))
 print('Recorded',len(rows),'sampled stage anchors; no dense annotation claim')
if __name__=='__main__':main()
