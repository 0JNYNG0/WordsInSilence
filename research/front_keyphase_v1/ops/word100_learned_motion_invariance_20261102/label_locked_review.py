"""Record visible physical intervals reviewed before any candidate prediction."""
from prepare_real_fit import N,read,exclusive,sha,clock

def main():
 assert not (N/'locked_predictions').exists()
 manifest=read(N/'LOCKED_REVIEW_MANIFEST.json');lookup={r['clip_id']:r for r in manifest['rows']};rows=[]
 def add(word,person,frames,target,kind,note):
  cid=f'NIA_SL_{word}_REAL{person:02}_F';r=lookup[cid];y=[0]*6;m=[False]*6
  if target in [2,4]:m[2]=m[4]=True
  else:m[target]=True
  if kind=='directed':y[target]=1
  y[5]=1;m[5]=True
  rows.append(dict(clip_id=cid,person=person,word_id=word,role='locked_task_evaluation',frames=frames,kind=kind,y=y,mask=m,
   evidence=note,contact_sha256=r['contact_sha256'],source_sha256=r['source_sha256']))
 for person,directed,stable in [(14,[48,60],[60,72]),(15,[52,72],[72,80])]:
  add('WORD1133',person,directed,0,'directed','Right hand visibly travels from proximal left arm toward left wrist; endpoint location is not contact-depth proof.')
  add('WORD1133',person,stable,0,'stable','Right hand remains near left wrist; ordinary body drift allowed, no continued along-arm travel asserted.')
 for person,directed,stable in [(14,[44,56],[56,64]),(15,[56,64],[64,72])]:
  add('WORD2523',person,directed,1,'directed','Two hand regions separate after fingertips were together; no required linguistic distance claimed.')
  add('WORD2523',person,stable,1,'stable','Hands retain separated configuration without further obvious separation.')
 for person,directed,stable in [(14,[60,72],[84,96]),(15,[48,60],[72,80])]:
  add('WORD0739',person,directed,3,'directed','Right hand moves from face-front down toward upper torso; rotation is not labeled as translation.')
  add('WORD0739',person,stable,3,'stable','Right hand held to side of upper torso; no additional face-to-torso descent.')
 for person in [14,15]:
  add('WORD1567',person,[48,56],4,'directed','Right hand changes from above left to below left during visible alternation.')
  add('WORD1567',person,[60,68],2,'directed','Right hand changes from below left to above left during visible alternation.')
  add('WORD1567',person,[80,88],2,'stable','Left remains higher than right at end configuration; no required repeat count annotated.')
 exclusive(N/'LOCKED_PHYSICAL_TARGETS.json',dict(at=clock(),rows=rows,reviewer='Codex visual inspection of eight raw chronological RGB sheets',
  not_independent_human_gold=True,model_predictions_viewed=False,training_use=False,selection_use=False,
  boundary_scope='Conservative visibly directed/stable intervals on four-frame contact sampling; not exact onset/end or semantic gold',
  contact_manifest_sha256=sha(N/'LOCKED_REVIEW_MANIFEST.json'),old_tcn_evaluation_exposure=True,
  recorded_ids_not_verified_biometric_identity=True,unknown_dimensions_masked=True))
 print('Recorded18 physical evaluation intervals before inference; excluded from training and selection')

if __name__=='__main__':main()
