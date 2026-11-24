from arrival_candidate import *
class AugmentedArrivalLesson(ArrivalLesson):
 def __init__(self):PlaneViewLesson.__init__(self);self.body_relation=ArrivalRelation(K/'runs_augmented')

def main():
 rows=[]
 for person,groups in[(2,{1:[49,58,64,66,68,70],2:[52,54,60,62]}),(5,{1:[41,50],2:[44,53]})]:
  for view in['L','R']:
   cid=f'NIA_SL_WORD1290_REAL{person:02d}_{view}'
   for state,frames in groups.items():
    for i in frames:rows.append(dict(clip=cid,frame=i,state=state,split='fit',source_review=[str(K/'fit_review'/(cid+'.jpg')),str(K/'fit_review_second'/(cid+'.jpg'))]if person==2 else[str(K/'fit_review'/(cid+'.jpg'))],label_scope='Visible projected separation vs near, not physical contact; independently reviewed each camera.'))
 write(K/'AUGMENTED_SOURCE_TRUTH.json',dict(rows=rows,reviewer='Codex source-native RGB review; no external KSL expert.',note='Person2 second near poses at60/62 in side cameras, NOT copied from front64. Person5 frames46/55 excluded as transition ambiguity.',private_data=False))
 train(rows,read(K/'fit_02/INPUTS.json')['rows']+read(K/'fit_05/INPUTS.json')['rows'],'runs_augmented')
if __name__=='__main__':main()
