from common_split import *
G=[1,1,1,1];B=[1,1,1,0];QL=[1,1,1,-1];I=[1,0,0,0];F=[0,0,0,0];T=[1,1,1,0];E=[1,1,1,1];X=[-1,0,0,0]
rows=[]
def add(role,w,p,indices,q,y,note='Center RGB and +/-3 frame context; overlay checked separately'):
 src=read(U/f'{role}_REVIEW_INPUTS.json')['rows'];cid=f'NIA_SL_WORD{w}_REAL{p:02d}_F'
 for k in indices:
  r=next(r for r in src if r['clip_id']==cid and r['index']==k)
  rows.append(r|dict(quality=q,shape=y,note=note,annotator='Codex visual review, not independent expert'))
add('fit','2700',2,[4],G,I);add('fit','2700',2,[5],G,F);add('fit','2700',2,[6],QL,[1,-1,-1,0]);add('fit','2700',2,[7,8,9,10],B,T)
add('fit','2700',5,[4],G,I);add('fit','2700',5,[5],G,X);add('fit','2700',5,[6],G,F);add('fit','2700',5,[7],G,T);add('fit','2700',5,[8,9,10],B,T)
add('fit','2700',7,[0],G,I);add('fit','2700',7,[1,2,3],G,F);add('fit','2700',7,[4,5,6,7],B,T)
add('fit','2718',2,[2,3],B,T);add('fit','2718',2,[4],B,F);add('fit','2718',2,[5],G,F);add('fit','2718',2,[6,7,8,9,10],G,I)
add('fit','2718',5,[3],QL,T);add('fit','2718',5,[4],B,T);add('fit','2718',5,[5],B,F);add('fit','2718',5,[6],G,F);add('fit','2718',5,[7],G,X);add('fit','2718',5,[8,9,10],G,I)
add('fit','2718',7,[1,2],B,T);add('fit','2718',7,[3,4],QL,F);add('fit','2718',7,[5],G,X);add('fit','2718',7,[6,7,8,9],G,I)
add('diagnostic','2700',10,[2,3],G,I);add('diagnostic','2700',10,[4],G,F);add('diagnostic','2700',10,[5],G,X);add('diagnostic','2700',10,[6,7,8],B,T)
add('diagnostic','2718',10,[2],B,T);add('diagnostic','2718',10,[3],QL,[-1,-1,-1,0]);add('diagnostic','2718',10,[4],G,F);add('diagnostic','2718',10,[5],G,X);add('diagnostic','2718',10,[6,7,8,9],G,I)
add('diagnostic','0071',10,[2,3,5],G,E);add('diagnostic','0071',10,[4],[-1]*4,E);add('diagnostic','0071',10,[6,7,8,9],G,I)
add('diagnostic','0949',10,list(range(3,12)),G,I)
add('diagnostic','1290',10,list(range(2,10)),G,E)
add('diagnostic','2748',10,list(range(2,10)),G,I)
add('confirmation_reserved','2700',16,[2,3,4],G,I);add('confirmation_reserved','2700',16,[5],G,X);add('confirmation_reserved','2700',16,[6],QL,[-1,-1,-1,0])
# The 60/65/70 little-finger chain needs full-resolution review before quality is finalized.
add('confirmation_reserved','2700',16,[7,8,9],QL,T,'Quality little pending full-resolution comparison; shape visible from source')
add('confirmation_reserved','2718',16,[2],QL,T);add('confirmation_reserved','2718',16,[3],G,T);add('confirmation_reserved','2718',16,[4],QL,[-1,-1,-1,0]);add('confirmation_reserved','2718',16,[5],G,X);add('confirmation_reserved','2718',16,[6,7,8,9,10],G,I)
add('confirmation_reserved','0071',16,[2,3,4],G,E);add('confirmation_reserved','0071',16,[5],G,[1,-1,-1,-1]);add('confirmation_reserved','0071',16,[6,7,8,9,10],G,I)
add('confirmation_reserved','0949',16,list(range(1,10)),G,I)
add('confirmation_reserved','1290',16,list(range(3,10)),G,E)
add('confirmation_reserved','2748',16,[2,3,4,5],G,I);add('confirmation_reserved','2748',16,[6],G,F);add('confirmation_reserved','2748',16,[7],G,X);add('confirmation_reserved','2748',16,[8,9,10],G,I)
write(U/'LABELS_DRAFT.json',dict(rows=rows,legend=dict(quality='1 own visible chain consistent;0 visibly wrong track;-1 unresolved',shape='index/middle/ring/little;1 extended;0 flexed;-1 unresolved'),notes=['Source RGB research labels by Codex, not expert certified','No prediction inspected on new performers10/16','No label inferred solely from word name','Original FIT retained; new unreliable chains excluded from shape training']))
audit=read(U/'PRIOR_LOSS_AUDIT.json')['rows']
write(U/'LOSS_CAUSE_REVIEW.json',dict(rows=[r|dict(review_quality=1 if i in list(range(8))+[26,27]else -1,review_reason='Own visible chain consistent'if i in list(range(8))+[26,27]else 'Individual chain identity unresolved at narrow or occluded view; not labeled corrupt')for i,r in enumerate(audit)],normal_source_known_posture=40,clear_quality_good=10,quality_unresolved=30,used_for_training=False))
print('draft',len(rows))
