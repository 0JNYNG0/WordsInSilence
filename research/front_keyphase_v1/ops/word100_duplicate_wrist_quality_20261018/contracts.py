import json
from pathlib import Path
import numpy as np
from candidate import propose,decision,PARAMS

tests=[]
def check(name,cond):
    assert cond,name
    tests.append(name)

# Stable synthetic palm; two overlapping projections differ only at the wrist.
q=np.array([[0,0],[-1,.5],[-1.5,1],[-1.8,1.5],[-2,2],[-1,2],[-1,3],[-1,4],[-1,5],
 [0,2.2],[0,3.2],[0,4.2],[0,5.2],[.6,2.1],[.6,3.1],[.6,4.1],[.6,5.1],[1,2],[1,2.8],[1,3.6],[1,4.4]],float)*20+200
bad=q.copy();bad[0]=q[[5,9,13,17]].mean(0)
pts=np.array([bad,q]);mask=np.ones((2,21),bool)
def run(p=pts,m=mask,s=[0,1],identity=[-1,-1],prev=q[None],gap=33333):
    return propose(p,m,s,identity,200,prev,np.ones((len(prev),21),bool),gap)
r=run();check('joint_conflict_withholds_hint_only',r['sides']==[-1,1])
check('confirmed_protected',run(identity=[0,-1])['sides']==[0,1])
check('already_withheld_protected',run(s=[-1,1])['sides']==[-1,1])
check('raw_coordinates_unchanged',np.array_equal(pts,np.array([bad,q])))
check('reverse_candidates',run(pts[::-1],mask[::-1],[1,0])['sides']==r['sides'][::-1])
check('two_prior_hands_do_not_suppress',run(prev=np.array([q,q]))['sides']==[0,1])
check('old_history_ignored',run(gap=100001)['sides']==[0,1])
check('future_history_ignored',run(gap=-1)['sides']==[0,1])
check('healthy_overlapping_palms_unchanged',run(np.array([q,q]))['sides']==[0,1])
check('both_collapsed_unchanged',run(np.array([bad,bad]))['sides']==[0,1])
separate=pts.copy();separate[1,:,0]+=200
check('separate_hands_unchanged',run(separate)['sides']==[0,1])
partial=mask.copy();partial[0,0]=False
check('unobserved_wrist_no_suppression',run(m=partial)['sides']==[0,1])
one=propose(pts[:1],mask[:1],[0],[-1],200,q[None],mask[:1],33333)
check('foreshortened_single_hand_preserved',one['sides']==[0])
for factor,shift in [(2.,np.array([10.,25.])),(.5,np.array([-70.,3.]))]:
    t=propose(pts*factor+shift,mask,[0,1],[-1,-1],200*factor,q[None]*factor+shift,mask[:1],33333)
    check('scale_translation_'+str(factor),t['sides']==r['sides'])
mir=pts.copy();mir[:,:,0]*=-1;prev=q[None].copy();prev[:,:,0]*=-1
check('reflection_invariant',propose(mir,mask,[0,1],[-1,-1],200,prev,mask[:1],33333)['sides']==r['sides'])
invalid=pts.copy();invalid[0,0]=np.nan
check('nonfinite_wrist_preserved',run(invalid)['sides']==[0,1])
g=[{'usable':True,'ratio':.1},{'usable':True,'ratio':.8}]
check('insufficient_overlap_no_suppression',decision(g,.70,1,[True,True],[0,1],[-1,-1])[0]==[0,1])
check('close_wrists_no_suppression',decision(g,.9,.4,[True,True],[0,1],[-1,-1])[0]==[0,1])
Path(__file__).with_name('CONTRACTS.json').write_text(json.dumps({'status':'PASS','count':len(tests),'tests':tests,'parameters':PARAMS},indent=2)+'\n',encoding='utf8')
print('CONTRACTS PASS',len(tests))
