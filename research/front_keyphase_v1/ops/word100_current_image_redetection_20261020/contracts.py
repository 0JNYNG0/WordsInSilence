from pathlib import Path
import json
import numpy as np
from candidate import decide,crops

p=np.full((33,2),np.nan);g=np.zeros(33,bool)
for i,xy in {11:[350,200],12:[650,200],13:[300,300],14:[700,300],15:[260,400],16:[740,400]}.items():p[i]=xy;g[i]=True
def hand(x,y):
 a=np.zeros((21,2));a[:]=[x,y];a[[5,9,13,17]] += [[-10,-25],[0,-30],[10,-25],[20,-20]];return a
l=hand(260,400);r=hand(740,400);passed=[]
def test(name,ok):assert ok,name;passed.append(name)
def call(d,h=(),s=()):return decide([dict(xy=x.tolist()) for x in d],np.asarray(h).reshape(-1,21,2),list(s),p,g,1000,800)
test('unique_current_arm',call([l])[0]['accepted'] and call([l])[0]['side']==0)
test('existing_not_replaced',not call([l],[l],[0])[0]['accepted'])
test('duplicate_unassigned_not_added',not call([l],[l],[-1])[0]['accepted'])
test('duplicate_new_candidates',not any(x['accepted'] for x in call([l,l])))
test('out_of_frame',not call([hand(260,850)])[0]['accepted'])
test('current_arm_far',not call([hand(500,400)])[0]['accepted'])
test('both_missing_distinct',len([x for x in call([l,r]) if x['accepted']])==2)
boxes=crops(p,g,[],1000,800);test('crop_clipped_sensor',all(0<=b[0]<b[2]<=1000 and 0<=b[1]<b[3]<=800 for b in boxes.values()))
test('existing_side_no_crop',0 not in crops(p,g,[0],1000,800))
saved=p.copy();p[[15,16]]=p[[16,15]];p[[13,14]]=p[[14,13]]
test('crossing_not_screen_half',call([l])[0]['side']==1 and call([l])[0]['accepted'])
p[:]=saved
d=dict(xy=l.tolist(),label='Right',handedness_score=.999)
test('raw_label_not_identity',decide([d],[],[],p,g,1000,800)[0]['side']==0)
(Path(__file__).parent/'CONTRACTS.json').write_text(json.dumps(dict(status='PASS',tests=passed),indent=2),encoding='utf8')
print('PASS',len(passed))
