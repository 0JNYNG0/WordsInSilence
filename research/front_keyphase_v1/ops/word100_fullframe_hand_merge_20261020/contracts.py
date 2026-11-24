import json
from pathlib import Path
import numpy as np
from candidate import merge,crop_boxes
p=np.full((33,2),np.nan);good=np.zeros(33,bool)
for j,v in {11:[350,200],12:[650,200],13:[300,300],14:[700,300],15:[260,400],16:[740,400]}.items():p[j]=v;good[j]=True
def hand(x,y):
    a=np.tile([x,y],(21,1)).astype(float)
    a[[5,9,13,17]] += [[-10,-25],[0,-30],[10,-25],[20,-20]]
    return a
l,r=hand(260,400),hand(740,400);passed=[]
def d(h):return dict(xy=h.tolist(),label='IGNORED')
def call(h,base=(),sides=(),c=None):
    ds=[d(x) for x in h]
    crops=[dict(d(x),box=[0,0,1000,800],requested_side=s) for x,s in [(l,0),(r,1)]] if c is None else c
    return merge(ds,crops,base,sides,p,good,1000,800)
def test(n,ok):assert ok,n;passed.append(n)
test('two_new_distinct_hands',sum(q['accepted'] for q in call([l,r]))==2)
test('existing_coordinates_not_replaced',call([l],[l],[0])[0]['reason']=='confirm_existing_hand')
test('wrong_label_same_physical_hand_not_double',call([l],[l],[1])[0]['reason']=='existing_identity_conflict')
test('unassigned_existing_not_duplicate',call([l],[l],[-1])[0]['reason']=='existing_identity_conflict')
test('no_corroboration_no_add',not call([l],c=[])[0]['accepted'])
test('no_raw_label_dependency',call([l])[0]['side']==0)
test('new_side_tie_not_added',not any(q['accepted'] for q in call([l,l])))
partial=l.copy();partial[8]=[1200,400]
q=call([partial])[0];test('unsupported_finger_not_filled',q['accepted'] and not q['joint_supported'][8])
test('source_outside_hand_not_added',not call([hand(260,850)])[0]['accepted'])
oldp=p.copy();p[[13,14]]=p[[14,13]];p[[15,16]]=p[[16,15]]
c=[dict(d(l),box=[0,0,1000,800],requested_side=1)]
test('crossing_not_screen_half',call([l],c=c)[0]['accepted'] and call([l],c=c)[0]['side']==1)
p[:]=oldp
test('existing_side_no_crop',0 not in crop_boxes(p,good,[0],1000,800))
assert all(0<=b[0]<b[2]<=1000 and 0<=b[1]<b[3]<=800 for b in crop_boxes(p,good,[],1000,800).values())
test('crop_never_uses_outside_source',True)
# Known prior duplicate counterexample, separate regression fixture only, not a sensor input for new extraction.
u=Path(__file__).resolve().parent;prev=u.parent/'word100_current_image_redetection_20261020'
t=json.loads((prev/'REVIEW_TARGETS.json').read_text('utf8'))[307]
fr=json.loads((prev/'FROZEN.json').read_text('utf8'))
row=next(x for x in fr['rows'] if x['clip_id']==t['clip_id'])
z=np.load(row['views']['sensor']['raw']);a=np.load(row['views']['sensor']['assignment'])['sides']
e=json.loads((prev/'records'/f"{t['clip_id']}.json").read_text('utf8'))['frames'][t['index']]
i=t['index'];n=int(z['raw_count'][i])
out=merge(e['image'],e['crop'],z['hand_xy_px'][i,:n],a[i,:n].tolist(),z['pose_xy_px'][i],z['pose_usable'][i],int(z['width'][i]),int(z['height'][i]))
test('prior_airport_duplicate_no_new_hand',not any(q['accepted'] for q in out))
test('prior_airport_conflict_recorded',any(q['reason']=='existing_identity_conflict' for q in out))
(u/'CONTRACTS.json').write_text(json.dumps(dict(status='PASS',tests=passed,prior_fixture_results=out),ensure_ascii=False,indent=2),encoding='utf8')
print('PASS',len(passed))
