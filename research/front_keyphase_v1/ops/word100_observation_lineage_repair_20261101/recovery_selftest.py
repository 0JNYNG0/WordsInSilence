"""Pure recovery contracts; no public frames, model objects or inference."""
import sys
sys.dont_write_bytecode = True
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import recovery as r


def run():
    count = 0; tests = []
    def check(value, label):
        nonlocal count
        count += 1
        if not value:
            raise AssertionError(label)
        tests.append(label)
    pose = np.full((33, 2), np.nan)
    good = np.zeros(33, dtype=bool)
    for j, p in {11:[350,200],12:[650,200],13:[300,300],14:[700,300],15:[260,400],16:[740,400]}.items():
        pose[j] = p; good[j] = True
    box = [0,0,1000,650]
    def hand(x, y):
        a = np.tile([x,y], (21,1)).astype(float)
        for j in range(1,21):
            a[j] += [(j%4-1)*8, -(j//4+1)*7]
        return a
    left, right = hand(260,400), hand(740,400)
    def frame():
        a = dict(xy_px=np.full((75,2), np.nan, dtype=np.float32), observed=np.zeros(75,bool),
                 usable=np.zeros(75,bool), conf=np.full(75,np.nan,np.float32), hand_score=np.full(2,np.nan,np.float32))
        a['xy_px'][:33] = pose; a['observed'][:33] = good; a['usable'][:33] = good
        return a
    def detection(p, score=.99):return dict(xy=p.tolist(),score=score,label='IGNORED')
    def crop(p, side, score=.99):return dict(detection(p,score),box=[0,0,1000,800],requested_side=side)
    a = frame()
    eg = r.eligibility(a['observed'],pose,good,box,1000,800,[])
    check(all(e['eligible'] for e in eg), 'Both truly absent sides with current pose are eligible')
    eg = r.eligibility(a['observed'],pose,good,box,1000,800,[0])
    check(not eg[0]['eligible'] and eg[0]['reason']=='EXISTING_RAW_ASSIGNED_SIDE_PRESERVED', 'Raw assigned side blocks inference even if inconsistent native mask')
    a['observed'][33] = True
    eg = r.eligibility(a['observed'],pose,good,box,1000,800,[])
    check(not eg[0]['eligible'], 'Single observed point protects entire partial native hand')
    check(eg[1]['eligible'], 'Partial left hand does not suppress genuinely absent right hand')
    a = frame(); low = pose.copy(); low[15,1] = 700
    check(not r.eligibility(a['observed'],low,good,box,1000,800,[])[0]['eligible'], 'Normal lowered pose wrist outside fixed ROI does not trigger')
    check(not r.eligibility(a['observed'],pose,good & (np.arange(33)!=13),box,1000,800,[])[0]['eligible'], 'Missing current elbow cannot force side')
    check(not r.eligibility(a['observed'],pose,good & (np.arange(33)!=11),box,1000,800,[])[0]['eligible'], 'Missing shoulder scale cannot force side')

    before = deepcopy(a)
    added, ledger = r.merge_frame(a,np.empty((0,21,2)),[],pose,good,1000,800,box,
                                  [detection(left),detection(right)],[crop(left,0),crop(right,1)],.5)
    check(len(ledger['additions'])==2, 'Two distinct current arms can add two observed hands')
    check(all(np.array_equal(a[k],before[k],equal_nan=True) for k in a), 'Pure merge input unchanged')
    check(np.array_equal(added['xy_px'][33:54],left.astype(np.float32)), 'New points are actual same-frame full-image points')
    check(np.array_equal(added['observed'][33:],np.ones(42,bool)), 'Same-scale complete agreement supports exact observed joints')
    check(np.array_equal(added['xy_px'][:33],before['xy_px'][:33],equal_nan=True), 'Original pose unchanged')
    check(np.all(added['hand_score']==np.float32(.99)), 'Actual handedness score retained, not 1.0')
    unchanged, ld = r.merge_frame(added,[left,right],[0,1],pose,good,1000,800,box,
                                  [detection(left+2),detection(right+2)],[],.5)
    check(not ld['additions'] and all(np.array_equal(unchanged[k],added[k],equal_nan=True) for k in added), 'Existing complete hand values and masks preserved')
    partial = deepcopy(added); partial['observed'][34:54]=False; partial['usable'][34:54]=False
    unchanged, ld = r.merge_frame(partial,[left,right],[0,1],pose,good,1000,800,box,
                                  [detection(left)],[crop(left,0)],.5)
    check(not ld['additions'] and all(np.array_equal(unchanged[k],partial[k],equal_nan=True) for k in partial), 'Existing partial and poor-quality observations never filled or overwritten')
    _, ld = r.merge_frame(a,[left],[-1],pose,good,1000,800,box,[detection(left)],[crop(left,0)],.5)
    check(not ld['additions'] and ld['decisions'][0]['reason']=='existing_identity_conflict', 'Raw unassigned hypothesis participates in duplicate protection')
    _, ld = r.merge_frame(a,[left],[1],pose,good,1000,800,box,[detection(left)],[crop(left,0)],.5)
    check(not ld['additions'], 'Existing conflicting identity cannot create a second hand')
    _, ld = r.merge_frame(a,np.empty((0,21,2)),[],pose,good,1000,800,box,[detection(left)],[],.5)
    check(not ld['additions'], 'Full image alone is insufficient without same-frame crop agreement')
    _, ld = r.merge_frame(a,np.empty((0,21,2)),[],pose,good,1000,800,box,[detection(left),detection(left)],[crop(left,0)],.5)
    check(not ld['additions'], 'Multiple candidates for the same side are withheld')
    bad = left.copy(); bad[8] += [100,0]
    added, ld = r.merge_frame(a,np.empty((0,21,2)),[],pose,good,1000,800,box,[detection(bad)],[crop(left,0)],.5)
    check(len(ld['additions'])==1 and not added['observed'][41] and added['observed'][40], 'Per-joint disagreement stays unknown while other points remain')
    added, ld = r.merge_frame(a,np.empty((0,21,2)),[],pose,good,1000,800,box,[detection(left,.49)],[crop(left,0,.98)],.5)
    check(added['observed'][33:54].all() and not added['usable'][33:54].any(), 'Low actual handedness score remains unusable under original .5 contract')
    check(np.isclose(added['hand_score'][0],.49), 'Low actual score is not replaced with detector confidence option')
    changed_labels=[dict(detection(left),label='Right')]
    added, ld = r.merge_frame(a,np.empty((0,21,2)),[],pose,good,1000,800,box,changed_labels,[crop(left,0)],.5)
    check(ld['additions'][0]['side']==0, 'Raw handedness word does not override current arm identity')
    crossed = pose.copy(); crossed[[13,14]]=crossed[[14,13]]; crossed[[15,16]]=crossed[[16,15]]
    _, ld = r.merge_frame(a,np.empty((0,21,2)),[],crossed,good,1000,800,box,[detection(left)],[crop(left,1)],.5)
    check(ld['additions'][0]['side']==1, 'Normal hand crossing does not use screen left-right')
    edge = left.copy(); edge[8]=[270,650]
    added, ld = r.merge_frame(a,np.empty((0,21,2)),[],pose,good,1000,800,box,[detection(edge)],[crop(edge,0)],.5)
    check(ld['additions'][0]['roi_usable_points']==20 and added['observed'][41], 'Fixed ROI excludes out-of-scope point after native evidence merge')
    check(not r.inside_box(np.array([270,650]),box), 'Fixed ROI upper endpoint remains exclusive')

    oldx=np.zeros((4,69),np.float32); oldm=np.zeros((4,69),bool);oldx[0,0]=12;oldm[0,0]=True
    x=oldx.copy();m=oldm.copy();x[1,0]=30;m[1,0]=True;x[2,1]=40;m[2,1]=True
    outx,outm,discarded=r.lock_candidate_features(oldx,oldm,x,m,np.array([1]))
    check(discarded==1 and outm[1,0] and not outm[2,1] and outx[2,1]==0, 'Next-frame derived support outside303 is discarded and locked')
    check(outx[0,0]==12 and outm[0,0], 'Original supported feature outside scope remains exact')
    check(np.array_equal(x[2],[0,40]+[0]*67), 'Scope lock does not modify caller arrays')
    badx=x.copy();badx[0,0]=13
    try:r.lock_candidate_features(oldx,oldm,badx,m,np.array([1]));rejected=False
    except AssertionError:rejected=True
    check(rejected, 'Changing an original supported value fails explicitly')
    badm=m.copy();badm[0,0]=False
    try:r.lock_candidate_features(oldx,oldm,x,badm,np.array([1]));rejected=False
    except AssertionError:rejected=True
    check(rejected, 'Losing an original supported mask fails explicitly')
    return dict(status='PASS_PURE_RECOVERY_CONTRACTS',checks=count,tests=tests,
                module_sha256=r.sha(r.__file__),selftest_sha256=r.sha(__file__),
                original_merge_sha256=r.sha(r.MERGE_PATH),public_frames_processed=0,
                detector_objects_created=0,detector_calls=0,feature_comparisons=0,educational_accuracy=None)


if __name__=='__main__':
    result=run();r.write(Path(__file__).with_name('RECOVERY_SELFTEST.json'),result)
    print(json.dumps(result,ensure_ascii=False))
