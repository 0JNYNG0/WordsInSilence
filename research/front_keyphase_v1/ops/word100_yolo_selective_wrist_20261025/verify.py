"""Independent saved arithmetic; no candidate imports or model calls."""
from common import *
from collections import Counter
f=read(U/'FROZEN.json');idx=read(U/'INDEX.json');manifest={r['clip_id']:r for r in f['old_rows']+f['new_rows']};summary=read(U/'SUMMARY.json');adds=read(U/'ADDITIONS.json')
assert len(idx)==46 and sum(i['frames'] for i in idx)==5637
counts=Counter();seen=set();checks=[]
for x in idx:
    for key in ('arrays','records'):assert sha(x[key])==x[key+'_sha256']
    r=manifest[x['clip_id']];z=load(r['arrays']);o=load(x['arrays']);rr=read(x['records'])['frames'];n=r['frames']
    bv=z['redetected_roi_usable'][:,33:].reshape(n,2,21);base=z['redetected_roi_xy_px'][:,33:].reshape(n,2,21,2)
    assert np.array_equal(o['frame_id'],z['frame_id']);assert np.array_equal(np.flatnonzero(o['called']),[q['i'] for q in rr])
    for pre in ('s','p'):
        assert o[pre+'_valid'][bv].all() and np.array_equal(o[pre+'_xy'][bv],base[bv])
        assert np.array_equal(o[pre+'_valid'][~o['called']],bv[~o['called']])
        assert np.array_equal(o[pre+'_xy'][~o['called']][bv[~o['called']]],base[~o['called']][bv[~o['called']]])
    counts['preserved']+=int(bv.sum());counts['calls']+=int(o['called'].sum());counts['slots']+=int(o['called_sides'].sum())
    for q in rr:
        i=q['i'];assert o['called'][i]
        for side in map(int,q['boxes']):
            assert not z['native_usable'][i,33+side*21:54+side*21].any()
            assert not bv[i,side].any() # full native availability checked separately in contract reconstruction
        for d in q['decisions']:
            if not d['accepted']:continue
            s=d['side'];assert not bv[i,s].any();assert o['called_sides'][i,s]
            seen.add((x['clip_id'],i,s));p=points(q['full'][d['index']]);c=points(q['crop'][d['crop_index']]);pose=z['raw_pose_xy_px'][i];scale=np.linalg.norm(pose[11]-pose[12])
            assert np.linalg.norm(p[0]-c[0])<=.08*scale
            if d['position_only']:
                assert o['p_valid'][i,s].sum()==1 and o['p_valid'][i,s,0]
                assert not o['s_valid'][i,s].any();counts['position_only']+=1
            else:assert np.array_equal(o['s_valid'][i,s],o['p_valid'][i,s])
    # Only additions; no denominator/required-duration trimming or frame deletion.
    assert len(o['frame_id'])==n and np.array_equal(o['frame_id'],np.arange(n))
assert seen=={(a['clip_id'],a['frame_id'],a['side']) for a in adds}
assert counts['preserved']==summary['preserved_M1_joint_frames']['p']==101680
assert counts['position_only']==6
# Direct rectangle arithmetic, independent from previous evaluate_benchmark scorer.
ra={r['sample_id']:r for r in read(O/'word100_recovery_image_benchmark_20261025/BLIND_REVIEW_A.json')['rows']};rb={r['sample_id']:r for r in read(O/'word100_recovery_image_benchmark_20261025/BLIND_REVIEW_B.json')['rows']}
vl={r['sample_id']:r for r in read(O/'word100_temporal_refiner_pilot_20261025/VISIBLE_LABELS_FROZEN.json')['labels']}
rows=read(U/'SCORED_ROWS.json')
for q in rows:
    sid=q['id']
    if q['status']=='unscorable':continue
    if not q['available']:assert q['status']=='unavailable';continue
    a,b=(ra[sid]['wrist_box'],rb[sid]['wrist_box']) if sid.startswith('R') else (vl[sid]['rectangle'],vl[sid]['rectangle'])
    x,y=q['point'];v=[box[0]<=x<=box[2] and box[1]<=y<=box[3] for box in (a,b)]
    expected='inside_both' if all(v) else 'inside_one_ambiguous' if any(v) else 'outside_both'
    assert expected==q['status'],(sid,expected,q['status'])
for group,g in summary['scoring'].items():
    for c,a in g.items():
        rr=[q for q in rows if q['condition']==c and (group=='ALL42' or q['group']==group)]
        assert dict(Counter(q['status'] for q in rr))==a['counts']
for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
write(U/'VERIFY_SAVED.json',{'status':'PASS','saved_score_rows':len(rows),'clips':46,'frames':5637,'counts':dict(counts),'additions':len(seen),'protected_files':len(f['protected']),
 'checks':['source and output hashes','all inherited M1 points exact','untriggered outputs unchanged','absent sides only','wrist-only cannot supply finger angles','timestamps/frame count preserved','score arithmetic independently reconstructed','C/T3-F/runtime pins unchanged'],
 'operator':'Codex separate arithmetic implementation; not a separate human/agent blind review'})
print('PASS',dict(counts),len(rows))
