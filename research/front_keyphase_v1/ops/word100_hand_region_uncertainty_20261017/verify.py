"""Saved-result scalar audit. Independent hull/clipping; imports no candidate."""
import json,hashlib,math
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime
import numpy as np
U=Path(__file__).resolve().parent;B=U.parent/'word100_hand_visibility_20261017'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(p):return all(math.isfinite(v) for v in p)
def dist(a,b):return math.hypot(a[0]-b[0],a[1]-b[1])
def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def hull(points):
    points=sorted(set(map(tuple,points)))
    if len(points)<3:return points
    lo=[];hi=[]
    for p in points:
        while len(lo)>1 and cross(lo[-2],lo[-1],p)<=0:lo.pop()
        lo.append(p)
    for p in reversed(points):
        while len(hi)>1 and cross(hi[-2],hi[-1],p)<=0:hi.pop()
        hi.append(p)
    return lo[:-1]+hi[:-1]
def area(poly):
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1])))/2 if len(poly)>2 else 0.
def intersect(subject,clip):
    # Sutherland-Hodgman convex polygon clipping in double precision.
    if len(subject)<3 or len(clip)<3:return 0.
    out=list(subject)
    for a,b in zip(clip,clip[1:]+clip[:1]):
        inp=out;out=[]
        if not inp:break
        old=inp[-1];ds=cross(a,b,old)
        for new in inp:
            de=cross(a,b,new)
            if (de>=0)!=(ds>=0):
                f=ds/(ds-de)
                out.append((old[0]+f*(new[0]-old[0]),old[1]+f*(new[1]-old[1])))
            if de>=0:out.append(new)
            old,ds=new,de
    return area(out)

s=read(U/'SCOPE.json');d=read(U/'PREDICTIONS.json');p=s['parameters'];b=s['base_parameters']
assert sha(U/'region.py')==s['code_sha256']['region.py']
assert sha(B/'visibility.py')==s['code_sha256']['baseline']
assert sha(U/'SCOPE.json')==d['scope_sha256'] and sha(U/'VISUAL_LABELS.json')==d['labels_sha256']
assert read(U/'VISUAL_LABELS.json')['predictions_seen'] is False
protected=read(B/'INDEPENDENT_CHECK.json')['protected_pins']
for path,h in {**s['pins'],**protected}.items():assert sha(path)==h,path
lookup={r['clip_id']:r for r in d['records']}
old={r['clip_id']:r for r in read(B/'PREDICTIONS.json')['records']}
checked=regions=immutable=0;max_u_error=0.;max_bounds_error=0.;errors=[];changed_confident=[]
counts={g:{k:Counter() for k in ['baseline_whole','candidate_whole','baseline_core','candidate_core']} for g in ['previous48','fresh12']}
for r in s['records']:
    obs=np.load(r['paths']['candidate']);raw=np.load(r['paths']['raw']);frames=lookup[r['clip_id']]['frames']
    assert len(frames)==len(obs['timestamp_us'])
    tracks=[[],[]];last_t=None;last_size=None
    for i,f in enumerate(frames):
        t=int(raw['timestamp_us'][i])/1e6;w,h=int(raw['width'][i]),int(raw['height'][i])
        assert f['frame']==i and f['timestamp_us']==int(raw['timestamp_us'][i])
        assert f['core']==any(a<=t<=z for a,z in r['intervals'])
        if (last_t is not None and t<=last_t) or last_size not in (None,(w,h)):tracks=[[],[]]
        last_t,last_size=t,(w,h)
        xy=obs['xy_px'][i].astype(float).tolist();mask=obs['usable'][i].tolist();conf=raw['pose_conf'][i].astype(float).tolist()
        scale=dist(xy[11],xy[12])
        def inside(v):return finite(v) and 0<=v[0]<=w and 0<=v[1]<=h
        def anchor(side):
            ix=[11,12,13+side,15+side]
            return (math.isfinite(scale) and scale>1e-6 and all(finite(xy[j]) and math.isfinite(conf[j]) and conf[j]>=b['pose_confidence'] for j in ix)
                and all(mask[j] and inside(xy[j]) for j in ix[:3])
                and all(b['arm_segment_min_shoulders']<=v<=b['arm_segment_max_shoulders'] for v in [dist(xy[11+side],xy[13+side])/scale,dist(xy[13+side],xy[15+side])/scale]))
        current=[]
        for side in (0,1):
            ix=range(33+side*21,54+side*21)
            current.append([xy[j] for j in ix] if anchor(side) and all(mask[j] and inside(xy[j]) for j in ix) else None)
            tracks[side]=[z for z in tracks[side] if t-z[0]<=p['history_seconds']]
        for side in (0,1):
            baseline=f['baseline'][side];candidate=f['hands'][side];state=baseline['state']
            if r['clip_id'] in old:assert baseline==old[r['clip_id']]['frames'][i]['hands'][side]
            if state in ('observed','partial_observation','assignment_uncertain','unknown'):
                assert state==candidate['state'];immutable+=1
            if state in ('inside_missing','overlap_likely','boundary_outside_likely') and tracks[side] and anchor(side):
                ts,points,a=tracks[side][-1];age=t-ts
                offsets=[[q[k]-a[k] for k in (0,1)] for q in points]
                nominal=[[v[k]+xy[15+side][k] for k in (0,1)] for v in offsets]
                poly=hull(nominal)
                lo=[min(v[k] for v in points) for k in (0,1)];hi=[max(v[k] for v in points) for k in (0,1)]
                diameter=dist(lo,hi)
                if len(poly)>=3 and diameter>1e-6 and area(poly)>0:
                    speed=0.
                    if len(tracks[side])>1:
                        earlier=tracks[side][-2];dt=ts-earlier[0]
                        speed=max(dist(v,[q[k]-earlier[2][k] for k in (0,1)]) for v,q in zip(offsets,earlier[1]))/dt
                    u=p['base_uncertainty_shoulders']*scale+age*(diameter*p['unknown_deformation_hand_diameters_per_second']+speed)
                    low=[min(v[k] for v in poly) for k in (0,1)];high=[max(v[k] for v in poly) for k in (0,1)]
                    e=candidate['evidence'];assert candidate['region_status']=='estimated_envelope'
                    max_u_error=max(max_u_error,abs(u-e['uncertainty_pixels']))
                    refbounds=[[v-u for v in low],[v+u for v in high]]
                    max_bounds_error=max(max_bounds_error,max(abs(refbounds[j][k]-e['outer_bounds'][j][k]) for j in (0,1) for k in (0,1)))
                    regions+=1
                    if u>p['max_uncertainty_hand_diameters']*diameter:state='region_uncertain'
                    elif low[0]+u<0 or low[1]+u<0 or high[0]-u>w or high[1]-u>h:state='boundary_outside_likely'
                    elif low[0]-u<0 or low[1]-u<0 or high[0]+u>w or high[1]+u>h:state='boundary_uncertain'
                    else:
                        state='inside_missing'
                        if current[1-side] is not None:
                            other=hull(current[1-side])
                            areas=[intersect([(q[0]+dx,q[1]+dy) for q in poly],other) for dx in (-u,u) for dy in (-u,u)]
                            if min(areas)>0:state='overlap_likely'
            checked+=1
            if state!=candidate['state']:errors.append([r['clip_id'],i,side,state,candidate['state']])
            assert candidate['coordinate_repair'] is False
            for prefix,row in [('baseline',baseline),('candidate',candidate)]:
                counts[r['group']][prefix+'_whole'][row['state']]+=1
                if f['core']:counts[r['group']][prefix+'_core'][row['state']]+=1
            if f['core'] and baseline['state']!=candidate['state'] and candidate['state'] in ('boundary_outside_likely','overlap_likely'):
                changed_confident.append([r['clip_id'],i,side,candidate['state']])
        for side in (0,1):
            if current[side] is not None:
                tracks[side].append((t,current[side],xy[15+side]));tracks[side]=tracks[side][-2:]
assert not errors,errors[:5]
assert max_u_error<.01 and max_bounds_error<.01,(max_u_error,max_bounds_error)
assert counts==d['counts']
summary=defaultdict(lambda:{'baseline':Counter(),'candidate':Counter()});confusion=defaultdict(lambda:defaultdict(Counter));specific=defaultdict(lambda:{'baseline':Counter(),'candidate':Counter()})
labels=read(U/'VISUAL_LABELS.json')['labels'];assert len(labels)==len(d['visual_comparisons'])
for x,z in zip(labels,d['visual_comparisons']):
    assert all(z[k]==v for k,v in x.items())
    f=lookup[x['clip_id']]['frames'][x['frame']];side=x['side']
    assert z['baseline']==f['baseline'][side] and z['candidate']==f['hands'][side]
    actual=x['visual_category'];origin=x['origin']
    for key in ('baseline','candidate'):
        state=z[key]['state'];confusion[origin][actual][key+':'+state]+=1
        if actual=='observed_control':
            assert state=='observed';summary[origin][key]['controls_preserved']+=1;continue
        loc='out' if state=='boundary_outside_likely' else 'in' if state in ('inside_missing','overlap_likely') else '?'
        target='out' if actual=='boundary_or_outside' else 'in'
        summary[origin][key]['abstain' if loc=='?' else 'agree' if loc==target else 'disagree']+=1
        if actual=='inside_overlap':specific[origin][key]['overlap_found' if state=='overlap_likely' else 'overlap_not_found']+=1
        elif state=='overlap_likely':specific[origin][key]['overlap_false_positive']+=1
for path,h in {**s['pins'],**protected}.items():assert sha(path)==h
out={'status':'PASS','at':datetime.now().astimezone().isoformat(timespec='seconds'),
 'operator':'Codex root, scalar hull/clipping reference code; not second human/agent',
 'source_pins_checked':len(s['pins']),'protected_pins':protected,'states_checked':checked,'regions_checked':regions,
 'max_uncertainty_difference_pixels':max_u_error,'max_bounds_difference_pixels':max_bounds_error,
 'state_mismatches':errors,'immutable_states_checked':immutable,'changed_confident_core_states':changed_confident,
 'counts':counts,'visual_location_summary':summary,'visual_confusion':confusion,'overlap_specific':specific,
 'limits':'Arithmetic not biological truth; conservative abstentions not correct labels; sample is TRAIN selected correlated frames.',
 'evidence_sha256':{n:sha(U/n) for n in ['region.py','SCOPE.json','VISUAL_LABELS.json','PREDICTIONS.json','SYNTHETIC_TESTS.json']}}
with (U/'INDEPENDENT_CHECK.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({k:out[k] for k in ['status','states_checked','regions_checked','max_uncertainty_difference_pixels','max_bounds_difference_pixels','visual_location_summary','overlap_specific']},ensure_ascii=False,indent=2))
