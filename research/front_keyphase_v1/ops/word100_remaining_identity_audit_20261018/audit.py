"""Public TRAIN-only diagnostic. No source assignment, coordinates or model edited.

The delayed proposal is explicitly retrospective (maximum100ms), never called
an instantaneous observation. No fit/search over its gates is performed.
"""
import json,hashlib,sys,shutil,time
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import numpy as np

U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/'word100_uncertain_hand_preservation_20261018'
V=O/'word100_assignment_provenance_20261018'
S=O/'word100_single_arm_recheck_20261018'
G=O/'word100_visible_arm_duplicate_20261018'
PALM=[0,5,9,13,17];HORIZON_US=100000
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p

def prepare():
    assert not (U/'CONTEXT.json').exists() and not (P/'RUNNING.json').exists()
    assert read(P/'FINAL_AUDIT.json')['status']=='PASS'
    c=read(P/'CONTEXT.json');pins=dict(c['pins']);pins.update(read(P/'OUTPUT_PINS.json'));pins.update(read(P/'FINAL_AUDIT.json')['pins'])
    assert read(U/'CONTRACTS.json')['status']=='PASS'
    for p in [P/'FINAL_AUDIT.json',U/'CONTRACTS.json']+list(U.glob('*.py')):pins[str(p.resolve())]=sha(p)
    check(pins);(U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
    write(U/'CONTEXT.json',{'rows':c['rows'],'pins':pins,'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'authorization':'Analyze remaining9, find common safe methods, validate public material; preserve normal hands,C,user footage; no training until review.',
        'diagnostic_scope':'Cached899TRAIN clips x native/masked and cropped/sensor; no new detection, classifier, training, DEV/heldout/user/mentor reads.',
        'prior_observations':'Remaining9 short timeline inspected before design: partial entry, missing intervals, physical hand substitution and duplicate branches. KnownTRAIN diagnostic, not blind test.',
        'fixed_delayed_probe':'Only unknown identity, existing nonnegative proposal, nonempty actual observations, not previously withheld. Follow consecutive existing possible_links only if reciprocal degree1. Blank, ambiguous branching, duplicate coverage>=.8, timestamp gap stop. Two consecutive current-arm/unique-Pose anatomical confirmations of same side, both no more than100ms after original observation, required. Any original evidence-backed contrary identity stops. No past conflict overwrites and no repeated provisional label as anchor. Final frame-level same-side collision rejects all newly proposed collisions. Gates reused (100ms,existing3palm/.4links,.8coverage), no search.',
        'side_outputs':'Only a separate delayed diagnostic proposal is written. Original proposed/confirmed sides, all coordinates and masks unchanged. Availability time and evidence path recorded. Unresolved observations retained.',
        'alternative_probe':'Nearest usable elbow diagnostic only: both elbows finite, physical in-frame, usable, current Hand wrist observed. No artificial confidence lowering; no word exceptions; not applied.',
        'acceptance':'Do not claim remaining9 recovery unless supported by frozen causal/physical chain and prior visual truth; report known normal/crossing contrary predictions, even if rejecting hypothesis. No implicit promotion into production or training.',
        'review':'Same Codex implementation and visual review; separate scalar evidence checker, not independent human labels.'})
    print('FROZEN',len(pins),'protected pins',flush=True)

def anchor(z,p,pr,s,f,i):
    # Require current, not merely inherited, anatomy evidence.
    side=int(p['identity_side'][f,i])
    if side<0:return -1
    direct=bool(pr['current_pose'][f,i] and int(pr['sides'][f,i])==side)
    return side if direct or int(s['action'][f,i]) in (2,3) else -1

def duplicate(g,f,n):
    return bool(n==2 and (g['coverage'][f,0,1]>=.8 or g['coverage'][f,1,0]>=.8))

def probe(z,p,pr,s,g,f,i):
    n=int(z['raw_count'][f]);origin=int(z['timestamp_us'][f])
    if p['identity_side'][f,i]>=0:return None,'already_evidence_backed',[]
    if p['sides'][f,i]<0 or not p['point_mask'][f,i].any():return None,'withheld_or_unobserved',[]
    if duplicate(g,f,n):return None,'duplicate_region',[]
    j=i;path=[(f,i)];last=-1;last_f=-2;evidence_seen=set()
    for t in range(f+1,len(z['raw_count'])):
        us=int(z['timestamp_us'][t]);dt=us-origin
        if dt<=0 or int(z['timestamp_us'][t])-int(z['timestamp_us'][t-1])<=0:return None,'timestamp',path
        if dt>HORIZON_US:return None,'no_two_current_anchors_within100ms',path
        nn=int(z['raw_count'][t]);pn=int(z['raw_count'][t-1])
        if nn==0 or pn==0:return None,'blank_frame',path
        if duplicate(g,t,nn):return None,'duplicate_region',path
        m=p['possible_links'][t,:nn,:pn]
        choices=np.flatnonzero(m[:,j])
        if len(choices)==0:return None,'no_visible_palm_link',path
        if len(choices)!=1:return None,'branching_link',path
        k=int(choices[0])
        if int(m[k].sum())!=1:return None,'merging_link',path
        j=k;path.append((t,j));a=anchor(z,p,pr,s,t,j)
        existing=int(p['identity_side'][t,j])
        if existing>=0:evidence_seen.add(existing)
        if len(evidence_seen)>1:return None,'contrary_identity',path
        if a>=0:
            if a==last and t==last_f+1:
                return {'side':a,'available_at_us':us,'delay_us':dt,'path':[[a,b] for a,b in path],'anchor_frames':[last_f,t]},'delayed_supported',path
            last=a;last_f=t
        else:last=-1;last_f=-2
    return None,'end_of_clip',path

def elbow_probe(z,p,f,i):
    xy=z['pose_xy_px'][f,[13,14]].astype(float);mask=z['pose_usable'][f,[13,14]]
    w=int(z['width'][f]);h=int(z['height'][f]);valid=mask&np.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    if not valid.all() or not p['point_mask'][f,i,0]:return None
    dist=np.linalg.norm(xy-z['hand_xy_px'][f,i,0],axis=1)
    if dist[0]==dist[1]:return None
    return {'side':int(dist.argmin()),'distance_px':dist.tolist()}

def run():
    c=read(U/'CONTEXT.json');check(c['pins']);assert not (U/'SUMMARY.json').exists()
    truth=read(P/'ANNOTATED_COMPARISON.json');tk={(x['clip_id'],x['view'],x['index'],x['raw_index']):x for x in truth}
    cross=read(P/'CROSSING_COMPARISON.json');ck={(x['clip_id'],x['view'],x['index']) for x in cross}
    counts=Counter();proposals=[];ann=[];remaining=[];crossout=[];start=time.monotonic()
    for rn,r in enumerate(c['rows'],1):
        assert r['role']=='train_candidate' and 1<=r['signer_id']<=9
        for v in ('masked','sensor'):
            clip=r['clip_id'];z=load(r['views'][v]['raw']);p=load(P/'results'/f'{clip}_{v}.npz');pr=load(V/'results'/f'{clip}_{v}.npz');s=load(S/'results'/f'{clip}_{v}.npz');g=load(G/'results'/f'{clip}_{v}.npz')
            for f,n0 in enumerate(z['raw_count']):
                n=int(n0);counts['frames']+=1;counts['raw_candidates']+=n;counts['observed_points_preserved']+=int(p['point_mask'][f,:n].sum());cand={};reason={};paths={}
                for i in range(n):
                    x,why,path=probe(z,p,pr,s,g,f,i);reason[i]=why;paths[i]=path
                    if x is not None:cand[i]=x
                # A new delayed assignment must never occupy an already
                # evidenced slot or allow two raw hands to occupy that slot.
                denied=set()
                for i,x in cand.items():
                    for j in range(n):
                        if i==j:continue
                        ss=cand[j]['side'] if j in cand else int(p['identity_side'][f,j])
                        if ss==x['side']:denied.add(i)
                for i in denied:del cand[i];reason[i]='destination_collision'
                for i in range(n):
                    counts[reason[i]]+=1;x=cand.get(i);k=(clip,v,f,i);e=elbow_probe(z,p,f,i)
                    newside=int(p['sides'][f,i]) if x is None else x['side']
                    if x is not None:
                        proposals.append({'clip_id':clip,'word':r['label'],'view':v,'index':f,'raw_index':i,'old_hint':int(p['sides'][f,i]),**x})
                        counts['delayed_same_hint' if x['side']==p['sides'][f,i] else 'delayed_changed_hint']+=1
                    if k in tk:
                        t=tk[k];a={**t,'delayed_proposal':x,'delayed_block_reason':reason[i],'delay_probe_path':[[a,b] for a,b in paths[i]],'elbow_only':e,'hypothetical_side':newside};ann.append(a)
                        if t['verdict']=='error_identity' and t['new_side'] is not None and t['new_side']!=t['actual_side']:
                            near=[]
                            for q in range(max(0,f-1),min(len(z['raw_count']),f+13)):
                                nq=int(z['raw_count'][q]);near.append({'frame':q,'timestamp_us':int(z['timestamp_us'][q]),'raw_count':nq,'hand_wrist_xy':z['hand_xy_px'][q,:nq,0].tolist(),'observed_points':p['point_mask'][q,:nq].sum(1).tolist(),'visible_palm_points':p['point_mask'][q,:nq][:,PALM].sum(1).tolist(),'old_hint':p['sides'][q,:nq].tolist(),'evidence_side':p['identity_side'][q,:nq].tolist(),'current_anchor':[anchor(z,p,pr,s,q,j) for j in range(nq)],'possible_links':p['possible_links'][q,:nq].astype(int).tolist(),'pose_wrist_confidence':z['pose_conf'][q,[15,16]].astype(float).tolist()})
                            a={**a,'core_intervals':r['intervals'],'timestamp_us':int(z['timestamp_us'][f]),'all_observed_points':int(p['point_mask'][f,i].sum()),'visible_palm_points':int(p['point_mask'][f,i,PALM].sum()),'timeline':near};remaining.append(a)
                if (clip,v,f) in ck:
                    out={'clip_id':clip,'view':v,'index':f,'original_sides':p['sides'][f,:n].tolist(),'delayed_sides':[cand[i]['side'] if i in cand else int(p['sides'][f,i]) for i in range(n)],'elbow_sides':[None if (ee:=elbow_probe(z,p,f,i)) is None else ee['side'] for i in range(n)]};crossout.append(out)
            if rn%200==0 and v=='sensor':print('AUDIT',rn,'/899',flush=True)
    assert len(ann)==461 and len(remaining)==9 and len(crossout)==315
    normal=[x for x in ann if x['verdict']=='normal_identity'];errors=[x for x in ann if x['verdict']=='error_identity'];es=[x for x in normal if x['elbow_only'] is not None]
    summary={'status':'COMPLETE_PENDING_INDEPENDENT_AND_VISUAL','counts':dict(counts),'annotated_normal':len(normal),'delayed_normal_contrary':sum(x['hypothetical_side']!=x['actual_side'] for x in normal),'delayed_remaining9_correct':sum(x['hypothetical_side']==x['actual_side'] for x in remaining),'delayed_crossing_changed':sum(x['delayed_sides']!=x['original_sides'] for x in crossout),'elbow_normal_applicable':len(es),'elbow_normal_wrong':sum(x['elbow_only']['side']!=x['actual_side'] for x in es),'elbow_remaining9_applicable':sum(x['elbow_only'] is not None for x in remaining),'elbow_remaining9_correct':sum(x['elbow_only'] is not None and x['elbow_only']['side']==x['actual_side'] for x in remaining),'seconds':time.monotonic()-start,'original_arrays_modified':False,'training':0,'classifier_inference':0,'user_reads':0,'C_changed':False}
    for name,x in [('PROPOSALS',proposals),('ANNOTATED',ann),('REMAINING9',remaining),('CROSSING',crossout),('SUMMARY',summary)]:write(U/f'{name}.json',x)
    check(c['pins']);print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
