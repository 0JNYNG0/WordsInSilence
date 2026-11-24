import sys,json,hashlib,shutil,time,os,traceback
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/'word100_assignment_provenance_20261018';A=O/'word100_label_only_audit_20261018'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def pinscheck(pins):
    for p,h in pins.items():assert sha(p)==h,p
def prepare():
    assert read(P/'FINAL_AUDIT.json')['status']=='PASS'
    assert not (P/'RUNNING.json').exists() and not (U/'CONTEXT.json').exists()
    assert read(U/'CONTRACTS.json')['status']=='PASS'
    c=read(P/'CONTEXT.json');pins=dict(c['pins']);pins.update(read(P/'OUTPUT_PINS.json'))
    for p in [P/'candidate.py',P/'CONTEXT.json',P/'FINAL_AUDIT.json',P/'CHANGES.json',P/'ANNOTATED_COMPARISON.json',P/'CROSSING_COMPARISON.json',U/'candidate.py',U/'run.py',U/'contracts.py',U/'CONTRACTS.json']:pins[str(p)]=sha(p)
    (U/'results').mkdir();(U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
    write(U/'CONTEXT.json',{'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'rows':c['rows'],'pins':pins,
        'authorization':'Latest user approved separate visible-arm/distorted-duplicate candidate and public validation. Preserve C. No training until joint review.',
        'protocol':{'candidate':'One fixed candidate; no threshold search after public replay. Known TRAIN456 informed design; not independent test accuracy.',
            'geometry':'Strict in-frame usable shoulder/elbow/wrist. Hand wrist to forearm segment AND Pose wrist to Hand palm hull. Elbow-only distance and distal points are diagnostic, never override identity.',
            'duplicate':'Only provisional opposite proposal with confirmed anchor; both visible arm chains; both candidates prefer same arm; >=80% of 20 non-wrist landmarks inside anchor hull plus existing .2 palm-span tolerance. No forced flip, no invented coordinates.',
            'constants':'Existing pose_gate .35, ambiguity .7, duplicate_palm_fraction .2, containment .80; no new label-specific constants.',
            'history':'Baseline history/proposals preserved independently; new withheld outputs cannot contaminate later evidence.',
            'validation':'899 TRAIN clips, two saved raw streams; exact prior-output comparison. Audit441 normal and315 crossing controls. Inspect every hard change in original public RGB. Independent scalar reconstruction. Prefix/order/reflection/hidden-point contracts.',
            'acceptance':'No known normal/crossing regression. Any unresolved hard-change correctness is reported; no production adoption. No recognition-improvement claim.'},
        'prior_design_diagnostic':'Observed distal Pose points are unusable for most first-entry errors despite plausible coordinates. They cannot be promoted as observed evidence. Prior arm-history extension gave0 recovery, so not repeated.',
        'prohibited':['user/mentor/DEV/heldout','training','classifier inference','camera','download','external transfer','Git','production C/TCN/DTW/web edits','future/outside frame data','word-specific logic'],
        'operator':'Codex local implementation and separate arithmetic verification; no independent human annotation.'})
    print('FROZEN',len(c['rows']),len(pins),flush=True)
def main():
    from candidate import geometry,decide
    c=read(U/'CONTEXT.json');pinscheck(c['pins']);assert not (U/'SUMMARY.json').exists()
    write(U/'RUNNING.json',{'pid':os.getpid(),'task':'public cached geometry replay, no training'})
    ann=read(A/'VISUAL_REVIEW.json')['records'];lookup={(x['clip_id'],x['view'],x['index'],x['raw_index']):x for x in ann}
    crosskeys={(x['clip_id'],x['view'],x['index']) for x in read(P/'CROSSING_COMPARISON.json')}
    counts=Counter();changes=[];annotated=[];cross=[];pins={};start=time.monotonic()
    names={0:'unassigned',1:'provisional',2:'confirmed',3:'withheld_conflict'}
    def serial(a):return [[float(v) if np.isfinite(v) else None for v in row] for row in a]
    try:
        for rn,r in enumerate(c['rows'],1):
            assert r['role']=='train_candidate' and 1<=r['signer_id']<=9
            for v in ('masked','sensor'):
                raw=load(r['views'][v]['raw']);old=load(P/'results'/f"{r['clip_id']}_{v}.npz");N=len(raw['raw_count']);side=old['sides'].copy()
                arm=np.full((N,2,2),np.nan);elbow=arm.copy();distal=arm.copy();coverage=np.zeros((N,2,2));pref=np.full((N,2),-1,np.int8);vis=np.zeros((N,2),bool);inside=vis.copy();blocked=np.zeros((N,2),bool)
                for f in range(N):
                    n=int(raw['raw_count'][f]);g=geometry(raw['hand_xy_px'][f,:n],raw['pose_xy_px'][f],raw['pose_usable'][f],int(raw['width'][f]),int(raw['height'][f]))
                    prev=[None if s<0 else int(s) for s in old['sides'][f,:n]];states=[names[int(s)] for s in old['state'][f,:n]]
                    out,b=decide(prev,states,g);side[f,:n]=[-1 if s is None else s for s in out]
                    arm[f,:n]=np.where(np.isfinite(g['arm_cost']),g['arm_cost'],np.nan);elbow[f,:n]=np.where(np.isfinite(g['elbow_distance']),g['elbow_distance'],np.nan);distal[f,:n]=np.where(np.isfinite(g['distal_cost']),g['distal_cost'],np.nan);coverage[f,:n,:n]=g['coverage'];pref[f,:n]=g['preferred'];vis[f]=g['arms_visible'];inside[f,:n]=g['valid'];blocked[f,b]=True
                    counts['frames']+=1;counts['raw_candidates']+=n;counts['both_visible_arm_frames']+=int(all(g['arms_visible']))
                    trace={'arm_cost':serial(g['arm_cost']),'elbow_distance':serial(g['elbow_distance']),'distal_cost':serial(g['distal_cost']),'coverage':g['coverage'].tolist(),'arm_preference':g['preferred'],'arms_visible':g['arms_visible'],'inside':g['valid'],'prior_states':states}
                    for i in range(n):
                        counts['unchanged' if prev[i]==out[i] else 'new_withheld']+=1
                        if states[i]=='provisional':
                            counts['provisional_total']+=1
                            counts['provisional_arm_agrees' if g['preferred'][i]==prev[i] else 'provisional_arm_disagrees' if g['preferred'][i]>=0 else 'provisional_arm_inconclusive']+=1
                        if i in b:changes.append({'clip_id':r['clip_id'],'word':r['label'],'view':v,'index':f,'frame_id':int(raw['frame_id'][f]),'raw_index':i,'old_side':prev[i],'new_side':out[i],'trace':trace})
                        key=(r['clip_id'],v,f,i)
                        if key in lookup:
                            q=lookup[key];annotated.append({**{k:q[k] for k in ('id','clip_id','word','view','index','raw_index','verdict','actual_side')},'old_side':prev[i],'new_side':out[i],'trace':trace})
                    if (r['clip_id'],v,f) in crosskeys:cross.append({'clip_id':r['clip_id'],'view':v,'index':f,'preserved':prev==out})
                path=U/'results'/f"{r['clip_id']}_{v}.npz";assert not path.exists()
                np.savez_compressed(path,sides=side,arm_cost=arm,elbow_distance=elbow,distal_cost=distal,coverage=coverage,preference=pref,arms_visible=vis,strict_hand_inside=inside,new_withheld=blocked,frame_id=raw['frame_id'],timestamp_us=raw['timestamp_us']);pins[str(path)]=sha(path)
            if rn%100==0:print('REPLAY',rn,'/899',flush=True)
        assert len(annotated)==456 and len(cross)==315
        pinscheck(c['pins']);write(U/'CHANGES.json',changes);write(U/'ANNOTATED_COMPARISON.json',sorted(annotated,key=lambda r:r['id']));write(U/'CROSSING_COMPARISON.json',cross);write(U/'OUTPUT_PINS.json',pins)
        summary={'status':'REPLAY_COMPLETE_PENDING_VERIFY_VISUAL','counts':dict(counts),'changed_clips':len({x['clip_id'] for x in changes}),'normal_preserved':sum(x['verdict']=='normal_identity' and x['old_side']==x['new_side'] for x in annotated),'crossing_preserved':sum(x['preserved'] for x in cross),'audited_errors_new_withheld':[x['id'] for x in annotated if x['verdict']=='error_identity' and x['old_side']!=x['new_side']],'audited_normal_changed':[x['id'] for x in annotated if x['verdict']=='normal_identity' and x['old_side']!=x['new_side']],'seconds':time.monotonic()-start,'training':0,'production_changed':False,'user_mentor_dev_heldout_reads':0}
        write(U/'SUMMARY.json',summary);(U/'RUNNING.json').unlink();print(json.dumps(summary,ensure_ascii=False),flush=True)
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'traceback':traceback.format_exc()})
        raise
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else main()
