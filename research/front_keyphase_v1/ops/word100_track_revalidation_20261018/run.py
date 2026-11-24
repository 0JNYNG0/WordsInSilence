import sys,json,hashlib,shutil,time,os,traceback
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/'word100_visible_arm_duplicate_20261018';V=O/'word100_assignment_provenance_20261018';A=O/'word100_label_only_audit_20261018'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def pinscheck(pins):
    for p,h in pins.items():assert sha(p)==h,p
def prepare():
    assert read(P/'FINAL_AUDIT.json')['status']=='PASS' and read(U/'CONTRACTS.json')['status']=='PASS'
    assert not (P/'RUNNING.json').exists() and not (U/'CONTEXT.json').exists()
    c=read(P/'CONTEXT.json');pins=dict(c['pins']);pins.update(read(P/'OUTPUT_PINS.json'))
    for p in [P/'candidate.py',P/'CONTEXT.json',P/'FINAL_AUDIT.json',P/'VISUAL_REVIEW.json',P/'ARM_CONFLICTS.json',P/'CROSSING_COMPARISON.json',U/'candidate.py',U/'run.py',U/'contracts.py',U/'CONTRACTS.json']:pins[str(p)]=sha(p)
    (U/'results').mkdir();(U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
    write(U/'CONTEXT.json',{'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'rows':c['rows'],'pins':pins,
        'authorization':'User approved common causal track revalidation using current visible arm/hand evidence; preserve C; public validation only; no training until joint review.',
        'protocol':{'candidate':'Single frozen candidate. Known public TRAIN456 audit and5 additional prior error frames informed design; these are not independent test data. No post-result threshold search.',
            'inputs':'Reuse independently verified visible-arm geometry for899 TRAIN two streams. Only current raw Hand21 and usable strict in-frame shoulder/elbow/wrist enter geometry. No user/mentor/DEV/heldout.',
            'trigger':'Only previously provisional assigned hands. Current both-arm evidence uniquely prefers opposite side including competing raw candidates; confirmed destination protected.',
            'confirmation':'First conflict withheld. Same physical palm on consecutive frames within100ms and .4 shoulder gate must agree2 times to relabel. Count2/gap100ms/ratio.7/gates inherited unchanged.',
            'memory':'Separate corrected identity memory. Current arm evidence renews100ms expiry; mere temporal tracking cannot renew it. No carry against current contradictory arm preference. Prior confirmed assignments and prior withholding protected.',
            'collision':'Resolve destinations jointly, no duplicate slots, no forced unassigned fill, no synthesized coordinates. Normal crossing independent of screen half.',
            'validation':'226362 frames exact old/new comparison;441 known normal and315 crossing controls; old13-error and extra5-error sets reported separately. Every hard change source RGB reviewed. Separate scalar state replay, wrapper parity, raw-order, prefix, unseen-point perturbation contracts.',
            'failure':'Any regression/uncertain changed scene disclosed; no production deployment or training. No recognition accuracy claim.'},
        'prohibited':['training','classifier inference','user footage','mentor/DEV/heldout','camera','download','external transfer','Git','C/TCN/DTW/web edit','word/frame exception','future/outside input pixels'],
        'implementation':'Codex local; separate calculation verification and visual review by same operator, not independent human ground truth.'})
    print('FROZEN',len(c['rows']),len(pins),flush=True)

def geometry_from_saved(raw,saved,f,n):
    p=raw['pose_xy_px'][f];pu=raw['pose_usable'][f];w=int(raw['width'][f]);h=int(raw['height'][f]);sh=p[[11,12]]
    visible=pu[[11,12]].all() and np.isfinite(sh).all() and ((sh[:,0]>=0)&(sh[:,0]<w)&(sh[:,1]>=0)&(sh[:,1]<h)).all()
    sw=float(np.linalg.norm(sh[0]-sh[1])) if visible else 0.
    return {'valid':saved['strict_hand_inside'][f,:n].tolist(),'preferred':saved['preference'][f,:n].tolist(),'arms_visible':saved['arms_visible'][f].tolist(),'arm_cost':np.where(np.isfinite(saved['arm_cost'][f,:n]),saved['arm_cost'][f,:n],np.inf),'scale':sw}

def main():
    from candidate import Revalidator
    c=read(U/'CONTEXT.json');pinscheck(c['pins']);assert not (U/'SUMMARY.json').exists()
    write(U/'RUNNING.json',{'pid':os.getpid(),'task':'public cached track revalidation, no learning'})
    ann=read(A/'VISUAL_REVIEW.json')['records'];truth={(x['clip_id'],x['view'],x['index'],x['raw_index']):{**x,'set':'original456'} for x in ann}
    for x in read(P/'VISUAL_REVIEW.json')['records']:
        k=(x['clip_id'],x['view'],x['index'],x['raw_index'])
        if k not in truth:truth[k]={**x,'set':'additional5','id':f"additional_{x['evidence_id']}",'verdict':'error_identity'}
    assert len(truth)==461
    crosskeys={(x['clip_id'],x['view'],x['index']) for x in read(P/'CROSSING_COMPARISON.json')}
    counts=Counter();changes=[];annotations=[];cross=[];events=[];pins={};start=time.monotonic()
    state_names={0:'unassigned',1:'provisional',2:'confirmed',3:'withheld_conflict'}
    codes={'previous_candidate':0,'pending_arm_conflict':1,'revalidated_current_arm':2,'renewed_current_arm':3,'bounded_revalidated_history':4,'destination_conflict':5}
    try:
        for rn,r in enumerate(c['rows'],1):
            assert r['role']=='train_candidate' and 1<=r['signer_id']<=9
            for view in ('masked','sensor'):
                raw=load(r['views'][view]['raw']);old=load(P/'results'/f"{r['clip_id']}_{view}.npz");provenance=load(V/'results'/f"{r['clip_id']}_{view}.npz")
                N=len(raw['raw_count']);sides=old['sides'].copy();actions=np.zeros((N,2),np.int8);targets=np.full((N,2),-1,np.int8);votes=np.zeros((N,2),np.int16);roots=np.full((N,2),-1,np.int32);model=Revalidator()
                for f in range(N):
                    n=int(raw['raw_count'][f]);g=geometry_from_saved(raw,old,f,n);prev=[None if x<0 else int(x) for x in old['sides'][f,:n]];states=[state_names[int(x)] for x in provenance['state'][f,:n]]
                    out,d=model.step(raw['hand_xy_px'][f,:n],raw['raw_score'][f,:n],int(raw['timestamp_us'][f]),prev,states,g)
                    counts['frames']+=1;counts['raw_candidates']+=n
                    trace={**d,'prior_state':states,'prior_sides':prev,'arm_cost':[[None if not np.isfinite(x) else float(x) for x in rr] for rr in g['arm_cost']],'arms_visible':g['arms_visible'],'arm_preference':g['preferred']}
                    for i in range(n):
                        sides[f,i]=-1 if out[i] is None else out[i];actions[f,i]=codes[d['recheck_reason'][i]];targets[f,i]=d['current_arm_target'][i];votes[f,i]=d['confirmation_count'][i];roots[f,i]=-1 if d['evidence_origin_frame'][i] is None else d['evidence_origin_frame'][i]
                        counts[d['recheck_reason'][i]]+=1
                        if out[i]==prev[i]:counts['unchanged_slots']+=1
                        else:
                            kind='new_withheld' if out[i] is None else 'reassigned';counts[kind]+=1
                            changes.append({'clip_id':r['clip_id'],'word':r['label'],'view':view,'index':f,'frame_id':int(raw['frame_id'][f]),'raw_index':i,'old_side':prev[i],'new_side':out[i],'kind':kind,'trace':trace})
                        if states[i]=='confirmed':counts['prior_confirmed']+=1;counts['prior_confirmed_preserved']+=int(out[i]==prev[i])
                        if prev[i] is None:counts['prior_unassigned']+=1;counts['prior_unassigned_preserved']+=int(out[i] is None)
                        key=(r['clip_id'],view,f,i)
                        if key in truth:
                            t=truth[key];annotations.append({**{k:t[k] for k in ('id','set','verdict','actual_side')},'clip_id':r['clip_id'],'word':r['label'],'view':view,'index':f,'raw_index':i,'old_side':prev[i],'new_side':out[i],'trace':trace})
                        if d['recheck_reason'][i]!='previous_candidate':events.append({'clip_id':r['clip_id'],'view':view,'index':f,'raw_index':i,'trace':trace})
                    if (r['clip_id'],view,f) in crosskeys:cross.append({'clip_id':r['clip_id'],'view':view,'index':f,'preserved':prev==out})
                path=U/'results'/f"{r['clip_id']}_{view}.npz";assert not path.exists()
                np.savez_compressed(path,sides=sides,action=actions,target=targets,votes=votes,origin=roots,frame_id=raw['frame_id'],timestamp_us=raw['timestamp_us']);pins[str(path)]=sha(path)
            if rn%200==0:print('REPLAY',rn,'/899',flush=True)
        assert len(annotations)==461 and len(cross)==315
        pinscheck(c['pins']);write(U/'CHANGES.json',changes);write(U/'ANNOTATED_COMPARISON.json',annotations);write(U/'CROSSING_COMPARISON.json',cross);write(U/'EVENTS.json',events);write(U/'OUTPUT_PINS.json',pins)
        summ={'status':'REPLAY_COMPLETE_PENDING_VERIFICATION_VISUAL','counts':dict(counts),'changed_clips':len({x['clip_id'] for x in changes}),'normal441_preserved':sum(x['verdict']=='normal_identity' and x['old_side']==x['new_side'] for x in annotations),'crossing315_preserved':sum(x['preserved'] for x in cross),'audited_normal_changes':[x['id'] for x in annotations if x['verdict']=='normal_identity' and x['old_side']!=x['new_side']],
            'error_sets':{ss:dict(Counter('correct' if x['new_side']==x['actual_side'] else 'withheld' if x['new_side'] is None else 'wrong' for x in annotations if x['set']==ss and x['verdict']=='error_identity')) for ss in ('original456','additional5')},'seconds':time.monotonic()-start,'training':0,'classifier_inference':0,'production_changed':False,'user_mentor_dev_heldout_reads':0}
        write(U/'SUMMARY.json',summ);(U/'RUNNING.json').unlink();print(json.dumps(summ,ensure_ascii=False),flush=True)
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'traceback':traceback.format_exc()})
        raise
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else main()
