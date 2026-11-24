import json,hashlib,shutil,os,time,traceback,sys
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/'word100_single_arm_recheck_20261018';G=O/'word100_visible_arm_duplicate_20261018';V=O/'word100_assignment_provenance_20261018';A=O/'word100_label_only_audit_20261018'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def scale_of(z,f):
    sh=z['pose_xy_px'][f,[11,12]];pu=z['pose_usable'][f,[11,12]];w=z['width'][f];h=z['height'][f]
    good=pu.all() and np.isfinite(sh).all() and ((sh[:,0]>=0)&(sh[:,0]<w)&(sh[:,1]>=0)&(sh[:,1]<h)).all()
    return float(np.linalg.norm(sh[0]-sh[1])) if good else 0.
def prepare():
    assert not (U/'CONTEXT.json').exists() and not (P/'RUNNING.json').exists()
    assert read(P/'FINAL_AUDIT.json')['status']=='PASS' and read(U/'CONTRACTS.json')['status']=='PASS'
    c=read(P/'CONTEXT.json');pins=dict(c['pins']);pins.update(read(P/'OUTPUT_PINS.json'))
    for p in [P/'FINAL_AUDIT.json',P/'CONTEXT.json',P/'ANNOTATED_COMPARISON.json',P/'CROSSING_COMPARISON.json',P/'RESULT.json',U/'CONTRACTS.json']+list(U.glob('*.py')):pins[str(p)]=sha(p)
    check(pins);(U/'results').mkdir();(U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
    protocol={
        'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'authorization':'User: continue low-wrist confidence and overlap/duplicate repair, preserve normal hands, no forced identities, public validation, preserve C, no user footage/training until reviewed.',
        'fixed_hypothesis':'Separate observed geometry from anatomical certainty. Add directional finger-region coverage to recent confirmed-history duplicate evidence. No parameter search.',
        'observation_output':'Unordered raw-hand-indexed observed XY, per-point strict frame/Hand-score mask, wrist-relative local XY only when current Hand wrist itself usable. Pose wrist quality is separate. Existing proposed side retained; evidence-backed identity side separate; uncertain identity never erases Hand geometry.',
        'association_output':'Possible previous-frame physical links using >=3 matching visible palm nodes, mean distance<=.4 shoulder width, gap<=100ms. Links neither commit anatomical side nor synthesize missing hand. No candidate label required.',
        'duplicate_guard':'Only previously provisional opposite candidate next to confirmed anchor, full observed palm, directed20nonwrist point coverage>=.8 within inherited .2palmspan hull margin, reachable opposite confirmed history<=.4shoulder, NO own confirmed history, anchor current-Pose confirmed or independently rooted<=100ms. Withhold conflicting anatomical assignment but retain raw coordinates. All gates inherited; no word rules.',
        'data':'899 TRAIN public clips, two saved raw streams, no extraction/classifier/user/mentor/DEV/heldout. Prior frozen geometry reused. Known audit events inform development, not independent test.',
        'validation':'All226362frames305602candidates;441normal315crossing protection; all new side changes visually reviewed;10 remaining errors +2 uncertain audited; independent scalar masks/local coordinates/links/identity/duplicate decisions; actual wrapper/order/prefix.',
        'acceptance':'No known normal/crossing regression; all observations preserved; uncertain changes not treated as successes. If regression, report not-adopted without retuning.',
        'prohibited':['training','classifier inference','user/mentor data','DEV/heldout','production C/TCN/DTW/web','camera','download','external transfer','Git','coordinate repair','forcing left/right by screen half','lowering Pose visibility threshold'],
        'roles':'Same Codex implementation/source review; separate numeric verifier, not independent human labels.'}
    write(U/'CONTEXT.json',{'rows':c['rows'],'pins':pins,'protocol':protocol});print('FROZEN',len(pins),'pins',flush=True)
def main():
    from candidate import duplicate_guard,ObservationLedger
    c=read(U/'CONTEXT.json');check(c['pins']);assert not (U/'SUMMARY.json').exists();write(U/'RUNNING.json',{'pid':os.getpid(),'task':'public observation preservation replay'})
    truth={(x['clip_id'],x['view'],x['index'],x['raw_index']):x for x in read(P/'ANNOTATED_COMPARISON.json')};crosskeys={(x['clip_id'],x['view'],x['index']) for x in read(P/'CROSSING_COMPARISON.json')}
    counts=Counter();changes=[];annotations=[];cross=[];opins={};start=time.monotonic()
    try:
        for rn,r in enumerate(c['rows'],1):
            assert r['role']=='train_candidate' and 1<=r['signer_id']<=9
            for v in ('masked','sensor'):
                z=load(r['views'][v]['raw']);old=load(P/'results'/f"{r['clip_id']}_{v}.npz");g=load(G/'results'/f"{r['clip_id']}_{v}.npz");pr=load(V/'results'/f"{r['clip_id']}_{v}.npz");N=len(z['raw_count']);ledger=ObservationLedger()
                out={k:old[k].copy() for k in ('sides','frame_id','timestamp_us')};out['blocked']=np.zeros((N,2),bool);out['anchor']=np.full((N,2),-1,np.int8)
                shapes={'observed_xy':((N,2,21,2),float,np.nan),'point_mask':((N,2,21),bool,False),'local_xy':((N,2,21,2),float,np.nan),'local_mask':((N,2,21),bool,False),'identity_side':((N,2),np.int8,-1),'identity_state':((N,2),np.int8,0),'pose_wrist_usable':((N,2),bool,False),'possible_links':((N,2,2),bool,False),'link_cost':((N,2,2),float,np.nan),'common_palm_points':((N,2,2),np.int8,0),'suggested_side':((N,2),np.int8,-1),'effective_side':((N,2),np.int8,-1)}
                out.update({k:np.full(sh,fill,dtype) for k,(sh,dtype,fill) in shapes.items()})
                for f,n0 in enumerate(z['raw_count']):
                    n=int(n0);sides=[None if x<0 else int(x) for x in old['sides'][f,:n]];origin=[-1 if q<0 else int(z['timestamp_us'][int(q)]) for q in pr['origin_frame'][f,:n]]
                    eff,blocked,anchor=duplicate_guard(sides,pr['state'][f,:n],pr['current_pose'][f,:n],origin,int(z['timestamp_us'][f]),pr['confirmed_track_distance'][f,:n],g['coverage'][f,:n,:n],g['strict_hand_inside'][f,:n])
                    o=ledger.step(z['hand_xy_px'][f,:n],z['raw_score'][f,:n],z['pose_xy_px'][f],z['pose_usable'][f],int(z['timestamp_us'][f]),int(z['width'][f]),int(z['height'][f]),sides,eff,pr['state'][f,:n],old['action'][f,:n],blocked,scale_of(z,f))
                    out['sides'][f,:n]=[-1 if s is None else s for s in eff];out['blocked'][f,blocked]=True;out['anchor'][f,:n]=anchor
                    for k,arr in o.items():
                        if k=='pose_wrist_usable':out[k][f]=arr
                        else:out[k][f,:n]=arr
                    counts['frames']+=1;counts['raw_candidates']+=n;counts['retained_raw_candidates']+=n;counts['usable_hand_points']+=int(o['point_mask'].sum());counts['usable_local_points']+=int(o['local_mask'].sum());counts['possible_link_edges']+=int(o['possible_links'].sum())
                    for i in range(n):
                        counts['unchanged_assignment']+=eff[i]==sides[i];counts['evidence_backed_identity']+=int(o['identity_side'][i]>=0);counts['uncertain_identity_observation_retained']+=int(o['identity_side'][i]<0 and o['point_mask'][i].any())
                        counts['observed_hand_both_pose_wrists_unusable']+=int(not o['pose_wrist_usable'].any() and o['point_mask'][i].any())
                        if eff[i]!=sides[i]:
                            counts['new_duplicate_withheld']+=1;changes.append({'clip_id':r['clip_id'],'word':r['label'],'view':v,'index':f,'frame_id':int(z['frame_id'][f]),'raw_index':i,'old_side':sides[i],'new_side':eff[i],'anchor_raw':anchor[i],'coverage':float(g['coverage'][f,i,anchor[i]]),'confirmed_distance':[None if not np.isfinite(x) else float(x) for x in pr['confirmed_track_distance'][f,i]],'anchor_origin_us':origin[anchor[i]],'timestamp_us':int(z['timestamp_us'][f]),'retained_observed_points':int(o['point_mask'][i].sum())})
                        k=(r['clip_id'],v,f,i)
                        if k in truth:
                            t=truth[k];annotations.append({**{k:t[k] for k in ('id','set','verdict','actual_side')},'clip_id':r['clip_id'],'word':r['label'],'view':v,'index':f,'raw_index':i,'old_side':sides[i],'new_side':eff[i],'suggested_side':int(o['suggested_side'][i]),'identity_side':int(o['identity_side'][i]),'identity_state':int(o['identity_state'][i]),'usable_points':int(o['point_mask'][i].sum()),'local_points':int(o['local_mask'][i].sum()),'possible_previous_raw_links':np.flatnonzero(o['possible_links'][i]).tolist()})
                    if (r['clip_id'],v,f) in crosskeys:cross.append({'clip_id':r['clip_id'],'view':v,'index':f,'preserved':sides==eff})
                path=U/'results'/f"{r['clip_id']}_{v}.npz";assert not path.exists();np.savez_compressed(path,**out);opins[str(path)]=sha(path)
            if rn%200==0:print('REPLAY',rn,'/899',flush=True)
        assert len(annotations)==461 and len(cross)==315;check(c['pins']);write(U/'CHANGES.json',changes);write(U/'ANNOTATED_COMPARISON.json',annotations);write(U/'CROSSING_COMPARISON.json',cross);write(U/'OUTPUT_PINS.json',opins)
        s={'status':'REPLAY_COMPLETE_PENDING_VERIFY_VISUAL','counts':dict(counts),'changed_clips':len({x['clip_id'] for x in changes}),'normal441_preserved':sum(x['verdict']=='normal_identity' and x['old_side']==x['new_side'] for x in annotations),'crossing315_preserved':sum(x['preserved'] for x in cross),'error_sets':{ss:dict(Counter('correct' if x['new_side']==x['actual_side'] else 'withheld' if x['new_side'] is None else 'wrong' for x in annotations if x['set']==ss and x['verdict']=='error_identity')) for ss in ('original456','additional5')},'training':0,'classifier_inference':0,'user_mentor_dev_heldout_reads':0,'production_changed':False,'seconds':time.monotonic()-start}
        write(U/'SUMMARY.json',s);(U/'RUNNING.json').unlink();print(json.dumps(s,ensure_ascii=False),flush=True)
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'traceback':traceback.format_exc()})
        raise
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else main()
