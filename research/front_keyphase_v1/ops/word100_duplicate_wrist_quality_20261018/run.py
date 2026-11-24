import json,hashlib,shutil,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import numpy as np
from candidate import propose,PARAMS
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
F=O/'word100_forearm_contour_20261018';R=O/'word100_remaining_identity_audit_20261018';B=O/'word100_uncertain_hand_preservation_20261018'
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def freeze():
    assert not (U/'CONTEXT.json').exists() and read(U/'CONTRACTS.json')['status']=='PASS'
    rows=read(R/'CONTEXT.json')['rows'];assert len(rows)==899
    assert all(r['role']=='train_candidate' and 1<=r['signer_id']<=9 for r in rows)
    pins=read(F/'CONTEXT.json')['pins'];pins.update(read(F/'OUTPUT_PINS.json'))
    for p in [F/'RESULTS.json',F/'ANNOTATED.json',F/'CROSSING.json',F/'FINAL_AUDIT.json',F/'RESULT.json',F/'INDEPENDENT.json',R/'CONTEXT.json',U/'candidate.py',U/'run.py',U/'contracts.py',U/'CONTRACTS.json']:pins[str(p)]=sha(p)
    for r in rows:
        for view in ['masked','sensor']:
            for p in [Path(r['views'][view]['raw']),B/'results'/f"{r['clip_id']}_{view}.npz"]:
                assert str(p) in pins,('unprotected_input',p)
    for p,h in pins.items():assert sha(p)==h,p
    (U/'before').mkdir(exist_ok=True)
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
    write(U/'CONTEXT.json',{'at':now(),'rows':rows,'parameters':PARAMS,'pins':pins,
      'authorization':'Latest user: public-data common duplicate/wrist repair, preserve normal crossing and C; no user training/tuning, no learning until joint review.',
      'development':'Known airport publicTRAIN case inspected to formulate a generic ambiguity guard; coefficients fixed before replay. These are development results, not heldout/generalization accuracy.',
      'method':'>=16 observed points including all5palm; wrist-to-meanMCP / maxMCPspan <.25 versus alternative>=.5; overlapping convex hull intersection/smaller>=.75; wrist separation/maxMCPspan>=.5; immediate causal frame <=100ms has exactly1 sufficiently observed noncollapsed hand; both current MCP median distances<=.4shoulders from that hand. Only unconfirmed positive side can be withheld. All coordinates/masks/identity unchanged. A2Dcollapse alone is not proof of wrong detection. No side promotion.',
      'scope':'All899publicTRAIN clips in native-detection/masked and actual-cropped sensor views; 226362frame-view observations. Preserve previous F side proposals on its785known frames; elsewhere compare U. This does not extend F RGB contour method to allframes.',
      'acceptance':'Normal441/crossing315 and prior13Fchanges must remain. Source review every newly changed hand. Regression means reject candidate, not tune to outcome. Fullscan unlabelled unchanged frames are not guaranteed correct. No claims of wrist-coordinate recovery or recognition improvement.',
      'prohibited':['user/mentor/DEV/heldout reads','training/classifier inference','C/TCN/DTW/liveweb changes','source/frozen output writes','camera/download/external transfer/Git'],
      'reviewers':'Codex implementation and visual review; separate reference implementation, not separate agent/person.'})
    print('FROZEN',len(pins),flush=True)

def run():
    c=read(U/'CONTEXT.json');assert c['parameters']==PARAMS
    assert not (U/'RUNNING.json').exists() and not (U/'SUMMARY.json').exists()
    for p,h in c['pins'].items():assert sha(p)==h,p
    write(U/'RUNNING.json',{'at':now()});start=time.monotonic();(U/'results').mkdir(exist_ok=True)
    old=read(F/'RESULTS.json');lookup={(x['clip_id'],x['view'],x['index']):x for x in old}
    counts=Counter();changes=[];selected=[];pins={}
    for z,r in enumerate(c['rows']):
        for view in ['masked','sensor']:
            with np.load(r['views'][view]['raw'],allow_pickle=False) as n, np.load(B/'results'/f"{r['clip_id']}_{view}.npz",allow_pickle=False) as b:
                q=n['hand_xy_px'];mask=b['point_mask'];identity=b['identity_side'];sides=b['sides'].copy();base=sides.copy();cnt=n['raw_count'];ts=n['timestamp_us']
                T=len(cnt);ratios=np.full((T,2),np.nan);overlap=np.zeros(T);sep=np.zeros(T);prev=np.zeros((T,2),bool)
                for t in range(T):
                    h=int(cnt[t]);k=(r['clip_id'],view,t);f=lookup.get(k)
                    if f:base[t,:h]=f['sides'];sides[t,:h]=f['sides']
                    scale=float(np.linalg.norm(n['pose_xy_px'][t,11]-n['pose_xy_px'][t,12]))
                    ph=int(cnt[t-1]) if t else 0
                    o=propose(q[t,:h],mask[t,:h],base[t,:h],identity[t,:h],scale,q[t-1,:ph] if t else None,mask[t-1,:ph] if t else None,int(ts[t]-ts[t-1]) if t else None)
                    sides[t,:h]=o['sides'];overlap[t]=o['overlap_smaller'];sep[t]=o['wrist_separation'];prev[t,:h]=o['previous_support']
                    for i,g in enumerate(o['geometry']):
                        if 'ratio' in g:ratios[t,i]=g['ratio']
                        counts['collapsed_projections']+=g['usable'] and g['ratio']<PARAMS['collapse_ratio']
                    counts.update(o['reason']);counts['frames']+=1;counts['hand_candidates']+=h;counts['observed_points']+=int(mask[t,:h].sum())
                    changed=np.flatnonzero(sides[t,:h]!=base[t,:h]).tolist()
                    entry={'clip_id':r['clip_id'],'word':r['label'],'view':view,'index':t,'frame_id':int(n['frame_id'][t]),'before':base[t,:h].tolist(),'after':sides[t,:h].tolist(),'identity':identity[t,:h].tolist(),'out':o,'changed_raw':changed,'observed_points':mask[t,:h].sum(axis=1).tolist()}
                    if changed:changes.append(entry)
                    if f:selected.append(entry)
                    if changed or f:
                        rev=propose(q[t,:h][::-1],mask[t,:h][::-1],base[t,:h][::-1],identity[t,:h][::-1],scale,q[t-1,:ph] if t else None,mask[t-1,:ph] if t else None,int(ts[t]-ts[t-1]) if t else None)
                        assert rev['sides'][::-1]==o['sides'];counts['order_tests']+=1
                        hidden=q[t,:h].copy();hidden[~mask[t,:h]]=1e9
                        prevhidden=q[t-1,:ph].copy() if t else None
                        if t:prevhidden[~mask[t-1,:ph]]=-1e9
                        hid=propose(hidden,mask[t,:h],base[t,:h],identity[t,:h],scale,prevhidden,mask[t-1,:ph] if t else None,int(ts[t]-ts[t-1]) if t else None)
                        assert hid['sides']==o['sides'];counts['hidden_point_tests']+=1
                path=U/'results'/f"{r['clip_id']}_{view}.npz"
                np.savez_compressed(path,sides=sides,baseline_sides=base,ratio=ratios,overlap=overlap,wrist_separation=sep,previous_support=prev)
                pins[str(path)]=sha(path)
        if (z+1)%100==0:print('REPLAY',z+1,'/899','changed_frames',len(changes),flush=True)
    by={(x['clip_id'],x['view'],x['index']):x for x in selected}
    ann=read(F/'ANNOTATED.json')
    for a in ann:
        x=by[(a['clip_id'],a['view'],a['index'])];i=a['raw_index'];a['before']=x['before'][i];a['after']=x['after'][i];a['reason']=x['out']['reason'][i]
    cross=read(F/'CROSSING.json')
    for a in cross:
        x=by[(a['clip_id'],a['view'],a['index'])];a['before']=x['before'];a['after']=x['after']
    normal=[a for a in ann if a['verdict']=='normal_identity'];rem=[a for a in ann if a['remaining9']]
    summary={'status':'EXECUTED_PENDING_VERIFY_VISUAL','counts':dict(counts),'changed_frames':len(changes),'changed_hands':sum(len(x['changed_raw']) for x in changes),'changed_clips':len({x['clip_id'] for x in changes}),'normal441_preserved':sum(a['after']==a['before']==a['actual_side'] for a in normal),'crossing315_preserved':sum(a['before']==a['after'] for a in cross),'prior13changes_preserved':sum(by[(x['clip_id'],x['view'],x['index'])]['after'][i]==s for x in old for i,s in enumerate(x['sides']) if s!=x['old_sides'][i]),'remaining9_correct':sum(a['after']==a['actual_side'] for a in rem),'remaining9_withheld':sum(a['after']<0 for a in rem),'remaining9_wrong_hint':sum(a['after']>=0 and a['after']!=a['actual_side'] for a in rem),'elapsed_seconds':time.monotonic()-start,'training':0,'user_reads':0,'production_changed':False}
    for name,value in [('RESULTS',selected),('CHANGES',changes),('ANNOTATED',ann),('CROSSING',cross),('SUMMARY',summary),('OUTPUT_PINS',pins)]:write(U/(name+'.json'),value)
    for p,h in c['pins'].items():assert sha(p)==h,p
    (U/'RUNNING.json').unlink();print(json.dumps(summary),flush=True)
if __name__=='__main__':
    try:freeze() if sys.argv[1]=='freeze' else run()
    except BaseException:
        if not (U/'FIRST_ERROR.json').exists():write(U/'FIRST_ERROR.json',{'at':now(),'error':traceback.format_exc()})
        raise
