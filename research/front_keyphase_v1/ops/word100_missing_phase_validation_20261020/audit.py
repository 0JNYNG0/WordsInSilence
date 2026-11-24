"""Frozen public observation-loss stress test. No model inference or semantic error labels."""
from pathlib import Path
import json,hashlib,sys,time,traceback
from datetime import datetime,timezone,timedelta
from collections import Counter
import numpy as np

U=Path(__file__).resolve().parent;OPS=U.parent;B=OPS.parent;ROOT=B.parents[1]
PREV=OPS/'word100_metric_evidence_20261020';OLD=OPS/'word100_multiref_observation_compare_20261019'
sys.dont_write_bytecode=True;sys.path.insert(0,str(PREV))
import candidate as C
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def core_mask(ts,intervals):
    out=np.zeros(len(ts),bool)
    for a,b in intervals:out|=(ts>=a*1e6)&(ts<=b*1e6)
    return out

def windows(q,side,intervals):
    """Fixed mid-window and max observed wrist-travel proxy. No lexical phase oracle."""
    ts=q['timestamp_us'].astype(float)/1e6;pos=q['position_'+side];valid=q['position_valid_'+side]
    a,b=max(intervals,key=lambda x:x[1]-x[0]);ids=np.flatnonzero((ts>a)&(ts<b))
    outputs={'whole':(np.ones(len(ts),bool),{'duration_s':float(ts[-1]-ts[0])})}
    travel=np.zeros(len(ts));adj=valid[1:]&valid[:-1]
    travel[1:]=np.where(adj,np.linalg.norm(np.nan_to_num(pos[1:]-pos[:-1]),axis=1),0)
    for mode,fraction in [('center10',.1),('peak10',.1),('peak20',.2)]:
        width=(b-a)*fraction
        center=(a+b)/2
        if mode.startswith('peak') and len(ids):
            centers=ts[ids[(ts[ids]>=a+width/2)&(ts[ids]<=b-width/2)]]
            if len(centers):
                masses=[float(travel[(ts>=c-width/2)&(ts<=c+width/2)].sum()) for c in centers]
                center=float(centers[int(np.argmax(masses))])
        mask=(ts>=center-width/2)&(ts<=center+width/2)&(ts>a)&(ts<b)
        outputs[mode]=(mask,{'start_s':float(center-width/2),'end_s':float(center+width/2),'duration_s':float(width),'travel_proxy':float(travel[mask].sum()),'is_semantic_phase':False})
    return outputs

def compare(q,r,side,path,mask,aligned):
    qi,ri=np.array(path,dtype=int).T
    qa=q['angles_'+side].copy();qa[mask]=np.nan
    aa=C.compared_angles(qa,r['angles_'+side],path,aligned)
    qv=q['position_valid_'+side]&~mask;rv=r['position_valid_'+side]
    good=qv[qi]&rv[ri];eligible=aligned and good.mean()>=.5 and min(len(np.unique(qi[good])),len(np.unique(ri[good])))>=8
    location=float(np.linalg.norm(q['position_'+side][qi[good]]-r['position_'+side][ri[good]],axis=1).mean()*100) if eligible else None
    # Fixed original origin: controlled internal visibility loss must not silently redefine movement.
    movement=None
    if eligible:
        oq=q['position_'+side][np.flatnonzero(q['position_valid_'+side])[0]]
        rr=r['position_'+side][np.flatnonzero(rv)[0]]
        delta=(q['position_'+side][qi[good]]-oq)-(r['position_'+side][ri[good]]-rr)
        movement=float(np.linalg.norm(delta,axis=1).mean()*100)
    return {'angles':aa,'location':location,'movement':movement}

def aggregate(pairs):
    a=C.aggregate_angles([p['angles'] for p in pairs]);out={'handshape':a['whole_hand'],'angles_ready':a['angles_ready'],'angles':a['angles']}
    for k in ('location','movement'):
        values=[p[k] for p in pairs if p[k] is not None]
        out[k]=float(np.median(values)) if len(values)>=5 else None
        out[k+'_refs']=len(values)
    return out

def freeze():
    assert not (U/'FROZEN.json').exists()
    prev=read(PREV/'FROZEN.json');audit=read(PREV/'FINAL_AUDIT.json')
    for p,h in prev['protected'].items():assert sha(p)==h
    for name,h in audit['artifact_sha256'].items():assert sha(PREV/name)==h
    (U/'before').mkdir(exist_ok=True)
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']:(U/'before'/p.name).write_bytes(p.read_bytes())
    write(U/'FROZEN.json',{'at':now(),'authorization':'2026-10-20 user: public entire-hand/short-motion missing-evidence validation and plan only; retain C/T3-F; no private data, training, recognition assistance or deployment',
      'scope':'100 words. TRAIN899 reference observations reused; DEV300 query observations only; persons14-18 excluded. Sensor view only. 591 preexisting required-or-conservative hand units. No new hand role decisions.',
      'conditions':['whole','center10','peak10','peak20'],'window':'Longest public annotation interval; internal10% middle, internal10% and20% maximum observed wrist travel; unchanged timestamps and remaining coordinates. Peak is a motion proxy, not a certified distinguishing phase. Earliest center breaks ties.',
      'comparison':'Saved original DTW paths and alignment eligibility reused to isolate descriptive measurement. No new alignment, preprocessing, classification or rescoring of original baseline. Existing per-angle gate .5 support,8distinctframes,5references; all15 for wholehand. Original trajectory origin retained.',
      'success':'Whole masked hand must return unavailable on all3 metrics, never zero. Quantify short masks leaving a finite or reduced mean. No low-score cutoff tuned. New absent evidence must be explicit. Existing descriptive values never become a pass verdict.',
      'real_gap_audit':'Separate source-inside missing observations vs source-lowered and unknown. Full source is an offline estimated witness, not ground truth. Report scalar availability with witnessed gaps, not accepted wrong signs.',
      'positive_control':'Same public observation against itself, identity alignment. Short masked intervals can retain exact0 differences with adequate remaining support. These are synthetic sensor losses, not independent accuracy or actual incorrect signing.',
      'public_protocol_limits':'No labels identifying essential phases, no invented linguistic omissions; word-specific thresholds forbidden; cannot certify rejection of unseen incorrect signs.',
      'protected':prev['protected'],'previous_artifact_sha256':audit['artifact_sha256'],'old_frozen_sha256':sha(OLD/'FROZEN.json'),'candidate_sha256':sha(PREV/'candidate.py'),'runner_sha256':sha(__file__)})
    write(U/'STATUS.json',{'status':'FROZEN','at':now()})
    state=read(OPS/'STATE.json');state['current_stage']='WORD100_MISSING_PHASE_VALIDATION_RUNNING';state['next_single_action']='公開100단어 저장 관측으로 전체손·짧은구간 관측누락을 검증하고 결과/계획 보고. 학습/운영변경 없음.';state['word100_missing_phase_validation_20261020']={'status':'RUNNING','unit':str(U),'at':now()};write(OPS/'STATE.json',state)
    print('FROZEN',flush=True)

def selftest():
    n=40;p=np.stack([np.linspace(0,1,n),np.zeros(n)],1).astype(np.float32)
    q={'timestamp_us':np.arange(n)*100000,'angles_L':np.full((n,15),50.),'position_L':p,'position_valid_L':np.ones(n,bool)}
    path=[[i,i] for i in range(n)];mask=np.zeros(n,bool);mask[18:22]=True
    a=compare(q,q,'L',path,mask,True);z=aggregate([a]*5)
    assert all(z[k]==0 for k in ('handshape','location','movement'))
    whole=aggregate([compare(q,q,'L',path,np.ones(n,bool),True)]*5)
    assert all(whole[k] is None for k in ('handshape','location','movement'))
    hidden={k:v.copy() for k,v in q.items()};hidden['angles_L'][mask]=9999;hidden['position_L'][mask]=9999
    assert compare(hidden,q,'L',path,mask,True)==a
    noalign=aggregate([compare(q,q,'L',path,mask,False)]*5);assert noalign['handshape'] is None
    fewer=aggregate([a]*4);assert all(fewer[k] is None for k in ('handshape','location','movement'))
    w=windows(q,'L',[[.1,3.8]]);assert all(len(m)==n for m,_ in w.values())
    assert all(not m[0] and not m[-1] for k,(m,_) in w.items() if k!='whole')
    assert np.all(q['angles_L']==50)
    write(U/'SELFTEST.json',{'status':'PASS','checks':8,'checks_description':['short missing evidence can retain zero','wholehand withheld','hidden value perturbation invariant','alignment gate','reference count gate','mask dimensions','short windows internal','source unchanged']})

def run():
    f=read(U/'FROZEN.json');assert sha(__file__)==f['runner_sha256'];assert sha(PREV/'candidate.py')==f['candidate_sha256']
    assert not (U/'SUMMARY.json').exists();assert not (U/'RUNNING.json').exists()
    write(U/'RUNNING.json',{'at':now()});beg=time.time();rows=[];pair_count=0;input_hashes={};word_hashes={};selfrows=[];real=[]
    index={x['clip_id']:x for x in read(OLD/'FROZEN.json')['inputs']}
    for wi,wp in enumerate(sorted((OLD/'words/sensor').glob('*.json'))):
        old=read(wp);previous=read(PREV/'words/sensor'/wp.name);word_hashes[str(wp)]=sha(wp);word_hashes[str(PREV/'words/sensor'/wp.name)]=sha(PREV/'words/sensor'/wp.name)
        feats={};prevqueries={q['clip_id']:q for q in previous['queries']}
        for x in old['queries']:
            cid=x['clip_id'];p=PREV/'features/sensor'/(cid+'.npz');meta=read(PREV/'inputs/sensor'/(cid+'.json'));assert sha(p)==meta['sha256'];input_hashes[str(p)]=meta['sha256'];feats[cid]=load(p)
        for oq in old['queries']:
            if oq['role']!='development_only':continue
            cid=oq['clip_id'];q=feats[cid];info=index[cid];pairs=[p for p in old['pairs'] if p['query_clip']==cid]
            assert 11<=oq['signer_id']<=13 and all(1<=p['reference_signer']<=9 for p in pairs)
            for side in old['alignment_sides']:
                pa=prevqueries[cid]['sides'][side]
                baseline={'handshape':pa['whole_hand'],'location':next(s['value'] for m in oq['metrics'] if m['key']=='location' for s in m['sides'] if s['side']==side),'movement':pa['trajectory']['old']}
                nativecore=q['full_'+side+'_annotated_action'];localmiss=~q['full_'+side+'_any_hand_points'];inside=q['full_'+side+'_inside_source_but_view_missing']&nativecore
                lowered=(q['full_'+side+'_offline_full_view_state']==3)&localmiss&nativecore
                real.append({'clip_id':cid,'word':old['word_code'],'label':old['label'],'side':side,'baseline':baseline,'inside_missing_frames':int(inside.sum()),'lowered_missing_frames':int(lowered.sum()),'action_missing_frames':int((nativecore&localmiss).sum()),'action_frames':int(nativecore.sum())})
                for mode,(mask,win) in windows(q,side,info['intervals']).items():
                    out=[]
                    for pair in pairs:
                        out.append(compare(q,feats[pair['reference_clip']],side,pair['diagnostics']['path'],mask,pair['result']['status']=='OK'));pair_count+=1
                    a=aggregate(out)
                    observed_angles=np.isfinite(q['angles_'+side]);observed_pos=q['position_valid_'+side]
                    r={'clip_id':cid,'word':old['word_code'],'label':old['label'],'signer_id':oq['signer_id'],'side':side,'condition':mode,'window':win,'mask_indices':np.flatnonzero(mask).tolist(),'samples':len(mask),'removed_position_frames':int((observed_pos&mask).sum()),'removed_angle_cells':int((observed_angles&mask[:,None]).sum()),'baseline':baseline,'result':a}
                    rows.append(r)
                    if mode=='peak10':
                        path=[[i,i] for i in range(len(mask))];ctrl=compare(q,q,side,path,mask,True)
                        selfrows.append({'clip_id':cid,'side':side,'removed_position_frames':r['removed_position_frames'],'removed_angle_cells':r['removed_angle_cells'],'angle_zero_count':sum(v==0 for v in ctrl['angles']['values'] if v is not None),'location':ctrl['location'],'movement':ctrl['movement'],'not_cross_person_evaluation':True})
        if (wi+1)%20==0:write(U/'STATUS.json',{'status':'RUNNING','words':wi+1,'at':now()});print('words',wi+1,flush=True)
    summary={}
    for mode in f['conditions']:
        rr=[r for r in rows if r['condition']==mode];data={'hand_units':len(rr),'effective_angle_removal':sum(r['removed_angle_cells']>0 for r in rr),'effective_position_removal':sum(r['removed_position_frames']>0 for r in rr),'metrics':{}}
        for key in ('handshape','location','movement'):
            effective=lambda r:r['removed_angle_cells']>0 if key=='handshape' else r['removed_position_frames']>0
            paired=[r for r in rr if effective(r) and r['baseline'][key] is not None and r['result'][key] is not None]
            delta=np.array([r['result'][key]-r['baseline'][key] for r in paired])
            data['metrics'][key]={'baseline_available':sum(r['baseline'][key] is not None for r in rr),'masked_available':sum(r['result'][key] is not None for r in rr),'paired_effective':len(paired),'decreased':int((delta < -1e-4).sum()),'unchanged':int((np.abs(delta)<=1e-4).sum()),'increased':int((delta>1e-4).sum()),'delta_median':float(np.median(delta)) if len(delta) else None,'delta_min':float(delta.min()) if len(delta) else None,'delta_max':float(delta.max()) if len(delta) else None}
        summary[mode]=data
    assert len(rows)==591*4 and len(real)==591 and len(input_hashes)==1199
    assert all(r['result'][k] is None for r in rows if r['condition']=='whole' for k in ('handshape','location','movement'))
    write(U/'ROWS.json',rows);write(U/'ACTUAL_GAPS.json',real);write(U/'SELF_CONTROLS.json',selfrows)
    write(U/'INPUT_PINS.json',{'features':input_hashes,'words':word_hashes})
    write(U/'SUMMARY.json',{'scope':{'words':100,'public_clips':1199,'train_refs':899,'dev_queries':300,'hand_units':591,'conditions':4,'counterfactual_pair_hand_computations':pair_count},'conditions':summary,'actual_gap':{'hand_units_with_source_inside_missing':sum(r['inside_missing_frames']>0 for r in real),'source_inside_missing_handframes':sum(r['inside_missing_frames'] for r in real),'whole_angle_available_despite_witnessed_gap':sum(r['inside_missing_frames']>0 and r['baseline']['handshape'] is not None for r in real),'location_available_despite_witnessed_gap':sum(r['inside_missing_frames']>0 and r['baseline']['location'] is not None for r in real)},'self_control':{'hand_units':len(selfrows),'removed_angles_and_all15_zero':sum(r['removed_angle_cells']>0 and r['angle_zero_count']==15 for r in selfrows),'removed_position_and_zero':sum(r['removed_position_frames']>0 and r['location']==0 for r in selfrows)},'seconds':time.time()-beg,'no_new_training':True,'no_correctness_labels':True,'deployment':False})
    write(U/'STATUS.json',{'status':'CALCULATION_COMPLETE_PENDING_VERIFICATION','at':now()});(U/'RUNNING.json').unlink();print(json.dumps(read(U/'SUMMARY.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        p=U/('FIRST_ERROR_'+sys.argv[1]+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
