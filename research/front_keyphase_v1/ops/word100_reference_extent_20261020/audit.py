from pathlib import Path
import sys,json,hashlib,time,traceback,copy
from datetime import datetime,timezone,timedelta
from collections import Counter,defaultdict
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent;B=OPS.parent;ROOT=B.parents[1]
O=OPS/'word100_multiref_observation_compare_20261019';P=OPS/'word100_metric_evidence_20261020';G=OPS/'word100_evidence_gate_20261020';M=OPS/'word100_missing_phase_validation_20261020'
sys.dont_write_bytecode=True;sys.path.insert(0,str(B/'src'))
from wis_lab.core.observation import Observation
from wis_lab.service import word100_observation_coverage as C
import candidate as E
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(name,x):
    p=U/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def serial(d):
    if isinstance(d,np.ndarray):return d.tolist()
    if isinstance(d,dict):return {k:serial(v) for k,v in d.items()}
    if isinstance(d,list):return [serial(v) for v in d]
    return d
def freeze():
    assert not (U/'FROZEN.json').exists()
    prev=read(G/'FROZEN.json');protected=prev['protected']
    for p,h in protected.items():assert sha(p)==h
    (U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']:(U/'before'/p.name).write_bytes(p.read_bytes())
    write('FROZEN.json',dict(at=now(),authorization='User: public multi-person reference extent robust to hand disappearance, normal lowering/singlehand/repeat verification; maintain C/T3-F; no private training or tuning; report before further training.',
        scope='Isolated offline candidate. No live web/model/scoring change, classifier inference, new detector extraction, training, private or heldout14-18 data, downloads or Git.',
        input='TRAIN899 references/DEV300 queries,100words,legacy+sensor paths. Query annotation and full-view source states exclusively evaluation, not candidate inputs. TRAIN annotation and geometry may label reference intervals.',
        algorithm=['Full recording96nearest real-time samples,body-normalized elbows and pose wrists; no hand21 coordinates or hand-based active_span for alignment.',
                   'Closed endpoint DTW band.2,mean normalized point distance; missing point cost1.5; path-support.35 and8distinct supported frames. Arm support>=half available selected points. Roleknownsinglehand uses only that arm; unreviewed words both arms/perhand analysis.',
                   'At least5distinct TRAIN people vote at each time. Mixed core/outside reference matches are unknown. Reference core from TRAIN annotations. Rest needs existing full-source stable-lowered state3; expected observation needs reference core,not rest,inside_source state2,and original metric eligibility. No absent=>rest inference.',
                   'Map reference labels to full query recording; no trimming based on query hand disappearance. Missing expected evidence is unknown action, never incorrect action. Low difference cannot change this. No new distance calculation or global observation threshold.'],
        controls=['All600query-view baselines: compare projected core to DEVannotation for localization-only diagnostics,not groundtruth essential sign phases.',
                  'All previous whole/center10/peak10/peak20 DEV hand masking windows; fixed pose alignment; hidden geometry perturbation and everypublicclip hand21 disappearance must leave pose input identical.',
                  'Normal-lowering offline DEV witness overlap with expected/rest/unknown; quantify potential inappropriate expectation,not certified false-sign-error rate.',
                  'Known3singlehand words: hide optional hand,required result unchanged. Speed.75/1.5 labels invariant,actual camera speed not tested.',
                  'Fullquery sequence repeated twice against original single-cycle references; no known cycle boundary given to algorithm. Measure projection to2annotatedcycles as stress,not a claim allwords permit double repetition.',
                  'Query pose wrists additionally masked in peak20 windows,elbows retained. Recompute alignment to expose arm-dependent failure; never count unknown as preserved semantic action.'],
        success='Report extent invariance for hand-only loss,extra revealed gaps,localization precision/recall and role/rest/repeat regressions. Preserve unknown. Do not promote if repeated cycles collapse or lowered hands expected incorrectly; no retuning against DEV.',
        protected=protected,previous={str(p):sha(p) for p in [G/'FINAL_AUDIT.json',G/'STRESS.json',G/'DECISION.json',M/'ROWS.json',O/'FROZEN.json']},code={str(p):sha(p) for p in [Path(__file__),U/'candidate.py']}))
    state=read(OPS/'STATE.json');state.update(updated_at=now(),current_stage='WORD100_REFERENCE_EXTENT_VALIDATING',activity_state='RUNNING',next_single_action='공개다중참고 구간 후보 구현·검증; C/T3-F/웹유지,추가학습없음.');state[U.name]=dict(status='FROZEN',unit=str(U));(OPS/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8');print('FROZEN',flush=True)

def selftest():
    n=40;ts=np.arange(n)*100000;v=np.ones((n,4),bool);x=np.tile(np.sin(np.linspace(0,np.pi,n))[:,None,None],(1,4,2))
    q=dict(xy=x,valid=v,timestamp_us=ts,full_timestamp_us=ts,picked_rows=np.arange(n))
    core=np.zeros(n,bool);core[10:30]=True;rest=np.zeros(n,bool)
    ref=dict(clip_id='public_synthetic',signer_id=1,pose=q,core=core,rest={s:rest.copy() for s in E.SIDES},expected={s:{k:core.copy() for k in E.AXES} for s in E.SIDES})
    refs=[dict(ref,signer_id=i) for i in range(1,10)];a=E.project(q,refs,['L','R']);assert np.array_equal(a['core'],core)
    absent={s:{k:np.zeros(n,bool) for k in E.AXES} for s in E.SIDES};d=E.describe(ts,absent,a,['L','R']);assert all(z['status']=='UNOBSERVED_REFERENCE_ACTION' for z in d)
    b=E.project(q,refs,['R']);assert not b['expected']['L']['location'].any()
    insufficient=E.project(q,refs[:4],['L','R']);assert insufficient['unknown'].all()
    blank=dict(q,valid=np.zeros_like(v));noarm=E.project(blank,refs,['L','R']);assert noarm['unknown'].all()
    for factor in (.75,1.5):
        fast=dict(q,timestamp_us=ts*factor,full_timestamp_us=ts*factor);p=E.project(fast,refs,['L','R']);assert np.array_equal(p['core'],a['core'])
    mixed=dict(ref,core=~core);rr=refs[:4]+[dict(mixed,signer_id=i) for i in range(5,10)];mm=E.project(q,rr,['L','R']);assert np.array_equal(mm['core'],~core)
    assert all(z['wrong_action'] is None for z in d)
    write('SELFTEST.json',dict(status='PASS',checks=9));print('SELFTEST PASS',flush=True)

def prepare():
    assert not (U/'PREPARED.json').exists();f=read(U/'FROZEN.json')
    for p,h in f['code'].items():assert sha(p)==h
    old=read(O/'FROZEN.json');pins={};c=Counter();records=[];beg=time.time()
    for i,e in enumerate(old['inputs']):
        assert e['signer_id'] in list(range(1,10))+[11,12,13]
        for view in ('legacy','sensor'):
            path=e['inputs'][view];assert sha(path)==e['input_hashes'][view];pins[path]=sha(path);obs=Observation.load(path)
            pose=E.pose_sample(obs);flags=C.metric_flags(obs)
            hidden=copy.deepcopy(obs);hidden.xy_px[:,33:]=np.nan;hidden.usable[:,33:]=False
            hp=E.pose_sample(hidden)
            assert all(np.array_equal(v,hp[k],equal_nan=True) for k,v in pose.items());c['hand_loss_pose_invariant']+=1
            arrays={**{'pose_'+k:v for k,v in pose.items()},**{s+'_'+k:v for s in E.SIDES for k,v in flags[s].items()}}
            # Reference geometry states are used ONLY for TRAIN. DEV witness loaded later for audit.
            if e['signer_id']<=9:
                pp=P/'features'/view/(e['clip_id']+'.npz');pm=read(P/'inputs'/view/(e['clip_id']+'.json'));assert sha(pp)==pm['sha256'];pins[str(pp)]=pm['sha256']
                with np.load(pp,allow_pickle=False) as z:
                    for s in E.SIDES:arrays[s+'_source_state']=z['full_'+s+'_offline_full_view_state']
                core=np.zeros(obs.n_frames,bool)
                for a,b in e['intervals']:core|=(obs.timestamp_us>=a*1e6)&(obs.timestamp_us<=b*1e6)
                arrays['reference_core']=core
            target=U/'prepared'/view/(e['clip_id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();np.savez_compressed(target,**arrays)
            records.append(dict(clip_id=e['clip_id'],word=e['word_code'],label=e['canonical_label'],signer_id=e['signer_id'],role=e['role'],view=view,path=str(target),sha256=sha(target)))
        if (i+1)%200==0:print('prepared',i+1,flush=True)
    write('INPUT_PINS.json',pins);write('INDEX.json',records);write('PREPARED.json',dict(status='PASS',records=len(records),counts=dict(c),seconds=time.time()-beg));print('PREPARED',len(records),flush=True)

def load(rec):
    assert sha(rec['path'])==rec['sha256']
    with np.load(rec['path'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
    pose={k[5:]:v for k,v in d.items() if k.startswith('pose_')};flags={s:{k:d[s+'_'+k] for k in E.AXES} for s in E.SIDES}
    return pose,flags,d

def make_ref(rec):
    pose,flags,d=load(rec);ids=pose['picked_rows'];core=d['reference_core'][ids]
    rest={s:d[s+'_source_state'][ids]==3 for s in E.SIDES}
    expected={s:{k:core&~rest[s]&(d[s+'_source_state'][ids]==2)&flags[s][k][ids] for k in E.AXES} for s in E.SIDES}
    return dict(clip_id=rec['clip_id'],signer_id=rec['signer_id'],pose=pose,core=core,rest=rest,expected=expected)

def localized(ts,extent,truth):
    w=E.weights(ts);a=extent['core'];inter=float(w@(a&truth));pred=float(w@a);total=float(w@truth)
    return dict(core_s=pred,annotation_s=total,intersection_s=inter,precision=inter/pred if pred else None,recall=inter/total if total else None,
        outside_s=float(w@extent['outside']),unknown_s=float(w@extent['unknown']))

def run():
    assert not (U/'SUMMARY.json').exists();f=read(U/'FROZEN.json')
    for p,h in f['code'].items():assert sha(p)==h
    idx=read(U/'INDEX.json');old=read(O/'FROZEN.json');rawidx={e['clip_id']:e for e in old['inputs']};windows=defaultdict(list)
    for row in read(M/'ROWS.json'):windows[row['clip_id']].append(row)
    words=defaultdict(list)
    for r in idx:words[r['word'],r['view']].append(r)
    baseline=[];stress=[];controls=[];witness=[];armcontrols=[];pins={};c=Counter();beg=time.time()
    for wi,((code,view),rr) in enumerate(sorted(words.items())):
        refs=[make_ref(r) for r in rr if r['signer_id']<=9];required=old['roles'].get(code,{}).get('required_sides',['L','R']);known=code in old['roles']
        assert len(refs) in (8,9)
        for rec in rr:
            if rec['signer_id']<=9:continue
            cid=rec['clip_id'];pose,flags,data=load(rec);ts=pose['full_timestamp_us'];proj=E.project(pose,refs,required)
            # DEV annotations arrive only after inference for evaluation diagnostics.
            truth=np.zeros(len(ts),bool)
            for a,b in rawidx[cid]['intervals']:truth|=(ts>=a*1e6)&(ts<=b*1e6)
            metrics=E.describe(ts,flags,proj,required,known);loc=localized(ts,proj,truth)
            baseline.append(dict(clip_id=cid,word=code,label=rec['label'],view=view,role_known=known,required_sides=required,metrics=metrics,localization=loc,
                available_references=sum(p['status']=='AVAILABLE' for p in proj['pairs']),mean_arm_support=float(np.mean([p['support'] for p in proj['pairs']]))))
            write('projections/'+view+'/'+cid+'.json',serial(dict(clip_id=cid,view=view,timestamp_us=ts,**proj)))
            # Transform only observation timestamps; feature/pose samples and mapping remain fixed.
            for factor in (.75,1.5):
                p=dict(pose,timestamp_us=(pose['timestamp_us']-ts[0])*factor,full_timestamp_us=(ts-ts[0])*factor)
                lifts=E.nearest(p['timestamp_us'],p['full_timestamp_us']);original=E.nearest(pose['timestamp_us'],ts)
                assert np.array_equal(lifts,original);c['speed_mapping_invariant']+=1
            if len(required)==1:
                other=next(s for s in E.SIDES if s not in required);ff=copy.deepcopy(flags)
                for k in E.AXES:ff[other][k][:]=False
                assert E.describe(ts,ff,proj,required,known)==metrics;c['optional_loss_invariant']+=1
            # Repeated whole query against single-cycle references; no oracle split supplied.
            dt=int(np.median(np.diff(ts)));rt=np.r_[ts-ts[0],ts-ts[0]+ts[-1]-ts[0]+dt]
            sampled_t=np.r_[pose['timestamp_us']-ts[0],pose['timestamp_us']-ts[0]+ts[-1]-ts[0]+dt]
            picks=np.unique(E.nearest(sampled_t,np.linspace(rt[0],rt[-1],96)))
            repeated=dict(timestamp_us=sampled_t[picks],full_timestamp_us=rt,xy=np.concatenate([pose['xy'],pose['xy']])[picks],valid=np.concatenate([pose['valid'],pose['valid']])[picks],picked_rows=picks)
            rp=E.project(repeated,refs,required);rf={s:{k:np.tile(v,2) for k,v in flags[s].items()} for s in E.SIDES}
            rmetrics=E.describe(rt,rf,rp,required,known);rloc=localized(rt,rp,np.tile(truth,2))
            controls.append(dict(clip_id=cid,view=view,kind='repeat_full_query_no_oracle_cycle',baseline_localization=loc,repeat_localization=rloc,metrics=rmetrics,wrong_action=None))
            if view=='sensor':
                pp=P/'features/sensor'/(cid+'.npz');pm=read(P/'inputs/sensor'/(cid+'.json'));assert sha(pp)==pm['sha256'];pins[str(pp)]=pm['sha256']
                with np.load(pp,allow_pickle=False) as z:
                    w=E.weights(ts)
                    for side in E.SIDES:
                        rest=(z['full_'+side+'_offline_full_view_state']==3);inside=z['full_'+side+'_inside_source_but_view_missing']
                        for k in E.AXES:
                            expected=proj['expected'][side][k];missing=~flags[side][k]
                            witness.append(dict(clip_id=cid,side=side,key=k,required=side in required,rest_missing_s=float(w@(rest&missing)),
                                rest_expected_s=float(w@(rest&missing&expected)),rest_reference_hint_s=float(w@(rest&proj['rest'][side])),
                                inside_missing_s=float(w@inside),inside_missing_expected_s=float(w@(inside&expected))))
                for oldrow in windows[cid]:
                    side=oldrow['side'];mode=oldrow['condition'];win=oldrow['window'];mask=np.ones(len(ts),bool) if mode=='whole' else ((ts/1e6>=win['start_s'])&(ts/1e6<=win['end_s']))
                    ff=copy.deepcopy(flags)
                    for k in E.AXES:ff[side][k]&=~mask
                    mm=E.describe(ts,ff,proj,required,known);w=E.weights(ts)
                    for k in E.AXES:
                        expected=proj['expected'][side][k];removed=mask&flags[side][k];before=next(x for x in metrics if x['side']==side and x['key']==k);after=next(x for x in mm if x['side']==side and x['key']==k)
                        assert after['expected_s']==before['expected_s'] and after['missing_expected_s']+1e-8>=before['missing_expected_s']
                        stress.append(dict(clip_id=cid,word=code,label=rec['label'],side=side,key=k,condition=mode,removed_frames=int(removed.sum()),
                            removed_s=float(w@removed),removed_in_reference_core_s=float(w@(removed&proj['core'])),removed_expected_s=float(w@(removed&expected)),
                            before=before,after=after,extent_unchanged=True))
                    if mode=='peak20':
                        # Additional pose-wrist disappearance, not just hand21; actual anchor limitation.
                        p=copy.deepcopy(pose);spmask=(p['timestamp_us']/1e6>=win['start_s'])&(p['timestamp_us']/1e6<=win['end_s']);j=1 if side=='L' else 3
                        p['valid'][spmask,j]=False;p['xy'][spmask,j]=np.nan;ap=E.project(p,refs,required)
                        am=E.describe(ts,ff,ap,required,known)
                        armcontrols.append(dict(clip_id=cid,side=side,condition='peak20_hand_plus_pose_wrist_loss',core_changed_frames=int((proj['core']!=ap['core']).sum()),
                            localization=localized(ts,ap,truth),baseline_localization=loc,unknown_s=float(w@ap['unknown']),metrics=am))
        if (wi+1)%20==0:write('STATUS.json',dict(status='RUNNING',word_views=wi+1,at=now()));print('wordviews',wi+1,flush=True)
    def dist(vals):
        a=[v for v in vals if v is not None]
        return dict(n=len(a),median=float(np.median(a)) if a else None,q10=float(np.quantile(a,.1)) if a else None,q90=float(np.quantile(a,.9)) if a else None)
    summary=dict(status='CALCULATED',at=now(),seconds=time.time()-beg,query_views=len(baseline),stress_metric_hands=len(stress),counts=dict(c),
        localization={v:{k:dist([x['localization'][k] for x in baseline if x['view']==v]) for k in ('precision','recall','unknown_s')} for v in ('legacy','sensor')},
        baseline_status=dict(Counter(m['status'] for r in baseline for m in r['metrics'])),
        stress={mode:dict(total=sum(r['condition']==mode for r in stress),effective=sum(r['condition']==mode and r['removed_frames']>0 for r in stress),
            removal_in_expected=sum(r['condition']==mode and r['removed_expected_s']>1e-8 for r in stress),removed_in_core=sum(r['condition']==mode and r['removed_in_reference_core_s']>1e-8 for r in stress)) for mode in ('whole','center10','peak10','peak20')},
        repeat={v:{k:dist([x['repeat_localization'][k] for x in controls if x['view']==v]) for k in ('precision','recall','unknown_s')} for v in ('legacy','sensor')},
        lowering={k:sum(x[k] for x in witness) for k in ('rest_missing_s','rest_expected_s','inside_missing_s','inside_missing_expected_s')},
        arm_loss=dict(cases=len(armcontrols),core_changes=sum(x['core_changed_frames']>0 for x in armcontrols),recall=dist([x['localization']['recall'] for x in armcontrols])),
        production_changes=0,training=0,private_data_read=0,wrong_action_verdicts=0,
        limits='Reference core projections are temporal proxies,not essential sign semantics. Label lift is96point nearest-time approximation. Hand-only invariance assumes pose remains; repetition and pose loss may degrade localization. Unknown roles do not become whole-sign judgments.')
    for p,h in f['protected'].items():assert sha(p)==h
    for p,h in f['previous'].items():assert sha(p)==h
    write('BASELINE.json',baseline);write('STRESS.json',stress);write('CONTROLS.json',controls);write('LOWERING.json',witness);write('ARM_LOSS.json',armcontrols);write('EVALUATION_PINS.json',pins);write('SUMMARY.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        p=U/('FIRST_ERROR_'+sys.argv[1]+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
