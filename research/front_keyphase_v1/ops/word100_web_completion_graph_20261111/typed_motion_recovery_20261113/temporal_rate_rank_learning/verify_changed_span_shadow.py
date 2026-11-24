"""Saved arithmetic only: no torch, network import, inference or rescoring."""
from pathlib import Path
import ast, collections, hashlib, json, time, traceback
import numpy as np
U=Path(__file__).resolve().parent;H=U.parent;G=H.parent;O=G.parent;E=U/'changed_span_shadow'
S=H/'stability_rank_blend/changed_span_shadow';J=H/'splice_rank_learning/changed_span_shadow'
MODES=['rate_head','old_local_full_plus_rate_half'];checks=collections.Counter()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8')as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def ck(n,b):
    assert bool(b),n
    checks[n]+=1
def rowkey(r):return r['seed'],r['clip_id'],r['kind']
def reqkey(cid,kind,card,s):return cid,kind,card,None if s is None else tuple(s)
def movkey(cid,kind,s,axis,side):return cid,kind,tuple(s),int(axis),side
def featkey(v):return v['task'],v.get('feature',v.get('axis','sequence'))
def state_signature(c):return c['status'],{featkey(v):v['state']for v in c['evidence']}
def independent_math():
    # These are separately implemented arithmetic routines from J's verifier,
    # not the evaluator's measurement/decision functions. AST loads three pure
    # functions only; the original verifier main is never invoked.
    p=H/'splice_rank_learning/verify_changed_span_shadow.py';tree=ast.parse(p.read_text('utf-8'))
    names={'shape_available','old_direction_geometry','direction_state'}
    fns=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    assert len(fns)==3
    ns={'np':np};exec(compile(ast.Module(body=fns,type_ignores=[]),str(p),'exec'),ns)
    return ns,p
def shape_state(p,ok,qualified):
    if not ok:return'observation_unknown'
    if not qualified:return'association_unknown'
    return'supported'if p>=.65 else'difference'if p<=.35 else'uncertain'
def status(items):
    st=[v['state']for v in items]
    return'difference_evidence'if'difference'in st or'opposite_direction'in st else'supported_features_only'if st and all(s=='supported'for s in st)else'insufficient_evidence'

def main():
    ck('never_overwrite_verification',not(E/'VERIFICATION.json').exists())
    cfg=read(E/'FROZEN.json');r=read(E/'RESULTS.json');old=read(S/'RESULTS.json');rank=read(U/'evaluation/COMPARISON.json');jc=read(J/'RESULTS.json')
    math,mathpath=independent_math();av_fn=math['shape_available'];geometry=math['old_direction_geometry'];dir_state=math['direction_state']
    for p,h in cfg['pins'].items():ck('frozen_pin',sha(p)==h)
    protections=read(G/'PROTECTIONS.json');ck('protected_count477',len(protections)==477)
    for p,h in protections.items():ck('protected477_exact',cfg['pins'][p]==h and sha(p)==h)
    ck('result_contract_hash',r['frozen_sha256']==sha(E/'FROZEN.json'))
    ck('no_inference_replay_or_adoption',r['no_old_result_reinference']and r['no_training']and r['no_pose_or_encoder']and r['no_TCN_or_DTW']and r['no_PD_or_new_hand']and r['no_web']and r['no_final8']and r['G2_pass']is None and r['G3_pass']is None and r['semantic_completion']is None)
    ck('old_summaries_exact',r['summary'][:len(old['summary'])]==old['summary'])
    ck('execution_plan_exact',r['execution']==cfg['plan'])
    ck('counts_exact',r['counts']==cfg['counts'])
    om={rowkey(x):x for x in old['rows']};rm={rowkey(x):x for x in r['rows']};km={rowkey(x):x for x in rank['rows']}
    ck('all246_rows',len(r['rows'])==len(rm)==len(om)==len(km)==246 and set(rm)==set(om)==set(km))
    ck('observer_selector_separate',r['observer_seed']==42 and r['selector_seeds']==[42,43,44])
    sources={(v['clip_id'],v['kind']):v for v in cfg['sources']};loaded={}
    def source(sid):
        if sid not in loaded:
            m=sources[sid];ck('source_pin',sha(m['source'])==m['sha256'])
            with np.load(m['source'],allow_pickle=False)as z:x=z['x'].copy();ts=z['ts'].copy()
            ck('source_finite_pts',x.ndim==2 and x.shape[1]==280 and np.isfinite(x).all()and len(x)==len(ts)and(np.diff(ts)>0).all())
            loaded[sid]=(x,ts,av_fn(x))
        return loaded[sid]
    saved={}
    for v in old['rows']:
        for cs in v['conditions'].values():
            for c in cs:
                k=reqkey(v['clip_id'],v['kind'],c['card'],c['span'])
                if k in saved:ck('saved_semantic_consistent',saved[k]==c)
                else:saved[k]=c
    # All old cached frames are immutable; only the explicit new120 are added.
    oldattrs={};attrs={};newattrs={}
    ck('reused_pointwise_index_exact',r['reused_pointwise_cache']==jc['pointwise_cache'])
    for isnew,records in[(False,jc['pointwise_cache']),(True,r['new_pointwise_cache'])]:
        for rec in records:
            sid=rec['clip_id'],rec['kind'];ck('pointwise_pin',sha(rec['path'])==rec['sha256'])
            ck('pointwise_source_match',rec['source_sha256']==sources[sid]['sha256']and rec['source']==sources[sid]['source'])
            with np.load(rec['path'],allow_pickle=False)as z:fi=z['frame_indices'].copy();vv=z['attribute_probabilities'].copy()
            ck('cache_probability_shape',vv.shape==(len(fi),7)and np.isfinite(vv).all()and((vv>=0)&(vv<=1)).all())
            ck('cache_frame_unique_sorted',np.array_equal(fi,np.unique(fi)))
            if isnew:ck('cache_selected_count',rec['selected_valid_frames']==len(fi))
            dest=newattrs if isnew else oldattrs;ck('cache_source_unique_per_layer',sid not in dest)
            dest[sid]={int(f):p for f,p in zip(fi,vv)}
            combined=attrs.setdefault(sid,{})
            ck('no_frame_reinference',not(set(map(int,fi))&set(combined)))
            combined.update(dest[sid])
    ck('new120_cached_frames',sum(len(v)for v in newattrs.values())==120==cfg['counts']['new_attribute_frames'])
    cards_cfg=read(O/'word100_action_feedback_demo_20261108/CARDS.json')['words'];family_cfg=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
    guides={(w,c['id']):c for w,cs in cards_cfg.items()for c in cs};families={(w,c['id']):c for w,cs in family_cfg.items()for c in cs}
    caps={(a['word'],a['card'],a['feature'],a['expected']):a['qualified']for a in read(O/'word100_guide_requirement_learning_20261111/GUIDE_CAPABILITY.json')['rows']}
    requests={reqkey(q['clip_id'],q['kind'],q['card'],q['span']):q for q in cfg['requests']}
    newreq={k:q for k,q in requests.items()if q['action']=='new_request'};needed=collections.defaultdict(set)
    for q in newreq.values():
        if q['span']is None:continue
        sid=q['clip_id'],q['kind'];x,ts,av=source(sid);a,b=q['span'];word=q['clip_id'].split('_')[2];guide=guides[word,q['card']]
        aa=a+(b-a)//2 if guide.get('shape_scope')=='late'else a
        for j in guide.get('features',{}):needed[sid].update(map(int,np.arange(aa,b+1)[av[aa:b+1,int(j)]]))
    ck('needed197_unique_frames',sum(len(v)for v in needed.values())==197)
    ck('reused77_frames',sum(len(v&set(oldattrs.get(sid,{})))for sid,v in needed.items())==77)
    for sid,wanted in needed.items():
        ck('new_frames_only_needed_missing',set(newattrs.get(sid,{}))==wanted-set(oldattrs.get(sid,{})))
        ck('all_needed_frames_available',wanted<=set(attrs.get(sid,{})))
    ck('no_unrequested_new_source',set(newattrs)<=set(needed))
    movements={}
    for m in r['new_movements']:
        sid=m['clip_id'],m['kind'];x,ts,av=source(sid);a,b=m['span'];mk=movkey(*sid,m['span'],m['axis'],m['moving'])
        ck('movement_unique',mk not in movements);movements[mk]=m
        g,ok,cov,observed=geometry(x,ts,a,b,m['axis'],m['moving'])
        ck('alternate_old_geometry58_exact',np.array_equal(g,np.asarray(m['geometry'],np.float32)))
        ck('geometry_mask_eligibility_exact',bool(ok)==m['eligible']and cov==m['info']['coverage']and observed==m['info']['observed'])
        if ok:ck('geometry_signed_delta_consistent',abs(float(g[40])-m['info']['signed_change'])<1e-6)
        ck('movement_source_pin',m['source_sha256']==sources[sid]['sha256'])
        p=np.asarray(m['probabilities']);ck('movement_probabilities_valid',p.shape==(3,)and np.isfinite(p).all()and(p>=0).all()and abs(p.sum()-1)<2e-6)
    ck('62_new_movements_no_reused_direction',len(movements)==62==cfg['counts']['new_direction_requests']and cfg['counts']['reused_direction_requests']==0)
    # Derive complete expected semantic cards from saved probabilities and masks.
    expected={}
    for k,q in newreq.items():
        if q['span']is None:expected[k]=dict(card=q['card'],span=None,status='association_unknown',evidence=[],completion=None);continue
        sid=q['clip_id'],q['kind'];x,ts,av=source(sid);a,b=q['span'];word=q['clip_id'].split('_')[2];guide=guides[word,q['card']];spec=families[word,q['card']]
        aa=a+(b-a)//2 if guide.get('shape_scope')=='late'else a;items=[]
        for j,want in guide.get('features',{}).items():
            j=int(j);mask=av[aa:b+1,j];ids=np.arange(aa,b+1)[mask];ok=bool(mask.mean()>=.8);qualified=caps.get((word,q['card'],j,want),False)
            pp=np.array([attrs[sid][int(f)][j]for f in ids],np.float32);p=float(pp.mean())if len(pp)else None
            if p is not None and not want:p=1-p
            state='observation_unknown'if p is None else shape_state(p,ok,qualified)
            items.append(dict(task='shape',feature=j,expected=want,span=[aa,b],probability=p,state=state,qualified=qualified,observed_count=int(mask.sum())))
        axis=spec['axis'];side=spec['hands'][0]
        if axis is not None:
            if spec['sign']is None:items.append(dict(task='direction',state='sequence_required',reason='Alternation cannot be judged from first/last displacement alone'))
            else:
                m=movements[movkey(*sid,[a,b],axis,side)];want=int(spec['sign']+1)
                items.append(dict(task='direction',axis=axis,desired=want,span=[a,b],probabilities=m['probabilities'],state=dir_state(m['probabilities'],want,m['eligible'])))
        expected[k]=dict(card=q['card'],span=[a,b],status=status(items),evidence=items,completion=None)
    ck('62_new_semantics',len(expected)==62)
    changes=[];feature_changes=[];metrics=[];new_seen=set();counts={}
    for mode in MODES:
        changed=0;unique=set();same=0
        for rk,row in rm.items():
            prior=om[rk];bc={c['card']:c for c in prior['conditions']['local_full']};chosen={c['card']:c for c in km[rk]['conditions'][mode]['cards']}
            for oldmode,cs in prior['conditions'].items():ck('all_old6conditions_exact',row['conditions'][oldmode]==cs)
            for c in row['conditions'][mode]:
                cid,kind=row['clip_id'],row['kind'];k=reqkey(cid,kind,c['card'],c['span']);before=bc[c['card']];rankcard=chosen[c['card']]
                sp=rankcard['chosen']['span']if rankcard['chosen']else None
                ck('selected_span_exact_completion_none',c['span']==sp and c['completion']is None)
                if sp==before['span']:same+=1;ck('unchanged_span_exact_copy',c==before)
                else:
                    changed+=1;unique.add(k);q=requests[k]
                    if q['action']=='reuse_saved_identical_request':ck('saved_request_exact_copy',c==saved[k])
                    else:new_seen.add(k);ck('new_request_all_arithmetic_exact',c==expected[k])
                if rankcard['family']not in[2,3]:ck('other_families_exact_copy',c==before)
                if kind=='all_hand_missing':
                    for it in c['evidence']:
                        if it['task']=='shape':ck('missing_shape_unknown',it['state']=='observation_unknown')
                if c['status']!=before['status']:
                    changes.append(dict(condition=mode,seed=row['seed'],clip_id=cid,kind=kind,card=c['card'],old_span=before['span'],new_span=sp,
                        before=before['status'],after=c['status']))
                bef={featkey(v):v for v in before['evidence']};aft={featkey(v):v for v in c['evidence']}
                ck('same_feature_tasks',set(bef)==set(aft))
                for fk,v in aft.items():
                    if v['state']!=bef[fk]['state']:
                        feature_changes.append(dict(condition=mode,seed=row['seed'],clip_id=cid,kind=kind,card=c['card'],
                            task=fk[0],feature_or_axis=fk[1],old_span=before['span'],new_span=sp,before=bef[fk]['state'],after=v['state']))
        counts[mode]=dict(cards=same+changed,changed_cards=changed,unique_changed_requests=len(unique),
            saved_identical_requests=sum(k in saved for k in unique),new_requests=sum(k not in saved for k in unique))
        ck('per_condition_counts',counts[mode]==cfg['counts']['per_condition'][mode])
    ck('all_new_requests_used',new_seen==set(expected))
    # Recount every summary, including all original conditions, without inference.
    for summary in r['summary']:
        cs=[c for row in r['rows']if row['seed']==summary['seed']and row['kind']==summary['kind']for c in row['conditions'][summary['condition']]]
        its=[i for c in cs for i in c['evidence']]
        calc=dict(cards=len(cs),statuses=dict(collections.Counter(c['status']for c in cs)),
            shape=dict(collections.Counter(i['state']for i in its if i['task']=='shape')),
            direction=dict(collections.Counter(i['state']for i in its if i['task']=='direction')))
        ck('summary_saved_arithmetic',all(summary[k]==v for k,v in calc.items()))
    ck('all264_summaries',len(r['summary'])==264)
    for mode in MODES:
        for seed in[42,43,44]:
            for kind in sorted({x['kind']for x in r['rows']}):
                cc=[x for x in changes if x['condition']==mode and x['seed']==seed and x['kind']==kind]
                ff=[x for x in feature_changes if x['condition']==mode and x['seed']==seed and x['kind']==kind]
                metrics.append(dict(condition=mode,seed=seed,kind=kind,status_changes=len(cc),feature_state_changes=len(ff),
                    new_difference_cards=sum(x['after']=='difference_evidence'for x in cc),
                    lost_difference_cards=sum(x['before']=='difference_evidence'for x in cc),
                    new_supported_feature_cards=sum(x['after']=='supported_features_only'for x in cc),
                    new_feature_warnings=sum(x['after']in['difference','opposite_direction']and x['before']not in['difference','opposite_direction']for x in ff)))
    newwarn=[v for v in changes if v['after']=='difference_evidence']
    featurewarn=[v for v in feature_changes if v['after']in['difference','opposite_direction']and v['before']not in['difference','opposite_direction']]
    normalchange=[v for v in changes if v['kind']in['full_normal','speed075','speed125']]
    normalfeature=[v for v in feature_changes if v['kind']in['full_normal','speed075','speed125']]
    passpromote=[v for v in changes if v['kind']in['omit_final_event','mask_final_event']and v['after']=='supported_features_only']
    # These are exact observed contract facts, not educational error labels.
    ck('no_new_difference_card_on_any_kind',not newwarn)
    ck('normal_speed_card_status_unchanged',not normalchange)
    ck('no_absence_promoted_to_supported_card',not passpromote)
    write(E/'PAIRED_CHANGES.json',dict(status='SAVED_DIFFERENCES_NOT_HUMAN_TRUTH',counts=counts,card_changes=changes,
        feature_changes=feature_changes,metrics=metrics,new_difference_cards=newwarn,new_feature_warnings=featurewarn,
        normal_speed_card_changes=normalchange,normal_speed_feature_changes=normalfeature,absence_supported_promotions=passpromote,
        interpretation='An order boundary issue and an explicit wrong shape/direction warning are separate outputs. Loss of opposite_direction on omitted/masked input is not a measured loss of a true opposite movement. No completion labels or whole-sign correctness inferred.'))
    write(E/'VERIFICATION.json',dict(status='SAVED_ARITHMETIC_AND_PRESERVATION_PASS_NOT_G2_G3_PASS',at=time.time(),
        checks=dict(checks),total_checks=sum(checks.values()),protected477=True,counts=cfg['counts'],rows=246,summary_rows=264,
        old_conditions_exact=6,new_conditions_checked=MODES,new_direction58_geometry_exact=62,new_attribute_frames=120,
        new_difference_cards=len(newwarn),new_feature_warnings=len(featurewarn),normal_speed_status_changes=len(normalchange),
        normal_speed_feature_state_changes=len(normalfeature),omission_mask_supported_promotions=len(passpromote),
        source_pins={str(Path(__file__)):sha(Path(__file__)),str(mathpath):sha(mathpath),str(E/'FROZEN.json'):sha(E/'FROZEN.json'),str(E/'RESULTS.json'):sha(E/'RESULTS.json')},
        no_model_or_old_result_inference=True,independent_human_truth=False,semantic_completion=None,G2_pass=None,G3_pass=None))
    print('K semantic saved verification PASS',sum(checks.values()),flush=True)
    print(json.dumps(dict(card_changes=len(changes),feature_changes=len(feature_changes),new_warning_cards=len(newwarn),
        new_feature_warnings=len(featurewarn),normal_speed_features=normalfeature),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        E.mkdir(parents=True,exist_ok=True)
        (E/f'VERIFY_FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise

