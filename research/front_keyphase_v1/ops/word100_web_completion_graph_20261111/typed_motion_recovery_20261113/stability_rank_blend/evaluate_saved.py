"""One predeclared 0.5 old + 0.5 J logit blend; saved arrays only.

No network/encoder is imported. Old cards/scores remain immutable. This is a
cheap retention control, not a coefficient search or automatic G2 acceptance.
"""
from pathlib import Path
from collections import Counter
import ast,copy,datetime,hashlib,itertools,json,sys,time,traceback
import numpy as np
S=Path(__file__).resolve().parent
H=S.parent;J=H/'splice_rank_learning';W=H.parent/'candidate_description_supervision'
L=W/'local_rank_learning';BOUNDARY=H/'null_candidate_learning/shared_boundary_audit.py'
MODE='equal_rank_blend'
AX={2:0,3:1,4:2,5:4,7:3}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def key(r):return r['seed'],r['clip_id'],r['kind']
def checkpins(pins):
    for p,h in pins.items():assert sha(p)==h,p
def pts_function():
    tree=ast.parse(BOUNDARY.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='temporal_relation')
    ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(BOUNDARY),'exec'),ns)
    return ns['temporal_relation']

def freeze():
    assert not(S/'FROZEN.json').exists()
    current=read(J/'evaluation/COMPARISON.json');old=read(L/'evaluation/COMPARISON.json')
    ok=set(key(r)for r in current['rows'])==set(key(r)for r in old['rows'])
    assert ok and len(current['rows'])==len(old['rows'])==246
    assert read(J/'POSTTRAIN_VERIFICATION.json')['status']=='SAVED_CONTRACT_PASS_RESEARCH_RESULTS_REQUIRE_REVIEW'
    paths=[Path(__file__),J/'evaluation/COMPARISON.json',L/'evaluation/COMPARISON.json',J/'TRAINING_COMPLETE.json',
           J/'TRAINING_FROZEN.json',J/'POSTTRAIN_VERIFICATION.json',J/'evaluation/PTS_ORDER_AUDIT.json',BOUNDARY,W/'core.py']
    pins={str(p):sha(p)for p in paths}
    for r in current['predictions']:
        for p,h in [(r['path'],r['sha256']),(r['original_prediction'],r['original_prediction_sha256']),(r['source'],r['source_sha256'])]:
            assert sha(p)==h;pins[p]=h
    # Numeric formula/selector contract only, without reading candidate scores.
    a=np.array([0.,4.,-8.]);b=np.array([4.,0.,8.]);v=.5*a+.5*b
    assert np.array_equal(v,[2.,2.,0.]) and int(v.argmax())==0
    write(S/'FROZEN.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='ONE_FIXED_SAVED_LOGIT_CONTROL_BEFORE_NEW_SELECTIONS',primary=MODE,alpha=.5,
        formula='float64(0.5*saved_local_full_logits+0.5*saved_splice_rank_logits), matchingseed/candidate/family. No softmax or probability interpretation.',
        families=[2,3],other_families='Copy original local_full scores/cards/qualifiers exactly; no blending.',
        selector='Exactly existing candidate ordering/presence>=0/rawargmax, firstindexforties; body andhand observation masks unchanged.',
        data='All82same frozenpublicscenarios x3seeds=246rows. No source/case/seed selection.',
        order='Existing temporal_relation on actual first/last integerPTS, samefunction onold/J/blend. Positive-duration overlap retained, sharedinstant boundaryallowed. Rawdiscreteorder preserved separately.',
        semantics='No changedhandshape/direction model, no omission/completion/educationalcorrectness inference. semantic_completion=None.',
        interpretation='Fixed retention ensemble control only. Scores sharearchitecture/normalization/initialization, but finetuningchangeslogitscale: equalweightsnotnecessarilyequalinfluence.',
        cost='Eventualdeploymentwouldrun two smallrankheads ononeexistingencoderfeaturecache. No doubledencoderextraction implied; no actuallatencymeasuredbythissavedarithmetictest.',
        criteria='Report everyseed andscenario versusold andJ; preservingnormal/speed while retainingcorrectdynamic/reversewitnessassociation isrequired. RootownsG2decision. Anyfailurepreservedwithoutalphasweeporautomaticnextvariant.',
        prohibited=['training','modelinference','newlabels','expectedorderinscoring','alphatuning','seedselection','webchange','final8/privateinputs','G2passdeclaration'],
        numeric_formula_test='Exact1:1mean andoriginalfirstargmaxtie behavior verifiedon3elementfixture.',pins=pins))
    print('Fixed equal-weight rank blend frozen; no evaluation yet',flush=True)

def select(original,spans,presence,score,qr):
    admitted=np.flatnonzero(presence>=0);chosen=None
    if len(admitted):
        pick=int(admitted[np.argmax(score[admitted])]);a,b=map(int,spans[pick]);axis=AX[original['family']]*2
        observed=np.arange(a,b+1);observed=observed[qr[observed,axis,8]>.5]
        chosen=dict(span=[a,b],source=MODE,body_observed=observed.tolist(),
            hand_qualified=observed[qr[observed,axis,9]>.5].tolist(),score=float(score[pick]),semantic_completion=None)
    card=dict(card=original['card'],family=original['family'],chosen=chosen,semantic_completion=None)
    if 'truth' in original:
        seen=set(chosen['body_observed'])if chosen else set()
        card['truth']=dict(original['truth'],all_witness=set(original['truth']['witness'])<=seen,
            wrong_points=len(set(original['truth']['wrong_witness'])&seen))
    return card

def orders(cards,t,relation):
    raw=[];pts=[]
    for i,j in itertools.combinations(range(len(cards)),2):
        if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
        chosen=[cards[k]['chosen']for k in [i,j]];sp=[c['span']if c else None for c in chosen]
        state='unobserved'if None in sp else 'ambiguous_overlap'if max(sp[0][0],sp[1][0])<=min(sp[0][1],sp[1][1])else'forward'if sp[0][1]<sp[1][0]else'reverse'
        names=[cards[i]['card'],cards[j]['card']];raw.append(dict(cards=names,state=state))
        times=[[int(t[k])for k in s]if s is not None else None for s in sp]
        state,detail,overlap=relation(*times)
        pts.append(dict(cards=names,state=state,detail=detail,overlap_duration_us=overlap,spans=sp,time_extents_us=times))
    return raw,pts

def run():
    assert not(S/'RESULTS.json').exists() and not(S/'predictions').exists()
    cfg=read(S/'FROZEN.json');checkpins(cfg['pins']);assert cfg['alpha']==.5
    comparison=read(J/'evaluation/COMPARISON.json');metadata={key(r):r for r in comparison['predictions']}
    relation=pts_function();rows=[];predictions=[];sources={};protected=0;selected_changes=[]
    for oldrow in comparison['rows']:
        k=key(oldrow);meta=metadata[k];old=dict(np.load(meta['original_prediction']));new=dict(np.load(meta['path']))
        if meta['source']not in sources:sources[meta['source']]=dict(np.load(meta['source']))
        source=sources[meta['source']];ts=source['ts'];qr=source['z'][:,292:392].reshape(-1,10,10)
        assert ts.ndim==1 and np.issubdtype(ts.dtype,np.integer)and(np.diff(ts)>0).all()
        basecards=oldrow['conditions']['local_full']['cards'];jcards=oldrow['conditions']['splice_rank']['cards'];cards=[];arrays={}
        for i,oc in enumerate(basecards):
            sp=old[f'spans_{i}'];pr=old[f'presence_{i}'];a=old[f'local_full_logits_{i}'];b=new[f'splice_rank_logits_{i}']
            assert np.array_equal(sp,new[f'spans_{i}'])and np.array_equal(pr,new[f'presence_{i}'])
            assert a.shape==b.shape==pr.shape and np.isfinite(a).all()and np.isfinite(b).all()
            if oc['family']in[2,3]:
                score=.5*a.astype(np.float64)+.5*b.astype(np.float64);card=select(oc,sp,pr,score,qr)
            else:
                score=a.copy();card=copy.deepcopy(oc);protected+=1
                assert card==oc and np.array_equal(score,a)
            arrays[f'spans_{i}']=sp;arrays[f'presence_{i}']=pr;arrays[f'{MODE}_logits_{i}']=score;cards.append(card)
            spans=[v['chosen']['span']if v['chosen']else None for v in [oc,jcards[i],card]]
            if spans[0]!=spans[2]or spans[1]!=spans[2]:selected_changes.append(dict(seed=k[0],clip_id=k[1],kind=k[2],card=oc['card'],family=oc['family'],old_span=spans[0],J_span=spans[1],blend_span=spans[2]))
        raw,pts=orders(cards,ts,relation);conditions=copy.deepcopy(oldrow['conditions']);conditions[MODE]=dict(cards=cards,order=raw)
        pts_by={MODE:pts}
        for mode in ['local_full','splice_rank']:
            rawold,ptsold=orders(conditions[mode]['cards'],ts,relation)
            assert rawold==conditions[mode]['order'];pts_by[mode]=ptsold
        assert all(conditions[mode]==oldrow['conditions'][mode]for mode in oldrow['conditions'])
        rows.append(dict(seed=k[0],clip_id=k[1],kind=k[2],conditions=conditions,pts_order=pts_by))
        path=S/'predictions'/str(k[0])/f'{k[1]}_{k[2]}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**arrays);predictions.append(dict(meta,path=str(path),sha256=sha(path),J_prediction=meta['path'],J_prediction_sha256=meta['sha256']))
    assert len(rows)==246
    summary=[]
    for seed,mode,kind in itertools.product([42,43,44],['local_full','splice_rank',MODE],sorted({r['kind']for r in rows})):
        rr=[r for r in rows if r['seed']==seed and r['kind']==kind];cc=[c for r in rr for c in r['conditions'][mode]['cards']];truth=[c['truth']for c in cc if'truth'in c]
        summary.append(dict(seed=seed,condition=mode,kind=kind,scenarios=len(rr),cards=len(cc),missing=sum(c['chosen']is None for c in cc),
            truth_cards=len(truth),all_witness=sum(t['all_witness']for t in truth),wrong_points=sum(t['wrong_points']for t in truth),
            PTS_order=dict(Counter(o['state']for r in rr for o in r['pts_order'][mode])),
            raw_discrete_order=dict(Counter(o['state']for r in rr for o in r['conditions'][mode]['order']))))
    checkpins(cfg['pins'])
    write(S/'RESULTS.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='ONE_FIXED_BLEND_COMPLETE_ROOT_G2_REVIEW_REQUIRED',
        alpha=.5,rows=rows,predictions=predictions,summary=summary,selection_changes=selected_changes,protected_other_family_cards=protected,
        prior_conditions_preserved=True,old_scores_unchanged=True,new_training=False,new_inference=False,new_encoder_inference=False,
        runtime_rank_heads=2,runtime_encoder_passes=1,actual_runtime_latency_not_measured=True,semantic_completion=None,web_changed=False,g2_pass_claim=False,
        frozen_sha256=sha(S/'FROZEN.json')))
    write(S/'COMPLETE.json',dict(results=str(S/'RESULTS.json'),sha256=sha(S/'RESULTS.json'),inference=False,training=False,alpha_search=False))
    print('Saved equalblend246rows, protectedotherfamilycards',protected,flush=True)
    for r in summary:
        if r['condition']==MODE and r['kind']in['full_normal','speed075','speed125','dynamic_complete_blocks','omit_final_event']:
            print(r,flush=True)

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1]in['freeze','run']
        freeze()if sys.argv[1]=='freeze'else run()
    except BaseException:
        (S/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
