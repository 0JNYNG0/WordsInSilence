"""K's two fixed rank conditions: reuse old/SB/J semantics; score new requests only.

freeze is saved arithmetic and mask accounting, with no neural inference.
Root reviews the frozen script before authorizing run-reviewed. No G2/G3 claim.
"""
from pathlib import Path
import ast, collections, copy, hashlib, importlib.util, json, sys, time, traceback
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;H=U.parent;G=H.parent;O=G.parent
S=H/'stability_rank_blend';J=H/'splice_rank_learning/changed_span_shadow'
E=U/'changed_span_shadow';REQ=O/'word100_guide_requirement_learning_20261111'
OBSERVER=O/'word100_broad_factorized_learning_20261110'
SHARED=O/'word100_shared_motion_learning_20261109/core.py'
MODES=['rate_head','old_local_full_plus_rate_half']

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8')as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rowkey(r):return r['seed'],r['clip_id'],r['kind']
def reqkey(cid,kind,card,s):return cid,kind,card,None if s is None else tuple(s)
def reqtext(k):return '|'.join([k[0],k[1],k[2],'none'if k[3]is None else f'{k[3][0]}:{k[3][1]}'])
def movkey(cid,kind,s,axis,side):return cid,kind,tuple(s),int(axis),side
def movtext(k):return '|'.join([k[0],k[1],f'{k[2][0]}:{k[2][1]}',str(k[3]),k[4]])
def span(c):return c['chosen']['span']if c['chosen']is not None else None
def checkpins(pins):
    for p,h in pins.items():assert sha(p)==h,('changed pin',p)
def stop():
    for p in[G,H,U]:
        for n in['STOP','STOP.txt','STOP_REQUESTED','STOP_REQUESTED.json']:assert not(p/n).exists(),str(p/n)
def available_function():
    # The exact frozen pure mask function, no model module or checkpoint imported.
    tree=ast.parse(SHARED.read_text('utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='available')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SHARED),'exec'),ns)
    return ns['available']
def lookups():
    cards=read(O/'word100_action_feedback_demo_20261108/CARDS.json')['words']
    fam=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
    return ({(w,c['id']):c for w,cs in cards.items()for c in cs},
            {(w,c['id']):c for w,cs in fam.items()for c in cs})

def contract():
    """No models, predictions, threshold choices or label changes are executed."""
    old=read(S/'changed_span_shadow/RESULTS.json');new=read(U/'evaluation/COMPARISON.json')
    prior_rank=read(S/'RESULTS.json');jc=read(J/'RESULTS.json')
    om={rowkey(r):r for r in old['rows']};nm={rowkey(r):r for r in new['rows']}
    pm={rowkey(r):r for r in new['predictions']};bm={rowkey(r):r for r in prior_rank['predictions']}
    assert len(old['rows'])==len(new['rows'])==len(om)==246 and set(om)==set(nm)==set(pm)==set(bm)
    assert new['semantic_completion']is None and new['G2_pass']is None
    cards,families=lookups();saved={};origins={};moves={};move_origins={}
    for r in old['rows']:
        for mode,cs in r['conditions'].items():
            for c in cs:
                k=reqkey(r['clip_id'],r['kind'],c['card'],c['span'])
                if k in saved:assert saved[k]==c,('semantic cache disagreement',k)
                else:saved[k]=c;origins[k]=dict(seed=r['seed'],condition=mode)
                word=r['clip_id'].split('_')[2];spec=families[word,c['card']]
                for item in c['evidence']:
                    if item['task']!='direction'or'probabilities'not in item:continue
                    mk=movkey(r['clip_id'],r['kind'],item['span'],item['axis'],spec['hands'][0])
                    rec=dict(probabilities=item['probabilities'],eligible=item['state']!='observation_unknown')
                    if mk in moves:assert moves[mk]==rec,('movement cache disagreement',mk)
                    else:moves[mk]=rec;move_origins[mk]=dict(source='saved_semantic_evidence',seed=r['seed'],condition=mode,card=c['card'])
    # Validate exact eligibility/probabilities against the independently stored requests.
    for origin,rs in [('J',jc['new_movement_requests']),('SB',old['new_movements'])]:
        for r in rs:
            mk=movkey(r['clip_id'],r['kind'],r['span'],r['axis'],r['moving'])
            rec=dict(probabilities=r['probabilities'],eligible=r['eligible'])
            if mk in moves:assert moves[mk]==rec,('explicit movement differs',mk)
            else:moves[mk]=rec;move_origins[mk]=dict(source=origin)
    sources={};plan=[];requests={};per_mode={}
    for mode in MODES:
        changed=0;total=0;mode_requests=set()
        for r in new['rows']:
            rk=rowkey(r);prior=om[rk];p=pm[rk];bp=bm[rk]
            assert p['source']==bp['source']and p['source_sha256']==bp['source_sha256']
            assert all(r['conditions'][m]==next(x for x in prior_rank['rows']if rowkey(x)==rk)['conditions'][m]
                       for m in prior['conditions'])
            sid=r['clip_id'],r['kind'];sr=dict(clip_id=sid[0],kind=sid[1],source=p['source'],sha256=p['source_sha256'])
            assert sid[0].startswith('NIA_SL_WORD')and sid[0].split('_')[3]in['REAL11','REAL12','REAL13']
            if sid in sources:assert sources[sid]==sr
            sources[sid]=sr;bc={c['card']:c for c in prior['conditions']['local_full']}
            ranked=r['conditions'][mode]['cards'];assert set(bc)=={c['card']for c in ranked}
            for c in ranked:
                total+=1;s=span(c);k=reqkey(*sid,c['card'],s);action='copy_same_local_full'
                if s!=bc[c['card']]['span']:
                    changed+=1;assert c['family']in[2,3];mode_requests.add(k)
                    action='reuse_saved_identical_request'if k in saved else'new_request'
                    if k not in requests:requests[k]=dict(request_id=reqtext(k),clip_id=sid[0],kind=sid[1],
                        card=c['card'],family=c['family'],span=s,action=action,saved_origin=origins.get(k))
                plan.append(dict(condition=mode,seed=r['seed'],clip_id=sid[0],kind=sid[1],card=c['card'],
                    old_span=bc[c['card']]['span'],new_span=s,action=action,request_id=reqtext(k)))
        assert total==318
        per_mode[mode]=dict(cards=total,changed_cards=changed,unique_changed_requests=len(mode_requests),
            saved_identical_requests=sum(k in saved for k in mode_requests),new_requests=sum(k not in saved for k in mode_requests))
    # Pointwise cache lookup is source-specific; edited controls cannot borrow another source's frames.
    cache_index={}
    for r in jc['pointwise_cache']:
        sid=r['clip_id'],r['kind'];assert sid not in cache_index
        assert r['source_sha256']==sources[sid]['sha256']and sha(r['path'])==r['sha256']
        with np.load(r['path'],allow_pickle=False)as z:fi=z['frame_indices'].copy();vv=z['attribute_probabilities']
        assert fi.ndim==1 and len(np.unique(fi))==len(fi)and vv.shape==(len(fi),7)and np.isfinite(vv).all()
        cache_index[sid]=dict(record=r,frames=set(map(int,fi)))
    available=available_function();loaded={};needed=collections.defaultdict(set);new_movements={};reused_movements={}
    request_detail=[]
    for q in requests.values():
        if q['action']!='new_request':continue
        detail=dict(q,attribute_scope=None,valid_attribute_frame_uses=0,movement_request=None,movement_action=None)
        if q['span']is None:request_detail.append(detail);continue
        sid=q['clip_id'],q['kind'];s=sources[sid]
        if sid not in loaded:
            assert sha(s['source'])==s['sha256']
            with np.load(s['source'],allow_pickle=False)as z:x=z['x'].copy();ts=z['ts'].copy()
            assert x.ndim==2 and x.shape[1]==280 and np.isfinite(x).all()and len(ts)==len(x)and(np.diff(ts)>0).all()
            loaded[sid]=dict(x=x,ts=ts,av=available(x))
        v=loaded[sid];a,b=q['span'];assert 0<=a<=b<len(v['x'])
        word=q['clip_id'].split('_')[2];guide=cards[word,q['card']];spec=families[word,q['card']]
        aa=a+(b-a)//2 if guide.get('shape_scope')=='late'else a;detail['attribute_scope']=[aa,b]
        for j in guide.get('features',{}):
            valid=np.arange(aa,b+1)[v['av'][aa:b+1,int(j)]]
            needed[sid].update(map(int,valid));detail['valid_attribute_frame_uses']+=len(valid)
        if spec['axis']is not None and spec['sign']is not None:
            mk=movkey(*sid,[a,b],spec['axis'],spec['hands'][0]);detail['movement_request']=movtext(mk)
            mr=dict(request_id=movtext(mk),clip_id=sid[0],kind=sid[1],span=[a,b],axis=spec['axis'],moving=spec['hands'][0])
            if mk in moves:
                detail['movement_action']='reuse_saved_movement';reused_movements[mk]=dict(mr,origin=move_origins[mk])
            else:detail['movement_action']='new_movement';new_movements[mk]=mr
        request_detail.append(detail)
    pointwise_plan=[]
    for sid in sorted(needed):
        wanted=needed[sid];existing=cache_index.get(sid,{}).get('frames',set())
        pointwise_plan.append(dict(clip_id=sid[0],kind=sid[1],needed_frames=sorted(wanted),
            reused_frames=sorted(wanted&existing),new_frames=sorted(wanted-existing),
            cache_record=cache_index[sid]['record']if sid in cache_index else None))
    count=dict(per_condition=per_mode,changed_card_occurrences=sum(v['changed_cards']for v in per_mode.values()),
        unique_changed_requests=len(requests),saved_identical_requests=sum(q['action']!='new_request'for q in requests.values()),
        new_semantic_requests=len(request_detail),new_no_span_requests=sum(q['span']is None for q in request_detail),
        new_direction_requests=len(new_movements),reused_direction_requests=len(reused_movements),
        required_attribute_frames=sum(len(r['needed_frames'])for r in pointwise_plan),
        reused_attribute_frames=sum(len(r['reused_frames'])for r in pointwise_plan),
        new_attribute_frames=sum(len(r['new_frames'])for r in pointwise_plan))
    return dict(old=old,new=new,oldmap=om,saved=saved,moves=moves,plan=plan,requests=requests,sources=sources,
        detail=request_detail,pointwise_plan=pointwise_plan,new_movements=list(new_movements.values()),
        reused_movements=list(reused_movements.values()),counts=count,jcache=jc['pointwise_cache'])

def freeze():
    stop();assert not(E/'FROZEN.json').exists();d=contract()
    pins=read(G/'PROTECTIONS.json').copy();assert len(pins)==477;checkpins(pins)
    files=[Path(__file__),U/'evaluation/COMPARISON.json',U/'evaluation/FROZEN.json',U/'TRAINING_COMPLETE.json',
        S/'RESULTS.json',S/'changed_span_shadow/RESULTS.json',S/'changed_span_shadow/FROZEN.json',
        J/'RESULTS.json',J/'FROZEN.json',J/'VERIFICATION.json',REQ/'requirement_core.py',REQ/'GUIDE_CAPABILITY.json',
        OBSERVER/'core.py',OBSERVER/'NORMALIZATION.json',OBSERVER/'runs/coupled_42/model.pt',SHARED,
        O/'word100_motion_components_20261109/motion.py',O/'word100_rtmw_joint_learning_20261108/feedback_model.py',
        O/'word100_rtmw_joint_learning_20261108/feedback_viewfix/FEEDBACK_NORMALIZATION.json',
        O/'word100_action_feedback_demo_20261108/CARDS.json',O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json']
    pins.update({str(p):sha(p)for p in files});pins.update({s['source']:s['sha256']for s in d['sources'].values()})
    pins.update({r['path']:r['sha256']for r in d['jcache']});checkpins(pins)
    write(E/'FROZEN.json',dict(at=time.time(),status='REQUEST_PLAN_FROZEN_AWAITING_ROOT_CODE_REVIEW_NO_INFERENCE',
        purpose='G2 cause isolation of BOTH fixed K conditions; no score-based condition choice and no G2/G3 declaration.',
        modes=MODES,rows=246,source_scenarios=82,counts=d['counts'],plan=d['plan'],
        requests=list(d['requests'].values()),new_request_detail=d['detail'],pointwise_plan=d['pointwise_plan'],
        new_movements=d['new_movements'],reused_movements=d['reused_movements'],sources=list(d['sources'].values()),
        observer_checkpoint='word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt',
        observer_seed=42,selector_seeds=[42,43,44],
        semantics='Exact old coupled42, old body-wrist geometry, .8 availability, late-half shape scope, .65/.35 logic and original capability. No PD or R/new hand models.',
        cache='All six previous semantic conditions/summary copied. Identical requests copied, saved pointwise values reused. Only previously unscored direction requests and missing required attribute frames are inferred.',
        no_training=True,no_pose_or_encoder=True,no_TCN_or_DTW=True,no_web=True,no_final8=True,no_private=True,no_threshold_change=True,
        semantic_completion=None,G2_pass=None,G3_pass=None,pins=pins))
    print(json.dumps(d['counts'],ensure_ascii=False),flush=True)

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def execute():
    stop();cfg=read(E/'FROZEN.json');checkpins(cfg['pins'])
    assert not(E/'RUN_STARTED.json').exists()and not(E/'RESULTS.json').exists()
    d=contract()
    for k,v in [('counts',d['counts']),('plan',d['plan']),('requests',list(d['requests'].values())),
                ('new_request_detail',d['detail']),('pointwise_plan',d['pointwise_plan']),
                ('new_movements',d['new_movements']),('reused_movements',d['reused_movements'])]:assert cfg[k]==v,k
    write(E/'RUN_STARTED.json',dict(at=time.time(),frozen_sha256=sha(E/'FROZEN.json'),
        scope='Parent-reviewed new requests only; no completed request reinference.',counts=cfg['counts']))
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rc=load('_K_changed_semantics_req',REQ/'requirement_core.py');net=rc.frozen_observer()
    assert isinstance(net.shared.base.attributes,torch.nn.Linear)and net.shared.base.attributes.in_features==64
    caps={(r['word'],r['card'],r['feature'],r['expected']):r['qualified']for r in read(REQ/'GUIDE_CAPABILITY.json')['rows']}
    pp={(r['clip_id'],r['kind']):r for r in cfg['pointwise_plan']}
    sources={};probabilities={};attrs_records=[];movement_records=[];computed={};moves=dict(d['moves'])
    tick=time.perf_counter()
    def source(sid):
        if sid not in sources:
            s=d['sources'][sid];assert sha(s['source'])==s['sha256']
            with np.load(s['source'],allow_pickle=False)as z:v={k:z[k].copy()for k in ['x','fh','ts']}
            assert v['fh'].shape==(len(v['x']),64)and np.isfinite(v['fh']).all()
            v['av']=rc.prior.shared.available(v['x']);sources[sid]=v
        return sources[sid]
    for sid,p in pp.items():
        stop();v=source(sid);values={}
        if p['cache_record']:
            cr=p['cache_record'];assert sha(cr['path'])==cr['sha256']
            with np.load(cr['path'],allow_pickle=False)as z:
                values.update({int(f):row.copy()for f,row in zip(z['frame_indices'],z['attribute_probabilities'])})
        fi=np.asarray(p['new_frames'],np.int64);new=np.empty((0,7),np.float32)
        if len(fi):
            with torch.inference_mode():new=net.shared.base.attributes(torch.tensor(v['fh'][fi])).sigmoid().numpy()
            assert new.shape==(len(fi),7)and np.isfinite(new).all()
            assert not set(map(int,fi))&set(values)
            values.update({int(f):row.copy()for f,row in zip(fi,new)})
            path=E/'pointwise_cache'/f'{sid[0]}_{sid[1]}.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
            np.savez_compressed(path,frame_indices=fi,attribute_probabilities=new)
            attrs_records.append(dict(clip_id=sid[0],kind=sid[1],source=d['sources'][sid]['source'],source_sha256=d['sources'][sid]['sha256'],
                path=str(path),sha256=sha(path),selected_valid_frames=len(fi),reason='Only frames missing from old J cache required by new K spans.'))
        assert set(p['needed_frames'])<=set(values);probabilities[sid]=values
    for q in cfg['new_movements']:
        stop();sid=q['clip_id'],q['kind'];v=source(sid);a,b=q['span'];axis=q['axis'];side=q['moving']
        mk=movkey(*sid,[a,b],axis,side);assert mk not in moves
        g,ok,info=rc.prior.geometry(v['x'],v['ts'],a,b,axis,side,'L'if side=='R'else'R')
        with torch.inference_mode():p=net.movement(torch.tensor(v['fh'][None]),torch.tensor(g[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy().tolist()
        moves[mk]=dict(probabilities=p,eligible=bool(ok))
        movement_records.append(dict(q,geometry=g.tolist(),eligible=bool(ok),info=info,probabilities=p,source_sha256=d['sources'][sid]['sha256']))
    for q in cfg['new_request_detail']:
        stop()
        if q['span']is None:
            computed[q['request_id']]=dict(card=q['card'],span=None,status='association_unknown',evidence=[],completion=None);continue
        sid=q['clip_id'],q['kind'];v=source(sid);word=q['clip_id'].split('_')[2];a,b=q['span']
        guide=next(c for c in rc.CARDS[word]if c['id']==q['card']);spec=next(c for c in rc.FAMILIES[word]if c['id']==q['card'])
        aa=a+(b-a)//2 if guide.get('shape_scope')=='late'else a;assert q['attribute_scope']==[aa,b];items=[]
        for j,want in guide.get('features',{}).items():
            j=int(j);valid=v['av'][aa:b+1,j];ids=np.arange(aa,b+1)[valid];ok=bool(valid.mean()>=.8)
            vv=np.asarray([probabilities[sid][int(f)][j]for f in ids],np.float32)
            p=float(vv.mean())if len(vv)else None;p=p if want or p is None else 1-p
            qualified=caps.get((word,q['card'],j,want),False);state=rc.decide(p,ok,qualified)if p is not None else'observation_unknown'
            items.append(dict(task='shape',feature=j,expected=want,span=[aa,b],probability=p,state=state,qualified=qualified,observed_count=int(valid.sum())))
        axis=spec['axis'];side=spec['hands'][0]
        if axis is not None:
            if spec['sign']is None:items.append(dict(task='direction',state='sequence_required',reason='Alternation cannot be judged from first/last displacement alone'))
            else:
                z=moves[movkey(*sid,[a,b],axis,side)];want=int(spec['sign']+1)
                items.append(dict(task='direction',axis=axis,desired=want,span=[a,b],probabilities=z['probabilities'],
                    state=rc.direction_verdict(z['probabilities'],want,z['eligible'])))
        states=[r['state']for r in items]
        status='difference_evidence'if'difference'in states or'opposite_direction'in states else'supported_features_only'if states and all(s=='supported'for s in states)else'insufficient_evidence'
        computed[q['request_id']]=dict(card=q['card'],span=[a,b],status=status,evidence=items,completion=None)
    assert len(computed)==cfg['counts']['new_semantic_requests']
    assert len(movement_records)==cfg['counts']['new_direction_requests']
    assert sum(r['selected_valid_frames']for r in attrs_records)==cfg['counts']['new_attribute_frames']
    out=[];execution=[]
    for r in d['new']['rows']:
        prior=d['oldmap'][rowkey(r)];conditions=copy.deepcopy(prior['conditions']);bc={c['card']:c for c in prior['conditions']['local_full']}
        for mode in MODES:
            cards=[]
            for c in r['conditions'][mode]['cards']:
                s=span(c);k=reqkey(r['clip_id'],r['kind'],c['card'],s)
                if s==bc[c['card']]['span']:after=copy.deepcopy(bc[c['card']]);action='copy_same_local_full'
                elif k in d['saved']:after=copy.deepcopy(d['saved'][k]);action='reuse_saved_identical_request'
                else:after=copy.deepcopy(computed[reqtext(k)]);action='new_request'
                assert after['span']==s and after['card']==c['card']and after['completion']is None
                cards.append(after);execution.append(dict(condition=mode,seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],
                    old_span=bc[c['card']]['span'],new_span=s,action=action,request_id=reqtext(k)))
            conditions[mode]=cards
        assert all(conditions[m]==prior['conditions'][m]for m in prior['conditions'])
        out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],conditions=conditions))
    # Construction groups row then condition; frozen plan groups condition then row.
    execution.sort(key=lambda v:(MODES.index(v['condition']),next(i for i,r in enumerate(d['new']['rows'])if rowkey(r)==(v['seed'],v['clip_id'],v['kind'])),
                                 next(i for i,c in enumerate(d['oldmap'][v['seed'],v['clip_id'],v['kind']]['conditions']['local_full'])if c['card']==v['card'])))
    assert execution==cfg['plan'];summary=copy.deepcopy(d['old']['summary'])
    for mode in MODES:
        for seed in[42,43,44]:
            for kind in sorted({r['kind']for r in out}):
                cs=[c for r in out if r['seed']==seed and r['kind']==kind for c in r['conditions'][mode]];items=[v for c in cs for v in c['evidence']]
                summary.append(dict(seed=seed,condition=mode,kind=kind,cards=len(cs),statuses=dict(collections.Counter(c['status']for c in cs)),
                    shape=dict(collections.Counter(v['state']for v in items if v['task']=='shape')),direction=dict(collections.Counter(v['state']for v in items if v['task']=='direction'))))
    assert all(v['state']=='observation_unknown'for r in out if r['kind']=='all_hand_missing'for mode in MODES for c in r['conditions'][mode]for v in c['evidence']if v['task']=='shape')
    stop();checkpins(cfg['pins'])
    write(E/'RESULTS.json',dict(at=time.time(),status='K_BOTH_CONDITIONS_CHANGED_SPAN_CAUSE_ISOLATION_NOT_G2_G3_PASS',rows=out,summary=summary,
        execution=execution,counts=cfg['counts'],new_movements=movement_records,new_pointwise_cache=attrs_records,reused_pointwise_cache=d['jcache'],
        previous_six_conditions_exact=True,no_old_result_reinference=True,no_training=True,no_pose_or_encoder=True,no_TCN_or_DTW=True,
        no_PD_or_new_hand=True,no_web=True,no_final8=True,observer_seed=42,selector_seeds=[42,43,44],semantic_completion=None,G2_pass=None,G3_pass=None,
        seconds=time.perf_counter()-tick,timing_scope='Only missing cached-source semantic requests plus assembly and pin checking; not user latency.',
        frozen_sha256=sha(E/'FROZEN.json')))
    print('Saved both fixed K conditions; old requests preserved; root gate review required.',flush=True)

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1]in['freeze','run-reviewed']
        freeze()if sys.argv[1]=='freeze'else execute()
    except BaseException:
        E.mkdir(parents=True,exist_ok=True)
        (E/f'FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise

