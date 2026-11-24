"""Eight new blended-rank semantic requests; all old/J output reused.

Parent authorized freeze, static review and one execution. This is G2 cause
isolation, not a G2/G3 acceptance, new training or runtime integration.
"""
from pathlib import Path
import collections, copy, hashlib, importlib.util, json, sys, time, traceback

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
H = U.parent; G = H.parent; O = G.parent
J = H/'splice_rank_learning/changed_span_shadow'
E = U/'changed_span_shadow'
REQ = O/'word100_guide_requirement_learning_20261111'
OBSERVER = O/'word100_broad_factorized_learning_20261110'
MODE = 'equal_rank_blend'

def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rowkey(r): return r['seed'],r['clip_id'],r['kind']
def span(c): return c['chosen']['span'] if c['chosen'] else None
def requestkey(cid,kind,card,s): return cid,kind,card,None if s is None else tuple(s)
def keytext(k): return '|'.join([k[0],k[1],k[2],'none' if k[3] is None else f'{k[3][0]}:{k[3][1]}'])
def write(p,r):
    p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with p.open('x',encoding='utf-8')as f:f.write(text)
def checkpins(pins):
    for p,h in pins.items():assert sha(p)==h,('pin',p)
def stop():
    for root in[G,H,U]:
        for name in['STOP','STOP.txt','STOP_REQUESTED','STOP_REQUESTED.json']:
            assert not(root/name).exists(),('stop marker',root/name)

def contract():
    old=read(J/'RESULTS.json');new=read(U/'RESULTS.json')
    assert read(J/'VERIFICATION.json')['status']=='SAVED_ARITHMETIC_AND_PRESERVATION_PASS_NOT_G3_PASS'
    assert new['alpha']==.5 and not new['new_inference']
    om={rowkey(r):r for r in old['rows']};nm={rowkey(r):r for r in new['rows']};pm={rowkey(r):r for r in new['predictions']}
    assert set(om)==set(nm)==set(pm) and len(om)==246
    saved={};saved_origin={}
    for r in old['rows']:
        for mode,cs in r['conditions'].items():
            for c in cs:
                k=requestkey(r['clip_id'],r['kind'],c['card'],c['span'])
                if k in saved:assert saved[k]==c
                else:saved[k]=c;saved_origin[k]=dict(seed=r['seed'],condition=mode)
    plan=[];requests={};sources={};changed=0;total=0
    for r in new['rows']:
        oldcards={c['card']:c for c in om[rowkey(r)]['conditions']['local_full']}
        sid=r['clip_id'],r['kind'];p=pm[rowkey(r)]
        sr=dict(clip_id=sid[0],kind=sid[1],source=p['source'],sha256=p['source_sha256'])
        if sid in sources:assert sources[sid]==sr
        sources[sid]=sr
        assert r['clip_id'].split('_')[3]in['REAL11','REAL12','REAL13']
        for c in r['conditions'][MODE]['cards']:
            total+=1;s=span(c);previous=oldcards[c['card']]['span'];k=requestkey(*sid,c['card'],s)
            action='copy_same_local_full'
            if s!=previous:
                changed+=1;assert c['family']in[2,3]
                action='reuse_saved_old_or_J' if k in saved else 'new_direction_only'
                if k not in requests:requests[k]=dict(request_id=keytext(k),clip_id=sid[0],kind=sid[1],card=c['card'],span=s,
                    action=action,saved_origin=saved_origin.get(k))
            plan.append(dict(seed=r['seed'],clip_id=sid[0],kind=sid[1],card=c['card'],old_span=previous,new_span=s,action=action,request_id=keytext(k)))
    assert total==318 and changed==63 and len(requests)==48
    counts=collections.Counter(r['action']for r in requests.values())
    assert counts=={'reuse_saved_old_or_J':40,'new_direction_only':8}
    return dict(old=old,new=new,oldmap=om,saved=saved,plan=plan,requests=requests,sources=sources)

def freeze():
    stop();assert not(E/'FROZEN.json').exists();d=contract()
    protections=read(G/'PROTECTIONS.json');assert len(protections)==477;checkpins(protections)
    pins=dict(protections)
    files=[Path(__file__),U/'RESULTS.json',U/'FROZEN.json',U/'VERIFICATION.json',J/'RESULTS.json',J/'FROZEN.json',J/'VERIFICATION.json',
        J/'BLEND_SHADOW_PRECHECK.json',REQ/'requirement_core.py',REQ/'GUIDE_CAPABILITY.json',OBSERVER/'core.py',OBSERVER/'NORMALIZATION.json',
        OBSERVER/'runs/coupled_42/model.pt',O/'word100_shared_motion_learning_20261109/core.py',O/'word100_motion_components_20261109/motion.py',
        O/'word100_rtmw_joint_learning_20261108/feedback_model.py',O/'word100_rtmw_joint_learning_20261108/feedback_viewfix/FEEDBACK_NORMALIZATION.json',
        O/'word100_action_feedback_demo_20261108/CARDS.json',O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json']
    pins.update({str(p):sha(p)for p in files});pins.update({r['source']:r['sha256']for r in d['sources'].values()})
    for r in d['old']['pointwise_cache']:pins[r['path']]=r['sha256']
    checkpins(pins)
    assert all(pins[p]==h for p,h in protections.items())
    write(E/'FROZEN.json',dict(at=time.time(),status='FROZEN_BEFORE_EIGHT_REQUEST_EXECUTION',purpose='G2 changed-span cause isolation only; root owns gates.',
        mode=MODE,rows=246,total_cards=318,changed_cards=63,unique_changed_requests=48,saved_reused_requests=40,new_direction_requests=8,new_attribute_frames=0,
        plan=d['plan'],requests=list(d['requests'].values()),sources=list(d['sources'].values()),
        observer='word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt',observer_seed=42,selector_seeds=[42,43,44],
        measurement='Same old body-wrist geometry/eligibility, same movement model, .65/.35 output logic. PD and new hand candidates excluded.',
        shape='All eight newly requested spans have zero eligible shape frames. Assert this before any inference and output observation_unknown. No attribute layer call.',
        reuse='All existing five semantic conditions and summaries copied exactly; same local_full spans copied, all existing old/J identical requests reused. Pointwise caches pinned but no new frame is needed.',
        no_training=True,no_pose_or_encoder=True,no_TCN_or_DTW=True,no_web=True,no_final8=True,no_private=True,no_threshold_change=True,
        semantic_completion=None,G2_pass=None,G3_pass=None,pins=pins))
    print('Frozen63 changed cards/48requests:40 saved,8 movement only,0 attribute frames.',flush=True)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def execute():
    stop();cfg=read(E/'FROZEN.json');checkpins(cfg['pins']);d=contract()
    assert cfg['plan']==d['plan'] and cfg['requests']==list(d['requests'].values())
    assert not(E/'RUN_STARTED.json').exists()and not(E/'RESULTS.json').exists()
    import numpy as np
    import torch
    rc=load('_blend_changed_semantics_req',REQ/'requirement_core.py')
    requests=[r for r in cfg['requests']if r['action']=='new_direction_only']
    sources={};specs={}
    # Check all shape masks before loading/calling a neural model.
    for q in requests:
        sid=q['clip_id'],q['kind'];s=d['sources'][sid]
        if sid not in sources:
            assert sha(s['source'])==s['sha256']
            with np.load(s['source'],allow_pickle=False)as z:values={k:z[k].copy()for k in['x','fh','ts']}
            assert values['fh'].shape==(len(values['x']),64) and values['x'].shape[1]==280 and np.all(np.diff(values['ts'])>0)
            assert np.isfinite(values['fh']).all() and np.isfinite(values['x']).all()
            values['av']=rc.prior.shared.available(values['x']);sources[sid]=values
        a,b=q['span'];word=q['clip_id'].split('_')[2];guide=next(c for c in rc.CARDS[word]if c['id']==q['card'])
        spec=next(c for c in rc.FAMILIES[word]if c['id']==q['card']);aa=a+(b-a)//2 if guide.get('shape_scope')=='late'else a
        assert 0<=a<=b<len(sources[sid]['x']) and spec['axis']in[0,1]and spec['sign']is not None
        assert all(not sources[sid]['av'][aa:b+1,int(j)].any()for j in guide.get('features',{})),q
        specs[q['request_id']]=dict(guide=guide,spec=spec,aa=aa)
    write(E/'RUN_STARTED.json',dict(at=time.time(),frozen_sha256=sha(E/'FROZEN.json'),all8_shape_masks_checked=True,attribute_calls=0))
    torch.set_num_threads(4);torch.set_num_interop_threads(1);net=rc.frozen_observer()
    caps={(r['word'],r['card'],r['feature'],r['expected']):r['qualified']for r in read(REQ/'GUIDE_CAPABILITY.json')['rows']}
    computed={};new_movements=[];tick=time.perf_counter()
    for q in requests:
        stop();sid=q['clip_id'],q['kind'];v=sources[sid];word=q['clip_id'].split('_')[2];a,b=q['span']
        meta=specs[q['request_id']];guide=meta['guide'];spec=meta['spec'];aa=meta['aa'];items=[]
        for j,want in guide.get('features',{}).items():
            j=int(j);items.append(dict(task='shape',feature=j,expected=want,span=[aa,b],probability=None,state='observation_unknown',
                qualified=caps.get((word,q['card'],j,want),False),observed_count=0))
        axis=spec['axis'];side=spec['hands'][0]
        g,ok,info=rc.prior.geometry(v['x'],v['ts'],a,b,axis,side,'L'if side=='R'else'R')
        with torch.inference_mode():p=net.movement(torch.tensor(v['fh'][None]),torch.tensor(g[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy().tolist()
        want=int(spec['sign']+1);state=rc.direction_verdict(p,want,bool(ok))
        items.append(dict(task='direction',axis=axis,desired=want,span=[a,b],probabilities=p,state=state))
        states=[r['state']for r in items]
        status='difference_evidence'if 'difference'in states or'opposite_direction'in states else'supported_features_only'if states and all(s=='supported'for s in states)else'insufficient_evidence'
        computed[q['request_id']]=dict(card=q['card'],span=[a,b],status=status,evidence=items,completion=None)
        new_movements.append(dict(request_id=q['request_id'],clip_id=sid[0],kind=sid[1],span=[a,b],axis=axis,moving=side,
            geometry=g.tolist(),eligible=bool(ok),info=info,probabilities=p,source_sha256=d['sources'][sid]['sha256']))
    assert len(computed)==len(new_movements)==8
    out=[];execution=[]
    for r in d['new']['rows']:
        prior=d['oldmap'][rowkey(r)];conditions=copy.deepcopy(prior['conditions']);bc={c['card']:c for c in prior['conditions']['local_full']};cards=[]
        for c in r['conditions'][MODE]['cards']:
            s=span(c);k=requestkey(r['clip_id'],r['kind'],c['card'],s)
            if s==bc[c['card']]['span']:after=copy.deepcopy(bc[c['card']]);action='copy_same_local_full'
            elif k in d['saved']:after=copy.deepcopy(d['saved'][k]);action='reuse_saved_old_or_J'
            else:after=copy.deepcopy(computed[keytext(k)]);action='new_direction_only'
            assert after['span']==s and after['completion']is None
            cards.append(after);execution.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],old_span=bc[c['card']]['span'],new_span=s,action=action,request_id=keytext(k)))
        conditions[MODE]=cards;assert all(conditions[m]==prior['conditions'][m]for m in prior['conditions'])
        out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],conditions=conditions))
    assert execution==cfg['plan'];summary=copy.deepcopy(d['old']['summary'])
    for seed in[42,43,44]:
        for kind in sorted({r['kind']for r in out}):
            cs=[c for r in out if r['seed']==seed and r['kind']==kind for c in r['conditions'][MODE]];it=[v for c in cs for v in c['evidence']]
            summary.append(dict(seed=seed,condition=MODE,kind=kind,cards=len(cs),statuses=dict(collections.Counter(c['status']for c in cs)),
                shape=dict(collections.Counter(v['state']for v in it if v['task']=='shape')),direction=dict(collections.Counter(v['state']for v in it if v['task']=='direction'))))
    assert all(v['state']=='observation_unknown'for r in out if r['kind']=='all_hand_missing'for c in r['conditions'][MODE]for v in c['evidence']if v['task']=='shape')
    checkpins(cfg['pins']);stop()
    write(E/'RESULTS.json',dict(at=time.time(),status='G2_BLEND_CHANGED_SPAN_CAUSE_ISOLATION_NOT_ACCEPTANCE',rows=out,summary=summary,execution=execution,
        changed_cards=63,unique_changed_requests=48,saved_reused_requests=40,new_direction_requests=8,new_attribute_frames=0,attribute_layer_calls=0,
        new_movements=new_movements,old_conditions_exact=True,observer_seed=42,selector_seeds=[42,43,44],compute_and_preservation_check_seconds=time.perf_counter()-tick,
        timing_scope='Eight intervals from cached h/x plus result assembly and preservation hashing; not camera/RTMW/end-to-end latency.',frozen_sha256=sha(E/'FROZEN.json'),
        previous_pointwise_cache_preserved=True,no_pointwise_cache_lookup_needed=True,no_old_result_reinference=True,no_training=True,no_web=True,no_final8=True,semantic_completion=None,G2_pass=None,G3_pass=None))
    print('Saved8 new movement requests;0 attribute calls;old/J unchanged. Root gate review pending.',flush=True)

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1]in['freeze','run-reviewed']
        freeze()if sys.argv[1]=='freeze'else execute()
    except BaseException:
        E.mkdir(parents=True,exist_ok=True)
        (E/f'FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
