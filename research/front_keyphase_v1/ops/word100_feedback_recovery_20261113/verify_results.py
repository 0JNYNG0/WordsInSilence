"""Independent saved-output audit. No training, model forward, or evaluation rerun."""
from pathlib import Path
from collections import Counter
import json,hashlib,math,statistics,traceback,time,sys
import numpy as np
import torch

U=Path(__file__).resolve().parent
P=U.parent/'word100_feature_expansion_20261113'
B=U.parent/'word100_shared_motion_learning_20261109'
S=U.parent/'word100_gru_semantic_finetune_20261109'
EXPECTED_FROZEN_SHA='347871fc62348edef0a72f01b79a75f7bb14b29585eab1a639b1a3bd03daf8a4'
CONDITIONS=['frozen','finetune'];SEEDS=[42,43,44]
FINE=['normal','upper','slow','fast','fine_missing_2','fine_missing_12']
LEGACY=['normal','upper','slow','fast','missing_2','missing_long','reversed_main']
checks=[];failures=[]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def check(ok,what,detail=None):
    checks.append(what)
    if not bool(ok):failures.append(dict(check=what,detail=detail))
def eq(a,b,path):
    if isinstance(a,dict) and isinstance(b,dict):
        check(set(a)==set(b),path+'.keys',dict(left=list(a),right=list(b)))
        for k in set(a)&set(b):eq(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,list) and isinstance(b,list):
        check(len(a)==len(b),path+'.length',[len(a),len(b)])
        for i,(x,y) in enumerate(zip(a,b)):eq(x,y,path+f'[{i}]')
    elif isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
        check(math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),path,[a,b])
    else:check(a==b,path,[a,b])
def metric(records):
    cm=[[0]*3 for _ in range(3)]
    accepted=0;accepted_correct=0
    for r in records:
        t=int(r['target']);p=int(r['pred'])
        check(0<=t<3 and 0<=p<3,'state_range')
        if not(0<=t<3 and 0<=p<3):continue
        cm[t][p]+=1
        if r['accepted']:
            accepted+=1;accepted_correct+=int(t==p)
    n=sum(map(sum,cm));correct=sum(cm[k][k] for k in range(3));f1=[]
    for k in range(3):
        den=sum(cm[k])+sum(cm[j][k] for j in range(3))
        f1.append(2*cm[k][k]/den if den else None)
    usable=[v for v in f1 if v is not None]
    return dict(total=n,correct=correct,accuracy=correct/n if n else None,
      macro_f1=sum(usable)/len(usable) if usable else None,state_f1=f1,confusion=cm,
      accepted=accepted,accepted_correct=accepted_correct,withheld=n-accepted)
def legacy_metric(records):
    out={}
    for task in sorted({r['task'] for r in records}):
        rs=[r for r in records if r['task']==task]
        ac=[r for r in rs if r['eligible'] and r['confident']]
        out[task]=dict(n=len(rs),raw_correct=sum(r['target']==r['pred'] for r in rs),
          accepted=len(ac),accepted_correct=sum(r['target']==r['pred'] for r in ac))
    return out
def expected_mask(cache,variant):
    m=cache['hand'][:,:,42:]>0.5;n=len(m);ids=np.arange(n)
    if variant=='upper':
        vis=cache['x_upper'][:,186:237]>.5
        for side,start in enumerate([9,30]):
            v=vis[:,start:start+21];m[:,side]&=v
            m[:,side]&=v[:,[0,5,9,13,17]].all(1)[:,None]
    if variant in ['slow','fast']:
        ids=np.rint(np.linspace(0,n-1,round(n*(1.25 if variant=='slow' else .75)))).astype(int);m=m[ids]
    if variant in ['fine_missing_2','fine_missing_12']:
        first=int(np.flatnonzero((cache['fy']>=0).any((1,2)))[0]);gap=2 if variant=='fine_missing_2' else 12
        start=max(0,first-gap//2);m[start:min(n,start+gap)]=False
    return m,ids
def check_acceptance(r,cache,mask,ids):
    probs=np.asarray(r['probability'],float)
    check(probs.shape==(3,) and np.isfinite(probs).all(),'finite_three_state_probs')
    check(abs(probs.sum()-1)<1e-5 and (probs>=0).all() and (probs<=1).all(),'probability_simplex')
    check(int(np.argmax(probs))==r['pred'],'saved_argmax')
    check(bool(r['accepted'])==bool(r['eligible'] and probs[r['pred']]>=.85),'fixed_acceptance')
    f=r['finger'];side=0 if r['side']=='L' else 1
    ii=int(np.flatnonzero(cache['frame_id']==r['frame'])[0])
    at=int(np.argmin(abs(ids-ii)));points=[0]+list(range(1+f*4,5+f*4))
    check(bool(mask[at,side,points].all())==r['eligible'],'eligible_recomputed_from_input_mask')
    check(int(cache['fy'][ii,side,f])==r['target'],'fine_truth_native')
    check(int(cache['timestamp_us'][ii])==r['pts_us'],'fine_native_pts')
def checkpoints(name,cfg):
    condition,seed=name.rsplit('_',1);d=U/'runs'/name;c=read(d/'COMPLETE.json')
    check(c['checkpoint_sha256']==sha(d/'model.pt'),name+'.checkpoint_sha')
    check(c['frozen_sha256']==sha(U/'FROZEN.json'),name+'.frozen_sha')
    check(c['condition']==condition and c['seed']==int(seed),name+'.identity')
    check(c['shared100_seen']==899 and c['development_training'] is False,name+'.FIT_scope')
    check([h['epoch'] for h in c['history']]==list(range(1,13)),name+'.fixed_12_epochs')
    check(all(math.isfinite(float(v)) for h in c['history'] for k,v in h.items() if isinstance(v,(int,float))),name+'.finite_history')
    base=torch.load(B/f'runs/shared100_{seed}/model.pt',map_location='cpu',weights_only=True)
    now=torch.load(d/'model.pt',map_location='cpu',weights_only=True)
    changed=[];same=[]
    for k,v in base.items():
        check(k in now,name+'.base_key_present',k)
        if k not in now:continue
        equal=torch.equal(v,now[k]);(same if equal else changed).append(k)
        if condition=='frozen':check(equal,name+'.frozen_all_base_equal',k)
        elif not k.startswith(('base.input','base.temporal','base.context')):
            check(equal,name+'.fixed_heads_buffers_equal',k)
    if condition=='finetune':
        check(any(k.startswith('base.temporal') for k in changed),name+'.temporal_updated')
        check(any(k.startswith('base.context') for k in changed),name+'.GRU_updated')
    delta={}
    for prefix in ['base.input','base.temporal','base.context','base.attributes','base.stage_query','base.prototype','word_head']:
        delta[prefix]=float(sum((now[k]-v).float().square().sum() for k,v in base.items() if k.startswith(prefix)).sqrt())
    eq(delta,c['deltas'],name+'.saved_delta')
    return dict(name=name,checkpoint_sha256=c['checkpoint_sha256'],base_changed_keys=changed,
      original_heads_and_buffers_preserved=not any(not k.startswith(('base.input','base.temporal','base.context')) for k in changed))
def main():
    assert not (U/'RESULT_VERIFIED.json').exists(),'Completed verification must not be silently overwritten'
    cfg=read(U/'FROZEN.json');ev=read(U/'EVALUATION_COMPLETE.json')
    check(sha(U/'FROZEN.json')==EXPECTED_FROZEN_SHA,'specific_approved_freeze_sha')
    check(cfg['conditions']==CONDITIONS and cfg['seeds']==SEEDS and cfg['max_runs']==6,'fixed_six_runs')
    for p,h in {**cfg['pins'],**cfg['protected']}.items():check(sha(p)==h,'frozen_protected_pin',p)
    for p,h in read(U/'EVAL_FROZEN.json')['pins'].items():check(sha(p)==h,'evaluation_pin',p)
    expected_names=[f'{c}_{s}' for c in CONDITIONS for s in SEEDS]
    actual_names=sorted(p.parent.name for p in (U/'runs').glob('*/COMPLETE.json'))
    check(actual_names==sorted(expected_names),'exact_six_completed_runs',actual_names)
    tc=read(U/'TRAINING_COMPLETE.json')
    check(tc['status']=='SIX_FIXED_RUNS_COMPLETE' and len(tc['runs'])==6,'training_complete_record')
    check(set(ev['models'])==set(expected_names),'evaluation_six_models')
    info=read(U/'DATA_READY.json');prev={r['clip_id']:r for r in read(P/'DATA_READY.json')['rows']}
    fit=[m for m in info['rows'] if m['role']=='fit' and m['origin']=='common100']
    dev=[m for m in info['rows'] if m['role']=='development' and m['origin']=='common100']
    check(len(fit)==899 and len(dev)==300,'common899_DEV300')
    check(len({m['class_index'] for m in fit})==100,'common100_labels')
    check({m['person'] for m in fit}<=set(range(1,10)) and {m['person'] for m in dev}=={11,12,13},'record_scope')
    check(not {m['clip_id'] for m in fit}&{m['clip_id'] for m in dev},'FIT_DEV_disjoint')
    cache={};gt={}
    for m in dev:
        check(m==prev[m['clip_id']],'development_metadata_unchanged',m['clip_id'])
        check(sha(m['path'])==m['sha256'],'dev_cache_sha',m['clip_id'])
        with np.load(m['path'],allow_pickle=False) as z:cache[m['clip_id']]={k:z[k].copy() for k in z.files}
        z=cache[m['clip_id']]
        for i,s,f in zip(*np.where(z['fy']>=0)):
            gt[(m['clip_id'],int(z['frame_id'][i]),'L' if s==0 else 'R',int(f))]=int(z['fy'][i,s,f])
    check(len(gt)==389 and len({k[0] for k in gt})==36,'fixed_fine36_clips389_cells')
    class_truth={m['clip_id']:m['class_index'] for m in dev};metadata={m['clip_id']:m for m in dev}
    groups={s:i for i,g in enumerate(read(S/'FROZEN.json')['semantic_stage_groups']) for s in g}
    expanded={}
    for r in read(B/'EXTRA_EVAL_LABELS.json')['rows']:
        for j,t in enumerate(r['targets']):
            if t is not None:expanded[(r['clip_id'],r['frame'],j)]=t
    recomputed={};checkpoint_checks=[]
    for name in expected_names:
        checkpoint_checks.append(checkpoints(name,cfg))
        score=read(U/'scores'/f'{name}.json')
        fingers=read(U/'scores'/f'{name}_fingers.json')
        legacy=read(U/'scores'/f'{name}_legacy.json')
        words=read(U/'scores'/f'{name}_words.json')
        check(set(r['variant'] for r in fingers)==set(FINE),name+'.fine_variants')
        byvariant={}
        for variant in FINE:
            rr=[r for r in fingers if r['variant']==variant]
            masks={cid:expected_mask(cache[cid],variant) for cid in {r['clip_id'] for r in rr}}
            keys=[(r['clip_id'],r['frame'],r['side'],r['finger']) for r in rr]
            check(len(rr)==389 and len(set(keys))==389 and set(keys)==set(gt),name+'.'+variant+'.all_original_truth')
            for r in rr:
                check(r['target']==gt.get((r['clip_id'],r['frame'],r['side'],r['finger'])),name+'.target_from_immutable_gold')
                check(r['word']==metadata[r['clip_id']]['word_id'] and r['person']==metadata[r['clip_id']]['person'],name+'.fine_source_metadata')
                check_acceptance(r,cache[r['clip_id']],*masks[r['clip_id']])
            byvariant[variant]=metric(rr)
        normal=[r for r in fingers if r['variant']=='normal']
        byhand={s:metric([r for r in normal if r['side']==s]) for s in ['L','R']}
        byrecord={str(p):metric([r for r in normal if r['person']==p]) for p in [11,12,13]}
        byword={w:metric([r for r in normal if r['word']==w]) for w in sorted({r['word'] for r in normal})}
        byfinger={str(f):metric([r for r in normal if r['finger']==f]) for f in range(5)}
        for r in legacy:
            check(r['variant'] in LEGACY,name+'.legacy_variant')
            task=r['task'];cid=r['clip_id'];z=cache[cid]
            if task=='normal_order':
                check(r['target']==int(r['variant']!='reversed_main'),name+'.order_test_truth')
                continue
            ix=np.flatnonzero(z['frame_id']==r['frame']);check(len(ix)==1,name+'.legacy_native_frame')
            if not len(ix):continue
            i=int(ix[0])
            if task=='expanded_shape':truth=expanded[(cid,r['frame'],r['feature'])]
            elif task=='reviewed_shape':
                check(z['direct_ym'][i,r['feature']]>0,name+'.reviewed_truth_present')
                truth=z['y'][i,r['feature']]
            elif task=='original_shape_labels':
                check(z['old_ym'][i,r['feature']]>0,name+'.old_truth_present')
                truth=z['old_y'][i,r['feature']]
            elif task=='semantic_stage':truth=groups[int(z['stage_gold'][i])]
            else:raise AssertionError('Unexpected legacy task '+task)
            check(r['target']==truth,name+'.legacy_truth_not_generated')
            if 'probability' in r:
                check(r['pred']==int(r['probability']>=.5),name+'.legacy_threshold')
                check(r['confident']==bool(r['probability']<=.35 or r['probability']>=.65),name+'.legacy_confidence')
        old={v:legacy_metric([r for r in legacy if r['variant']==v]) for v in LEGACY}
        check(len(words)==300 and len({r['clip_id'] for r in words})==300 and {r['clip_id'] for r in words}==set(class_truth),name+'.word300_denominator')
        for r in words:check(r['target']==class_truth[r['clip_id']],name+'.word_target')
        word=dict(correct=sum(r['pred']==r['target'] for r in words),total=300)
        l=read(U/'scores'/f'{name}_latency.json');times=l['seconds']
        check(len(times)==36 and len(set(l['clip_ids']))==36 and set(l['clip_ids'])=={k[0] for k in gt},name+'.latency36_distinct_clips')
        check(all(math.isfinite(t) and t>=0 for t in times),name+'.finite_latencies')
        latency=dict(p50_seconds=statistics.median(times),p95_seconds=float(np.percentile(times,95)),clips=36,scope=l['scope'])
        result=dict(finger_by_variant=byvariant,finger_by_hand=byhand,finger_by_record=byrecord,
          finger_by_word=byword,finger_by_finger=byfinger,legacy=old,word100=word,latency=latency)
        eq(result,score,name+'.score_independent_recalculation')
        eq(result,ev['models'][name],name+'.aggregate_independent_recalculation')
        recomputed[name]=result
    ret=[];gates=[];scope=[]
    for condition in CONDITIONS:
      for seed in SEEDS:
        name=f'{condition}_{seed}';r=recomputed[name];baseline=read(B/'scores'/f'shared100_{seed}.json');a=read(P/'scores'/f'A_{seed}.json')
        loss=[]
        for v, tasks in r['legacy'].items():
            check(set(tasks)==set(baseline['by_variant'][v]),name+'.baseline_task_denominators')
            for task,q in tasks.items():
                oq=baseline['by_variant'][v][task];check(q['n']==oq['n'],name+'.baseline_observation_denominator')
                if q['raw_correct']<oq['raw_correct']:loss.append(dict(variant=v,task=task,before=oq['raw_correct'],after=q['raw_correct'],n=q['n']))
        rr=dict(condition=condition,seed=seed,legacy_failures=loss,word100_before=baseline['aux_word']['correct'],word100_after=r['word100']['correct'],full_preservation=not loss and r['word100']['correct']>=baseline['aux_word']['correct'])
        ret.append(rr);cn=r['finger_by_variant']['normal'];an=a['finger_by_variant']['normal'];variant_fail=[]
        for v in FINE:
            cv=r['finger_by_variant'][v];av=a['finger_by_variant'][v]
            check(cv['total']==av['total']==389,name+'.historic_fine_denominator')
            if cv['correct']<av['correct'] or any((x or 0)<(y or 0) for x,y in zip(cv['state_f1'],av['state_f1'])):variant_fail.append(dict(variant=v,before=av,after=cv))
        gain=cn['macro_f1']>=an['macro_f1']+.01
        hand=all(r['finger_by_hand'][s]['accuracy']>=a['finger_by_hand'][s]['accuracy'] for s in ['L','R'])
        state=all((cv or 0)>=(av or 0) for av,cv in zip(an['state_f1'],cn['state_f1']))
        cover=cn['withheld']<=an['withheld']
        gates.append(dict(condition=condition,seed=seed,consistent_fine_gain=gain,hands_preserved=hand,states_preserved=state,normal_abstention_not_increased=cover,variant_failures=variant_fail,legacy_preserved=rr['full_preservation'],pass_candidate=bool(gain and hand and state and cover and rr['full_preservation'] and not variant_fail)))
    passed={c:all(x['pass_candidate'] for x in gates if x['condition']==c) for c in CONDITIONS}
    eq(ret,ev['retention_all_conditions'],'retention_all_conditions')
    eq(gates,ev['gates'],'all_gates')
    eq(passed,ev['condition_pass'],'condition_pass')
    check(any(passed.values())==ev['any_pass'],'any_pass')
    check(ev['no_production_promotion'] is True and ev['development_historically_exposed'] is True,'honest_adoption_and_historical_scope')
    check('historical' in cfg['eval']['status'].lower() and 'no extra word name loss' in cfg['architecture'],'frozen_scope_text')
    check(all('excludes' in r['latency']['scope'] and 'UI' in r['latency']['scope'] for r in recomputed.values()),'timing_excludes_real_user_wait')
    for p,h in {**cfg['pins'],**cfg['protected']}.items():check(sha(p)==h,'postaudit_frozen_protected_pin',p)
    report=dict(status='PASS' if not failures else 'FAIL',verification='Independent arithmetic from saved records plus checkpoint/file verification; no neural inference',
      checks=len(checks),failures=failures,frozen_sha256=sha(U/'FROZEN.json'),script_sha256=sha(__file__),conditions=CONDITIONS,seeds=SEEDS,
      fine_denominator_per_variant=389,word_denominator=300,model_checkpoints=checkpoint_checks,condition_pass=passed,
      gates=gates,retention=ret,results=recomputed,
      limitations=['Historical DEV11..13, not new untouched-user evaluation','Physical finger labels, not educational acceptable-error boundaries',
        'Thumb B/F, spread and exact contact unsupported','Left middle/ring/little B FIT evidence remains one-expression',
        'No automatic web adoption; phase binding/motion/order/omission require separate graph gates',
        'Cached neural latency excludes capture/detector/queue/network/UI','Same Codex separate audit, not independent human or KSL expert'])
    write(U/'RESULT_VERIFIED.json',report)
    print(json.dumps(dict(status=report['status'],checks=len(checks),failures=len(failures),condition_pass=passed),ensure_ascii=False))
    if failures:raise SystemExit(1)
if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except BaseException:
        write(U/f'FAILURE_VERIFY_{time.time_ns()}.json',dict(status='ERROR',traceback=traceback.format_exc(),no_inference=True))
        raise

