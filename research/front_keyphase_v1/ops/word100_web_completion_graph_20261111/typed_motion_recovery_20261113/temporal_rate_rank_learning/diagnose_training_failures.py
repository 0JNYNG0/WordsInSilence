"""FIT-only endpoint/sampling/consistency diagnosis of saved K failures.

No DEV data, model construction/forward, optimization, feature recomputation,
label rewrite, sampler change or training restart.
"""
from pathlib import Path
import collections,json,hashlib
import numpy as np
K=Path(__file__).resolve().parent;H=K.parent;J=H/'splice_rank_learning';F=[2,3,4,5,7]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r):return r['clip_id'],r['view'],r['reverse'],r['family']
def main():
    assert not(K/'FIT_FAILURE_CAUSES.json').exists()
    meta=read(K/'PREPARATION_INPUTS_COMPLETE.json');cfg=read(K/'FROZEN.json');fitness=read(K/'FIT_TRAINING_DIAGNOSIS.json')
    runs={r['seed']:r for r in read(K/'TRAINING_COMPLETE.json')['rows']};jruns={r['seed']:r for r in read(J/'TRAINING_COMPLETE.json')['rows']}
    allmeta=[dict(r,base_kind='normal',rate=1.)for r in meta['replay_original']]+[dict(r,base_kind='splice',rate=1.)for r in meta['replay_splice']]+[r for r in meta['bags']if not r['skipped']]
    bypath={r['path']:r for r in allmeta};allcases={(*key(r),r['base_kind'],r['rate']):r for r in allmeta}
    for r in meta['bags']:
        if r['skipped']:allcases[(*key(r),r['base_kind'],r['rate'])]=r
    source={(r['clip_id'],r['view'],r['reverse']):r for r in cfg['sequences']}
    splice={(r['clip_id'],r['view'],r['reverse']):r for r in cfg['splice_sequences']}
    pcache={};ccache={};bags={};contexts={}
    def bag(path):
        if path not in bags:
            z=np.load(path);bags[path]={k:z[k]for k in ['y','spans']}
        return bags[path]
    def logits(seed,path):
        if seed not in pcache:
            run=runs[seed];idx=read(Path(run['model']).parent/'FIT_PREDICTION_INDEX.json')
            pcache[seed]=(np.load(run['fit_predictions']),{r['path']:r['key']for r in idx})
        p,idx=pcache[seed];return p[idx[path]][:,F.index(bypath[path]['family'])]
    def context(r):
        sid=(r['clip_id'],r['view'],r['reverse']);cid=(*sid,r['base_kind'],r['rate'])
        if cid in contexts:return contexts[cid]
        src=np.load(source[sid]['path']);j=np.load(splice[sid]['path']);n=len(j['x'])
        if r['rate']==1.:
            if r['base_kind']=='normal':m=np.arange(n);ss=j['source_index'][np.argsort(j['baseline_index'])];block=np.zeros(n,int)
            else:m=j['baseline_index'];ss=j['source_index'];block=np.searchsorted(j['join_right'],np.arange(n),side='right')
            base=np.arange(n)
        else:
            q=np.load(r['canonical_path']);m=q['baseline_index'];ss=q['source_index'];block=q['source_block_id'];base=q['base_query_index']
        contexts[cid]=dict(source=src,m=m,source_index=ss,block=block,base=base);return contexts[cid]
    def endpoint(r,span):
        c=context(r);a,b=map(int,span);m=c['m'][a:b+1];j=F.index(r['family']);src=c['source'];va=src['valid'][:,j].any(1)
        pos=(src['mask'][:,j]>0)&(src['y'][:,j]>0);neg=(src['mask'][:,j]>0)&(src['y'][:,j]==0)
        onset=np.flatnonzero(src['bands'][:,j,0]);offset=np.flatnonzero(src['bands'][:,j,1]);required=np.flatnonzero(pos);retained=np.intersect1d(required,np.unique(m))
        return dict(query_span=[a,b],base_query_endpoints=c['base'][[a,b]].tolist(),baseline_source_endpoints=c['m'][[a,b]].tolist(),
            original_source_endpoints=c['source_index'][[a,b]].tolist(),source_ids=c['source_index'][a:b+1].tolist(),
            source_onset_band=onset.tolist(),source_offset_band=offset.tolist(),onset_in_band=bool(src['bands'][m[0],j,0]),
            offset_in_band=bool(src['bands'][m[-1],j,1]),required_witnesses=required.tolist(),retained_witnesses=retained.tolist(),
            all_required_witnesses_retained=np.array_equal(required,retained),observed_other_negative_ids=np.intersect1d(np.flatnonzero(neg),np.unique(m)).tolist(),
            unique_observed_source_frames=int(len(np.unique(m[va[m]]))),body_coverage=float(va[m].mean()),body_endpoints_valid=bool(va[m[0]]and va[m[-1]]),
            crosses_known_join=bool(c['block'][a]!=c['block'][b]))
    failed=[r for r in fitness['rows']if not r['top_full_positive']];details=[];interesting={}
    for row in failed:
        r=bypath[row['path']];z=bag(row['path']);score=logits(row['seed'],row['path']);top=int(score.argmax());pi=np.flatnonzero(z['y']);best=int(pi[np.argmax(score[pi])])
        entry=dict(row,top_index=top,best_positive_index=best,top=endpoint(r,z['spans'][top]),best_positive=endpoint(r,z['spans'][best]),
            top_score=float(score[top]),best_positive_score=float(score[best]),top5=[dict(index=int(i),span=z['spans'][i].tolist(),full_positive=bool(z['y'][i]),score=float(score[i]))for i in np.argsort(-score)[:5]])
        interesting[row['seed'],row['path']]=top
        # Exact source/view/orientation/family siblings, with held rates explicit.
        siblings=[]
        for kind in ['normal','splice']:
            for rate in [1.,.8,1.2]:
                other=allcases[(*key(r),kind,rate)]
                if other.get('skipped',False):
                    siblings.append(dict(base_kind=kind,rate=rate,held=True,reasons=other['hold_reasons'],missing_witnesses=other['missing_witnesses']));continue
                zz=bag(other['path']);ss=logits(row['seed'],other['path']);ii=int(ss.argmax());pp=np.flatnonzero(zz['y']);bi=int(pp[np.argmax(ss[pp])])
                siblings.append(dict(base_kind=kind,rate=rate,held=False,top_full_positive=bool(zz['y'][ii]),
                    margin=float(ss[bi]-ss[~zz['y']].max()),top=endpoint(other,zz['spans'][ii]),best_positive=endpoint(other,zz['spans'][bi])))
        entry['same_source_variants']=siblings
        if row['rate']==1.:
            jr=jruns[row['seed']];jp=np.load(jr['fit_predictions']);ji=read(Path(jr['model']).parent/'FIT_PREDICTION_INDEX.json');jk=next(t['key']for t in ji if t['path']==row['path'])
            js=jp[jk][:,F.index(row['family'])];jb=int(pi[np.argmax(js[pi])])
            entry['J_same_bag']=dict(top_candidate_score=float(js[top]),best_positive_score=float(js[jb]),
                best_positive_span=z['spans'][jb].tolist(),positive_margin_over_K_top=float(js[jb]-js[top]),
                rank_of_K_top=int((js>js[top]).sum()+1))
        details.append(entry)
    # Exact negative-sampling exposure of each observed failure, not a new fit.
    original=allmeta[:178];variants=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in allmeta:
        if r['family']in[2,3]:variants[key(r)][r['base_kind']].append(r)
    groups={f:[i for i,r in enumerate(original)if r['family']==f]for f in F};exposure=collections.Counter();visits=collections.Counter()
    neg={r['path']:np.flatnonzero(~bag(r['path'])['y'])for r in allmeta}
    for seed in sorted({r['seed']for r in failed}):
        rng=np.random.default_rng(seed)
        for _ in range(1200):
            for f in F:
                for i in rng.choice(groups[f],2,replace=False):
                    r=original[int(i)]
                    if f in[2,3]:
                        v=variants[key(r)];selected=[v[b][int(rng.integers(len(v[b])))]for b in['normal','splice']]
                    else:selected=[r]
                    for r in selected:
                        drawn=rng.choice(neg[r['path']],min(len(neg[r['path']]),56),replace=False);k=seed,r['path']
                        if k in interesting:
                            visits[k]+=1;exposure[k]+=int(interesting[k]in drawn)
    for d in details:
        k=d['seed'],d['path'];assert visits[k]==d['sampled_steps']
        d['winning_negative_sampled_steps']=exposure[k]
        d['all_positive_sampled_steps']=visits[k]
    # Feasibility of same-source interval consistency, before choosing any loss.
    matching=[]
    for k,v in variants.items():
        for kind,rr in v.items():
            parent=next(r for r in rr if r['rate']==1.);p=bag(parent['path']);pm={tuple(s):bool(y)for s,y in zip(p['spans'],p['y'])}
            for r in rr:
                if r['rate']==1.:continue
                z=bag(r['path']);c=context(r);eq=collections.defaultdict(set)
                for s,y in zip(z['spans'],z['y']):eq[tuple(c['base'][s])].add(bool(y))
                common=set(pm)&set(eq);conflicts=[e for e in common if eq[e]!={pm[e]}]
                matching.append(dict(clip_id=k[0],view=k[1],reverse=k[2],family=k[3],base_kind=kind,rate=r['rate'],
                    rate1_selected_windows=len(pm),new_unique_source_endpoint_windows=len(eq),common_endpoint_windows=len(common),
                    common_full_positive_windows=sum(pm[e]for e in common),label_conflicts=len(conflicts),
                    limitation='Same endpoint source interval, but interior retained observations may differ. Shared endpoints alone do not prove equal evidence or justify raw-logit equality.'))
    assert not any(r['label_conflicts']for r in matching)
    report=dict(status='FIT_FAILURE_AND_CONSISTENCY_DIAGNOSIS_ONLY',failures=details,matching=matching,
        failure_count=len(details),rate1_parent_regressions=2,endpoint_label_conflicts=sum(r['label_conflicts']for r in matching),
        no_DEV=True,no_model_inference=True,no_training=True,no_label_rewrite=True,
        conclusions='All6 saved failures are normal-branch family2, upper/crop views. Analyze endpoint bands and retained witnesses separately from educational action correctness. Exact sample exposure and rate siblings below distinguish missing-label/sample causes from optimization/generalization hypotheses; they do not prove causality.',
        pins={str(p):sha(p)for p in [K/'TRAINING_COMPLETE.json',K/'FIT_TRAINING_DIAGNOSIS.json',K/'PREPARATION_INPUTS_COMPLETE.json',J/'TRAINING_COMPLETE.json',Path(__file__)]})
    write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write(K/'FIT_FAILURE_CAUSES.json',report)
    print('FIT saved failure diagnosis complete',len(details),'cases;',len(matching),'rate-pair groups; no label conflicts')
if __name__=='__main__':main()
