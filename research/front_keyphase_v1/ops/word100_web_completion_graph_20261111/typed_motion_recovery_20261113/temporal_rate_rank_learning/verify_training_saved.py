"""Saved training audit: parameters/arrays/RNG counts only, NEVER model forward."""
from pathlib import Path
import collections,hashlib,json,time,traceback
import numpy as np
import torch
K=Path(__file__).resolve().parent;H=K.parent;G=H.parent;J=H/'splice_rank_learning';L=G/'candidate_description_supervision/local_rank_learning'
FAMILIES=[2,3,4,5,7]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def key(r):return r['clip_id'],r['view'],r['reverse'],r['family']

def main():
    assert (K/'TRAINING_COMPLETE.json').exists() and not(K/'TRAINING_SAVED_VERIFICATION.json').exists()
    cfg=read(K/'TRAINING_FROZEN.json');trained=read(K/'TRAINING_COMPLETE.json');meta=read(K/'PREPARATION_INPUTS_COMPLETE.json')
    checks=collections.Counter()
    def ck(ok,label):assert bool(ok),label;checks[label]+=1
    for p,h in cfg['pins'].items():ck(sha(p)==h,'training_pin_unchanged')
    protected=read(G/'PROTECTIONS.json');ck(len(protected)==477,'protected_manifest_count')
    for p,h in protected.items():ck(sha(p)==h,'protected_file_unchanged')
    ck((cfg['steps'],cfg['lr'],cfg['weight_decay'])==(1200,.001,.001),'fixed_recipe')
    ck(trained['status']=='FIXED_RATE_HEAD_TRAINING_COMPLETE_NOT_ADOPTED','completed_not_adopted')
    allbags=[];original=[];variants=collections.defaultdict(lambda:collections.defaultdict(list))
    def item(r,kind,rate):
        z=np.load(r['path']);y=z['y'];ck(y.dtype==np.bool_ and y.any()and(~y).any(),'fit_rank_labels_valid')
        ck(len(y)==r['positive']+r['negative'] and int(y.sum())==r['positive'],'fit_label_counts')
        if 'rank_mask'in z.files:ck(z['rank_mask'].all(),'all_saved_targets_known')
        d=dict(row=r,kind=kind,rate=rate,y=y,neg=np.flatnonzero(~y),identity=(r['path'],kind,rate));allbags.append(d);return d
    for r in meta['replay_original']:
        d=item(r,'normal',1.);original.append(d)
        if r['family']in[2,3]:variants[key(r)]['normal'].append(d)
    for r in meta['replay_splice']:variants[key(r)]['splice'].append(item(r,'splice',1.))
    held=[]
    for r in meta['bags']:
        if r['skipped']:held.append(r);continue
        variants[key(r)][r['base_kind']].append(item(r,r['base_kind'],r['rate']))
    ck(len(allbags)==460 and len(original)==178 and len(variants)==72 and len(held)==78,'exact_training_data_denominator')
    ck({d['row']['person']for d in allbags}==set(range(2,8)) and all(d['row']['split']=='train'for d in allbags),'FIT_only_people_split')
    groups={f:[i for i,d in enumerate(original)if d['row']['family']==f]for f in FAMILIES}
    ck({f:len(v)for f,v in groups.items()}=={2:36,3:36,4:34,5:36,7:36},'old_family_source_groups')
    for v in variants.values():
        ck(set(v)=={'normal','splice'},'paired_branches')
        for branch in v.values():
            rates=[d['rate']for d in branch];ck(1. in rates and len(rates)==len(set(rates)) and set(rates)<={.8,1.,1.2},'unique_available_rates')
    jtrain=read(J/'TRAINING_COMPLETE.json');jmodels={r['seed']:r for r in jtrain['rows']}
    oldmeta=read(L/'INPUTS_COMPLETE.json');oldidx={r['path']:i for i,r in enumerate(oldmeta['bags'])}
    oldruns={r['seed']:r for r in read(L/'TRAINING_COMPLETE.json')['rows']if r['condition']=='local_full'}
    ck({r['seed']for r in trained['rows']}=={42,43,44}and len(trained['rows'])==3,'all_matching_seeds')
    rows=[];sampling=[];models=[]
    for run in trained['rows']:
        seed=run['seed'];parent=jmodels[seed]
        ck(run['initial_model']==parent['model']and run['initial_model_sha256']==parent['model_sha256'],'matching_J_initial_reference')
        ck(sha(run['initial_model'])==run['initial_model_sha256']and sha(run['model'])==run['model_sha256'],'initial_final_checkpoint_pins')
        st=torch.load(run['model'],weights_only=True,map_location='cpu');before=torch.load(parent['model'],weights_only=True,map_location='cpu')
        ck(set(st)==set(before),'same_parameter_schema')
        for k,v in st.items():
            ck(v.shape==before[k].shape and v.dtype==before[k].dtype and bool(torch.isfinite(v).all()),'parameter_shape_dtype_finite')
        ck(torch.equal(st['mean'],before['mean'])and torch.equal(st['std'],before['std']),'matching_J_normalization_exact')
        changed=[k for k in st if not torch.equal(st[k],before[k])]
        ck(bool(changed)and all(k.startswith('head.')for k in changed),'head_only_parameter_changes')
        ck(run['steps']==1200 and [r['step']for r in run['log']]==[1,301,601,901,1200]and all(np.isfinite(r['loss'])for r in run['log']),'fixed_final_steps_finite_log')
        models.append(dict(seed=seed,initial_model=run['initial_model'],model=run['model'],changed_keys=changed,
            normalization_exact=True,initial_all_weight_copy_evidence='Pinned executed train.py loads matching parent and asserts every state_dict tensor equal before optimizer; final checkpoint cannot independently reconstruct historical initialization.'))
        # Replay ONLY RNG/label-index arithmetic, never x, network or optimizer.
        rng=np.random.default_rng(seed);counts=collections.Counter();family_mass=collections.Counter();branch_mass=collections.Counter()
        for step in range(1200):
            total=0.
            for family in FAMILIES:
                for oi in rng.choice(groups[family],2,replace=False):
                    base=original[int(oi)]
                    if family in[2,3]:
                        branch=variants[key(base['row'])]
                        selected=[(branch[b][int(rng.integers(len(branch[b])))],.5)for b in['normal','splice']]
                    else:selected=[(base,1.)]
                    for d,w in selected:
                        rng.choice(d['neg'],min(len(d['neg']),56),replace=False)
                        counts[d['identity']]+=1;mass=w/10;total+=mass;family_mass[family]+=mass
                        branch_mass[family,d['kind']]+=mass
            ck(np.isclose(total,1),'unit_loss_mass_each_step')
        for family in FAMILIES:ck(np.isclose(family_mass[family],240),'equal_family_total_mass')
        for family in[2,3]:
            ck(np.isclose(branch_mass[family,'normal'],120)and np.isclose(branch_mass[family,'splice'],120),'equal_normal_splice_mass')
        ck(sha(run['fit_predictions'])==run['fit_predictions_sha256'],'savedFIT_prediction_pin')
        pred=np.load(run['fit_predictions']);index=read(Path(run['model']).parent/'FIT_PREDICTION_INDEX.json')
        ck(len(index)==460 and set(pred.files)=={f'bag_{i}'for i in range(460)},'savedFIT_full_index')
        jpred=np.load(parent['fit_predictions']);ji=read(Path(parent['model']).parent/'FIT_PREDICTION_INDEX.json');jlookup={r['path']:r['key']for r in ji}
        oldrun=oldruns[seed];op=np.load(oldrun['predictions'])
        ck(sha(parent['fit_predictions'])==parent['fit_predictions_sha256']and sha(oldrun['predictions'])==oldrun['predictions_sha256'],'saved_J_original_prediction_pins')
        for i,(r,d)in enumerate(zip(index,allbags)):
            ck(r['key']==f'bag_{i}'and r['path']==d['row']['path']and r['sha256']==d['row']['sha256']and r['family']==d['row']['family']and r['base_kind']==d['kind']and r['rate']==d['rate'],'FIT_index_identity')
            ck(r['sampled_steps']==counts[d['identity']],'exact_RNG_sample_count')
            p=pred[r['key']];ck(p.shape==(len(d['y']),5)and np.isfinite(p).all(),'FIT_prediction_shape_finite')
            j=FAMILIES.index(r['family']);score=p[:,j];y=d['y'];top=int(score.argmax())
            out=dict(seed=seed,path=r['path'],clip_id=d['row']['clip_id'],view=d['row']['view'],reverse=d['row']['reverse'],
                family=r['family'],base_kind=d['kind'],rate=d['rate'],sampled_steps=r['sampled_steps'],positive=int(y.sum()),negative=int((~y).sum()),
                top_full_positive=bool(y[top]),positive_minus_negative_margin=float(score[y].max()-score[~y].max()))
            if d['rate']==1.:
                jp=jpred[jlookup[r['path']]][:,j];ck(len(jp)==len(y),'same_rate1_J_candidate_count')
                out['J_top_full_positive']=bool(y[int(jp.argmax())]);out['J_margin']=float(jp[y].max()-jp[~y].max())
                if d['kind']=='normal':
                    oo=op[f"logits_{oldidx[r['path']]}"][:,j];ck(len(oo)==len(y),'same_rate1_original_candidate_count')
                    out['original_top_full_positive']=bool(y[int(oo.argmax())]);out['original_margin']=float(oo[y].max()-oo[~y].max())
            rows.append(out)
        sampled_flag=all(counts[d['identity']]>0 for d in allbags)
        ck(run['all_training_variants_sampled']==sampled_flag,'sampling_flag_arithmetic_only')
        sampling.append(dict(seed=seed,all_bags_sampled=sampled_flag,zero_draw_bags=[d['identity']for d in allbags if not counts[d['identity']]],
            minimum_bag_draws=min(counts[d['identity']]for d in allbags),maximum_bag_draws=max(counts[d['identity']]for d in allbags),
            total_bag_draws=sum(counts.values()),family_weight=dict(family_mass),
            scope='Finite-sampling diagnostic only. No retry or seed change if a bag is unvisited. Does not imply every negative candidate/frame was used.'))
    summary=[]
    for seed in[42,43,44]:
        for kind,rate,family in sorted({(r['base_kind'],r['rate'],r['family'])for r in rows}):
            rr=[r for r in rows if(r['seed'],r['base_kind'],r['rate'],r['family'])==(seed,kind,rate,family)]
            summary.append(dict(seed=seed,base_kind=kind,rate=rate,family=family,bags=len(rr),top_full_positive=sum(r['top_full_positive']for r in rr),
                J_top_full_positive=sum(r['J_top_full_positive']for r in rr)if rate==1 else None,
                original_top_full_positive=sum(r['original_top_full_positive']for r in rr)if rate==1 and kind=='normal'else None,
                minimum_margin=min(r['positive_minus_negative_margin']for r in rr)))
    write(K/'FIT_TRAINING_DIAGNOSIS.json',dict(status='FIT_FITNESS_ONLY_NOT_GENERALIZATION',summary=summary,rows=rows,sampling=sampling,
        comparison='Same saved rate1 bag y/spans and old/J/K saved logits arithmetic. New rates have no newly run original/J comparator.',
        held_denominator=78,no_retraining_required_from_sampling_flag=True,no_model_inference=True))
    write(K/'TRAINING_SAVED_VERIFICATION.json',dict(status='SAVED_TRAINING_CONTRACT_PASS_NOT_ADOPTED',at=time.time(),checks=dict(checks),total_checks=sum(checks.values()),
        training_code_sha256=sha(K/'train.py'),training_complete_sha256=sha(K/'TRAINING_COMPLETE.json'),protected_files=477,models=models,
        fit_prediction_bags=460*3,source_rate_rng_replayed_without_learning=True,held_cases_preserved=78,
        no_forward_pass=True,no_optimizer=True,no_training=True,old_artifacts_unchanged=True,
        verifier='Delegated Codex saved-data arithmetic; distinct from root implementation/execution and human/KSL truth.',script_sha256=sha(__file__)))
    print('Saved training verification PASS',sum(checks.values()),'477 protected;1380 FIT prediction bags; no inference')
if __name__=='__main__':
    try:main()
    except BaseException:
        write(K/f'TRAIN_VERIFICATION_FAILURE_{time.time_ns()}.json',dict(traceback=traceback.format_exc(),no_model_inference=True));raise
