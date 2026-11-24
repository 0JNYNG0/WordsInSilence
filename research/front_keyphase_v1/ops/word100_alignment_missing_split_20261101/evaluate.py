"""Evaluate frozen outputs, never select a new path or run a model."""
from pathlib import Path
import sys,json,statistics,collections,copy
sys.dont_write_bytecode=True
import numpy as np
import run as driver
U=Path(__file__).resolve().parent;OLD=driver.OLD
old_eval=driver.load(OLD/'evaluate.py','old_evaluation_readonly')
read,new,sha=driver.read,driver.new,driver.sha

def local_axis(axis, frames):
    own=np.asarray(axis['own_observed'],bool)[frames]
    support=np.asarray(axis['matched_support'],bool)[frames]
    av=np.asarray(axis['mean_residual'],float)[frames]
    peak=np.asarray(axis['max_residual'],float)[frames]
    mixed=np.asarray(axis['mixed_counterpart_missing'],bool)[frames]
    missing_links=np.asarray(axis['missing_counterpart_link_count'],int)[frames]
    assert np.isfinite(av).sum()==support.sum()
    assert np.isfinite(peak).sum()==support.sum()
    assert not (support & ~own).any()
    return dict(possible=int(own.size),own_observed=int(own.sum()),shared=int(support.sum()),
        own_known_uncompared=int((own&~support).sum()),own_unknown=int((~own).sum()),
        mixed_support_cells=int(mixed.sum()),missing_counterpart_link_cells=int(missing_links.sum()),
        residual_mean_conditions_on_shared_links=True,shared_does_not_imply_all_links_observed=True,
        mean_residual_sum=float(np.nansum(av)),peak_residual_sum=float(np.nansum(peak)),
        supported_mean=float(np.nanmean(av)) if support.any() else None,
        supported_peak_mean=float(np.nanmean(peak)) if support.any() else None)

def main():
    driver.checkpins()
    import split_alignment as candidate
    tasks=read(U/'TASKS.json');prior=read(OLD/'EVALUATION.json')
    physical={r['alias']:r for r in read(old_eval.PHYS)['clips']}
    rows=copy.deepcopy(prior['pairs']);controls=copy.deepcopy(prior['controls'])
    # All earlier path metrics are reused verbatim; no old score reproduction.
    for c in controls:c['scope']='previous16'
    store=driver.previous.original.RuntimeStore(read(driver.previous.PREV/'REFERENCE_BANK.json'))
    shorts=[];accounting=[]
    for rec in read(U/'RESULT_MANIFEST.json')['results']:
        assert sha(rec['path'])==rec['sha256']
        d=read(rec['path']);p=d['task'];ev=d['evidence'];path=d['path']
        full=({i for i,j in path}==set(range(d['query_frames'])) and {j for i,j in path}==set(range(d['reference_frames'])))
        assert full
        for g,axes in ev['groups'].items():
            for side,n in [('query',d['query_frames']),('reference',d['reference_frames'])]:
                a=axes[side];v=local_axis(a,list(range(n)))
                assert v['shared']+v['own_known_uncompared']+v['own_unknown']==v['possible']
                accounting.append(dict(name=d['name'],group=g,side=side,**v))
        if d['kind']=='pair':
            audit=old_eval.phase_audit(physical[p['query']],physical[p['reference']],{'*':path})
            rows.append(dict(method='split',seed=d['seed'],query=p['query'],reference=p['reference'],role=p['role'],word_id=p['word_id'],
                source=rec['path'],source_sha256=rec['sha256'],audit=audit,full_scope=full,ambiguous=d['optimal_path_ambiguous'],seconds=d['elapsed_seconds']))
        else:
            audit=old_eval.control_audit(d,p)
            controls.append(dict(method='split',seed=d['seed'],alias=p['alias'],role=p['role'],control=p['control'],name=p['name'],scope=p['scope'],
                audit=audit,full_scope=full,ambiguous=d['optimal_path_ambiguous'],seconds=d['elapsed_seconds']))
            if p['control'] in ('short_shape_change_5f','left_short_shape_change_5f'):
                group='L_shape' if p['control'].startswith('left_') else 'R_shape';frames=p['metadata']['changed_frames']
                item=dict(alias=p['alias'],seed=d['seed'],scope=p['scope'],control=p['control'],changed_frames=frames,
                    actual_observed_changes=p['metadata']['actual_observed_changed_cells'],
                    new_query=local_axis(ev['groups'][group]['query'],frames),new_reference=local_axis(ev['groups'][group]['reference'],frames))
                if p['scope']=='previous16':
                    oldpath=read(OLD/'results'/f"control_{p['name']}_{d['seed']}.json")['path']
                    pp=copy.deepcopy(p);q=driver.control_input(pp,store.clip(p['alias']),d['seed']);q.pop('original_indices_evaluation_only',None)
                    r=driver.previous.select(store.clip(p['alias']),d['seed'])
                    old_evidence=candidate.readout(candidate.features(q),candidate.features(r),oldpath)
                    item['old_path_query']=local_axis(old_evidence['groups'][group]['query'],frames)
                    item['old_path_reference']=local_axis(old_evidence['groups'][group]['reference'],frames)
                shorts.append(item)
    summary={};byword={}
    for row in rows:
        for target,key in [(summary,f"{row['role']}/{row['method']}/{row['seed']}"),(byword,f"{row['role']}/{row['word_id']}/{row['method']}/{row['seed']}")]:
            s=target.setdefault(key,dict(pairs=0,full_reference_frames=0,certain_reference_frames=0,states=collections.Counter(),ambiguous_paths=0))
            s['pairs']+=1;s['full_reference_frames']+=row['audit']['reference_full_frames'];s['certain_reference_frames']+=row['audit']['expected_certain_frames']
            s['states'].update(row['audit']['states']);s['ambiguous_paths']+=bool(row.get('ambiguous',False))
    for group in (summary,byword):
        for s in group.values():s['same_kind_percent']=100*s['states']['same_observed_kind']/s['certain_reference_frames'];s['states']=dict(s['states'])
    cs={}
    for c in controls:
        key=f"{c['scope']}/{c['control']}/{c['method']}/{c['seed']}"
        s=cs.setdefault(key,dict(cases=0,errors=[],fullpaths=0,exact=[],ambiguous=0))
        s['cases']+=1;s['errors'].append(c['audit']['mean_original_frame_error']);s['exact'].append(c['audit']['exact_original_frame_fraction']);s['fullpaths']+=c['audit']['full_path'];s['ambiguous']+=bool(c.get('ambiguous',False))
    for s in cs.values():s['mean_original_frame_error']=statistics.mean(s['errors']);s['exact_original_frame_percent']=100*statistics.mean(s['exact'])
    scores=[summary[f'development/split/{s}']['same_kind_percent'] for s in driver.SEEDS]
    baseline=[summary[f'development/{m}/None']['same_kind_percent'] for m in ('global_dtw','relational_global')]
    regress=[]
    for w in sorted({p['word_id'] for p in tasks['pairs']}):
        best=max(byword[f'development/{w}/{m}/None']['same_kind_percent'] for m in ('global_dtw','relational_global'))
        for s in driver.SEEDS:regress.append(dict(word=w,seed=s,pp=byword[f'development/{w}/split/{s}']['same_kind_percent']-best))
    speed=[]
    for c in ('speed_075','speed_125'):
        old=cs[f'previous16/{c}/saved_relational_global/None']['mean_original_frame_error']
        for s in driver.SEEDS:speed.append(dict(control=c,seed=s,increase=cs[f'previous16/{c}/split/{s}']['mean_original_frame_error']-old))
    technical=dict(full_scope_preserved=all(c['full_scope'] for c in rows+controls if c['method']=='split'),
        all_axis_opportunities_accounted=True,
        identity_exact=all(cs[f'{scope}/identity/split/{s}']['mean_original_frame_error']==0 for scope in ('previous16','expanded32') for s in driver.SEEDS),
        right_mask_same_clip_exact=all(cs[f'{scope}/right_hand_unobserved/split/{s}']['mean_original_frame_error']==0 for scope in ('previous16','expanded32') for s in driver.SEEDS),
        short_changed_cells_accounted=all(x['new_query']['own_observed']==x['actual_observed_changes'] for x in shorts),
        normal_speed_error_increase_at_most1=all(x['increase']<=1 for x in speed))
    broad=dict(mean_gain_at_least5pp_vs_both=all(statistics.mean(scores)>=b+5 for b in baseline),every_seed_above_both=all(x>max(baseline) for x in scores),
        no_word_regression_over5pp=all(x['pp']>=-5 for x in regress))
    new(U/'AXIS_ACCOUNTING.json',dict(rows=accounting,short_change_diagnosis=shorts,new_path_search=0,new_inference=0))
    out=dict(status='ONE_FROZEN_CANDIDATE_EVALUATED',summary=summary,byword=byword,control_summary=cs,
        decision=dict(technical_gates=technical,broad_gates=broad,technical_pass=all(technical.values()),clear_development_improvement=all(technical.values()) and all(broad.values()),
            dev_scores=scores,baselines=baseline,word_regressions=regress,speed_changes=speed,adopted=False,unresolved_missing_island=True,educational_accuracy=None),
        pairs=rows,controls=controls,shorts=shorts,new_training=0,new_inference=0,
        limitations=['Historical4word same-kind metric, not exact repeated instance or educational truth','Additional32clips8words are feature-space controls, not new annotated signs',
            'Zero alignment residual is not an agreement score; local missing islands can still attract paths','Both-axis peak evidence is not an error verdict','No new video observation, classifiers, reference selection or tuning'])
    new(U/'EVALUATION.json',out)
    print(json.dumps(out['decision'],ensure_ascii=False))
if __name__=='__main__':main()
