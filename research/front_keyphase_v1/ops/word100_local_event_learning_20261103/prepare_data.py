"""Build an immutable weak-bag dataset from completed direct reviews and caches.

No models are instantiated or loaded. Source video is decoded for native PTS
verification only; no pixels are extracted or new observations computed.
"""
import math
import time
import traceback
from pathlib import Path
from collections import Counter
import numpy as np
import av
from data import (U,O,P,M,R,N,V,OLD,AG,AC,WORDS,TARGETS,FIT_IDS,DEV_IDS,
                  read,sha,exclusive,geometry_valid,validate_role,observation_paths,
                  recipe_for,prepare_row,indices_for,evidence_for,composed_support)

def main():
    for filename in ['DATASET_FROZEN.json', 'INPUT_CHECKS.json']:
        if (U/filename).exists():
            raise FileExistsError('Completed artifact must not be overwritten: '+filename)
    reviews = [(U/(word+'_REVIEW.json')) for word in WORDS]
    if not all(p.is_file() for p in reviews):
        raise RuntimeError('Wait for all four completed direct-review files before preparation')
    started = time.perf_counter()
    checks = []
    pins = {}
    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise ValueError(name)
    def pin(path, expected=None):
        path = path.resolve()
        digest = sha(path)
        if expected is not None:
            check(str(path)+':sha256',digest == expected)
        if str(path) in pins:
            check(str(path)+':unchanged',pins[str(path)] == digest)
        pins[str(path)] = digest
        return digest
    initial_files = [U/'data.py',U/'prepare_data.py',U/'local_events.py',U/'PLAN_RESOLUTION.json',
                     U/'REVIEW_SCOPE.json',P/'INPUT_MANIFEST.json',M/'MOTION_TARGETS.json',
                     M/'NORMALIZATION.json',R/'RGB_NORMALIZATION.json',N/'TRAINING_CONTRACT.json',
                     R/'runs/rgb_residual_42/model.pt',AG/'episode_query.py',
                     M/'motion_model.py',R/'rgb_model.py',
                     O/'word100_learned_motion_path_20261102/path_model.py',
                     AC/'process_rgb_support.py',OLD/'PUBLIC_INPUTS.json'] + reviews
    for p in initial_files:
        pin(p)
    recipes = [OLD/'prepare_public.py',O/'word100_signrep_temporal_head_20261029/prepare_inputs.py',
               O/'word100_signrep_feasibility_20261029/common.py']
    recipe_pins = {str(p):pin(p) for p in recipes}
    public = read(OLD/'PUBLIC_INPUTS.json')
    public_sha = pins[str((OLD/'PUBLIC_INPUTS.json').resolve())]
    manifest = read(P/'INPUT_MANIFEST.json')
    selected = [{k:r[k] for k in ['clip_id','word_id','person','role','source','source_sha256','input','input_sha256','frames']} for r in manifest['rows']]
    check('source_count48',len(selected)==48)
    check('source_clip_unique48',len({r['clip_id'] for r in selected})==48)
    check('fit36dev12',Counter(r['role'] for r in selected)=={'fit':36,'development':12})
    for r in selected:
        validate_role(r)
    fit_sha={r['source_sha256'] for r in selected if r['role']=='fit'}
    dev_sha={r['source_sha256'] for r in selected if r['role']=='development'}
    check('source_sha_unique_and_split',len(fit_sha)==36 and len(dev_sha)==12 and not (fit_sha & dev_sha))
    stats = read(M/'NORMALIZATION.json')
    rgb_stats = read(R/'RGB_NORMALIZATION.json')
    check('geometry_norm73',len(stats['mean'])==len(stats['std'])==73 and all(v>0 for v in stats['std']))
    check('rgb_norm768',len(rgb_stats['mean'])==len(rgb_stats['std'])==768 and all(v>0 for v in rgb_stats['std']))
    check('existing_fit_normalization_only',set(stats['fit_people'])==set(FIT_IDS) and rgb_stats['fit_only'] is True)
    valid = geometry_valid()
    prepared = {}
    rows = []
    for row in selected:
        cid=row['clip_id']
        pin(row_path := Path(row['source']), row['source_sha256'])
        pin(Path(row['input']), row['input_sha256'])
        sp,rp=observation_paths(row)
        shape_sha,relation_sha=pin(sp),pin(rp)
        with np.load(row['input'],allow_pickle=False) as f:
            a={k:np.array(f[k],copy=True) for k in f.files}
        recipe=recipe_for(row,a,public,public_sha,recipe_pins)
        pp=prepare_row(row,stats,rgb_stats,recipe,valid)
        source_pts=[]
        with av.open(str(row_path)) as container:
            for frame in container.decode(video=0):
                if frame.pts is None:
                    raise ValueError('Missing original source PTS: '+cid)
                source_pts.append(round(float(frame.pts*frame.time_base)*1e6))
        check(cid+':source_native_pts_exact',source_pts==pp['ts'].tolist())
        check(cid+':source_frame_count',len(source_pts)==row['frames'])
        prepared[cid]=pp
        row.update(verified_recipe=recipe,rgb_lineage=pp['rgb_lineage'],
                   native_frame_ids=pp['ids'].tolist(),timestamp_us=pp['ts'].tolist(),
                   observed_fragments=pp['observed_fragments'],
                   evidence_available=pp['evidence_available'].tolist(),motion_anchor=pp['motion_anchor'].tolist(),
                   shape_observation=dict(path=str(sp),sha256=shape_sha),
                   relation_observation=dict(path=str(rp),sha256=relation_sha),
                   input_shape=list(pp['x'].shape),normalized_arrays_saved=False,
                   source_pts_redecoded=True,whole_clause_truth=None,educational_error=None)
        rows.append(row)
        print('prepared',cid,pp['rgb_lineage'],flush=True)
    lookup={r['clip_id']:r for r in rows}
    groups={}
    def group_for(row,j):
        key=(row['clip_id'],j)
        if key not in groups:
            groups[key]=dict(group_id=row['clip_id']+'__target'+str(j),clip_id=row['clip_id'],
                source_sha256=row['source_sha256'],word_id=row['word_id'],role=row['role'],person=row['person'],
                target_index=j,target=TARGETS[j],bags=[],train_eligible=False,
                group_weight=1.0,whole_clause_truth=None,educational_error=None)
        return groups[key]
    def add_bag(row,j,frames,label,kind,evidence,witnesses,fully_reviewed):
        pp=prepared[row['clip_id']]
        ix=indices_for(pp,frames)
        check(row['clip_id']+':witness_in_fixed_scope',bool(witnesses) and set(witnesses).issubset(range(frames[0],frames[1]+1)))
        ev=evidence_for(pp,ix,witnesses)
        eligible=bool(label is not None and fully_reviewed and ev['evidence_complete'] and ev['witness_complete_one_observed_fragment'])
        bag_id=row['clip_id']+'__t'+str(j)+'__'+kind+'_'+str(frames[0])+'_'+str(frames[1])
        g=group_for(row,j)
        check(bag_id+':no_duplicate',all(b['bag_id']!=bag_id for b in g['bags']))
        g['bags'].append(dict(bag_id=bag_id,kind=kind,frames=list(frames),indices=ix,label=label,
            timestamp_us=pp['ts'][ix[0]:ix[1]+1].tolist(),
            native_frames=pp['ids'][ix[0]:ix[1]+1].tolist(),
            train_eligible=bool(row['role']=='fit' and eligible),label_eligible=eligible,
            evidence_complete=ev['evidence_complete'],witness_frames=list(witnesses),
            witness_semantics=('All bracket frames are availability-required scope, not individually labelled action witnesses.'
                               if kind=='weak_positive_bracket' else 'Exact reviewer-selected native evidence frames; no dense labels are implied.'),
            availability_required_frames=pp['ids'][ix[0]:ix[1]+1].tolist(),
            witness_timestamp_us=[int(pp['ts'][int(np.flatnonzero(pp['ids']==f)[0])]) for f in witnesses],
            evidence=evidence,availability=ev,
            direct_full_scope_review=bool(fully_reviewed) if kind=='reviewed_local' else None,
            fixed_bracket_supervision=True,
            label_scope='weak_occurrence_in_fixed_bracket_not_dense_truth' if label==1 else 'fixed_local_nonoccurrence' if label==0 else 'unresolved',
            train_ineligible_reason=None if eligible else 'unknown_label_or_incomplete_observation_in_fixed_scope',
            composed_structural_support=composed_support(pp,ix),
            whole_clause_truth=None,educational_error=None))
    motion=read(M/'MOTION_TARGETS.json')
    check('motion_target_order_unchanged',motion['targets'][:5]==TARGETS)
    directed=[(i,r) for i,r in enumerate(motion['rows']) if r['kind']=='directed']
    check('old_directed45fit15dev',Counter(r['role'] for _,r in directed)=={'fit':45,'development':15})
    for mi,r in directed:
        row=lookup[r['clip_id']]
        for key in ['role','person','word_id','source_sha256','input_sha256']:
            check(r['id']+':'+key,row[key]==r[key])
        js=[j for j in range(5) if r['mask'][j] and r['y'][j]==1]
        check(r['id']+':one_positive_query',len(js)==1)
        pp=prepared[r['clip_id']];ix=indices_for(pp,r['frames'])
        check(r['id']+':positive_native_pts',r['timestamps_us']==[int(pp['ts'][ix[0]]),int(pp['ts'][ix[1]])])
        add_bag(row,js[0],r['frames'],1,'weak_positive_bracket',
                dict(path=str(M/'MOTION_TARGETS.json'),sha256=pins[str((M/'MOTION_TARGETS.json').resolve())],
                     json_pointer='/rows/'+str(mi),source_id=r['id'],review=r['review'],claim=r['claim'],
                     interpretation='Visible directed movement somewhere inside the original bracket; no dense1, exact boundary or semantic correctness.'),
                list(range(r['frames'][0],r['frames'][1]+1)),True)
    check('initial45fit15dev_groups',Counter(g['role'] for g in groups.values())=={'fit':45,'development':15})
    review_query_counts=Counter()
    for path in reviews:
        review=read(path)
        check(path.name+':twelve_clips',len(review['rows'])==12)
        for ri,r in enumerate(review['rows']):
            row=lookup[r['clip_id']]
            for key in ['role','person','source_sha256','input_sha256']:
                check(r['clip_id']+':review_'+key,row[key]==r[key])
            check(r['clip_id']+':actual_direct_review',r['actually_viewed'] is True)
            check(r['clip_id']+':review_null_semantics',r['whole_clause_truth'] is None and r['educational_error'] is None)
            pin(Path(r['sheet']['path']),r['sheet']['sha256'])
            if r.get('previous_review'):
                pin(Path(r['previous_review']['path']),r['previous_review']['sha256'])
            ids=r['native_frames'];frames=[ids[0],ids[-1]]
            check(r['clip_id']+':review_consecutive',ids==list(range(frames[0],frames[1]+1)))
            pp=prepared[r['clip_id']];ix=indices_for(pp,frames)
            check(r['clip_id']+':review_exact_pts',r['timestamp_us']==pp['ts'][ix[0]:ix[1]+1].tolist())
            for qi,q in enumerate(r['queries']):
                j=q['target_index'];label=q['presence']
                check(r['clip_id']+':review_target',type(j) is int and j in range(5) and (r['clip_id'],j) in groups)
                check(r['clip_id']+':label_domain',label is None or type(label) is int and label in [0,1])
                review_query_counts[(row['role'],str(label))]+=1
                add_bag(row,j,frames,label,'reviewed_local',dict(path=str(path),sha256=pins[str(path.resolve())],
                    json_pointer='/rows/'+str(ri)+'/queries/'+str(qi),sheet=r['sheet'],reason_ko=q['reason_ko'],
                    observability=q['observability'],reviewer=r['reviewer'],actually_viewed=True,
                    reused_verified_review=r.get('reused_verified_review',False)),q['witness_frames'],True)
    check('exact60local_queries',sum(review_query_counts.values())==60)
    support=[]
    for j in range(5):
        classes={}
        for label in [0,1]:
            ids=sorted({g['person'] for g in groups.values() if g['role']=='fit' and g['target_index']==j and
                        any(b['label']==label and b['label_eligible'] for b in g['bags'])})
            classes[str(label)]=ids
        support.append(dict(target_index=j,fit_ids_by_label=classes,
                            supported=all(len(classes[str(k)])>=3 for k in [0,1]),
                            minimum_distinct_fit_ids_per_class=3))
    supported=[s['target_index'] for s in support if s['supported']]
    for g in groups.values():
        pp=prepared[g['clip_id']]
        for b in g['bags']:
            b['train_eligible']=bool(b['train_eligible'] and g['target_index'] in supported)
            if g['target_index'] not in supported:
                b['train_ineligible_reason']='target_lacks_three_distinct_fit_ids_per_class'
        g['train_eligible']=any(b['train_eligible'] for b in g['bags'])
        g['eligible_class_labels']=sorted({b['label'] for b in g['bags'] if b['label_eligible']})
        g['broad_consistency']=dict(frames=[int(pp['ids'][0]),int(pp['ids'][-1])],indices=[0,len(pp['ids'])-1],
            label=1,kind='full_native_consistency_only',train_eligible=False,loss_weight=0.0,
            evidence='Contains the original positive bracket; evaluated from the same full-native score column only.',
            no_additional_forward_required=True,no_extra_training_weight=True,
            composed_structural_support=composed_support(pp,[0,len(pp['ids'])-1]),
            whole_clause_truth=None,educational_error=None)
    ordered=sorted(groups.values(),key=lambda g:(g['clip_id'],g['target_index']))
    training_ids=[g['group_id'] for g in ordered if g['train_eligible']]
    training_steps=80*math.ceil(len(training_ids)/12)
    denominators={role:dict(sources=sum(r['role']==role for r in rows),
        groups=sum(g['role']==role for g in ordered),
        weak_positive_bags=sum(b['kind']=='weak_positive_bracket' for g in ordered if g['role']==role for b in g['bags']),
        reviewed_local_bags=sum(b['kind']=='reviewed_local' for g in ordered if g['role']==role for b in g['bags']),
        local_labels={str(label):review_query_counts[(role,str(label))] for label in [0,1,None]},
        all_bags=sum(len(g['bags']) for g in ordered if g['role']==role),
        full_native_consistency_queries=sum(g['role']==role for g in ordered)) for role in ['fit','development']}
    for path,digest in pins.items():
        check(path+':final_pin',sha(path)==digest)
    check_data=dict(status='PASS',checks_count=len(checks),checks=checks,
        verification_scope='Source hashes and decoded native PTS, cache/auxiliary exact axes, normalized finite input, weak/local evidence binding, role and source separation; no model or semantic validation.',
        rows=[dict(clip_id=r['clip_id'],source_sha256=r['source_sha256'],input_sha256=r['input_sha256'],
                   frames=r['frames'],rgb_lineage=r['rgb_lineage'],source_pts_exact=True,
                   observed_fragments=r['observed_fragments']) for r in rows],
        denominators=denominators,supported_targets=supported,training_groups=len(training_ids),
        training_steps=training_steps,elapsed_seconds=time.perf_counter()-started,
        model_loads=0,model_forwards=0,training=0,new_rgb_or_detector_extractions=0,
        normalized_arrays_saved=0)
    exclusive(U/'INPUT_CHECKS.json',check_data)
    pins[str((U/'INPUT_CHECKS.json').resolve())]=sha(U/'INPUT_CHECKS.json')
    dataset=dict(status='FROZEN_LOCAL_EVENT_SUPERVISION_READY',rows=rows,groups=ordered,targets=TARGETS,
        supported_targets=supported,target_support=support,geometry_valid=valid.tolist(),
        input_dim=915,training_group_ids=training_ids,epochs=80,group_batch_size=12,
        steps_per_epoch=math.ceil(len(training_ids)/12),training_steps=training_steps,
        denominators=denominators,pins=pins,
        pretrained_checkpoint=dict(path=str(R/'runs/rgb_residual_42/model.pt'),sha256=pins[str((R/'runs/rgb_residual_42/model.pt').resolve())]),
        normalization=dict(geometry_path=str(M/'NORMALIZATION.json'),rgb_path=str(R/'RGB_NORMALIZATION.json'),refit=False),
        policy=dict(group_key='clip_id and physical target index',source_target_groups_equal=True,
                    broad_extra_loss_weight=0,one_full_native_forward_per_clip=True,
                    positives_are_weak_bags=True,dense_labels_created=False,
                    unknowns_preserved=True,all_bags_retained_in_evaluation_denominators=True,
                    negative_requires_entire_nominal_observation=True,
                    positive_requires_entire_witness_bracket_in_one_observed_fragment=True,
                    native_rows_never_removed=True,word_person_or_absolute_time_model_features=False),
        limitations=['Same Codex physical review, not independent KSL educator ground truth.',
                     'Registered DEV11-13 are repeated development material, not an independent new-person claim.',
                     'RGB/TCN contextual pixel support may exceed nominal label scope; structural support is disclosed, not dense truth.',
                     'No full source clause, required repetition, complete action or educational error labels.'],
        whole_clause_truth=None,educational_error=None,model_loads=0,training=0)
    exclusive(U/'DATASET_FROZEN.json',dataset)
    print(dict(status=dataset['status'],supported_targets=supported,training_groups=len(training_ids),
               training_steps=training_steps,denominators=denominators,checks=len(checks)),flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        failure=U/'DATA_PREP_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(),encoding='utf-8')
        raise
