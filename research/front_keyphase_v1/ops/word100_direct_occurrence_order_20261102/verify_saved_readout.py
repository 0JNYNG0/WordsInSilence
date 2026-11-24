"""Independent arithmetic of the completed Y comparison; no model forward.

Run ONLY after TRAIN_PROGRESS COMPLETE and EVALUATION/GATES exist. This module
does not import implementation models, losses, observe(), or summary/gate code.
All outputs go to this new Y directory; old inputs/checkpoints stay read-only.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json, hashlib, traceback
import numpy as np

Y=Path(__file__).resolve().parent; O=Y.parent
OUT=Path('W:/WordsInSilence_derived/word100_direct_occurrence_order_20261102')
BASELINE=Path('W:/WordsInSilence_derived/word100_seated_fit_transfer_20261102/predictions/seated_variation')
F=O/'word100_learned_flow_role_20261102'
B=O/'word100_learned_action_process_20261102'
L=O/'word100_seated_fit_transfer_20261102'
S=O/'word100_learned_phase_sequence_20261102'
K=O/'word100_coherent_sensor_input_20261102'
D=O/'word100_action_episode_supervision_20261102'
E=O/'word100_guide_conditioned_episode_20261102'
G=O/'word100_guided_event_order_20261102'
NORMAL={'native','original','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest','seated_variation'}
CONFIRMED={'OBSERVED_A_BEFORE_B':1,'OBSERVED_B_BEFORE_A':0}
hash_cache={}


def safe(path):
    p=Path(path);s=str(p).replace('\\','/').lower()
    assert not any(k in s for k in ['private_sessions','/mentor/','real14','real15','real16']),str(p)
    return p


def read(path):
    return json.loads(safe(path).read_text(encoding='utf-8-sig'))


def sha(path):
    p=safe(path)
    if str(p) not in hash_cache:
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        hash_cache[str(p)]=h.hexdigest()
    return hash_cache[str(p)]


def arrays(path):
    with np.load(safe(path),allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def equal(expected,actual,where='value',atol=2e-7,rtol=1e-6):
    if isinstance(expected,dict):
        assert set(expected)==set(actual),(where,'keys',set(expected)^set(actual))
        for k in expected:equal(expected[k],actual[k],where+'/'+str(k),atol,rtol)
    elif isinstance(expected,list):
        assert len(expected)==len(actual),(where,'length')
        for i,(a,b) in enumerate(zip(expected,actual)):equal(a,b,where+'/'+str(i),atol,rtol)
    elif isinstance(expected,(float,np.floating)):
        assert np.isfinite(expected) and np.isfinite(actual)
        assert np.isclose(expected,actual,atol=atol,rtol=rtol),(where,expected,actual)
    else:assert expected==actual,(where,expected,actual)


def pair_numeric_and_sets(a,first,second,condition,saved_probability):
    """Independent directed pair sum: explicit triangular products, not cumsum."""
    p=a['physical'];occ=a['occurrence'];av=np.asarray(a['image_available'],bool);times=a['times']
    weights=occ*p[:,:8] if condition=='tied_readout' else occ.copy()
    wa=weights[:,first].astype(np.float64)*av
    wb=weights[:,second].astype(np.float64)*av
    pair=np.outer(wa,wb)
    forward=float(np.triu(pair,1).sum(dtype=np.float64))
    reverse=float(np.tril(pair,-1).sum(dtype=np.float64))
    total=forward+reverse
    numerical=(forward+1e-8)/(total+2e-8)
    assert np.isclose(numerical,saved_probability,atol=3e-6,rtol=3e-6),(numerical,saved_probability)
    aa=np.flatnonzero(av&(weights[:,first]>=.8));bb=np.flatnonzero(av&(weights[:,second]>=.8))
    both=sorted(set(aa)&set(bb));missing=[];direction=None
    state='PAIR_OCCURRENCE_UNCONFIRMED'
    if len(aa) and len(bb):
        lower=min(int(aa[0]),int(bb[0]));upper=max(int(aa[-1]),int(bb[-1]))
        missing=[i for i in range(lower,upper+1) if not av[i]]
        if missing:state='PAIR_OBSERVATION_GAP'
        elif both:state='PAIR_OVERLAPPING_EVIDENCE'
        elif int(aa[-1])<int(bb[0]):
            direction=1;state='OBSERVED_A_BEFORE_B' if saved_probability>=.8 else 'PAIR_READOUT_DISAGREEMENT'
        elif int(bb[-1])<int(aa[0]):
            direction=0;state='OBSERVED_B_BEFORE_A' if saved_probability<=.2 else 'PAIR_READOUT_DISAGREEMENT'
        else:state='PAIR_INTERLEAVED_EVIDENCE'
    pa=np.flatnonzero(av&(p[:,first]>=.8));pb=np.flatnonzero(av&(p[:,second]>=.8));pd=None
    if len(pa) and len(pb):
        if pa[-1]<pb[0]:pd=1
        elif pb[-1]<pa[0]:pd=0
        elif not set(pa)&set(pb):pd='interleaved'
    learned_state=state
    if state in CONFIRMED and pd in [0,1] and pd!=direction:state='RETAINED_PHYSICAL_ORDER_DISAGREEMENT'
    checks=dict(first=int(first),second=int(second),pair_mass=total,
        first_evidence_frames=aa.tolist(),second_evidence_frames=bb.tolist(),overlap_frames=list(map(int,both)),
        first_evidence_seconds=times[aa].tolist(),second_evidence_seconds=times[bb].tolist(),
        gaps_inside_evidence=missing,status=state,observed_direction=direction,
        physical_direction=pd,learned_pair_status=learned_state,raw_physical_first_frames=pa.tolist(),
        raw_physical_second_frames=pb.tolist(),raw_physical_context_qualified=False,
        raw_physical_first_seconds=times[pa].tolist(),raw_physical_second_seconds=times[pb].tolist(),
        comparison_frames=len(av),unobserved_frames=np.flatnonzero(~av).tolist(),
        whole_sign_correctness=None,complete_action_verified=False,missing_action_verified=False,
        raw_prediction=1 if saved_probability>.5 else 0 if saved_probability<.5 else None,
        confirmed_prediction=CONFIRMED.get(state))
    return checks,numerical,weights


def stats(records,anchors):
    def section(rr):
        known=[r for r in rr if r['target']!=-1]
        correct_raw=sum(int(r['raw_prediction']==r['target']) for r in known)
        correct_conf=sum(int(r['confirmed_prediction']==r['target']) for r in known)
        wrong_conf=sum(int(r['confirmed_prediction'] is not None and r['confirmed_prediction']!=r['target']) for r in known)
        held=sum(r['confirmed_prediction'] is None for r in known)
        assert correct_conf+wrong_conf+held==len(known)
        return dict(known=len(known),raw_correct=correct_raw,confirmed_correct=correct_conf,
            confirmed_wrong=wrong_conf,held_known=held,unknown=len(rr)-len(known),
            interleaved_confirmed=sum(r['truth_reason']=='interleaved_witnesses' and r['confirmed_prediction'] is not None for r in rr))
    groups=defaultdict(list)
    for r in records:
        if r['target']>=0:groups[(r['clip_id'],r['first'],r['second'])].append(r)
    group_raw=[sum(r['raw_prediction']==r['target'] for r in rr)/len(rr) for rr in groups.values()]
    group_confirmed=[sum(r['confirmed_prediction']==r['target'] for r in rr)/len(rr) for rr in groups.values()]
    local=[r for r in records if 'local' in r['kind'] or 'short' in r['kind']]
    normal_anchors=[r for r in anchors if r['kind'] in NORMAL]
    def anchor_section(rr):
        return dict(positive_known=sum(r['target']==1 for r in rr),
            positive_confirmed=sum(r['available'] and r['target']==1 and r['effective_weight']>=.8 for r in rr),
            nonmain_known=sum(r['target']==0 for r in rr),
            nonmain_false_confirmed=sum(r['available'] and r['target']==0 and r['effective_weight']>=.8 for r in rr))
    return dict(all=section(records),normal=section([r for r in records if r['kind'] in NORMAL]),
        local=section(local),local_evaluation_status='EVALUATED' if local else 'NOT_EVALUATED_NO_LOCAL_OR_SHORT_KIND',
        contrary=section([r for r in records if r['target']==0]),
        main_reverse=section([r for r in records if r['kind']=='actual_reverse_main']),
        source_group_count=len(groups),source_group_macro_raw_accuracy=float(np.mean(group_raw)) if groups else None,
        source_group_macro_confirmed_correct_fraction=float(np.mean(group_confirmed)) if groups else None,
        by_kind={k:section([r for r in records if r['kind']==k]) for k in sorted({r['kind'] for r in records})},
        by_person={str(k):section([r for r in records if r['person']==k]) for k in sorted({r['person'] for r in records})},
        occurrence_positive_known=sum(r['target']==1 for r in anchors),
        occurrence_positive_confirmed=sum(r['available'] and r['target']==1 and r['effective_weight']>=.8 for r in anchors),
        occurrence_positive_raw_threshold=sum(r['target']==1 and r['probability']>=.8 for r in anchors),
        nonmain_negative_known=sum(r['target']==0 for r in anchors),
        nonmain_false_confirmed=sum(r['available'] and r['target']==0 and r['effective_weight']>=.8 for r in anchors),
        nonmain_raw_occurrence_false_confirmed=sum(r['available'] and r['target']==0 and r['probability']>=.8 for r in anchors),
        normal_anchors_by_family={str(k):anchor_section([r for r in normal_anchors if r['family']==k]) for k in sorted({r['family'] for r in normal_anchors})},
        normal_anchors_by_source={k:anchor_section([r for r in normal_anchors if r['clip_id']==k]) for k in sorted({r['clip_id'] for r in normal_anchors})},
        status_counts=dict(Counter(r['status'] for r in records)))


def source_metadata():
    fit=read(B/'PREPARED.json')['rows']+read(L/'PAIR_MANIFEST.json')['conditions']['seated_variation']
    byfit={(r['clip_id'],r['kind']):r for r in fit}
    dev=read(S/'EVALUATION_INPUTS.json')['rows']
    bydev={(r['clip_id'],r['kind']):r for r in dev}
    coherent={r['clip_id']:r for r in read(K/'INPUT_MANIFEST.json')['rows']}
    fit_labels={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(D/'ACTION_TARGETS.json')['rows']}
    fit_context={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(E/'CONTEXT_TARGETS.json')['rows']}
    dev_labels={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(D/'DEV_ACTION_TARGETS.json')['rows']}
    return byfit,bydev,coherent,fit_labels,fit_context,dev_labels


def restore_source(split,meta,tables,flow_saved):
    byfit,bydev,coherent,fit_labels,fit_context,dev_labels=tables
    cid=meta['clip_id'];kind=meta['kind']
    mapping=np.asarray(flow_saved['source_frame_id'],int);available=np.asarray(flow_saved['image_available'],bool)
    if split=='fit':
        src=byfit[(cid,kind)];assert sha(src['input'])==src['input_sha256']
        inp=arrays(src['input']);times=inp['timestamp_us']/1e6
        cache=arrays(src['cache']);assert sha(src['cache'])==src['cache_sha256']
        source_map=np.asarray(src['mapping']) if 'mapping' in src else cache['mapping']
        assert np.array_equal(mapping,source_map) and np.array_equal(available,cache['available'])
        current=cache['current'];labelmap=fit_labels[cid];contextmap=fit_context[cid]
    else:
        src=bydev[(cid,kind)]
        axis=coherent[cid] if kind=='native' else {'input':src['axis_path'],'input_sha256':src['axis_sha256']}
        assert sha(axis['input'])==axis['input_sha256']
        inp=arrays(axis['input']);times=inp['timestamp_us']/1e6
        assert np.array_equal(mapping,np.asarray(src['mapping']))
        old=arrays(BASELINE/(src['key']+'.npz'))
        assert np.array_equal(available,old['image_available'])
        assert np.array_equal(times,old['timestamp']) and np.array_equal(mapping,old['source_frame_id'])
        current=old['current_probability'];labelmap=dev_labels[cid];contextmap=None
    assert len(times)==len(mapping)==len(available) and np.all(np.diff(times)>0)
    labels=np.asarray([labelmap.get(int(frame),-1) for frame in mapping]);labels[~available]=-1
    seams=np.flatnonzero(np.abs(np.diff(mapping))>3)+1
    for f in seams:labels[max(0,f-2):min(len(labels),f+3)]=-1
    if split=='fit':
        context=np.asarray([contextmap.get(int(frame),-1) for frame in mapping]);context[~available]=-1
        for f in seams:context[max(0,f-2):min(len(context),f+3)]=-1
        role=np.where(context<0,-1,np.where(context<8,1,np.where(context==8,2,0)))
    else:role=np.where(labels<0,-1,np.where(labels<8,1,2))
    return dict(mapping=mapping,available=available,times=times,current=current,episode=labels,role=role)


def main():
    destination=Y/'SAVED_READOUT_VERIFIED.json'
    assert not destination.exists(),'Completed verification must not be repeated.'
    assert read(Y/'TRAIN_PROGRESS.json')['status']=='COMPLETE','Do not verify unfinished training.'
    assert (Y/'EVALUATION.json').exists() and (Y/'GATES.json').exists(),'Wait for saved comparison completion.'
    frozen=read(Y/'FROZEN.json');protocol=read(Y/'EVALUATION_PROTOCOL.json');evaluation=read(Y/'EVALUATION.json')
    assert frozen['epochs']==80 and frozen['seed']==42 and frozen['conditions']==['tied_readout','direct_readout']
    tables=source_metadata();flow=read(F/'FLOW_INPUTS.json')['items'];all_results={};source_cache={}
    expected_queries={split:{(q['flow_key'],q['first'],q['second']):q for q in protocol['query_manifest'][split]} for split in ['fit','dev']}
    total_arrays=0;total_pairs=0;total_anchors=0;old_exact=0;prob_errors=[];condition_exact=0;boundary_probability_cases=[]
    for condition in frozen['conditions']:
        checkpoint=OUT/'runs'/condition/'model.pt';complete=read(checkpoint.parent/'COMPLETE.json')
        assert sha(checkpoint)==complete['checkpoint_sha256']
        assert complete['history'][-1]['epoch']==80
        assert all(value==0 for name,value in complete['changes'].items() if name.startswith('base.'))
        for split in ['fit','dev']:
            saved=read(OUT/'evaluations'/f'{condition}_{split}.json')
            assert saved['condition']==condition and saved['split']==split
            metas={key:v for key,v in flow.items() if key.startswith(split+'_')}
            assert len(metas)==protocol['splits'][split]['videos_and_variants']
            rec={};expected_anchors={};expected_meta={}
            for key,meta in metas.items():
                assert meta['person'] in (range(1,10) if split=='fit' else [11,12,13])
                if key not in source_cache:
                    assert sha(meta['path'])==meta['sha256']
                    source_cache[key]=restore_source(split,meta,tables,arrays(meta['path']))
                source=source_cache[key]
                predpath=OUT/'predictions'/condition/split/f'{key}.npz';a=arrays(predpath);n=len(source['mapping'])
                for name,width in [('physical',9),('role',2),('occurrence',8),('current',10)]:
                    assert a[name].shape==(n,width) and np.isfinite(a[name]).all()
                    assert np.min(a[name])>=0 and np.max(a[name])<=1
                    if name!='occurrence':assert np.allclose(a[name].sum(1),1,atol=2e-6)
                for pfield,sfield in [('mapping','mapping'),('times','times'),('image_available','available'),('current','current')]:
                    assert np.array_equal(a[pfield],source[sfield]),(condition,key,pfield)
                if condition=='direct_readout':
                    prior=arrays(OUT/'predictions/tied_readout'/split/f'{key}.npz')
                    for name in ['physical','role','current','image_available','mapping','times']:
                        assert np.array_equal(a[name],prior[name]),(key,name)
                    condition_exact+=1
                if split=='dev':
                    old=arrays(BASELINE/f"{meta['clip_id']}_{meta['kind']}.npz")
                    assert np.array_equal(a['physical'],old['raw_action_probability'])
                    assert np.array_equal(a['current'],old['current_probability']);old_exact+=1
                expected_meta[key]=meta;rec[key]=a;total_arrays+=1
                labels=source['episode'];roles=source['role']
                targets=np.full((n,8),-1,dtype=int)
                targets[(roles>=0)&(roles!=1),:]=0
                for frame in np.flatnonzero((labels>=0)&(labels<8)):targets[frame,int(labels[frame])]=1
                for frame,family in zip(*np.where(targets>=0)):
                    effective=a['occurrence'][frame,family]*(a['physical'][frame,family] if condition=='tied_readout' else 1)
                    expected_anchors[(key,int(frame),int(family))]=dict(flow_key=key,clip_id=meta['clip_id'],person=meta['person'],kind=meta['kind'],
                        frame=int(frame),source_frame=int(source['mapping'][frame]),family=int(family),target=int(targets[frame,family]),
                        probability=float(a['occurrence'][frame,family]),available=bool(source['available'][frame]),effective_weight=float(effective))
            assert len(saved['pairs'])==len(expected_queries[split])
            seen=set()
            for p in saved['pairs']:
                key=(p['flow_key'],p['first'],p['second']);assert key not in seen;seen.add(key)
                q=expected_queries[split][key]
                for name,value in q.items():equal(value,p[name],'/pair/'+name)
                a=rec[p['flow_key']];source=source_cache[p['flow_key']]
                ia=np.flatnonzero(source['episode']==p['first']).tolist();ib=np.flatnonzero(source['episode']==p['second']).tolist()
                truth=1 if ia and ib and max(ia)<min(ib) else 0 if ia and ib and max(ib)<min(ia) else -1
                assert (ia,ib,truth)==(p['first_witnesses'],p['second_witnesses'],p['target'])
                reason='known_witness_order' if truth>=0 else 'interleaved_witnesses' if ia and ib else 'witness_missing'
                assert p['truth_reason']==reason
                checks,independent,_=pair_numeric_and_sets(a,p['first'],p['second'],condition,p['forward_probability'])
                for name,value in checks.items():equal(value,p[name],'/pair/'+name,atol=2e-5 if name=='pair_mass' else 2e-7,rtol=3e-6)
                prob_errors.append(abs(independent-p['forward_probability']))
                if abs(p['forward_probability']-.5)<=3e-6:boundary_probability_cases.append(dict(condition=condition,split=split,key=list(key),p=p['forward_probability']))
                assert p['timestamp_kind']=='native_input_PTS'
                total_pairs+=1
            assert seen==set(expected_queries[split])
            assert len(saved['anchors'])==len(expected_anchors)
            seen=set()
            for anchor in saved['anchors']:
                key=(anchor['flow_key'],anchor['frame'],anchor['family']);assert key not in seen;seen.add(key)
                equal(expected_anchors[key],anchor,'/anchor');total_anchors+=1
            assert seen==set(expected_anchors)
            summary=stats(saved['pairs'],saved['anchors']);equal(summary,saved['summary'],'/summary')
            main_report=next(r for r in evaluation['reports'] if r['condition']==condition and r['split']==split)
            equal(summary,main_report['summary'],'/evaluation_summary')
            assert main_report['head_median_seconds']==saved['head_median_seconds'] and saved['head_median_seconds']>=0
            all_results[(condition,split)]=summary
    a=all_results[('tied_readout','dev')];b=all_results[('direct_readout','dev')]
    gates={
        'raw_correct_strictly_higher': b['all']['raw_correct']>a['all']['raw_correct'],
        'source_macro_raw_strictly_higher': b['source_group_macro_raw_accuracy']>a['source_group_macro_raw_accuracy'],
        'confirmed_correct_strictly_higher': b['all']['confirmed_correct']>a['all']['confirmed_correct'],
        'source_macro_confirmed_strictly_higher': b['source_group_macro_confirmed_correct_fraction']>a['source_group_macro_confirmed_correct_fraction'],
        'normal_confident_wrong_zero': b['normal']['confirmed_wrong']==0,
        'normal_confirmed_correct_preserved': b['normal']['confirmed_correct']>=a['normal']['confirmed_correct'],
        'nonmain_false_effective_not_increased': b['nonmain_false_confirmed']<=a['nonmain_false_confirmed'],
        'normal_each_family_positive_preserved': all(b['normal_anchors_by_family'][k]['positive_confirmed']>=v['positive_confirmed'] for k,v in a['normal_anchors_by_family'].items()),
        'contrary_raw_correct_preserved': b['contrary']['raw_correct']>=a['contrary']['raw_correct'],
        'contrary_confirmed_correct_preserved': b['contrary']['confirmed_correct']>=a['contrary']['confirmed_correct'],
        'contrary_confident_wrong_not_increased': b['contrary']['confirmed_wrong']<=a['contrary']['confirmed_wrong'],
        'main_reverse_raw_correct_preserved': b['main_reverse']['raw_correct']>=a['main_reverse']['raw_correct'],
        'main_reverse_confirmed_correct_preserved': b['main_reverse']['confirmed_correct']>=a['main_reverse']['confirmed_correct'],
        'main_reverse_confident_wrong_not_increased': b['main_reverse']['confirmed_wrong']<=a['main_reverse']['confirmed_wrong'],
        'known_interleaved_never_confirmed': b['all']['interleaved_confirmed']==0}
    for condition in frozen['conditions']:
        dev=all_results[(condition,'dev')]
        assert dev['all']['known']==103 and dev['all']['unknown']==7 and dev['source_group_count']==15
        assert dev['contrary']['known']==20 and dev['main_reverse']['known']==5
        assert dev['local']['known']==dev['local']['unknown']==0
        assert dev['local_evaluation_status']=='NOT_EVALUATED_NO_LOCAL_OR_SHORT_KIND'
    stored_gates=read(Y/'GATES.json');equal(gates,stored_gates['checks'],'/gates')
    assert set(gates)==set(protocol['numeric_gates'])==set(frozen['gates'])
    assert stored_gates['passed']==all(gates.values())
    assert set(stored_gates['failed'])=={k for k,v in gates.items() if not v}
    source_anchor_regressions={k:dict(tied=v,direct=b['normal_anchors_by_source'][k])
        for k,v in a['normal_anchors_by_source'].items()
        if b['normal_anchors_by_source'][k]['positive_confirmed']<v['positive_confirmed']
        or b['normal_anchors_by_source'][k]['nonmain_false_confirmed']>v['nonmain_false_confirmed']}
    assert total_arrays==680 and condition_exact==340 and old_exact==176
    assert evaluation['new_model_clip_inferences']==680 and evaluation['dev_new_inferences']==176 and evaluation['old_model_reruns']==0
    # Tensor equality only; no model instantiation or forward call.
    import torch
    states={tag:torch.load(OUT/'runs'/tag/'model.pt',map_location='cpu',weights_only=True) for tag in frozen['conditions']}
    initial=torch.load(safe(frozen['initialization']),map_location='cpu',weights_only=True)
    base_tensors=0
    for key,value in states['tied_readout'].items():
        if key.startswith('base.'):
            assert torch.equal(value,states['direct_readout'][key]);base_tensors+=1
            originalkey=key[5:]
            if originalkey in initial:assert torch.equal(value,initial[originalkey]),key
            elif originalkey=='occurrence_query.weight':assert not torch.count_nonzero(value)
            else:assert originalkey=='family_guide_keys',originalkey
    head_count=sum(v.numel() for k,v in states['direct_readout'].items() if not k.startswith('base.'))
    assert head_count==13824==frozen['trainable_parameters']
    for bundle in [frozen['pins'],protocol['pins']]:
        for path,digest in bundle.items():assert sha(path)==digest,path
    output=dict(status='PASS_SAVED_ARITHMETIC_AND_CONTRACT' if all(gates.values()) else 'PASS_SAVED_ARITHMETIC_CANDIDATE_GATES_NOT_MET',
        at_utc=datetime.now(timezone.utc).isoformat(),verification='Separate Codex saved-array arithmetic, not independent KSL expert truth',
        prediction_arrays=total_arrays,between_condition_fixed_arrays_exact=condition_exact,
        old_seated_raw_action_and_current_comparisons_exact=old_exact,
        independently_checked_pairs=total_pairs,independently_checked_occurrence_anchors=total_anchors,
        max_pair_probability_absolute_error=max(prob_errors),classification_boundary_diagnostics=boundary_probability_cases,
        immutable_base_tensors=base_tensors,matched_trainable_parameters=head_count,
        source_axes_and_availability_exact=True,unknown_and_seam_masks_preserved=True,
        summaries={tag+'_'+split:value for (tag,split),value in all_results.items()},
        gates=gates,candidate_passed=all(gates.values()),failed_gates=[k for k,v in gates.items() if not v],
        normal_source_anchor_regressions=source_anchor_regressions,
        inputs_sha256_verified=len(hash_cache),new_inference=0,new_training=0,old_evaluation_recomputed=False,
        limits=['FIT630 swapped/variant questions are dependent45 source relations in27 clips; primary FIT has315 unswapped records.',
                'DEV110 unswapped records cover15 source relations and repeat-used IDs11–13; not independent final generalization.',
                'Prediction summaries retain every known query, including abstentions. Unknown queries are not normal labels.',
                'DEV has no local/short-kind subgroup, so this behavior is NOT_EVALUATED, not passed by an empty denominator.',
                'Only queried observed physical before/after is assessed; no complete sign, missing action, exact boundary, or required repeat claim.',
                'Model head latency excludes extraction and is not total feedback wait time.'],
        pins={str(p):sha(p) for p in [Path(__file__),Y/'FROZEN.json',Y/'EVALUATION_PROTOCOL.json',Y/'EVALUATION.json',Y/'GATES.json']},
        whole_goal_complete=False)
    with destination.open('x',encoding='utf8') as stream:json.dump(output,stream,ensure_ascii=False,indent=2)
    print(output['status'], 'arrays',total_arrays,'pairs',total_pairs,'failed gates',output['failed_gates'],flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        path=Y/'SAVED_VERIFY_FIRST_FAILURE.txt'
        if not path.exists():path.write_text(traceback.format_exc(),encoding='utf8')
        raise
