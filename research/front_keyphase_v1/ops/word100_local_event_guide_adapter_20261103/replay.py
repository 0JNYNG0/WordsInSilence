"""Freeze metadata-only proposals, then replay stored AH outputs once.

No source video reads, model loads/forwards, DTW or training. Run --freeze before
review; --run requires an explicit code-review pass. Existing artifacts remain
immutable and every new output is created exclusively.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
import traceback
import numpy as np
from adapter import attach,value_sha

sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent
AH=O/'word100_local_event_learning_20261103'
X=O/'word100_learned_component_extent_20261102'
AE=O/'word100_guide_feature_supervision_20261102'
G=O/'word100_learned_feedback_integration_20261102'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def axis_sha(ids,ts):return hashlib.sha256(np.asarray(ids).tobytes()+np.asarray(ts).tobytes()).hexdigest()

def freeze():
    if (U/'FROZEN.json').exists():raise FileExistsError('Do not replace already fixed scopes')
    review=read(AH/'RUNTIME_CONNECTION_REVIEW.json');dataset=read(AH/'DATASET_FROZEN.json')
    training=read(AH/'TRAIN_RESULT.json')
    contract=read(AE/'COMPARISON_CONTRACT.json')
    constraints=[c for c in contract['constraints'] if c['clause_id']=='well_brush']
    assert len(constraints)==1
    source_constraint=constraints[0]
    assert source_constraint['entry_id']=='NIKL_legacy:9543' and source_constraint['condition_relation']=='RELATED_ONLY'
    assert source_constraint['component_id']=='right_hand_traverses_left_arm_toward_wrist'
    bridge=module('frozen_source_bridge_catalog',X/'source_bridge.py').SourceBridge()
    catalog=bridge.catalog['well_brush']
    for k in ['text','url','source']:
        assert catalog[k]==source_constraint[k],k
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();v=sha(path)
        if expected is not None:assert v==expected,str(path)
        pins[str(path)]=v
        return v
    for p in [U/'PLAN.json',U/'adapter.py',U/'replay.py',U/'check_adapter.py',U/'STRUCTURAL_CHECKS_v2.json',AH/'RUNTIME_CONNECTION_REVIEW.json',
              AH/'DATASET_FROZEN.json',AH/'INPUT_CHECKS.json',AH/'DATA_REVIEW.json',AH/'SAVED_VERIFIED.json',
              AH/'TRAIN_RESULT.json',AH/'EXECUTION_FROZEN.json',AH/'data.py',AH/'local_events.py',
              X/'source_bridge.py',AE/'COMPARISON_CONTRACT.json',G/'learner_adapter.py',
              O/'word100_predicate_supervision_20261102/CLAUSE_COMPONENT_LINKS.json',
              O/'word100_source_component_connector_20261102/DIRECT_TARGETS.json']:
        pin(p)
    pin(training['model'],training['model_sha256'])
    pin(source_constraint['source']['path'],source_constraint['source']['sha256'])
    source_paths={str(Path(r['source']).resolve()) for r in dataset['rows']}
    # Reuse the AH cache loader without touching source video files. The source
    # hashes were checked in AH preparation and are compared across pinned JSON.
    for p,h in dataset['pins'].items():
        if str(Path(p).resolve()) not in source_paths:
            pin(p,h)
    rows=[]
    for r in review['rows']:
        d=next(q for q in dataset['rows'] if q['clip_id']==r['clip_id'])
        for k in ['word_id','role','person','source_sha256','input_sha256']:
            assert r[k]==d[k],(r['clip_id'],k)
        assert r['role']=='development' and r['person'] in [11,12,13]
        for key in ['existing_feedback','existing_path_base','AH_prediction','input']:
            pin(r[key],r[key+'_sha256'])
        base=read(r['existing_feedback']);pathbase=read(r['existing_path_base'])
        assert base['word_id']==r['word_id'] and base['path']==pathbase['path']
        assert base['path_axes']==['query_native_index','learned_canonical_index']
        path=np.asarray(base['path']);ids=np.asarray(d['native_frame_ids']);ts=np.asarray(d['timestamp_us'])
        assert path.dtype.kind in 'iu' and path.ndim==2 and path.shape[1]==2
        assert (path>=0).all() and path[:,0].max()<len(ids) and (np.diff(path,axis=0)>=0).all()
        stages={s['guide_id']:s for s in base['stages']};proposals=[]
        for route in r['routes']:
            j=route['target_index'];ms=[m for m in base['motion_observations'] if m['target']==j and m['guide_id']==route['guide_id']]
            assert len(ms)==1
            old=ms[0];legacy=old['frame_ids']
            assert [min(legacy),max(legacy)]==route['legacy_motion_frames']
            stage=stages[route['guide_id']]
            assert [min(stage['frame_ids']),max(stage['frame_ids'])]==route['legacy_stage_frames']
            sc=None;canonical=None
            if j==0:
                key=(r['word_id'],'right_hand_traverses_left_arm_toward_wrist')
                nominal=bridge.nominal[key]
                assert 'well_brush' in nominal['clauses']
                canonical=[min(nominal['frames']),max(nominal['frames'])]
                hit=np.unique(path[(path[:,1]>=canonical[0])&(path[:,1]<=canonical[1]),0])
                assert len(hit)
                indices=[int(hit.min()),int(hit.max())]
                sc=copy.deepcopy(source_constraint)
                family='existing_source_component_nominal_projected_by_existing_path'
            else:
                hit=np.flatnonzero(np.isin(ids,legacy))
                assert ids[hit].tolist()==legacy and np.array_equal(hit,np.arange(hit[0],hit[-1]+1))
                indices=[int(hit[0]),int(hit[-1])]
                family='existing_legacy_motion_native_proposal'
            lo,hi=indices
            proposals.append(dict(target_index=j,guide_id=route['guide_id'],scope_family=family,
                canonical_extent=canonical,indices=indices,native_frame_ids=ids[lo:hi+1].tolist(),
                timestamp_us=ts[lo:hi+1].tolist(),source_constraint=sc,
                source_component_id=sc['component_id'] if sc else None,
                source_binding_relation='RELATED_ONLY' if sc else 'REFERENCE_OBSERVATION_ONLY',
                other_scope_metadata=dict(legacy_motion_frame_ids=legacy,guide_stage_frame_ids=stage['frame_ids'],
                    canonical_legacy_motion_frames=old['canonical_frames'],not_alternative_score_candidates=True),
                selection='Fixed before scoped AH readout; no probability-based selection.',
                event_boundary_truth=None,source_variant_applicability_verified=None))
        envelope=dict(**r,native_frame_ids=d['native_frame_ids'],timestamp_us=d['timestamp_us'],
                      base_value_sha256=value_sha(base),proposals_value_sha256=value_sha(proposals),
                      AH_model_path=training['model'],AH_model_sha256=training['model_sha256'],
                      source_hash_verification='Inherited exact AH verified source identity; source media not reread in this replay.')
        rows.append(dict(identity=envelope,proposals=proposals))
    assert len(rows)==12 and sum(len(r['proposals']) for r in rows)==15
    frozen=dict(status='FROZEN_BEFORE_SCOPED_AH_READOUT',rows=rows,pins=pins,
        thresholds=dict(high=.8,low=.2,joint_support=.5),
        source_media_paths_not_to_read=sorted(source_paths),
        source_identity='Inherited from pinned AH decoded-PTS/input verification; no new media hash or decode.',
        base_field='local_event_feedback',source_variant_and_full_truth_remain_unknown=True,
        primary_scope_count=15,clip_count=12,new_model_loads=0,new_inferences=0,new_DTW=0,
        new_training=0,new_source_video_reads=0,new_scoped_scores_computed=0,
        conclusion_scope='Repeated DEV integration replay; not fresh accuracy, exact timing, or learner-error truth.')
    write(U/'FROZEN.json',frozen)
    print(json.dumps({'status':frozen['status'],'scopes':15,'clips':12,'sha256':sha(U/'FROZEN.json')}))

def run():
    if (U/'REPLAY_STARTED.json').exists():raise FileExistsError('Inspect existing run instead of repeating it')
    frozen=read(U/'FROZEN.json');code=read(U/'CODE_REVIEW.json')
    assert code['approved_for_saved_replay'] is True
    for p,h in code['code_pins'].items():assert sha(p)==h,p
    for p,h in frozen['pins'].items():assert sha(p)==h,p
    sys.path.insert(0,str(AH))
    import data as ah_data
    dataset=read(AH/'DATASET_FROZEN.json')
    original_source_paths={str(Path(r['source']).resolve()) for r in dataset['rows']}
    assert original_source_paths==set(frozen['source_media_paths_not_to_read'])
    cache_dataset=copy.deepcopy(dataset)
    wanted={r['identity']['clip_id'] for r in frozen['rows']}
    cache_dataset['rows']=[r for r in dataset['rows'] if r['clip_id'] in wanted]
    assert len(cache_dataset['rows'])==12
    cache_dataset['pins']={p:h for p,h in dataset['pins'].items() if str(Path(p).resolve()) not in original_source_paths}
    write(U/'REPLAY_STARTED.json',dict(time=time.time(),frozen_sha256=sha(U/'FROZEN.json'),mode='saved_outputs_only'))
    start=time.perf_counter();prepared=ah_data.load_inputs(cache_dataset)
    for r in cache_dataset['rows']:
        prepared[r['clip_id']]['input_identity']={k:r[k] for k in ['clip_id','word_id','source_sha256','input_sha256']}
    observations=[];result_rows=[];preservation=[]
    outputs=U/'outputs';outputs.mkdir(exist_ok=False)
    for row in frozen['rows']:
        e=row['identity'];cid=e['clip_id'];a=prepared[cid]
        with np.load(e['AH_prediction'],allow_pickle=False) as f:
            pred={k:np.array(f[k],copy=True) for k in f.files}
        assert axis_sha(pred['frame_id'],pred['timestamp_us'])==e['native_axis_sha256']
        support={q['target_index']:ah_data.composed_support(a,q['indices']) for q in row['proposals']}
        base=read(e['existing_feedback'])
        out=attach(base,e,a,pred,row['proposals'],support)
        assert set(out)==set(base)|{'local_event_feedback'}
        for k,v in base.items():assert out[k]==v,(cid,k)
        dest=outputs/(cid+'.json');write(dest,out)
        preservation.append(dict(clip_id=cid,all_old_fields_equal=True,old_keys=list(base),
            baseline_value_sha256=value_sha(base),output_without_addition_sha256=value_sha({k:v for k,v in out.items() if k!='local_event_feedback'})))
        for ob in out['local_event_feedback']['observations']:
            old=next(m for m in base['motion_observations'] if m['target']==ob['target_index'] and m['guide_id']==ob['proposal']['guide_id'])
            observations.append(dict(clip_id=cid,target_index=ob['target_index'],scope_frames=[ob['proposal']['native_frame_ids'][0],ob['proposal']['native_frame_ids'][-1]],
                status=ob['status'],probability=ob['probability'],stage_binding=ob['guide_time_binding']['status'],
                source_binding=ob['source_binding'],scope_evidence_complete=ob['scope_evidence_complete'],
                prior_motion_status=old['status'],prior_motion_value_preserved=True,
                same_native_scope_as_prior=old['frame_ids']==ob['proposal']['native_frame_ids'],
                direct_groundtruth_for_exact_primary_scope=None,educational_error=None))
        result_rows.append(dict(clip_id=cid,path=str(dest),sha256=sha(dest)))
    for p,h in frozen['pins'].items():assert sha(p)==h,p
    counts={s:sum(o['status']==s for o in observations) for s in sorted({o['status'] for o in observations})}
    result=dict(status='SAVED_OUTPUT_ADDITIVE_REPLAY_COMPLETE',outputs=result_rows,observations=observations,
        preservation=preservation,counts=counts,clips=12,primary_scopes=15,
        source_related_scopes=sum(o['target_index']==0 for o in observations),
        stage_guard_supported=sum(o['stage_binding']=='SUPPORTED' for o in observations),
        local_attachment_and_cache_validation_seconds=time.perf_counter()-start,
        timing_is_end_to_end_latency=False,source_videos_read=0,models_loaded=0,new_model_forwards=0,new_DTW=0,new_training=0,
        exact_primary_scope_accuracy=None,whole_sign_correctness=None,complete_action=None,educational_error=None,
        limitations=['Repeated DEV source observations, not a fresh accuracy test.',
          'Different observer targets: old net-direction/continuity and new local occurrence statuses are not an interchangeable accuracy metric.',
          'Source target0 uses the existing component proposal; others retain exact legacy motion scopes. Actual equality with legacy scopes is recorded, not assumed. No score selected the primary scope.',
          'Stage consistency is inherited on its original guide range, not direct verification of all source-proposal points.',
          'High anchor preserves observed evidence but does not certify absent gaps, contacts, thumb shape, repeat count or completion.',
          'R native snapshots do not all include later source_components/order outputs; synthetic preservation tests cover such extra fields without manufacturing native old results.'],
        whole_goal_complete=False)
    write(U/'REPLAY_RESULT.json',result)
    text='저장된 새 판정부 결과를 기존 공개 개발 영상 12개, 고정 동작 구간 15개에 추가 연결했습니다.\n'
    text+='기존 path·단계·손모양·관계·움직임 결과의 모든 값은 그대로입니다.\n'
    text+='새 구간 판정 집계: '+json.dumps(counts,ensure_ascii=False)+'\n'
    text+='이 결과는 연결 작동 확인입니다. 정확한 구간 정답률이나 수어 전체의 옳고 그름을 뜻하지 않습니다.\n'
    text+='원영상 읽기·새 모델 추론·DTW 재실행·학습은 모두 0회이며, 실제 앱 전체 대기시간 측정도 아닙니다.\n'
    with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as f:f.write(text)
    write(U/'REPLAY_FINISHED.json',dict(time=time.time(),result_sha256=sha(U/'REPLAY_RESULT.json'),status='COMPLETE'))
    print(json.dumps({'status':result['status'],'counts':counts,'preservation':len(preservation),'result_sha256':sha(U/'REPLAY_RESULT.json')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    try:
        if args.freeze==args.run:raise ValueError('Select exactly --freeze or --run')
        freeze() if args.freeze else run()
    except BaseException:
        p=U/'REPLAY_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
