"""New coherent-input contrast; old mixed-input scores remain immutable."""
from pathlib import Path
import sys,os,time,traceback,shutil,json,hashlib
from datetime import datetime,timezone,timedelta
import numpy as np
K=Path(__file__).resolve().parent;O=K.parent
J=O/'word100_sequence_supervision_audit_20261102'
P=O/'word100_learned_guide_feedback_20261102';B=O/'word100_learned_action_process_20261102'
T=O/'word100_learned_state_context_balance_20261102';D=O/'word100_action_episode_supervision_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def exclusive(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clock():return datetime.now(timezone(timedelta(hours=9))).isoformat()

def main():
    assert not (K/'INPUT_MANIFEST.json').exists()
    rows=[r for r in read(P/'INPUT_MANIFEST.json')['rows'] if r['role']=='development']
    assert len(rows)==12 and {r['person'] for r in rows}=={11,12,13}
    files=[J/'FROZEN.json',J/'VERIFIED.json',J/'PROVENANCE_AUDIT.json',T/'runs/current_42/model.pt',
        J/'runs/point_ce_42/model.pt',J/'runs/witness_ctc_42/model.pt',D/'DEV_ACTION_TARGETS.json',
        O/'word100_learned_body_relation_20261102/demo_adapter.py',P/'live_adapter.py',K/'prepare_coherent.py',K/'diagnose_coherent.py']
    plan=dict(at=clock(),status='FROZEN_NEW_INPUT_CONTRAST',source_count=12,recorded_ids=[11,12,13],
        purpose='Same full-sensor capture/RGB/native-axis contract as FIT216. Compare fixed J two heads on12 new input variants. No new training or model selection.',
        primary='witness_ctc_42',conditions=['point_ce_42','witness_ctc_42'],new_comparisons=24,
        causal_scope='Whole frontend package: RGB extent, observed coordinates, latent representation and T observer. Not isolated RGB-crop causality. Original video12same public sources; old evaluation results untouched.',
        output_scope='Input diagnosis only, not a replacement score or a full feedback/linguistic pass. No claim existing copied guide features changed automatically.',
        expected_contract='Full original sensor, no artificial lower boundary; source PTS and frame identity kept; original image availability; body-relative coordinates; existing SignRep checkpoint.',
        reserve_bytes=1024**3,estimated_max_new_bytes=256*1024**2,
        pins={str(p):sha(p) for p in files},source_rows=[{k:r[k] for k in ['clip_id','word_id','person','source','source_sha256','input','input_sha256','frames']} for r in rows],whole_goal_complete=False)
    plan['first_freeze_sha256']=sha(K/'FROZEN.json')
    plan['correction']='Before any extraction, avoid training module imports shadowing the runtime engine model.py. Same sources/conditions/metrics; original freeze retained.'
    if (K/'EXECUTION_FREEZE.json').exists():
        saved=read(K/'EXECUTION_FREEZE.json');assert saved['pins']==plan['pins'];plan=saved
    else:exclusive(K/'EXECUTION_FREEZE.json',plan)
    assert shutil.disk_usage(K).free>plan['reserve_bytes']+plan['estimated_max_new_bytes']
    sys.path.insert(0,str(O/'word100_learned_body_relation_20261102'));import demo_adapter
    assert Path(demo_adapter.__file__).parent.name=='word100_learned_body_relation_20261102'
    demo_adapter.U=K
    start=time.perf_counter();warm=demo_adapter.warmup();done=[]
    write(K/'CURRENT_EXECUTION.json',dict(at=clock(),status='PREPARING_NEW_INPUTS',pid=os.getpid(),warmup=warm,whole_goal_complete=False))
    for row in rows:
        cid=row['clip_id'];source=Path(row['source']);assert sha(source)==row['source_sha256']
        dest=K/'inputs'/cid
        if (dest/'RESULT.json').exists():result=read(dest/'RESULT.json')
        else:result=demo_adapter.analyze(source,dest,row['word_id'])
        a=np.load(dest/'analysis.npz');old=np.load(row['input'])
        assert len(a['frame_id'])==row['frames'] and np.array_equal(a['frame_id'],old['frame_id']) and np.array_equal(a['timestamp_us'],old['timestamp_us'])
        assert a['image_available'].all() and a['rgb'].shape==(row['frames'],768) and a['latent'].shape==(row['frames'],64)
        assert np.isfinite(a['rgb']).all() and np.isfinite(a['latent']).all() and np.isfinite(a['x'][a['mask']]).all()
        rec=dict(clip_id=cid,word_id=row['word_id'],person=row['person'],source=str(source),source_sha256=row['source_sha256'],
            input=str(dest/'analysis.npz'),input_sha256=sha(dest/'analysis.npz'),old_input=row['input'],old_input_sha256=row['input_sha256'],
            result=str(dest/'RESULT.json'),frames=row['frames'],source_axis_equal=True,actual_sensor_extent=result['actual_sensor_extent'],
            extraction_seconds=result['latency']['total_analysis_seconds'],device=result['device'])
        done.append(rec);write(K/'PREP_PROGRESS.json',dict(at=clock(),status='PREPARING',pid=os.getpid(),done=len(done),total=12,last=cid,elapsed_seconds=time.perf_counter()-start))
        assert shutil.disk_usage(K).free>plan['reserve_bytes'];print('prepared',len(done),12,cid,result['latency']['total_analysis_seconds'],flush=True)
    for p,h in plan['pins'].items():assert sha(p)==h
    exclusive(K/'INPUT_MANIFEST.json',dict(at=clock(),status='PREPARED_FULL_SENSOR_NATIVE12',rows=done,new_training=0,old_scores_revised=False,whole_goal_complete=False))
    write(K/'PREP_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None,done=12,total=12,elapsed_seconds=time.perf_counter()-start))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (K/'FIRST_FAILURE.txt').exists():(K/'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
