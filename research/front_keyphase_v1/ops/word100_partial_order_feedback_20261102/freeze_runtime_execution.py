"""Freeze the runtime closure and a single previously extracted public job."""
from order_runtime import *

def exclusive(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def main():
    assert not (Z/'RUNTIME_FROZEN.json').exists()
    scope=read(Z/'SCOPE_FROZEN.json')
    # The original scope's code hashes are retained as pre-review evidence.
    # Data/model/protocol pins remain immutable; new code fixes get new pins.
    for p,h in scope['pins'].items():
        if Path(p).parent.resolve()!=Z.resolve():assert sha(p)==h,p
    y=read(Y/'FROZEN.json')
    for p,h in y['pins'].items():assert sha(p)==h,p
    flow=read(model.FLOW_UNIT/'FLOW_INPUTS.json')
    context=dict(graphs=read(Y/'EVALUATION_PROTOCOL.json')['graphs'],
        families=read(model.D/'ACTION_TARGETS.json')['families'],
        guides=read(model.P/'GUIDE_SUPPORT.json')['guides'],
        flow_normalization={k:flow[k] for k in ['mean','std']},
        scope='Only FIT-derived reference context/normalization; no evaluation queries or input labels.')
    exclusive(Z/'RUNTIME_CONTEXT.json',context)
    files=set()
    for mod in list(sys.modules.values()):
        file=getattr(mod,'__file__',None)
        if file and Path(file).suffix=='.py' and Path(file).resolve().is_relative_to(O.resolve()):files.add(Path(file).resolve())
    files.update(Z/name for name in ['order_runtime.py','order_feedback.py','verify_runtime_input.py','SCOPE_FROZEN.json','RUNTIME_CONTEXT.json'])
    files.update(Path(p) for p in y['pins'] if Path(p).suffix in ['.py','.pt','.json','.npy'])
    files.update([
        model.OUT/'runs/tied_readout/model.pt',model.OUT/'runs/tied_readout/COMPLETE.json',
        model.D/'runs/episode_42/model.pt',model.B/'runs/joint_42/model.pt',model.A/'runs/temporal_42/model.pt',
        model.T/'runs/current_42/model.pt',model.S/'PHYSICAL_STATE_TARGETS.json',model.FLOW_UNIT/'TRAINING_RESULT.json',
        model.D/'ACTION_TARGETS.json',Y/'FROZEN.json',Y/'TRAINING_RESULT.json'])
    base=read(scope['base_result']); inp=Path(scope['new_runtime_input']); a=dict(np.load(inp,allow_pickle=False))
    validate_input(a,base['word_id'],base)
    pair=dict(job_id='fd785d173921436492c65e1703e46589',input=str(inp),input_sha256=sha(inp),
        base=scope['base_result'],base_sha256=sha(scope['base_result']),source_sha256=base['source_sha256'],
        word_id=base['word_id'],frames=base['frames'],native_timestamps_sha256=hashlib.sha256(a['timestamp_us'].tobytes()).hexdigest())
    files.update([inp,Path(scope['base_result'])])
    frozen=dict(status='FROZEN_AFTER_CODE_REVIEW_BEFORE_RUNTIME_INFERENCE',public_job=pair,
        reviewed_changes=['source input/base pair allowlist and hash binding','all current inference dependencies pinned','runtime context contains no evaluation queries'],
        old_y_pins_verified=len(y['pins']),pins={str(p):sha(p) for p in sorted(files)},
        max_tied_runtime_forwards=1,max_current_state_forwards=1,old_evaluation_rerun=False,new_training=0,whole_goal_complete=False)
    exclusive(Z/'RUNTIME_FROZEN.json',frozen);print('frozen',len(files),'files; prior pins',len(y['pins']))

if __name__=='__main__':main()
