"""Thin local orchestrator of verified Claude training and Codex audits."""
import datetime
import json
import os
import subprocess
import sys
import torch
from runtime_binding import U, RUN, OLD_RUN, read, setup, datasets, context, gw


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan,cfg,meta,factory,augment,_=setup()
    for name in ('METADATA_AUDIT.json','INTEGRATION_CHECK.json','CODE_REVIEW.json'):
        assert read(U/name)['status']=='PASS',name
    train,val,all_data=datasets(meta)
    ctx=context(plan,meta)
    resume='--resume' in sys.argv
    ctxpath=U/'EXECUTION_CONTEXT.json'
    if resume:
        assert read(ctxpath)==ctx
    else:
        assert not ctxpath.exists() and not RUN.exists(), 'Existing run: do not duplicate'
        ctxpath.write_text(json.dumps(ctx,ensure_ascii=False,indent=2),encoding='utf-8')
    fresh=factory()
    assert gw.state_digest(fresh.state_dict())==read(OLD_RUN/'INIT_SELECT.json')['state_digest']
    del fresh
    started={'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'run_dir':str(RUN),'resume':resume,'automatic_sequence':'selection -> fresh refit -> trace -> frozen855 -> independent audit'}
    (U/'LIVE_EXECUTION.json').write_text(json.dumps(started,indent=2),encoding='utf-8')
    print(json.dumps(started),flush=True)
    result=gw.run_experiment(factory,train,val,all_data,cfg,
                             dict(cfg,num_samples=plan['refit_num_samples']),str(RUN),ctx,
                             augment=augment,resume=resume)
    (U/'TRAINING_COMPLETION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TRAINING_AND_FREEZE_COMPLETE',flush=True)
    def run(script,*args):
        subprocess.run([sys.executable,'-X','utf8','-B',str(U/script),*args],cwd=U,check=True)
    run('check_training_trace.py')
    trace=read(U/'TRAINING_TRACE_AUDIT.json')
    assert trace['status']=='PASS' and trace['finalized']
    if not (RUN/'EVALUATION_SCORES.npz').exists():
        run('evaluate_frozen_candidate.py')
    else:
        assert (RUN/'EVALUATION_EXECUTION.json').exists(), 'Partial evaluation needs independent review'
    run('check_evaluation_inputs.py')
    if not (RUN/'AUDIT_COMPARISON.json').exists():
        run('check_comparison.py','--run-dir',str(RUN))
    assert read(RUN/'AUDIT_COMPARISON.json')['verification_passed'] is True
    (U/'COMPARISON_COMPLETION.json').write_text(json.dumps({'status':'COMPARED_AND_AUDITED_REPORT_PENDING',
        'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_dir':str(RUN)},indent=2),encoding='utf-8')
    print('COMPARISON_AND_INDEPENDENT_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':
    main()
