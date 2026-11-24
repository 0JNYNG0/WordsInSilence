"""Audited reuse of the original C optimizer/schedule. Never mutates original driver files."""
from pathlib import Path
import sys,json,hashlib,copy,datetime,os,traceback
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
B=O/'word100_body_boundary_20261017';I=O/'word100_wrist_only_inputs_20261026';D=O/'word100_wrist_only_dev_inputs_20261026'
sys.path.insert(0,str(B))
import training_driver as driver
import training_core as core
import torch
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return driver.sha(p)
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def now():return datetime.datetime.now().astimezone().isoformat()
def configure():
    # Output-only relocation; original numerical functions and scheduling are reused.
    driver.U=U;driver.CONTEXT=U/'CONTEXT.json';driver.PROGRESS=U/'PROGRESS.json'
    driver.configure_torch()

def plan():
    assert not (U/'PLAN.json').exists()
    original=read(B/'training_CONTEXT.json');pins={str(p):sha(p) for p in [B/'training_CONTEXT.json',B/'training_driver.py',B/'training_core.py',ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py',I/'FROZEN.json',D/'FROZEN.json',U/'control.py']}
    assert original['model']==core.MODEL and original['training']==core.TRAIN and original['runtime']==driver.runtime()
    assert original['pins'][str(B/'training_driver.py')]==sha(B/'training_driver.py')
    driver.configure_torch();checks=[]
    for seed in core.SEEDS:
        rp=B/f'training_C_{seed}_RESULT.json';cp=B/f'training_C_{seed}_best.pt';result=read(rp)
        assert result['status']=='COMPLETE_DEVELOPMENT_ONLY' and result['context_sha256']==sha(B/'training_CONTEXT.json') and result['checkpoint_sha256']==sha(cp)
        driver.set_seed(seed);model=driver.MaskedSignTCN(**core.MODEL)
        initial=driver.state_hash(model.state_dict());assert initial==result['initial_state_sha256']
        checks.append(dict(seed=seed,initial_state_sha256=initial,original_checkpoint_sha256=sha(cp),original_result_sha256=sha(rp),same_initial_state=True))
        pins[str(rp)]=sha(rp);pins[str(cp)]=sha(cp)
    # Inputs for the old controls are already reproduced exactly in the TRAIN and DEV preparation audits.
    ip=read(O/'word100_wrist_only_training_preflight_20261026/FROZEN.json')['pins']
    for r in original['ordered_rows']:
        for v in core.VIEWS:
            q=r['views'][v];assert q['channel_file'] is not None and sha(q['channel_file'])==q['sha256']
            if r['role']=='train_candidate':assert ip[q['channel_file']]==q['sha256']
    write(U/'PLAN.json',dict(at=now(),candidate='W: original C architecture + accepted missing-wrist bodyxy/body-valid only',
        pins=pins,protected=read(I/'FROZEN.json')['protected'],control_checks=checks,seeds=list(core.SEEDS),model=core.MODEL,training=core.TRAIN,runtime=driver.runtime(),
        comparison='Old C saved original pipeline versus W retrained new wrist-input pipeline. Same split/order/seeds/architecture/optimizer/schedule/early stop/selection. Old scores reused, never retrained or rescored unnecessarily.',
        selection='Unchanged arithmetic mean of masked/sensor DEV cross-entropy; min improvement1e-6, patience12, max80epochs. No best-seed deployment or test-based tuning.',
        diagnostic='After selecting W, fixed cross-input diagnostic: old C on modified DEV and W on original DEV. Unchanged-input rows reuse saved logits; changed-input/model pairs once only. Diagnostics never influence epochs or parameters.',
        gate='sensor mean correct count > old C; at least2 seeds nonregressing and at least1 improved; masked mean nondecreasing; no silent exclusions. All regressions reported. Not generalization proof.',
        readiness=[str(p/'ALL_VERIFIED.json') for p in (I,D)]+[str(p/'FULL_REVIEW.json') for p in (I,D)],
        data='Public899 TRAIN record IDs1-9; exposedDEV300 IDs11-13. Original1199 order preserved. User/mentor/fresh199/200 forbidden. No operating replacement.',
        trace='Per actual training batch store IDs, chosen view/time and actual padded x/mask/label hashes. This supplements original saved epoch aggregate traces.',
        no_training_yet=True))
    print('PLAN PASS: same initial state all3 seeds, runtime/config/input controls match; training not run',flush=True)

def freeze():
    plan=read(U/'PLAN.json');check(plan['pins']);check(plan['protected']);original=read(B/'training_CONTEXT.json')
    assert driver.runtime()==plan['runtime'];pins=plan['pins'].copy();pins[str(U/'PLAN.json')]=sha(U/'PLAN.json')
    for unit in (I,D):
        assert not (unit/'RUNNING.json').exists()
        verified=read(unit/'ALL_VERIFIED.json');review=read(unit/'FULL_REVIEW.json')
        assert verified['status']=='PASS' and review['ready_for_training'] is True
        assert review['verified_sha256']==sha(unit/'ALL_VERIFIED.json')
        assert verified['frozen_sha256']==sha(unit/'FROZEN.json')
        for p in (unit/'ALL_VERIFIED.json',unit/'FULL_REVIEW.json'):pins[str(p)]=sha(p)
    rows=[]
    for r in original['ordered_rows']:
        row={k:v for k,v in r.items() if k!='views'};row['views']={};unit=I if r['role']=='train_candidate' else D
        for view in core.VIEWS:
            uid=r['clip_id']+'__'+view;record=read(unit/'records'/(uid+'.json'));cp=unit/'features'/(uid+'.npz')
            assert record['outputs'][str(cp)]==sha(cp)
            with np.load(cp,allow_pickle=False) as z:
                for key in (('x_whole','x_context') if r['role']=='train_candidate' else ('x_whole',)):core.validate_channels(z[key])
            pins[str(cp)]=sha(cp);row['views'][view]=dict(channel_file=str(cp),sha256=pins[str(cp)],reason=None)
        rows.append(row)
    assert len(rows)==1199 and sum(r['role']=='train_candidate' for r in rows)==899
    context=dict(schema='wrist_only_C_retraining_v1',at=now(),pins=pins,runtime=driver.runtime(),ordered_rows=rows,labels=original['labels'],word_codes=original['word_codes'],
        model=core.MODEL,training=core.TRAIN,seeds=list(core.SEEDS),missingness_groups=original['missingness_groups'],train_count=899,development_count=300,
        controls=plan['control_checks'],original_context=str(B/'training_CONTEXT.json'),candidate_id='W',schedule_arm='C',
        score_status='Already exposed development, no final generalization claim',production_replacement=False)
    write(U/'CONTEXT.json',context);print('TRAINING CONTEXT FROZEN, no learning yet',flush=True)

def run():
    context=read(U/'CONTEXT.json');check(context['pins']);check(read(U/'PLAN.json')['protected']);configure()
    assert not (U/'RUNNING.json').exists();write(U/'RUNNING.json',dict(at=now(),pid=os.getpid(),context_sha256=sha(U/'CONTEXT.json')))
    train,dev=driver.load_data(context)
    original_batch=driver.torch_batch;original_selected=driver.selected_checkpoint;active={'seed':None}
    def audited_batch(rows,epoch=None,arm=None,view=None):
        x,p,y=original_batch(rows,epoch,arm,view)
        if epoch is not None:
            entry=dict(seed=active['seed'],epoch=epoch,clip_ids=[r['clip_id'] for r in rows],choices=[core.schedule(r['clip_id'],epoch,arm) for r in rows],
                x_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),pad_sha256=hashlib.sha256(p.numpy().tobytes()).hexdigest(),labels_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest())
            with (U/f'BATCH_TRACE_{active["seed"]}.jsonl').open('a',encoding='utf8') as out:out.write(json.dumps(entry,separators=(',',':'))+'\n')
        return x,p,y
    def selected(best,c,arm,seed):
        ck=original_selected(best,c,arm,seed);ck['candidate_id']='W';ck['feature_schema']=dict(builder='original pilot.data.build_channels + sparse accepted missing-wrist body coordinates; original scale/spans/local hand channels preserved',context_sha256=sha(U/'CONTEXT.json'))
        return ck
    driver.torch_batch=audited_batch;driver.selected_checkpoint=selected
    for seed in core.SEEDS:
        active['seed']=seed;result=driver.run_arm(context,train,dev,'C',seed)
        original=read(B/f'training_C_{seed}_RESULT.json');assert result['initial_state_sha256']==original['initial_state_sha256']
    check(context['pins']);check(read(U/'PLAN.json')['protected'])
    write(U/'TRAINING_DONE.json',dict(at=now(),status='THREE_CANDIDATE_RUNS_COMPLETE_PENDING_SCORE_TRACE_VERIFICATION',operating_changed=False))
    (U/'RUNNING.json').unlink()
if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except Exception:
        p=U/('ERROR_'+now().replace(':','_')+'.txt');p.write_text(traceback.format_exc(),encoding='utf8');raise
