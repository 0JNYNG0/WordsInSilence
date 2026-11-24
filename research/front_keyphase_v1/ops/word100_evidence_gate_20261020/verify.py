"""Independent scalar checks; deliberately does not import criterion or audit."""
from pathlib import Path
import json,hashlib,math,traceback
from collections import Counter
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def scalar(d,refs,required,t=None):
    if required is False:return 'NOT_REQUIRED'
    if d['coverage'] is None or d['observed_frames']<8 or refs<5:return 'INSUFFICIENT'
    if t and (d['coverage']<t['coverage_floor']-1e-7 or d['gap_fraction']>t['gap_ceiling']+1e-7):return 'INSUFFICIENT'
    return 'PARTIAL' if d['missing_s']>1e-8 else 'OBSERVED_ENVELOPE'
def quantile(xs,q):
    a=sorted(xs);v=(len(a)-1)*q;i=math.floor(v);j=math.ceil(v)
    return a[i]+(a[j]-a[i])*(v-i)
def check_evidence(d):
    if d['coverage'] is None:return
    assert 0<=d['coverage']<=100 and 0<=d['gap_fraction']<=1+1e-8
    missing=sum(x['end_s']-x['start_s'] for x in d['missing_intervals'])
    assert abs(missing-d['missing_s'])<1e-7
    assert abs(d['coverage']-100*(1-missing/d['duration_s']))<1e-6
    longest=max([x['end_s']-x['start_s'] for x in d['missing_intervals']],default=0)
    assert abs(d['gap_fraction']-longest/d['duration_s'])<1e-7
def run():
    f=read(U/'FROZEN.json');cal=read(U/'CALIBRATION.json');T=cal['thresholds']
    assert sha(U/'TRAIN_ROWS.json')==cal['train_rows_sha256']
    train=read(U/'TRAIN_ROWS.json');base=read(U/'BASELINE.json');stress=read(U/'STRESS.json');controls=read(U/'CONTROLS.json');summary=read(U/'SUMMARY.json')
    assert len(train)==899*2*6 and len(base)==300*2*6 and len(stress)==591*4*3
    assert {r['signer_id'] for r in train}==set(range(1,10))
    assert {r['signer_id'] for r in base}=={11,12,13}
    assert len({r['clip_id'] for r in train})==899 and len({r['clip_id'] for r in base})==300
    assert not {r['clip_id'] for r in train}&{r['clip_id'] for r in base}
    for k,t in T.items():
        rr=[r for r in train if r['key']==k and r['required'] is not False and scalar(r['evidence'],r['valid_refs'],r['required'])!='INSUFFICIENT']
        assert len(rr)==t['n']
        assert abs(quantile([r['evidence']['coverage'] for r in rr],.05)-t['coverage_floor'])<1e-8
        assert abs(quantile([r['evidence']['gap_fraction'] for r in rr],.95)-t['gap_ceiling'])<1e-8
    checks=0
    for r in base+stress:
        d=r['evidence'];check_evidence(d)
        assert scalar(d,r['valid_refs'],r['required'])==r['g0']['status']
        assert scalar(d,r['valid_refs'],r['required'],T[r['key']])==r['g1']['status']
        assert r['g0']['wrong_action'] is None and r['g1']['wrong_action'] is None
        checks+=2
        if 'fixed_evidence' in r:
            check_evidence(r['fixed_evidence']);assert scalar(r['fixed_evidence'],r['valid_refs'],r['required'],T[r['key']])==r['fixed_g1']['status'];checks+=1
    assert summary['baseline_g1']==dict(Counter(r['g1']['status'] for r in base))
    for cond,s in summary['stress'].items():
        rr=[r for r in stress if r['condition']==cond and r['removed_frames']>0]
        fall=[r for r in rr if r['witness_score_decreased']]
        assert s['g1']==dict(Counter(r['g1']['status'] for r in rr))
        assert s['score_fall_g1']==dict(Counter(r['g1']['status'] for r in fall))
        assert all(r['g1']['status']=='INSUFFICIENT' for r in rr if cond=='whole')
    assert summary['repeat']['changed']==sum(r['before']!=r['after'] for r in controls)
    pinchecks=0
    for filename in ('TRAIN_INPUT_PINS.json','DEV_INPUT_PINS.json'):
        for p,h in read(U/filename).items():assert sha(p)==h,p;pinchecks+=1
    for key in ('protected','previous','code'):
        for p,h in f[key].items():assert sha(p)==h,p;pinchecks+=1
    write('VERIFICATION.json',dict(status='PASS',scalar_gate_decisions=checks,train_calibration_only=True,split_disjoint=True,
        input_and_preservation_hashes=pinchecks,interval_arithmetic=True,whole_hand_unavailable=True,optional_hand_isolation=True,
        operator='Codex independent scalar implementation; not a separate reviewer',model_training=0,user_data_used=0))
    print('VERIFIED',checks,pinchecks)
if __name__=='__main__':
    try:run()
    except Exception:
        p=U/'FIRST_ERROR_verify.txt'
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
