"""Independent saved-logit/metadata/trace checks, without training or score regeneration."""
from pathlib import Path
import sys
import json
import hashlib
import math
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent
OLD=U.parent/'word100_training_20261013'
sys.path.insert(0,str(OLD))
from common import BASE,OUT,read,write,sha,pin_check

def main():
    assert not (U/'INDEPENDENT_CHECK.json').exists(),'Reuse existing verification'
    f=pin_check();frozen=read(U/'FROZEN_EXPERIMENT.json');run=Path(frozen['run'])
    assert all(sha(p)==h for p,h in frozen['source_sha256'].items())
    manifest=read(OLD/'MANIFEST.json');dev={r['clip_id']:r for r in manifest if r['role']=='development_only'}
    tr=[r for r in manifest if r['role']=='train_candidate'];assert len(tr)==899 and len(dev)==300
    assert {r['signer_id'] for r in tr}==set(range(1,10)) and {r['signer_id'] for r in dev.values()}=={11,12,13}
    bymethod={};predictions={}
    files={'baseline':BASE/'runs/word100_tcn_v1/whole_context/PREDICTIONS.json','candidate':run/'PREDICTIONS.json'}
    for fac in [.8,1.25]:
        for name in ['baseline','candidate']:files[f'{name}_speed{fac}']=run/f'STRESS_{name}_{fac}.json'
    for name,path in files.items():
        rr=read(path);assert len(rr)==len({r['clip_id'] for r in rr})==300 and {r['clip_id'] for r in rr}==set(dev)
        ce=[];right=Counter();cls=Counter();top5=0;observed={}
        for r in rr:
            mr=dev[r['clip_id']];assert mr['label_index']==r['label_index'] and mr['signer_id']==r['signer_id']
            z=np.asarray(r['logits'],np.float64);assert z.shape==(100,) and np.isfinite(z).all()
            arg=int(np.argmax(z));yes=arg==mr['label_index'];assert arg==r['prediction'] and yes==r['correct']
            right[r['signer_id']]+=yes;cls[r['label_index']]+=yes;top5+=mr['label_index'] in np.argsort(z)[-5:]
            high=z.max();ce.append(float(high+np.log(np.exp(z-high).sum())-z[mr['label_index']]))
            observed[r['clip_id']]=yes
        bymethod[name]=dict(correct=int(sum(right.values())),total=300,accuracy_percent=float(sum(right.values())/3),top5_correct=int(top5),top5_percent=float(top5/3),ce=float(np.mean(ce)),by_signer={str(s):int(right[s]) for s in [11,12,13]},macro_recall_percent=float(np.mean([cls[i]/3 for i in range(100)])*100))
        predictions[name]=observed
    result=read(run/'RESULT.json');metric=result['metrics'];h=read(run/'history.json');best=math.inf;best_ep=0
    assert len(h)==result['epochs_executed']<=80 and len(h)==len(result['order_sha256'])
    for ep,row in enumerate(h,1):
        assert row['epoch']==ep and row['step']==ep*math.ceil(899/32)
        assert np.isfinite([row['train_ce'],row['development_ce'],row['seconds']]).all()
        if row['development_ce']<best-1e-6:best=row['development_ce'];best_ep=ep
        assert row['best_epoch']==best_ep
        order=np.random.default_rng(f['train']['seed']+ep).permutation(899)
        expected=hashlib.sha256('|'.join(tr[i]['clip_id'] for i in order).encode()).hexdigest()
        assert result['order_sha256'][ep-1]==expected
    assert len(h)==80 or len(h)-best_ep==12
    assert best_ep==result['selected_epoch'] and abs(best-result['selected_development_ce'])<1e-6
    for name,summary in [('candidate',result),('baseline',frozen['baseline'])]:
        m=bymethod[name];assert m['correct']==summary['metrics']['correct']
        assert abs(m['ce']-summary['selected_development_ce'])<2e-6
        assert abs(m['accuracy_percent']-summary['metrics']['accuracy_percent'])<1e-9
        assert all(m['by_signer'][str(s)]==summary['metrics']['by_signer'][str(s)]['correct'] for s in [11,12,13])
    assert sha(run/'best.pt')==result['checkpoint_sha256'] and result['initial_state_sha256']==frozen['baseline']['initial_state_sha256']
    for row in read(run/'STRESS_SUMMARY.json'):
        m=bymethod[f"{row['model']}_speed{row['speed']}"];assert m['correct']==row['correct'] and abs(m['ce']-row['ce'])<2e-6
    a,b=predictions['baseline'],predictions['candidate']
    gains=sorted(k for k in a if not a[k] and b[k]);losses=sorted(k for k in a if a[k] and not b[k])
    detail=[]
    for change,ids in [('improved',gains),('regressed',losses)]:
        for k in ids:detail.append(dict(clip_id=k,label=dev[k]['canonical_label'],signer=dev[k]['signer_id'],change=change))
    success=bymethod['candidate']['correct']>290 and all(bymethod['candidate']['by_signer'][s]>=bymethod['baseline']['by_signer'][s] for s in ['11','12','13'])
    comparison=dict(status='VERIFIED_DEVELOPMENT_COMPARISON',success=success,decision='RESEARCH_CANDIDATE_ONLY_PENDING_USER_REVIEW' if success else 'DO_NOT_ADOPT',metrics=bymethod,
        delta_percentage_points=(bymethod['candidate']['correct']-290)/3,improved=gains,regressed=losses,changed_rows=detail,
        actual_training_runs=1,stress_conditions_per_model=600,blind_test=False,production_model_replaced=False,
        limitations=['same development set chooses epoch and evaluates candidate','one seed','nearest temporal sampling can omit brief details at faster speed','synthetic speed robustness is not new-user evidence','class recognition is not handshape correctness'])
    write(U/'COMPARISON.json',comparison)
    original=read(U/'ORIGINAL_GUARD.json');assert all(Path(p).is_file() and sha(p)==v for p,v in original.items())
    write(U/'INDEPENDENT_CHECK.json',dict(status='PASS',saved_prediction_rows_recounted=1800,trace_epochs=len(h),training_order_verified=True,checkpoint_sha256=result['checkpoint_sha256'],source_guards=len(original),source_guard_unchanged=True,verifier='Codex independent NumPy64 saved-logit/manifest/trace calculations; not another agent or external expert',new_training=0,new_inference=0))
    print(json.dumps(comparison,ensure_ascii=False))

if __name__=='__main__':main()
