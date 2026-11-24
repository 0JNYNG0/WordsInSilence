"""Read-only audit of the frozen AIHub100 inputs and saved development predictions."""
from pathlib import Path
import sys
import json
import numpy as np

U=Path(__file__).resolve().parent
OLD=U.parent/'word100_training_20261013'
sys.path.insert(0,str(OLD))
from common import OUT,BASE,ROOT,read,write,sha,pin_check

def describe(v):
    a=np.asarray(v,float)
    return dict(n=len(a),min=float(a.min()),q25=float(np.quantile(a,.25)),median=float(np.median(a)),q75=float(np.quantile(a,.75)),max=float(a.max())) if len(a) else None

def internal_gaps(valid,dt):
    ids=np.flatnonzero(valid)
    if not len(ids):return dict(observed=0,gaps=0,missing=0,max_seconds=0.)
    vv=valid[ids[0]:ids[-1]+1]; dd=dt[ids[0]:ids[-1]+1]
    runs=[];run=0.
    for ok,t in zip(vv,dd):
        if not ok:run+=float(t)
        elif run:runs.append(run);run=0.
    return dict(observed=int(valid.sum()),gaps=len(runs),missing=int((~vv).sum()),max_seconds=max(runs,default=0.))

def main():
    assert not (U/'AUDIT.json').exists(), 'Reuse completed audit'
    f=pin_check(); rows=read(OLD/'MANIFEST.json')
    preds={r['clip_id']:r for r in read(BASE/'runs/word100_tcn_v1/whole_context/PREDICTIONS.json')}
    guard_paths=list((BASE/'runs/ses_79977f57893d').rglob('*'))
    guard_paths+=[BASE/'runs/word100_tcn_v1/whole_context/best.pt',BASE/'runs/word100_tcn_v1/whole_context/PREDICTIONS.json',BASE/'config/word100_practice.json',BASE/'config/word100_analysis_references.json']
    guard_paths+=list((BASE/'src/wis_lab').rglob('*.py'))+list((BASE/'web').rglob('*'))
    write(U/'ORIGINAL_GUARD.json',{str(p):sha(p) for p in guard_paths if p.is_file()})
    write(U/'SCOPE.json',dict(authorization='2026-10-13 user requests audit, justified robustness candidate, DTW96 in that order; future report after results',implementer='Codex',verifier='Codex separate saved-data calculations; no external reviewer',protected='user/mentor data never training/tuning/selection; no heldout/model replacement/camera/download/Git',baseline=f['run'],baseline_hash=sha(BASE/'runs/word100_tcn_v1/whole_context/best.pt')))
    checked=read(OLD/'INPUT_CHECK.json'); out=[]
    for r in rows:
        p=OUT/'records'/f"{r['clip_id']}.json"; assert sha(p)==checked['record_hashes'][r['clip_id']]
        rec=read(p); assert sha(rec['channel_file'])==rec['channel_sha256']
        with np.load(rec['channel_file'],allow_pickle=False) as z:
            x=z['x_whole'];ts=z['whole_ts'];ids=z['whole_frame_id']
        valid=x[:,237:279].reshape(-1,2,21).any(2); dt=x[:,-1]
        a,b=r['intervals'][0];core=(ts>=round(a*1e6))&(ts<=round(b*1e6))
        item=dict(clip_id=r['clip_id'],role=r['role'],signer=r['signer_id'],code=r['word_code'],label=r['canonical_label'],
                  frames=len(x),seconds=float((ts[-1]-ts[0])/1e6),core_seconds=b-a,core_fraction=float(core.mean()),
                  max_dt_ms=float(dt.max()*1000),both_missing_fraction=float((~valid.any(1)).mean()),
                  shoulder_missing_fraction=float((x[:,187]*x[:,188]==0).mean()),
                  hands=[internal_gaps(valid[:,j],dt) for j in range(2)],
                  channel_sha256=rec['channel_sha256'],observation_sha256=rec['observation_sha256'])
        if r['clip_id'] in preds:
            pr=preds[r['clip_id']];item.update(correct=pr['correct'],prediction=f['labels'][pr['prediction']])
        out.append(item)
    tr=[r for r in out if r['role']=='train_candidate'];dv=[r for r in out if r['role']=='development_only']
    for r in out:
        peers=[q for q in tr if q['code']==r['code']]
        r['duration_over_training_word_median']=r['core_seconds']/np.median([q['core_seconds'] for q in peers])
        r['max_internal_gap_seconds']=max(q['max_seconds'] for q in r['hands'])
        r['internal_missing_frames']=sum(q['missing'] for q in r['hands'])
    fields=['frames','seconds','core_fraction','duration_over_training_word_median','max_dt_ms','both_missing_fraction','shoulder_missing_fraction','max_internal_gap_seconds','internal_missing_frames']
    groups={'train899':tr,'development_correct290':[r for r in dv if r['correct']],'development_errors10':[r for r in dv if not r['correct']]}
    summary={g:{k:describe([r[k] for r in rr]) for k in fields} for g,rr in groups.items()}
    write(U/'AUDIT_ROWS.json',out)
    result=dict(status='COMPLETE_DESCRIPTIVE_NOT_CAUSAL',rows=1199,groups=summary,errors=[r for r in dv if not r['correct']],
                by_signer={str(s):dict(total=100,correct=sum(r['correct'] for r in dv if r['signer']==s),duration_ratio=describe([r['duration_over_training_word_median'] for r in dv if r['signer']==s])) for s in [11,12,13]},
                note='Word annotation duration for descriptive audit only; development input unchanged; per-hand absence is not automatically detector error; no counterfactual inference.')
    write(U/'AUDIT.json',result)
    print(json.dumps(dict(groups=summary,errors=[{k:r[k] for k in ['label','signer','duration_over_training_word_median','core_fraction','max_internal_gap_seconds','both_missing_fraction']} for r in result['errors']],by_signer=result['by_signer']),ensure_ascii=False))

if __name__=='__main__':main()
