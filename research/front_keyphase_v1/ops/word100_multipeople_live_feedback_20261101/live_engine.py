"""Conservative evidence guard around the immutable base research pipeline."""
from pathlib import Path
import time,json
import numpy as np
import engine

def catalog():
    result=engine.catalog()
    result['evidence_guard']='arm_alignment_union_support_v1'
    return result

def analyze_submission(path,jobdir,progress=None):
    started=time.perf_counter()
    def base_progress(*args,**kwargs):
        if len(args)>1 and args[1]==100:return
        if progress:progress(*args,**kwargs)
    base=engine.analyze_submission(path,jobdir,base_progress)
    if not base.get('references'):return base
    if progress:progress('시간 대응의 관측 근거 확인',95,'손과 팔의 대응에서 빠진 중요한 관측이 없는지 확인합니다.')
    directory=Path(jobdir)/'adapter'
    with np.load(directory/'analysis.npz',allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in ['x','mask','rgb','probabilities','frame_id','timestamp_us']}
    arrays['feature_names']=engine.read(directory/'COMPLETE.json')['feature_names']
    bank,bank_arrays=engine.bank()
    if bank['observer']!='protected_baseline':raise ValueError('This guard adapter is frozen to the protected baseline probabilities.')
    rows=[r for r in bank['rows'] if r['word_id']==base['word']]
    from evidence_guard import guard_result
    t=time.perf_counter()
    guarded=guard_result(arrays,rows,bank_arrays,base)
    audit=guarded.pop('guard_audit')
    guarded['latency']={**base['latency'],'evidence_guard_seconds':time.perf_counter()-t,'total_analysis_seconds':time.perf_counter()-started}
    guarded['evidence_guard']=dict(applied=True,policy='arm_alignment_union_support_v1',only_similarity_can_be_downgraded=True,aggregate_changes=audit['aggregate_changes'],full_audit_saved_locally=True)
    engine.save(Path(jobdir)/'EVIDENCE_GUARD.json',audit)
    engine.save(Path(jobdir)/'FINAL_FEEDBACK.json',guarded)
    if progress:progress('분석 완료',100,'학습에 추가하지 않고, 확인할 수 있는 항목과 부족한 근거를 구분했습니다.')
    return guarded
