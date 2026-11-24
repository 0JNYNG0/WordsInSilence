"""Read-only artifact audit and design note; no learned inference or evaluation."""
from pathlib import Path
import json, hashlib
from collections import Counter
from datetime import datetime, timezone
import numpy as np

OUT = Path(__file__).resolve().parent
HERE = OUT.parent.parent
PRIOR = HERE.parent/'candidate_description_supervision'
N = HERE/'null_candidate_learning'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    target=OUT/'NULL_CONTRACT_REVIEW.json'
    assert not target.exists(), 'Preserve prior review'
    meta=read(N/'INPUTS_COMPLETE.json')
    audit=read(N/'FIT_BASELINE_AUDIT.json')
    original=read(PRIOR/'local_rank_learning/INPUTS_COMPLETE.json')
    reviews={r['clip_id']:r for p in [OUT.parent/'REVIEW_COMPLETE.json',OUT/'REVIEW_COMPLETE.json'] for r in read(p)['rows']}
    counts=Counter(); negative_lengths={}; checks=0
    for row in meta['bags']:
        assert sha(row['path'])==row['sha256']; checks+=1
        z=np.load(row['path']); r=reviews[row['clip_id']]; review=r['review']
        assert row['source_span']==review['hold_span']
        assert row['family']==review['target_family'] and row['family'] in [3,4,5,7]
        assert row['source_sha256']==r['source_sha256']
        a,b=row['source_span']; expected=np.arange(a,b+1)
        assert np.array_equal(z['source_frames'],expected)
        assert not set(expected)&set(review['uncertain_frames'])
        assert review['all_frames_inspected'] and set(expected)<=set(review['inspected_frames'])
        assert not z['y'].any() and len(z['y'])==row['candidate_count']
        assert len(z['local_full'])==len(z['body_features'])==len(z['spans'])==len(z['y'])
        assert np.isfinite(z['local_full']).all() and np.isfinite(z['body_features']).all()
        assert (z['spans'][:,0]>=0).all() and (z['spans'][:,1]<len(expected)).all()
        assert (z['spans'][:,1]-z['spans'][:,0]>=3).all()
        assert row['split']==('probe' if row['person'] in [8,9] else 'train')
        assert row['person'] in [2,3,4,5,6,7,8,9]
        assert row['semantic_completion'] is None
        counts[(row['split'],row['family'])]+=1
        if row['split']=='train':
            negative_lengths.setdefault(row['family'],[]).extend((z['spans'][:,1]-z['spans'][:,0]+1).tolist())
    assert not ({r['source_sha256'] for r in meta['bags'] if r['split']=='train'} & {r['source_sha256'] for r in meta['bags'] if r['split']=='probe'})
    assert meta['probe_predictions_computed'] is False and audit['no_probe_scoring'] is True
    positive_lengths={}
    for r in original['bags']:
        if r['split']!='train': continue
        assert sha(r['path'])==r['sha256']; checks+=1
        z=np.load(r['path']); sp=z['spans'][z['y']]
        positive_lengths.setdefault(r['family'],[]).extend((sp[:,1]-sp[:,0]+1).tolist())
    def limits(v): return dict(min=int(min(v)),median=float(np.median(v)),max=int(max(v)))
    findings=[
        dict(level='SUPPORTED',topic='Labels and split',detail='96 bags=32 public originals x3 observation views. 72 train bags from24 originals/people02..07;24 diagnostic bags from8 originals/people08,09. The latter have upstream exposure, not final evaluation. All negative candidates stay inside same-family continuously RGB-reviewed hold spans; no uncertain frames entered.'),
        dict(level='SUPPORTED',topic='Baseline need',detail='Saved FIT audit reports210/216 seed-bag decisions where frozen body_window admits at least one candidate and local_full necessarily selects it. This is a local non-progression false selection, NOT210 distinct videos, whole-sign omission errors, or educational errors. No diagnostic prediction read or recomputed.'),
        dict(level='CONDITION',topic='Family2 preservation',detail='Fine-tuning the shared96-unit hidden layer changes every output including family2 even if its final output weights are frozen. At inference route family2 through the complete original local_full model for that seed. New null is supported only for families3,4,5,7; do not apply it to family2 or other families.'),
        dict(level='CONDITION',topic='Short motion confound',detail='Family5 positive FIT compatible spans are23..43 frames (median30), whereas reviewed hold candidate spans are at most21. Short duration is therefore a plausible shortcut. Existing normal-speed/cross regression is necessary but not sufficient: separately retain attempts with observable short/partial family movement, and report unsupported if these lack independent labels. Do not discard short movement or call it a completed/omitted sign using this null target.'),
        dict(level='CONDITION',topic='Loss and candidate set',detail='For observed positive bags, use logsumexp(candidate scores plus family null)-logsumexp(compatible positive scores). For reviewed hold bags, use logsumexp(candidate scores plus null)-null. Preserve target-family selection only, no forced order/word rules. Pin positive/null sampling ratio and candidate cap before probe. Sampled-candidate training and exhaustive-candidate runtime differ in listwise mass; no post-probe threshold, multiplier, or cap tuning.'),
        dict(level='CONDITION',topic='Runtime semantics',detail='Keep exactly the same frozen body_window gate for baseline and candidate comparison. Choose null versus the best eligible candidate; family scalar null is a learned score boundary, not evidence-free threshold removal. If body observation does not support a candidate, retain observation-insufficient status rather than claiming non-progression. Null says no supported progressing candidate in this local search domain, not that the whole sign was omitted.'),
        dict(level='CONDITION',topic='Transfer and comparison',detail='Copy each seed original local_full head and its normalization; frozen common100 encoder and body_window. Do not update means/stds using probe. Freeze1200steps,seeds42..44,other configured hyperparameters before probe. Compare old and new on identical candidates/gates and inspect positive span retention, normal-speed/cross, wrong-shape/direction/order/omission separately. Reject any adoption that hides previously retained wrong attempts or loses normal movements.'),
        dict(level='LIMITATION',topic='Evidence quality',detail='RGB annotation by Codex agents/root review is not independent human/KSL expert truth. Hold ranges are short and terminal only; they do not cover all non-action, uncertainty, partial-action, or return-to-rest cases. This is a justified bounded research test, not a guarantee of general feedback correctness.'),
    ]
    files=[HERE/'prepare_absence.py',HERE/'common.py',N/'INPUTS_COMPLETE.json',N/'FIT_BASELINE_AUDIT.json',PRIOR/'local_rank_core.py',PRIOR/'prepare_local_rank.py',PRIOR/'train_local_rank.py',PRIOR/'evaluate_local_rank.py',PRIOR/'transfer_local_timefix_core.py',PRIOR/'local_rank_learning/FROZEN.json',PRIOR/'local_rank_learning/INPUTS_COMPLETE.json']
    report=dict(at=datetime.now(timezone.utc).isoformat(),reviewer='/root/absence_contract_review_1113',
        verdict='SUPPORTED_BOUNDED_EXPERIMENT_WITH_LISTED_CONDITIONS_NOT_ADOPTION_APPROVAL',
        basis='Source code, pinned saved metadata/labels/features, saved FIT audit only. No model execution, diagnostic evaluation, or final8 access.',
        file_hash_checks=checks,
        bags=len(meta['bags']),counts={f'{split}_family{f}':v for (split,f),v in counts.items()},
        positive_fit_span_lengths={str(k):limits(v) for k,v in positive_lengths.items()},
        hold_fit_candidate_lengths={str(k):limits(v) for k,v in negative_lengths.items()},
        fit_old_false_selections=audit['false_selection_count'],fit_old_decisions=audit['total'],
        probe_predictions_read_or_run=False,training_or_model_inference_run=False,
        findings=findings,files={str(p):sha(p) for p in files})
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    text='''공통 null 후보 학습 설계 검토

판정: 확인된 강제 선택 문제를 다루는 작은 비교 실험으로 타당하다. 적용 승인이나 전반적인 피드백 개선 증명은 아니다.

정답 적합성:32공개 원본×3관측 방식=96묶음. 학습24원본/72묶음, 진단8원본/24묶음으로 분리된다. 모든 null 후보는 같은 family의 연속 RGB 확인 유지 구간 안에만 있고 불확실 프레임은 포함되지 않았다. ID08/09는 상류 노출이 있어 독립 최종평가가 아니다.

필요성:기존 방식의 FIT 기록210/216은 유지 구간에서도 후보가 선택된 경우다. 216개 영상이나 사용자 오답 건수가 아니라72묶음×3seed다. 이 결과를 전체 수어 누락 판정 실패로 표현하면 안 된다.

반드시 지킬 조건
1. family2는 새 head의 출력만 얼려서는 보존되지 않는다. 공유 hidden이 바뀌므로 원본 모델 전체에서 나온 family2 출력을 그대로 사용한다.
2. family5의 기존 정상 정답 span은23~43프레임, 새 유지 후보는 최대21프레임이다. 단순히 짧으면 null을 택하는 문제를 확인해야 한다. 실제 진행이 있는 짧은·부분 동작이 사라지면 채택하지 않는다. 근거가 없는 부분은 보류로 보고한다.
3. 같은 몸 관측 gate와 후보들로 비교한다. null은 학습한 점수 경계이며, 별도 수동 임계값을 없앴다고 판단 근거가 자동 확보되는 것은 아니다. 관측 부족은 null/행동 부재와 따로 표시한다.
4. 정상/짧은 움직임/교차/속도 변화 및 잘못된 손모양·방향·순서 시도가 후보에서 사라지는지 함께 본다. 유지 오선택만 줄고 정상 또는 오류 시도를 놓치면 미채택한다.
5. 원본 head와 정규화를 복사하고100단어 인코더·body_window는 동결한다. seed,1200step,양성/null 묶음 비율,후보 표본 수를 probe 전 고정하고 probe를 보고 수치·규칙을 바꾸지 않는다.

국소 유지 구간에서 움직임이 더 진행되지 않는다는 정답일 뿐, 수어의 완료·생략·교육적 정오 정답이 아니다. family2나 다른 항목에 부재 정답을 자동 확대하지 않는다.
Codex의 RGB 주석은 독립 사람·한국수어 전문가 정답과 구분한다. root가 최종 학습 계약과 후속 채택을 판단한다. 이 검토에서는 학습·모델 추론·평가를 실행하지 않았다.
'''
    (OUT/'NULL_CONTRACT_REVIEW_KO.txt').write_text(text,encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['verdict','file_hash_checks','bags','positive_fit_span_lengths','hold_fit_candidate_lengths']},ensure_ascii=False))

if __name__=='__main__': main()
