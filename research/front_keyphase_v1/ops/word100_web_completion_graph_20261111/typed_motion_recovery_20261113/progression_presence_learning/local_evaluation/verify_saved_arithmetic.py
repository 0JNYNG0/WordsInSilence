"""Recount saved progression-head decisions; no models, re-scoring or labels edit."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json, hashlib

E=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check(rows, keys, saved, funcs):
    got={}
    for r in rows:
        key=tuple(r[k] for k in keys); c=got.setdefault(key,Counter());c['rows']+=1
        for name,fn in funcs.items():c[name]+=int(fn(r))
    want={tuple(r[k] for k in keys):{k:v for k,v in r.items() if k not in keys} for r in saved}
    assert {k:dict(v) for k,v in got.items()}==want,keys

def main():
    out=E/'INDEPENDENT_ARITHMETIC_VERIFICATION.json'
    assert not out.exists()
    d=read(E/'COMPARISON.json'); f=read(E/'RULES_FROZEN.json')
    assert sha(E/'RULES_FROZEN.json')==d['rules_sha256']
    assert sha(E.parent/'FROZEN.json')==f['training_frozen_sha256']
    for p,h in {**f['pins'],**d['prediction_pins']}.items(): assert sha(p)==h
    cm=dict(positive_labels=lambda r:r['expected_progression'],negative_labels=lambda r:not r['expected_progression'],
        true_positive=lambda r:r['expected_progression'] and r['predicted_progression'],
        false_negative=lambda r:r['expected_progression'] and not r['predicted_progression'],
        false_positive=lambda r:not r['expected_progression'] and r['predicted_progression'],
        true_negative=lambda r:not r['expected_progression'] and not r['predicted_progression'])
    bm={k:(lambda r,k=k:r[k]) for k in ['any_progression_raw','any_known_positive_supported','only_unknown_progression',
        'old_chosen_known_full','known_old_full_extent_supported','known_old_full_extent_unsupported',
        'old_hold_false_selection','previous_null_hold_false_selection','hold_false_positive_any_raw','hold_false_positive_old_chosen']}
    bm['old_chosen_present']=lambda r:r['old_chosen_index'] is not None
    bm['old_chosen_supported']=lambda r:r['old_chosen_supported'] is True
    pm=dict(body_gate_admitted=lambda r:r['previous_body_gate_admitted'],old_null_suppressed=lambda r:r['old_null_suppressed'],
        new_progression_positive=lambda r:r['predicted_progression'],new_positive_after_saved_body_gate=lambda r:r['after_same_partial_body_gate'],
        recovered_from_null_suppression=lambda r:r['recovered_from_null_suppression'],lost_previous_above_null=lambda r:r['lost_previous_above_null'])
    tables=[('candidate_summary','candidate_rows',['seed','split','family','label_kind','duration_bucket'],cm),
        ('candidate_all_durations','candidate_rows',['seed','split','family','label_kind'],cm),
        ('candidate_all_families','candidate_rows',['seed','split','label_kind'],cm),
        ('matched_duration_scope','candidate_rows',['seed','split','family','duration_bucket_in_fit_matched','label_kind'],cm),
        ('bag_summary','bag_rows',['seed','split','origin','family'],bm),
        ('bag_all_families','bag_rows',['seed','split','origin'],bm),
        ('partial_comparison','partial_rows',['seed','split','family','duration_bucket'],pm),
        ('partial_all_families','partial_rows',['seed','split'],pm)]
    for table,rows,keys,funcs in tables:check(d[rows],keys,d[table],funcs)
    known={(r['seed'],r['bag_index'],r['candidate_index']):r for r in d['candidate_rows']}
    assert len(known)==len(d['candidate_rows'])
    bagmap={(r['seed'],r['bag_index']):r for r in d['bag_rows']}
    unknown={(r['seed'],r['bag_index']):r for r in d['unknown_rows']}
    kc=Counter((r['seed'],r['bag_index']) for r in d['candidate_rows'])
    for r in d['candidate_rows']:
        assert r['predicted_progression']==(r['score']>=0)
        assert r['correct']==(r['expected_progression']==r['predicted_progression'])
        assert r['expected_progression']==(r['label_kind'] in ['full','partial'])
    for key,r in bagmap.items():
        assert r['known_candidates']==kc[key]
        assert kc[key]+unknown[key]['unknown_candidates']==r['candidate_count']
        assert unknown[key]['excluded_from_performance_denominator'] is True
        assert r['old_extent_unchanged'] is True
        chosen=known.get((*key,r['old_chosen_index']))
        if chosen:
            assert chosen['span']==r['old_chosen_span']
            assert chosen['predicted_progression']==r['old_chosen_supported']
    for r in d['partial_rows']:
        source=known[r['seed'],r['bag_index'],r['candidate_index']]
        assert source['label_kind']=='partial'
        assert source['score']==r['score'] and source['predicted_progression']==r['predicted_progression']
        assert r['recovered_from_null_suppression']==(r['old_null_suppressed'] and r['predicted_progression'])
    totals=[]
    for split in ['train','probe']:
        for kind in ['full','partial','hold']:
            rr=[r for r in d['candidate_rows'] if r['split']==split and r['label_kind']==kind]
            totals.append(dict(split=split,label_kind=kind,candidates=len(rr),predicted_progression=sum(r['predicted_progression'] for r in rr)))
    static=read(E.parent/'static_contract_audit/RESULTS.json')
    static_f=read(E.parent/'static_contract_audit/FROZEN.json')
    for p,h in static_f['pins'].items():assert sha(p)==h
    static_counts=[]
    for seed in [42,43,44]:
        rr=[r for r in static['rows'] if r['seed']==seed]
        assert len(rr)==72 and all(r['coordinates_and_masks_constant'] for r in rr)
        assert all(r['static_false_progression']==(r['static_logit']>=0) for r in rr)
        static_counts.append(dict(seed=seed,source_clips=len({r['clip_id'] for r in rr}),
            same_pose_inputs=len(rr),false_progression=sum(r['static_false_progression'] for r in rr),
            max_original_feature_reconstruction_error=max(r['original_feature_max_error'] for r in rr)))
    report=dict(at=datetime.now(timezone.utc).isoformat(),reviewer='/root/absence_contract_review_1113',
        status='PASS_SAVED_ARITHMETIC_BUT_STATIC_INPUT_CONTRACT_FAILED',
        comparison_sha256=sha(E/'COMPARISON.json'),script_sha256=sha(__file__),
        eight_summary_tables_exact=True,unknown_excluded=True,old_extent_unchanged=True,
        candidate_decisions=len(d['candidate_rows']),bag_decisions=len(d['bag_rows']),partial_decisions=len(d['partial_rows']),
        combined_candidate_totals=totals,static_input_audit=static_counts,
        old_known_full_spans_supported=sum(r['known_old_full_extent_supported'] for r in d['bag_rows']),
        old_known_full_spans_total=sum(r['old_chosen_known_full'] for r in d['bag_rows']),
        model_inference=False,rescoring=False,generator_modified=False,
        conclusion='Separating progression from extent restores most supported partial candidates while retaining old known extents and rejecting observed terminal holds. The separate head still mistakes constant copies of positive poses for progression; do not claim it learned motion robustly or deploy as omission detector.',
        next_plan_review=[
            'Add FIT-only static same-pose/same-duration negatives paired with real progressing windows, selected by source rules before scores. Include all predeclared start/middle/end anchors, not only failures.',
            'Repeat raw coordinates and masks, retain elapsed timing, recompute relation/velocity/normalized pose/local encoder without outside context. Preserve physical-source SHA, anchor, span, transform provenance.',
            'Keep real full and partial progression positives and real reviewed holds. Balance source/family/duration and synthetic/real sampling from FIT only; preserve old extent and family2.',
            'The synthetic label means zero observed keypoint progression, not an incorrect sign, absent real hand motion, semantic omission or completion. Static handshape signs must not be judged wrong.',
            'Do not treat the same72 synthetic inputs after training as fresh validation. Separately freeze held-out-source/static controls and real-data normal/partial/hold regressions. Existing probe remains upstream-exposed diagnostic.',
            'Perfectly repeated observations do not cover camera drift, detector noise, true small motion or occlusion. Synthetic-vs-real scores must remain separate.'
        ],static_audit_sha256=sha(E.parent/'static_contract_audit/RESULTS.json'),
        independent_human_gold=False)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    note='''진행존재 학습 결과 저장값 독립 검산

8개 집계표,미주석 분모 제외,기존 선택구간 보존,부분후보 대응이 모두 일치했다. 모델이나 생성기는 실행하지 않았다.

확인된 개선: FIT 부분진행은13113/13113, 노출진단 부분진행은4338/4356에서 진행있음으로 잡았다. 이전 null에서 지워졌던 부분진행이 대부분 회복됐다. 실제 유지 구간의 양성오류는0이고 기존에 맞던 전체구간505개 판단(학습426+진단79)을 모두 지원했다. 이들은 여러 관측방식·역순·seed가 겹치는 후보/판단 수이며 독립 사용자 수가 아니다.

추가 감사의 문제: 같은 자세의 관절을 같은 길이만큼 복사해 실제 관절변화가0인 입력72개에서, seed42/43/44가44/42/38개를 움직임으로 판단했다. 관절변화 대신 동작에 등장하는 자세를 보는 지름길이 남았을 가능성이 있다. 이것만으로 원인이 자세인지 인코더의 위치효과인지 단정할 수는 없다.

후속 보완은 타당하다. FIT의 실제 동작과 동일시간·동일자세 정지를 짝지어 진행존재 head만 보강하되, 실제 full/partial/hold 정답과 기존extent 선택기는 유지한다. 실패 입력만 골라 쓰지 말고 사전 지정한 시작·중간·끝 정지입력을 모두 포함한다.
정지 합성은 관절입력상 진행0이라는 정답이며 수어 오답이나 의미누락 정답이 아니다. 이72입력을 학습한 후 같은 입력 통과는 새검증으로 부르지 않는다. 별도사람/원본 정지진단과 실제 영상의 짧은진행·유지·정상 차이 회귀를 따로 본다.
완전정지 합성만으로 실제 웹캠 흔들림·검출오차·가림이나 의미있는 미세동작을 해결했다고 주장하면 안 된다. 현재는 웹 채택 근거가 아직 부족하다.
'''
    (E/'INDEPENDENT_REVIEW_KO.txt').write_text(note,encoding='utf-8')
    print(json.dumps(dict(status=report['status'],totals=totals,static=static_counts),ensure_ascii=False))

if __name__=='__main__':main()
