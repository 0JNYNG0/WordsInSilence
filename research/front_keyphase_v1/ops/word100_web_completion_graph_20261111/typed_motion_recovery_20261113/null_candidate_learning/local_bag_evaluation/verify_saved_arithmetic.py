"""Independent accumulation of saved decisions, no models or re-scoring."""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import json, hashlib

P=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def aggregate(rows, fields, partial=False):
    out={}
    for r in rows:
        k=tuple(r[f] for f in fields)
        c=out.setdefault(k,Counter()); c['rows']+=1
        if partial:
            c['body_gate_admitted']+=int(r['old_body_gate_admitted'])
            c['no_body_gate_admission']+=int(r['state']=='no_body_gate_admission')
            c['learned_null_suppressed']+=int(r['state']=='learned_null_suppressed')
            c['candidate_above_null']+=int(r['state']=='candidate_above_null')
            c['below_or_equal_null_ungated']+=int(not r['above_null_ungated'])
        else:
            c['baseline_candidate']+=int(r['baseline']['state']=='candidate')
            c['new_candidate']+=int(r['new']['state']=='candidate')
            c['no_body_gate_admission']+=int(r['new']['state']=='no_body_gate_admission')
            c['learned_null']+=int(r['new']['state']=='learned_null')
            c['baseline_known_positive']+=int(r['baseline']['known_positive'])
            c['new_known_positive']+=int(r['new']['known_positive'])
            c['new_ungated_known_positive']+=int(r['new_ungated']['known_positive'])
            c['lost_previously_positive']+=int(r['baseline']['known_positive'] and not r['new']['known_positive'])
            c['baseline_false_hold_selection']+=int(r['baseline_false_hold_selection'])
            c['new_false_hold_selection']+=int(r['new_false_hold_selection'])
    return {k:dict(v) for k,v in out.items()}

def check_summary(actual, saved, fields):
    expected={tuple(r[k] for k in fields):{k:v for k,v in r.items() if k not in fields} for r in saved}
    assert actual==expected, fields

def main():
    dst=P/'INDEPENDENT_ARITHMETIC_VERIFICATION.json'
    assert not dst.exists(), 'Preserve completed verification'
    d=read(P/'COMPARISON.json'); rules=read(P/'RULES_FROZEN.json')
    assert sha(P/'RULES_FROZEN.json')==d['rules_sha256']
    assert sha(P.parent/'FROZEN.json')==rules['training_frozen_sha256']
    for p,h in d['evidence_pins'].items(): assert sha(p)==h
    for p,h in rules['pins'].items(): assert sha(p)==h
    # Separate implementation: no import of evaluation code, no logits/model read.
    check_summary(aggregate(d['rows'],['seed','split','origin','family']),d['summary'],['seed','split','origin','family'])
    check_summary(aggregate(d['rows'],['seed','split','origin']),d['summary_all_families'],['seed','split','origin'])
    check_summary(aggregate(d['partial_rows'],['seed','split','family','duration_bucket'],True),d['partial_summary'],['seed','split','family','duration_bucket'])
    check_summary(aggregate(d['partial_rows'],['seed','split','family','frames'],True),d['partial_exact_duration'],['seed','split','family','frames'])
    assert len({(r['seed'],r['bag_index']) for r in d['rows']})==len(d['rows'])
    frozen_masks={(r['bag_index'],c['candidate_index']):(r,c) for r in rules['partial_masks'] for c in r['candidates']}
    assert len(d['partial_rows'])==3*len(frozen_masks)
    assert len({(r['seed'],r['bag_index'],r['candidate_index']) for r in d['partial_rows']})==len(d['partial_rows'])
    for r in d['partial_rows']:
        fr,c=frozen_masks[(r['bag_index'],r['candidate_index'])]
        for k,v in c.items(): assert r[k]==v
        a,b=c['transformed_span']; n=fr['source_length']
        assert c['original_span']==([n-1-b,n-1-a] if fr['reverse'] else [a,b])
        for g in c['groups']:
            assert 2<=len(g['witnessed_frames'])<len(g['all_group_witnesses'])
            assert g['outer_bounds'][0]<=c['original_span'][0]<=c['original_span'][1]<=g['outer_bounds'][1]
        # Verify stored arithmetic only; do not alter decisions or choose candidates.
        assert r['above_null_ungated']==(r['new_score']>r['null_logit'])
        assert r['old_body_gate_admitted']==(r['body_gate_logit']>=0)
        assert r['state']==('no_body_gate_admission' if not r['old_body_gate_admitted'] else 'candidate_above_null' if r['above_null_ungated'] else 'learned_null_suppressed')
    combined=aggregate(d['rows'],['split','origin'])
    family_partial=aggregate(d['partial_rows'],['split','family'],True)
    loss=[{k:r[k] for k in ['seed','bag_index','clip_id','split','view','reverse','family']} for r in d['rows'] if r['baseline']['known_positive'] and not r['new']['known_positive']]
    short=[r for r in d['partial_rows'] if r['split']=='probe' and r['family']==5 and r['frames']<=10]
    long=[r for r in d['partial_rows'] if r['split']=='probe' and r['family']==5 and r['frames']>=22]
    short_n=sum(r['state']=='learned_null_suppressed' for r in short)
    long_n=sum(r['state']=='learned_null_suppressed' for r in long)
    report=dict(at=datetime.now(timezone.utc).isoformat(),reviewer='/root/absence_contract_review_1113',
        result='PASS_SAVED_ARITHMETIC_NOT_ADOPTION',
        comparison_sha256=sha(P/'COMPARISON.json'),rules_sha256=d['rules_sha256'],script_sha256=sha(__file__),
        all_four_summary_tables_exact=True,unique_bag_decisions=len(d['rows']),
        unique_partial_candidate_decisions=len(d['partial_rows']),partial_masks_and_reverse_mapping_verified=True,
        combined=[dict(split=k[0],origin=k[1],**v) for k,v in combined.items()],
        partial_by_family=[dict(split=k[0],family=k[1],**v) for k,v in family_partial.items()],
        known_extent_losses=loss,
        family5_probe_duration_diagnostic=dict(short_at_most10_frames=dict(suppressed=short_n,total=len(short),ratio=short_n/len(short)),
            long_at_least22_frames=dict(suppressed=long_n,total=len(long),ratio=long_n/len(long))),
        interpretation=[
            'Hold false selections decrease210 to0 among216 FIT seed-bags;6 were already rejected by frozen body gate. Probe false selections72 to0 among72 seed-bags.',
            'Known full-extent-compatible selections decline426 to423/426 FIT decisions. Probe changes79 to82/144, but loses3 previous successes while gaining6. Aggregate gains do not eliminate regressions.',
            'Partial candidates supported by direct witnesses are frequently below null. These are overlapping correlated existing candidate windows, not17469 distinct clips/trials or educational errors.',
            'Family5 duration association supports possible short-duration/full-extent shortcut, not causal proof that duration alone is used. Terminal-pose or context shortcuts also remain possible.',
            'Probe partial evidence exists only for families4/5 under the frozen rule; no family3/7 partial success claim is supported.',
            'Observed movement existence and full source-extent compatibility should be separately supervised. A partial observed attempt is not equivalent to no movement or an incorrect sign.',
            'Local null mechanism should not be adopted as whole-sign omission/feedback failure judgment. Old extent selector preservation and separately learned existence are a justified next controlled experiment.',
            'Probe08/09 have upstream exposure; they are person-separated diagnostic, not independent final evaluation. Same-agent arithmetic review is not independent human/KSL gold truth.'
        ],new_model_inference=False,baseline_model_inference=False,rescoring=False,
        generator_source_modified=False,full_path82_comparison_not_reviewed_here=True)
    dst.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    note='''저장값 독립 산술검산 및 후속 설계 검토

집계4종,부분후보 마스크,역순 프레임 대응,해시 모두 일치했다. 평가 생성기나 모델을 실행하지 않았으며 저장된 판단을 합산·대조했다.

개선: 멈춰 유지하는 구간을 움직임으로 선택하던 경우가 학습210→0, 진단72→0으로 줄었다. 이는 원본24/8개에3관측방식과3seed를 적용한 관련된 판단 수다. 서로 다른282개 영상이 아니다. 학습의 나머지6건은 원래 body gate에서 빠졌다.

남은 문제: 전체 행동구간 정답에 맞는 선택은 학습426→423, 진단79→82다. 진단 합계는 늘었지만 기존정답3건을 잃고6건을 얻었다. 완전 보존 조건은 충족하지 못했다.
진행이 실제 확인되는 짧은 조상 계열 후보도 많이 사라졌다. 진단의10프레임 이하702후보 중683(97.3%),22프레임 이상885후보 중308(34.8%)이 null 아래였다. 겹치는 후보를 여러 방식·seed에서 본 수치이며, 교육적 오류율이나 새 영상 정답률이 아니다. 길이 자체만 이용했다는 인과증명도 아니다.

판단: 현재 학습 목표는 '전체 행동구간과 맞지 않는다'와 '움직임이 없다'를 함께 낮은 점수로 만드는 문제가 있다. 존재와 전체구간 적합성을 분리하는 방향이 맞다.

다음 최소 설계
1. 기존 전체구간 선택기는 동결한다. 별도 진행존재 판정부에 기존 전체구간과 직접 진행 표본2개 이상이 있는 부분구간을 진행있음으로 가르친다. 부분구간을 완료 정답으로 바꾸지 않는다.
2. 없음은 연속 RGB로 확인한 같은 family 유지 구간만 사용한다. 미주석·관측부족·다른family를 없음으로 확대하지 않는다. 역순도 방향오류가 드러나는 관측된 시도이므로 존재를 지우지 않는다.
3. FIT에서 길이별 겹침과 균형을 먼저 정하고 동결한다. 정지 마지막 자세만 보고 구별하는 지름길도 한계로 남긴다. family2는 원본 보존한다.
4. 모든 관측적격 후보의 진행존재와 기존 전체구간 결과를 따로 기록한다. 존재는 있으나 전체구간 근거가 부족하면 부분진행/구간불확실로 남긴다. 기존 선택기 한 후보만 거절하고 행동전체를 없음으로 바꾸지 않는다.

현재 null은 전체 수어 누락이나 교육적 오답 판정부로 채택할 수 없다. ID08/09 진단은 상류 노출이 있어 독립 최종평가가 아니다. 본 검산은 Codex 별도 산술검산이며 독립 사람·한국수어 전문가 정답은 아니다.
'''
    (P/'INDEPENDENT_REVIEW_KO.txt').write_text(note,encoding='utf-8')
    print(json.dumps(dict(result=report['result'],bag_decisions=len(d['rows']),partial_candidate_decisions=len(d['partial_rows']),short_null=[short_n,len(short)],long_null=[long_n,len(long)]),ensure_ascii=False))

if __name__=='__main__': main()
