from pathlib import Path
import json,hashlib,re
from collections import Counter
U=Path(__file__).resolve().parents[1]
def rd(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=rd(U/'COMPARISON.json');a=rd(U/'ADDITIONAL_REVIEW.json')
text=(U/'RESULT_KO.txt').read_text(encoding='utf-8-sig')
html=(U/'RESULT_KO.html').read_text(encoding='utf-8-sig')
narrative=re.sub('<script[^>]*>.*?</script>','',html,flags=re.S|re.I)
distribution=Counter(r['hidden_draft'] for r in c['masked_trials'])
checks={
 'mask_denominator_279':len(c['masked_trials'])==279,
 'class_imbalance_64_positive_215_negative':dict(distribution)=={0:215,1:64},
 'real_unknown_C_no_fills':c['center_summary']['C']['new_estimates']==c['all_frames_unknown_summary']['C']['new_estimates']==0,
 'secondary_recheck_four_conflicts_agree':len(a['masked_conflict_rechecks'])==4 and all(r['two_drafts_agree'] and r['original_draft']==r['second_image_draft'] for r in a['masked_conflict_rechecks']),
 'secondary_28_fields_sum':sum(a['secondary_field_agreement'].values())==28,
 'no_training_or_replacement_report':a['training'] is False and a['web_model_replaced'] is False,
 'same_AI_not_independent_human_label_status':a['not_independent_human_truth'] is True,
 'known_saved_comparison_unchanged':sha(U/'COMPARISON.json')==rd(U/'review/COMPARISON_VERIFY.json')['saved_comparison_sha256'],
 'own_image_labels_unchanged':sha(U/'review/IMAGE_LABELS.json')==sha(U/'review/IMAGE_LABELS_FIRST_PASS.json'),
}
assert all(checks.values()),checks
result={
 'status':'PASS_WITH_MINOR_CLARIFICATIONS','checks':checks,
 'scope_review':{
  'TCN_DTW_accuracy_claim':False,
  'automatic_RGB_model_claim':False,
  'all_temporal_models_impossible_claim':False,
  'same_AI_drafts_and_real_unknown_limit_explicit':True,
  'artificial_masking_and_correlated_trials_explicit':True,
  'C_coverage_loss_and_short_change_failure_explicit':True,
  'new_training_0_and_existing_preservation_explicit':True,
  'new_proposed_RGB_change_work_not_claimed_completed':True,
 },
 'major_numeric_or_logical_issues':[],
 'minor_clarifications':[
  {'location':'HTML 세 가지 방법 ②/③', 'reason':'최소 두 번 같은 상태라는 요약만 읽으면 나머지 관측이 반대여도 채우는 것으로 오해할 수 있음.', 'suggestion':'범위 안에서 확인된 모든 상태가 같아야 하며, 서로 다른 관측이 하나라도 있으면 보류한다는 한 문장 추가 권장.'},
  {'location':'TXT 결론 첫 문장 및 HTML 결론', 'reason':'실제로 가려졌던 상태에 대한 C 이득은 0이므로 안정된 모양의 누락 보완 효과는 인위적 마스킹 근거로 제한해야 함.', 'suggestion':'이번 인위적 가림 시험에서는 안정된 모양을 되찾는 경우가 있었다고 표현하면 범위가 더 정확함.'},
  {'location':'다음 단계 직전', 'reason':'현 규칙의 한계를 더 강한 학습형 시간 모델 전체의 한계로 확대하지 않도록 명시하면 좋음.', 'suggestion':'이번 결과는 앞뒤 상태 일치만 이용한 고정 규칙의 결과이며, RGB 변화까지 학습하는 시간 모델의 가능성을 부정하는 결과는 아니다.'}
 ],
 'reviewed_file_sha256':{p.name:sha(p) for p in [U/'RESULT_KO.html',U/'RESULT_KO.txt',U/'ADDITIONAL_REVIEW.json']},
 'review_only_original_files_unchanged':True,
 'visual_UI_render_review_in_this_pass':False,
}
(U/'review'/'REPORT_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'checks':checks,'major_issues':0,'minor_clarifications':3},ensure_ascii=False))
