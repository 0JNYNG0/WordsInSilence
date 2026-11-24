"""Audit only frozen fit arrays/results. Reproduce arithmetic on saved paths/candidates.
No model, new search, DTW, compare, or free_order call is made.
"""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import numpy as np

U=Path(__file__).resolve().parents[1]
OLD=U.parent/'word100_relational_sequence_compare_20261031'
PH=U.parent/'word100_core_stage_evidence_20261031'/'PHASE_DATA.json'
pins={};check_count=0
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):pins[str(p)]=sha(p)
def read(p):
    pin(p);return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def arrays(p):
    pin(p)
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ['x','mask','timestamp_us','frame_id']}
auth=read(OLD/'AUTHORIZATION.json');rows={r['alias']:r for r in auth['rows'] if r['role']=='fit'}
assert len(rows)==12 and {r['person'] for r in rows.values()}=={1,2,3}
phases={c['alias']:c for c in read(PH)['clips'] if c['alias'] in rows}
ranges=read(OLD/'REFERENCE_RANGES.json')['ranges']['C_relational_path']
lookup={(r['reference'],r['segment_id'],r['group']):r for r in ranges}
diagnostic=read(OLD/'CONTROL_RANGE_DIAGNOSTIC.json')
pin(OLD/'relational.py');pin(OLD/'run_compare.py');pin(OLD/'control_ranges.py');pin(Path(__file__))
freeze=read(OLD/'FROZEN_V2.json');assert freeze['pins'][str(OLD/'relational.py')]==pins[str(OLD/'relational.py')]
spec=importlib.util.spec_from_file_location('frozen_relational_features_only',OLD/'relational.py')
rel=importlib.util.module_from_spec(spec);spec.loader.exec_module(rel)
def prohibited(*args,**kwargs):raise RuntimeError('No search or comparison rerun in cause audit')
rel.dtw=rel.compare=rel.free_order=prohibited
base={a:arrays(r['input']) for a,r in rows.items()}
assert all(pins[r['input']]==r['input_sha256'] for r in rows.values())

identity=read(OLD/'controls'/'P063_identity.json')
identity_arr=arrays(OLD/'controls'/'P063_identity.npz')
assert sha(OLD/'controls'/'P063_identity.npz')==identity['array_sha256']
assert all(np.array_equal(identity_arr[k],base['P063'][k]) for k in identity_arr)
assert all(i==j for i,j in identity['path']) and len(identity['path'])==len(identity_arr['x'])
f=rel.features(identity_arr)
anchors={a['segment_id']:a for a in identity['free_order']['anchors']}
candidate_tables=[];stage_templates=[]
for sid in ['P063_s05','P063_s06']:
    saved=anchors[sid];ids=np.arange(saved['reference_core'][0],saved['reference_core'][1]+1)
    template={};support=[]
    for g in rel.GROUPS:
        x,m=f[g];counts=m[ids].sum(0);minimum=max(1,int(np.ceil(.75*len(ids))))
        expected=counts>=minimum
        mean=np.divide(np.where(m[ids],x[ids],0).sum(0),counts,out=np.zeros(x.shape[1]),where=counts>0)
        template[g]=(mean,expected)
        support.append({'group':g,'total_dimensions':int(x.shape[1]),'retained_dimensions':int(expected.sum()),
          'valid_native_frame_counts_per_dimension':counts.tolist(),'required_native_frames':minimum,
          'fully_observed_frame_ids':ids[m[ids].all(1)].tolist(),
          'partially_or_fully_missing_frame_ids':ids[~m[ids].all(1)].tolist()})
    stage_templates.append({'segment_id':sid,'core':saved['reference_core'],
      'description_ko':next(s['description_ko'] for s in phases['P063']['segments'] if s['segment_id']==sid),
      'support_threshold':.75,'retained_groups':[s['group'] for s in support if s['retained_dimensions']],
      'excluded_groups':[s['group'] for s in support if not s['retained_dimensions']],
      'retained_dimensions':sum(s['retained_dimensions'] for s in support),
      'all_possible_dimensions':sum(s['total_dimensions'] for s in support),'group_support':support})
    for cand in saved['five_nearest_candidates']:
        center=cand['center'];win=np.arange(max(0,center-2),min(len(identity_arr['x']),center+3))
        per=[];expected_total=0;shared_total=0
        for g in rel.GROUPS:
            x,m=f[g];mean,expected=template[g]
            if not expected.any():continue
            shared=m[win]&expected;den=int(expected.sum()*len(win));cnt=int(shared.sum())
            cost=float((np.where(shared,np.minimum(np.abs(x[win]-mean),1),0).sum()+(den-cnt))/den)
            per.append({'group':g,'cost':cost,'expected_cells':den,'observed_cells':cnt})
            expected_total+=den;shared_total+=cnt
        co=float(np.mean([v['cost'] for v in per]));coverage=shared_total/expected_total
        assert np.isclose(co,cand['cost'],atol=1e-12,rtol=0)
        assert np.isclose(coverage,cand['coverage'],atol=1e-12,rtol=0)
        check_count+=2
        candidate_tables.append({'segment_id':sid,'stored_candidate_center':center,'window':[int(win[0]),int(win[-1])],
          'stored_cost':cand['cost'],'recalculated_cost_fixed_candidate_only':co,
          'coverage_of_retained_template_dimensions':coverage,'group_costs':per})

short=[];recomputed=[]
for a,r in rows.items():
    saved=read(OLD/'controls'/f'{a}_left_short_shape_change_5f.json')
    ctrl=arrays(OLD/'controls'/f'{a}_left_short_shape_change_5f.npz')
    assert pins[str(OLD/'controls'/f'{a}_left_short_shape_change_5f.npz')]==saved['array_sha256']
    assert saved['source_input_sha256']==r['input_sha256']
    orig=base[a];assert np.array_equal(orig['mask'],ctrl['mask'])
    cols=np.arange(11,25);changed=np.asarray(saved['metadata']['changed_frames'])
    known=orig['mask'][np.ix_(changed,cols)]
    delta=np.abs(ctrl['x'][np.ix_(changed,cols)].astype(float)-orig['x'][np.ix_(changed,cols)].astype(float))
    assert int(known.sum())==saved['metadata']['actual_observed_changed_cells']
    direct_mean=float(delta[known].mean()) if known.any() else None
    direct_peak=float(delta[known].max()) if known.any() else None
    path=np.asarray(saved['path'],int);qx=ctrl['x'][:,cols].astype(float);rx=orig['x'][:,cols].astype(float)
    qm=ctrl['mask'][:,cols];rm=orig['mask'][:,cols]
    first=next(s for s in phases[a]['segments'] if s['role']=='core_candidate')
    target=next(s for s in saved['stages'] if s['segment_id']==first['segment_id'])
    for stage in saved['stages']:
        lo,hi=stage['reference_core'];pp=path[(path[:,1]>=lo)&(path[:,1]<=hi)]
        maxima=np.zeros_like(rx);compared=np.zeros_like(rm)
        for qi,ri in pp:
            shared=qm[qi]&rm[ri]
            maxima[ri]=np.maximum(maxima[ri],np.where(shared,np.minimum(np.abs(qx[qi]-rx[ri]),1),0))
            compared[ri]|=shared
        n=int(compared[lo:hi+1].sum());mean=float(maxima[lo:hi+1].sum()/n) if n else None
        peak=float(maxima[lo:hi+1].max()) if n else None
        old=stage['groups']['L_shape']
        assert n==old['compared']
        assert (mean is None and old['mean_difference'] is None) or np.isclose(mean,old['mean_difference'],atol=1e-12,rtol=0)
        assert (peak is None and old['peak_difference'] is None) or np.isclose(peak,old['peak_difference'],atol=1e-12,rtol=0)
        check_count+=3
        recomputed.append({'alias':a,'segment_id':stage['segment_id'],'mean':mean,'peak':peak,'compared_cells':n,
          'cells_with_difference_gt_1e_6':int((maxima[lo:hi+1][compared[lo:hi+1]]>1e-6).sum()),
          'arithmetic_matches_saved':True})
    rng=lookup[(a,first['segment_id'],'L_shape')]
    dec=next(x for x in diagnostic['rows'] if x['alias']==a and x['control']=='left_short_shape_change_5f')
    verdict=next(x for x in dec['decisions'] if x['segment_id']==first['segment_id'])
    calibration=[]
    for qa,qr in rows.items():
        if qa==a or qr['word_id']!=r['word_id']:continue
        pair=read(OLD/'pairs'/f'C_relational_path_{qa}_{a}.json')
        st=next(s for s in pair['stages'] if s['segment_id']==first['segment_id'])['groups']['L_shape']
        calibration.append({'fit_query':qa,'person':qr['person'],
          **{k:st[k] for k in ['mean_difference','peak_difference','expected','compared','coverage','path_missing_cells','sufficient','distinct_shared_query_frames','reference_unobserved']}})
    assert len(calibration)==2
    cal_ok=all(c['sufficient'] and c['mean_difference'] is not None for c in calibration)
    assert cal_ok==rng['supported']
    if cal_ok:assert np.isclose(max(c['mean_difference'] for c in calibration),rng['max_difference'],atol=1e-12,rtol=0)
    stats=target['groups']['L_shape'];re=next(z for z in recomputed if z['alias']==a and z['segment_id']==first['segment_id'])
    short.append({'alias':a,'word_id':r['word_id'],'person':r['person'],'changed_frames':changed.tolist(),
      'observed_edited_cells':int(known.sum()),'potential_edited_cells':int(known.size),
      'direct_same_frame_observed_edit_mean':direct_mean,'direct_same_frame_observed_edit_peak':direct_peak,
      'first_core':first['core'],'first_core_length':first['core'][1]-first['core'][0]+1,
      'saved_control_path_diagonal':bool(len(path)==len(qx) and np.array_equal(path[:,0],path[:,1])),
      'stage_mean_difference':stats['mean_difference'],'stage_peak_difference':stats['peak_difference'],
      'stage_observed_support':{k:stats[k] for k in ['expected','compared','coverage','path_missing_cells','sufficient','reference_unobserved']},
      'stage_nonzero_difference_cells':re['cells_with_difference_gt_1e_6'],
      'mean_to_direct_edit_mean_ratio':stats['mean_difference']/direct_mean if direct_mean and stats['mean_difference'] is not None else None,
      'fit_mean_range_max':rng['max_difference'],'saved_range_state':verdict['state'],
      'two_fit_calibration_pairs':calibration,'peak_compared_to_mean_threshold_is_valid_decision':False})

states=Counter(s['saved_range_state'] for s in short if s['observed_edited_cells'])
s05=anchors['P063_s05'];s06=anchors['P063_s06'];near76=next(c for c in s05['five_nearest_candidates'] if c['center']==76)
findings=[
 'P063 identity arrays and original frame/PTS are exact; its saved DTW is 142 diagonal pairs. The reverse is from independent free_order template retrieval, not from DTW warping or a changed performance.',
 'The 7-frame s05 transition becomes one mean posture per component. The >=75% reference support rule retains 6/13 groups (34/63 dimensions); both motion-direction groups and most right-hand/bilateral evidence are excluded. Missing groups contribute neither evidence nor a missing-information warning to the winner coverage.',
 'That reduced mean template selects center98/window96-100 at 0.0689201032 over center76/window74-78 at 0.0755550415. The latter is already stored as fourth nearest. The advantage is only 0.0066349383. The later hold s06 selects center90/window88-92, yielding a reverse relation in an unchanged clip.',
 'Coverage1.0 means complete support for retained expected dimensions, not complete support for the original13groups or dynamic transition. Numeric unique at tie tolerance1e-9 is not semantic uniqueness.',
 'The free_order template is constant across its core and candidates are scored independently over five frames. It has no entry-to-core-to-exit trajectory or repetition-instance identity; similar return/hold postures can be substituted.',
 'Short-left synthetic controls: 3/12 have zero observed edited cells and are not observed-error tests. Of9 observed edits, first-core states are4 reference-range unavailable,2 within mean range,3 above mean range.',
 'P046 and P047 have observed local peaks near1 but stage means0.1939618534 and0.2413029071, below fit mean maxima0.2575181913 and0.2682824721. This is duration/cell averaging relative to inter-person mean variability, not evidence that the synthetic local change is anatomically acceptable.',
 'P046 edits70 observed cells with direct-edit mean0.737055043; its core compares266 cells so the stage mean retains70/266=26.3158% of that edit mean. P047 edits70 of224 compared cells with direct-edit mean0.772169303, retaining31.25%. This quantifies dilution without fitting a new threshold.',
 'Unavailable ranges require both other fit comparisons to be sufficient. Missing support is a separate cause, not successful tolerance or detector failure on a visible RGB movement.',
 'The perturbation changes saved projected angle numbers only; it is not an actual changed hand image or a linguistic mistake. This audit cannot measure real 3D projection ambiguity; it documents the 2D feature limitation without attributing an unmeasured effect.'
]
contracts=[
 {'id':'C1_context_identity','proposal':'Compare ordered entry, core and exit observation blocks jointly for one candidate instance, using original PTS and distinct frame support. Keep several candidates when instances look alike.',
  'preserve':'Do not enforce the entire expected expression order during retrieval merely to make identity pass; diagnose cross-instance ambiguity separately.'},
 {'id':'C2_support_before_semantics','proposal':'Expose original possible dimensions, retained template dimensions and missing context separately. A missing direction or side makes transition identity uncertain even if retained static coverage is1.',
  'preserve':'Do not fill crossed/occluded hands or require an irrelevant lowered hand globally. Keep observed shape/location evidence from the other hand.'},
 {'id':'C3_motion_and_hold','proposal':'Compare entry-to-core and core-to-exit signed changes with local support and duration; keep hold/rest as a separate observable pattern and keep no-motion apart from no-observation.',
  'preserve':'Normal crossings can legitimately swap image ordering. Trace anatomical sides and avoid interpreting crossing itself as reversal.'},
 {'id':'C4_short_event_statistic','proposal':'Keep stage mean, local peak/short contiguous excursion, duration and observed-cell count as separate evidence. Calibrate any new peak/duration rule only on fit LOPO using the same statistic and freeze it before DEV.',
  'preserve':'Do not compare a peak to the old mean threshold, turn a noisy one-frame spike into a language error, or tune to these control outcomes.'},
 {'id':'C5_bounded_unknown','proposal':'Retain all original frames and uncertain boundaries; unknown intervals remain gaps. Report whether each ambiguity is lack of support, same-shape instance confusion or an actual supported numeric difference.',
  'preserve':'Do not concatenate across gaps as ground-truth order, remove normal preparation/return, or call a5-frame omission a proven important semantic omission.'}
]
assert all(sha(p)==h for p,h in pins.items())
out={'status':'CAUSE_AUDIT_SAVED_FIT_ONLY','author':'Codex agent A; same-family independent arithmetic/source review, not human KSL ground truth',
 'execution_scope':{'fit_clips':12,'identity_case':'P063','new_model_calls':0,'new_DTW_calls':0,'new_free_order_search_calls':0,
   'new_comparison_runs':0,'new_candidates_searched':0,'fixed_saved_candidates_recalculated':len(candidate_tables),
   'old_files_written':0,'DEV_arrays_or_results_used':False,'source_or_private_data_writes':False},
 'identity_failure':{'original_arrays_exact':True,'saved_DTW_diagonal':True,'native_frame_count':142,
   's05_saved':s05,'s06_saved':s06,'within_s05_stored_candidate':near76,
   'wrong_vs_within_core_cost_advantage':near76['cost']-s05['cost'],
   'candidate_center_displacement_from_within_core_frames':s05['query_center']-76,
   'stored_reverse_relation':next(x for x in identity['free_order']['relations'] if x['first']=='P063_s05'),
   'template_support':stage_templates,'fixed_candidate_cost_breakdown':candidate_tables},
 'short_left_change':{'fit_cases':12,'cases_with_observed_edits':sum(x['observed_edited_cells']>0 for x in short),
   'cases_without_observed_edits':sum(x['observed_edited_cells']==0 for x in short),
   'first_core_states_among_observed_edits':dict(states),'cases':short,'saved_path_arithmetic_checks':recomputed,
   'aggregation_definition':'For each reference frame/component take max clipped absolute difference over the saved path, then average over compared reference cells. A short excursion can be diluted by unchanged reference cells; it is not a simple average of DTW pair rows.',
   'units':'shape differences and mean ranges use projected2D interior angle/180; peak and mean have same units but require different calibration distributions'},
 'findings':findings,'generic_entry_core_exit_contracts':contracts,'arithmetic_checks':check_count,
 'limitations':['This explains existing fit artifacts; no new model score or method performance was measured.',
   'No camera/viewpoint/body-proportion or actual handshape error experiment occurred.',
   'RGB phases are sampled provisional drafts, not dense semantic labels or obligatory linguistic order.',
   'Source-grounded fit anchors remain limited to one person per expression and cannot be propagated as multi-person labels.'],
 'pins':pins,'all_read_artifacts_unchanged':True}
(U/'review'/'CAUSE_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=['이전 평균자세 검색·짧은 왼손 변화의 원인 감사',
 '공개 fit12의 저장 배열·결과만 확인했다. 새 모델/DTW/후보검색은 실행하지 않았다.', '',
 '1. 그대로인 영상에서 왜 역순이 나왔나',
 'P063의 identity 입력은 원본과 같고 DTW142쌍도 모두 같은 프레임끼리 연결돼 있다. 문제는 별도의 평균자세 검색이다.',
 's05(72–78프레임)는 왼손을 올리는 전환인데, 지원75% 기준을 지나며13그룹 중6개만 남았다. 양손 이동방향과 오른손/양손관계의 상당 부분이 빠졌다. 따라서 coverage100%도 원래 전체 동작 근거가 충분하다는 뜻은 아니다.',
 f"검색은 실제 구간 안74–78창(중심76,비용{near76['cost']:.9f})보다 뒤의96–100창(중심98,비용{s05['cost']:.9f})을 골랐다. 차이는{near76['cost']-s05['cost']:.9f}뿐이다.",
 '뒤이어야 할 유지 단계 s06(81–96프레임)은88–92창(중심90)을 골라98→90의 역순이 됐다. 평균 모양을 독립적으로 찾으므로 올라가는 과정과 이후 비슷한 모양을 구별할 문맥이 사라진 것이다.', '',
 '2. 짧은 왼손 변화는 왜 약하게 보였나',
 '12개 중3개는 바꾼 왼손 값의 관측 mask가 전부 꺼져 있어 실제 관측변화가0개였다. 이를 오류를 놓친 사례로 세면 안 된다.',
 '관측변화가 있는9개 중4개는 참고범위가 없고,2개는 평균범위 안,3개는 평균범위 초과다. 참고범위 없음과 평균 희석은 다른 원인이다.',
 'P046은 국소 최대0.993671인데20프레임 구간 평균0.193962, 참고 평균범위0.257518이다. P047은 국소 최대0.997573인데16프레임 평균0.241303, 참고 평균범위0.268282이다.',
 'P046은 관측266칸 중70칸을 바꿔 변경칸 평균0.737055의26.3%만 구간 평균에 남았다. P047은224칸 중70칸이라 변경칸 평균0.772169의31.25%가 남았다.',
 '최대차이가 큰데도 변하지 않은 프레임/각도와 함께 평균내면 작아진다. 하지만 최대값을 기존 평균 기준과 바로 비교해 새 오류판정을 만들 수도 없다.',
 '이 통제는2D 투영각도 숫자를 바꾼 것이다. 실제 손사진·실제3D 굽힘·교육적 수어 오류를 검증한 것이 아니다.', '',
 '3. 다음 공통 비교 조건',
 '진입→핵심→이탈을 한 후보 회차의 연결로 비교하되, 기대 순서를 강제로 맞춰 통과시키지 않는다. 비슷한 후보는 여러 개와 모호성을 남긴다.',
 '정지/휴식과 미관측을 구분하고, 정상적인 손 교차·아래로 둔 손·가려진 손을 오류로 바꾸지 않는다. 관측된 다른 손 근거는 유지한다.',
 '구간 평균과 짧은 국소 변화/지속시간/지원 셀 수를 분리한다. 새 통계의 범위는 같은 통계로 fit LOPO에서만 정하고 DEV 전에 고정한다.',
 '모든 원본 프레임과 불확실 경계는 보존한다. 미주석 구간을 정답 순서로 이어 붙이거나5프레임 삭제를 필수 동작 누락으로 단정하지 않는다.', '',
 '부록: fit별 왼손 통제 첫 핵심 구간']
for s in short:
    text.append(f"{s['alias']}: 관측변경{s['observed_edited_cells']}/70칸, 구간{s['first_core']}, 평균{s['stage_mean_difference']}, 최대{s['stage_peak_difference']}, 평균범위{s['fit_mean_range_max']}, 상태{s['saved_range_state']}")
(U/'review'/'CAUSE_AUDIT_KO.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'checks':check_count,'identity_retained_groups':stage_templates[0]['retained_groups'],
 'identity_retained_dimensions':stage_templates[0]['retained_dimensions'],'identity_possible_dimensions':stage_templates[0]['all_possible_dimensions'],
 'observed_left_cases':out['short_left_change']['cases_with_observed_edits'],'left_states':dict(states),
 'files_written':['CAUSE_AUDIT.json','CAUSE_AUDIT_KO.txt'],'all_read_artifacts_unchanged':True}))
