"""Arithmetic and plain-language report of one saved diagnostic; no models."""
from prepare import U,O,read,sha,exclusive
result=read(U/'RESULT.json');done=read(U/'RUN_FINISHED.json')
assert done['result_sha256']==sha(U/'RESULT.json')
review={r['clip_id']:r for r in read(U/'LATE_VISUAL_REVIEW.json')['rows']}
old={r['record']['clip_id']:r for r in read(O/'word100_component_rgb_context_20261102/SCOPE_RESULT.json')['rows']}
paired=[]
for cid in review:
    b=next(r for r in result['rows'] if r['clip_id']==cid and r['scope']=='broad')
    l=next(r for r in result['rows'] if r['clip_id']==cid and r['scope']=='late')
    original=old[cid];witness=original['record']['native_frames']
    assert original['record']['y']==1 and original['record']['mask'] is True
    assert b['frames'][0]<=min(witness)<=max(witness)<=b['frames'][1]
    p=dict(clip_id=cid,person=b['person'],broad_native_frames=b['frames'],
           existing_positive_witness=witness,
           old_narrow_scope=original['extent'],old_narrow_probability=original['probabilities'][0],
           broad_probability=b['observation']['probability'],broad_continuity=b['observation']['continuity_probability'],
           broad_status=b['observation']['raw_observer_status'],
           late_native_frames=l['frames'],late_direct_local_presence=review[cid]['local_motion_presence'],
           late_probability=l['observation']['probability'],late_continuity=l['observation']['continuity_probability'],
           late_status=l['observation']['raw_observer_status'],
           late_RGB_context_frames=l['observation']['rgb_context_support']['observed_context_source_frames'],
           late_short_window_limit=review[cid]['short_window_limit'])
    paired.append(p)
counts=dict(selected_fit_records=3,new_scopes=6,
            broad_with_known_motion_reported=int(sum(r['broad_status']=='DIRECTED_MOTION_OBSERVED' for r in paired)),
            locally_still_late_scopes_reported_as_motion=int(sum(r['late_direct_local_presence']==0 and r['late_status']=='DIRECTED_MOTION_OBSERVED' for r in paired)))
exclusive(U/'SAVED_SUMMARY.json',dict(status='SAVED_DIAGNOSTIC_ARITHMETIC_NOT_GENERALIZATION',rows=paired,counts=counts,
    limitations=['Selected FIT3, not unseen-person performance',
                 'Late absence is local, not a wrong sign or a full-action omission',
                 '16-frame RGB context crosses nominal interval; high probability does not prove motion inside nominal scope',
                 'New broad queries are manually proposed from independent body context; automatic new-user localization not validated'],
    pins={str(p):sha(p) for p in [U/'RESULT.json',U/'LATE_VISUAL_REVIEW.json',O/'word100_component_rgb_context_20261102/SCOPE_RESULT.json']}))
lines=['동작 비교 범위와 실제 움직임을 분리한 시험','',
       '기존에는 필요한 움직임이 있는 짧은 구간을 지정해 검사했다. 이번에는 받침팔을 올리고 유지하는 넓은 작업 범위를 먼저 잡고, 그 안에 쓸기 움직임의 근거가 있는지 같은 학습 모델에 물었다. 동작 점수가 낮다는 이유로 해당 시점을 비교 범위에서 빼지 않는다.',
       '', '공개 기록 ID01·04·07의 같은 표현3영상, 넓은 구간3개와 짧은 유지구간3개를 고정해서 한번 비교했다. 넓은 구간에는 기존 직접검토에서 확인한 쓸기 흐름이 포함된다. 짧은 유지구간의 연속 원RGB14프레임은 새로 직접 보았으며 큰 쓸기 이동이 보이지 않는다. 이 유지구간은 정상 동작의 일부이고 틀린 수어가 아니다.', '']
for r in paired:
    lines.append(f"ID{r['person']:02d}: 기존 좁은 움직임 구간 점수 {r['old_narrow_probability']:.4f} / 새 넓은 작업 범위 {r['broad_probability']:.4f} ({r['broad_status']}) / 짧은 유지 범위 {r['late_probability']:.4f} ({r['late_status']}).")
lines.extend(['',f"움직임이 포함된 넓은 범위에서 확인: {counts['broad_with_known_motion_reported']}/3. 실제 짧은 유지범위에 움직임을 표시한 경우: {counts['locally_still_late_scopes_reported_as_motion']}/3. 학습에 사용된 공개3자료의 진단값이며 새 사용자 정확도가 아니다.",
              '영상특징은 앞뒤16프레임을 함께 본다. 따라서 높은 움직임 점수가 짧은 검사범위 안의 움직임 때문인지, 바로 앞뒤 움직임 때문인지는 이 점수만으로 분리되지 않는다. 실제 영상지원범위를 결과에 함께 저장했다. 이전 정지 라벨과 직접 움직임 정답의 범위가 겹친 문제도 발견하여 옛 라벨을 새 정답으로 재사용하지 않았다.',
              '',f"추가 움직임 판정부6구간 계산은 {result['motion_head_batch_seconds']:.6f}초. 이미 준비된 입력에서 측정했으므로 전체 영상 분석시간 또는 실제 사용자의 대기시간이 아니다.",
              '모델·가중치·임계값을 바꾸거나 재학습하지 않았다. 이전 평가 재채점·기존웹 적용도 없다. C·T3-F·기존데모와 사용자·멘토자료 보호를 유지했다. 전체 학습형 피드백 데모 목표는 미완료이며, 이 결과를 전체 동작 완료나 교육적 정오 판단으로 사용하지 않는다.'])
with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(counts)
