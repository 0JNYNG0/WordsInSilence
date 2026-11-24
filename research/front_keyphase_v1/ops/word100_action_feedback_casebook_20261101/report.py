"""Report the frozen saved-evidence join; never recompute model or DTW outputs."""
import sys
sys.dont_write_bytecode=True
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
from pipeline import UNIT, read, sha, write


def readable(value):
    if isinstance(value,str):return value
    if isinstance(value,list):return '\n'.join('- '+readable(v) for v in value)
    if isinstance(value,dict):return '\n'.join(str(k)+': '+readable(v) for k,v in value.items())
    return json.dumps(value,ensure_ascii=False)


def main():
    if (UNIT/'EVALUATION.json').exists():
        raise RuntimeError('Existing report data: inspect and reuse, do not overwrite')
    inp=read(UNIT/'INPUTS.json'); manifest=read(UNIT/'RESULT_MANIFEST.json')
    verification=read(UNIT/'review/RESULT_VERIFICATION.json')
    assert verification['status'].startswith('PASS'), verification['status']
    visual=read(UNIT/'review/VISUAL_REVIEW.json'); casebook=read(UNIT/'CASEBOOK.json')
    rgb=read(UNIT/'review/RGB_MANIFEST.json')
    links={r['id']:r for r in manifest['results']}
    totals=Counter(); per_case={}; row_details=[]; review_count=0; source_statuses=Counter()
    for task in inp['natural']:
        link=links[task['id']]; assert sha(link['path'])==link['sha256']
        result=read(link['path']); summary=result['summary']; counts=Counter()
        for group in summary['groups'].values():
            for axis in ('query','reference'): counts.update(group[axis]['counts'])
        totals.update(counts); per_case.setdefault(task['query'],Counter()).update(counts)
        reviewed=summary['visible_difference_review']['status']=='REVIEWED_SCOPED_OBSERVATIONS'
        review_count+=int(reviewed)
        source_statuses.update(summary['description_support']['source_status_counts'])
        row_details.append({
            'id':task['id'],'query':task['query'],'reference':task['reference'],
            'parent_id':task['parent_id'],'query_range':task['query_range'],
            'reference_range':task['reference_range'],'source_cards':task['source_card_count'],
            'physical_description':task['parent_physical_description'],
            'counts':dict(counts),'scoped_RGB_review_attached':reviewed,
            'result':link,'educational_error':None})
    assert len(row_details)==39 and review_count==10
    assert totals['possible']==30560
    assert [totals[k] for k in ('direct_complete','partial_shared','not_comparable')]==[24174,32,6354]
    assert totals['direct_complete']+totals['partial_shared']+totals['not_comparable']==totals['possible']
    reviewed_queries={w['query'] for w in visual['windows']}
    assert reviewed_queries=={'P036','P048','P066','P072'}
    assert all(c['direct_complete']>0 for c in per_case.values())
    assert len(visual['windows'])==10
    assert rgb['unique_frames']==len(rgb['full_frames'])==303
    assert dict(source_statuses)=={'visible_partial_match':24,'visible_proximity_only':3,'unverified_edge_on':3}
    report_data={
        'status':'PASS_SAVED_EVIDENCE_LINKAGE_NOT_EDUCATIONAL_ACCURACY',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':inp['counts'],'natural_counts_both_axes':dict(totals),
        'case_counts':{k:dict(v) for k,v in per_case.items()},
        'reviewed_parent_records':review_count,'unreviewed_parent_records':39-review_count,
        'new_RGB_windows':10,'new_RGB_unique_frames':303,
        'description_source_parents':12,'physical_only_parents':27,
        'source_clause_status_counts':dict(source_statuses),
        'natural_rows':row_details,'technical_saved_summaries':810,
        'each_case_has_source_traceable_RGB_review':True,
        'all_39_opportunities_and_3_references_preserved':True,
        'new_training':0,'new_model_inference':0,'new_dtw':0,'new_coordinate_comparison':0,
        'source_verification':{'path':str(UNIT/'review/RESULT_VERIFICATION.json'),'sha256':sha(UNIT/'review/RESULT_VERIFICATION.json')},
        'visual_review':{'path':str(UNIT/'review/VISUAL_REVIEW.json'),'sha256':sha(UNIT/'review/VISUAL_REVIEW.json')},
        'educational_accuracy':None,'recognition_accuracy_improvement':None,
        'alignment_accuracy_improvement':None,'user_end_to_end_latency':None,
        'summary_compute_seconds':manifest['compute_seconds'],
        'summary_wall_seconds':manifest['wall_seconds'],
        'historical_development_not_fresh_generalization':True,
        'same_codex_review_not_independent_human_KSL_ground_truth':True,
        'production_changed':False,'C_T3F_changed':False,'user_mentor_data_used':False}
    write(UNIT/'EVALUATION.json',report_data)
    training={
        'status':'NO_NEW_TRAINING_JUSTIFIED_BY_THIS_BOUNDED_REVIEW',
        'decision':'기존 모델 유지. 이번 확인만으로 새 판정부 학습을 시작하지 않는다.',
        'classifier_failure_after_valid_observation_and_correspondence_established':False,
        'new_validated_training_targets':0,
        'reasons':[
            '현재 사례는 기존 정상 공개 개발영상의 저장 비교와 제한된 RGB 관측이다. 교육적 오답 또는 밀집 손모양 정답을 새로 만들지 않았다.',
            '영상에서 보이는 손의 저장 관측 누락과 서로 다른 순간의 대응 사례가 있어, 수치 차이를 판정부의 오답으로 바로 사용할 수 없다.',
            '사전 설명의 일부 연결은 참고 문맥이며 실제 비교영상의 각 프레임 정답이나 허용 경계가 아니다.',
            '810개 기술 대조는 저장 수치 변형의 보존시험이며 실제 사람이 틀리게 수어한 영상/정답이 아니다.'
        ],
        'does_not_claim_no_classifier_errors_exist':True,
        'future_gate':'관측과 동작 대응이 확인된 공개 학습자료에서 특정 판정부의 실패와 적절한 시간별 정답 지원을 분리하여 확인한 뒤 후보 하나를 고정한다.',
        'source':{'evaluation':str(UNIT/'EVALUATION.json'),'visual_review':str(UNIT/'review/VISUAL_REVIEW.json')},
        'new_training':0,'synthetic_or_guide_labels_promoted_to_truth':False}
    write(UNIT/'TRAINING_DECISION.json',training)
    lines=[
        '설명·동작 근거 연결과 차이 원인 검토 — 네 단계 완료',
        '',
        '이번에는 새 수어 인식 모델을 학습한 것이 아니라, 기존 비교가 왜 다르게 나오는지 설명할 근거를 보완했습니다.',
        '설명과 비교값을 연결하는 기능은 이미 있었습니다. 여기에 손가락별 화면상 각도, 화면상 손 방향, 손과 아래팔(팔꿈치에서 손목까지)의 상대 방향을 독립적으로 붙이고 원본 장면에서 확인한 내용을 함께 남겼습니다. 화면에서 구한 각도이지 실제 3차원 관절각 확정은 아닙니다.',
        '',
        '이전 → 현재',
        '- 이전: 설명 카드와 위치·모양 등의 비교값은 있었지만, 최신 독립 손 특징과 원본 확인 내용을 한 경로로 추적하기 어려웠습니다.',
        '- 현재: 무엇을 관측했는지 / 시점을 잘 맞췄는지 / 숫자가 얼마나 다른지 / 실제 영상에서 무엇이 보이는지 / 설명의 어느 부분과 연결되는지를 나눠 따라갈 수 있습니다.',
        '- 숫자 차이가 있어도 무조건 틀렸다고 하지 않고, 모르는 값은 0점 차이 또는 정상으로 바꾸지 않습니다. 반대로 보류 때문에 이미 보인 차이를 지우지도 않습니다.',
        '',
        '검토한 범위',
        '순종·잘하다·조상·이집트의 공개 개발영상 4개를 각 3개 참고와 연결했습니다. 총 12쌍·39개 주요 범위를 모두 유지했습니다.',
        '새로 원본을 확인할 범위는 결과 전에 10개로 고정했습니다. 8개 영상에서 303개 고유 프레임을 보았고, 나머지 참고/범위에는 원본 확인이 끝났다고 표시하지 않았습니다.',
        '39개 중 사전 설명 일부가 연결된 것은 12개, 영상의 물리적 설명만 있는 것은 27개입니다. 설명 연결 30항목 중 부분 확인 24, 가까움만 확인 3, 미확인 3이며 전체 정답 규칙이 아닙니다.',
        '',
        '사례별로 무엇을 확인했는지'
    ]
    for case in visual['cases']:
        label=next(c['label'] for c in casebook['cases'] if c['query']==case['query'])
        lines+=['',label+' ('+case['query']+')',readable(case['summary_ko']), '남은 한계: '+readable(case['limits_ko'])]
    lines += [
        '', '검증 결과',
        '39개 실제 비교와 810개 기술 대조를 기존 저장값에서 연결했습니다. 모델 추론·DTW·원래 좌표 비교는 다시 실행하지 않았습니다.',
        '양쪽 시간축의 특징 비교 기회 30,560개를 유지했습니다. 직접 비교 가능 24,174개, 일부 대응만 가능 32개, 비교 불가 6,354개입니다. 이 숫자는 프레임 수나 정확도가 아니라 특징×시점×참고별 기회의 합계입니다.',
        '공통 평행이동, 손 방향 변화, 짧은 변화, 손가락 일부 누락 등 기존 810개 기술 대조의 값과 근거가 새 요약에서 사라지지 않는지 별도 코드로 확인했습니다.',
        '별도 검산은 같은 Codex의 다른 에이전트가 작성한 검산이며, 수어 전문가의 독립 정답 검증을 뜻하지 않습니다.',
        '', '학습 여부와 실제 개선 범위',
        '이번에는 추가 학습을 하지 않았습니다. 확인된 관측 누락과 시간 대응 문제를 학습 오답으로 쓰면 잘못된 내용을 가르칠 수 있고, 이번 검토는 새 시간별 학습 정답을 만든 시험도 아닙니다.',
        '이는 새 학습을 정당화할 근거가 부족하다는 판단이며, 기존 판정부에 오류가 없다고 확인했다는 뜻은 아닙니다.',
        '영상에 보이는 손의 저장 정보가 빠졌다는 것과 MediaPipe가 처음부터 검출을 실패했다는 것은 다릅니다. 정상 손내림을 사용 범위에서 제외했을 수도 있습니다. 원검출·영역 제한·후처리 중 어느 단계에서 빠졌는지는 아직 확정하지 않았습니다.',
        '개선된 것은 비교 근거를 구분하고 원본까지 추적하는 기능입니다. TCN 인식률, DTW 정렬 정확도, 교육 피드백 정확도가 올랐다고 주장할 결과는 아직 없습니다.',
        '정상 손·좌우·교차의 기존 좌표와 마스크, 저장 경로는 그대로 유지했습니다. 따라서 이번 연결이 손을 지우거나 배정을 바꾸지는 않았습니다. 모든 정상 교차의 교육적 판정을 검증했다는 뜻은 아닙니다.',
        '3차원 주먹 회전, 정확한 손목 굽힘, 실제 접촉 여부와 표현별 중요한 차이·허용 범위는 여전히 별도 근거가 필요합니다.',
        '', '실행과 보존',
        f"저장 근거 요약 계산 {manifest['compute_seconds']:.3f}초, 해당 실행 전체 {manifest['wall_seconds']:.3f}초. 영상 처리나 사용자 피드백 전체 지연 측정이 아닙니다.",
        '기존 C·T3-F, 운영 웹·DTW는 변경하지 않았고, 사용자·멘토 촬영은 접근·학습·튜닝·선택에 사용하지 않았습니다.',
        '새 HTML은 만들지 않았습니다. 8800 주소는 이전 비교 보고서입니다.',
        '', '상세 사례와 원본 연결: CASEBOOK_KO.txt, CASEBOOK.json, review/VISUAL_REVIEW.json',
        '검산: review/RESULT_VERIFICATION.json / 학습 판단: TRAINING_DECISION.json'
    ]
    dest=UNIT/'RESULT_KO.txt'; assert not dest.exists();dest.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    detail=['공개 동작 사례 근거 목록','',
        '각 참고는 따로 유지합니다. 숫자 차이와 실제 RGB 확인, 설명 지원 범위는 서로 다른 정보입니다.',
        '이 목록의 검토된 사례도 틀린 수어 판정이나 학습용 오답 정답이 아닙니다.','']
    for case in casebook['cases']:
        detail += [case['label']+' / 비교영상 '+case['query'], '참고: '+', '.join(case['references'])]
        cv=next(c for c in visual['cases'] if c['query']==case['query'])
        detail += [readable(cv['summary_ko']),'한계: '+readable(cv['limits_ko'])]
        for w in visual['windows']:
            if w['query']!=case['query']:continue
            detail+=['','원본 확인 창 '+w['window_id']+' / '+w['query']+' ↔ '+w['reference']+' / '+w['parent_id']]
            for axis in ('query_axis','reference_axis'):
                a=w[axis];detail += [axis+': 원본프레임 '+str(a['reviewed_frame_ids']),
                    '원본시각(마이크로초): '+str(a['reviewed_timestamp_us']),
                    '보이는 내용: '+readable(a['observations']),
                    '저장 관측 근거: '+readable(a['stored_mask_observations']),
                    '시간순 장면: '+readable(a['contact_pages'])]
            for key in ('visible_difference_review','observation_missing','correspondence_review','coexisting_causes'):
                detail += [key+': '+readable(w[key])]
        detail += ['','전체 범위(검토한 부분만 골라내지 않음)']
        for row in row_details:
            if row['query']!=case['query']:continue
            c=row['counts']
            detail += [f"{row['reference']} / {row['parent_id']} | 비교원프레임 {row['query_range']} / 참고원프레임 {row['reference_range']} | 설명연결 {row['source_cards']}개 | RGB부분검토 {row['scoped_RGB_review_attached']}",
                f"특징기회 {c['possible']}: 직접 {c['direct_complete']}, 부분 {c['partial_shared']}, 비교불가 {c['not_comparable']}",
                '원래 물리적 설명: '+str(row['physical_description']),
                '저장값/시점/출처: '+row['result']['path']]
        detail+=['','']
    dest=UNIT/'CASEBOOK_KO.txt';assert not dest.exists();dest.write_text('\n'.join(detail)+'\n',encoding='utf-8')
    print(json.dumps({'status':report_data['status'],'cases':4,'parents':39,'reviewed_parents':review_count,'training':0},ensure_ascii=False))


if __name__=='__main__':main()
