from prepare_coherent import *
from collections import Counter
ROOT=K.parents[3]
def main():
    assert not (K/'FINAL.json').exists();v=read(K/'VERIFIED.json');plan=read(K/'EXECUTION_FREEZE.json');r=read(K/'evaluations/witness_ctc_42.json');manifest=read(K/'INPUT_MANIFEST.json');at=clock()
    supported=[q for q in r['sequences'] if q['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED']
    caution=[dict(key=q['key'],action_errors=q['evaluation_action_errors'],core_unconfirmed=q['previous_reviewed_core_unconfirmed']) for q in supported if q['evaluation_action_errors'] or q['previous_reviewed_core_unconfirmed']]
    fit_prov=read(J/'PROVENANCE_AUDIT.json')['rows'];extents=Counter(tuple(q['actual_sensor_extent']) if q['actual_sensor_extent'] is not None else ('not_stored',) for q in fit_prov if q['split']=='FIT')
    crop_code=O/'word100_seated_web_framing_20261102/validate_prepare.py';controls=O/'word100_learned_feature_retention_20261102/VIDEO_CONTROLS.json'
    old_controls=read(controls)['rows'];normal={'reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
    crop_normal=[dict(word=q['word_id'],kind=q['kind'],source=q['source'],source_sha256=q['source_sha256']) for q in old_controls if q['kind'] in normal]
    assert len(crop_normal)==16 and all('public_crops' in q['source'] for q in crop_normal)
    audit=dict(at=at,supported_status_cases=len(supported),supported_with_other_known_concerns=caution,clean_supported_cases=len(supported)-len(caution),
        fit_cached_extent_counts={str(k):n for k,n in extents.items()},seated_normal_controls=crop_normal,
        conclusion='New full-sensor native input resolves7rest/action conflicts, but same-model performance still has errors. Existing normal reencode/speed/trim controls come from seated public pixel crops. Audit and add the same declared framing variability on FIT public inputs before further loss-only changes.',
        no_new_training=True,not_all_error_causes_proven=True,pins={str(crop_code):sha(crop_code),str(controls):sha(controls)},whole_goal_complete=False)
    exclusive(K/'CAUSE_AND_NEXT.json',audit)
    summary='K공개DEV원본12개를FIT와동일전체센서경로로GPU.0추출하고고정J2모델24진단·별도검산완료. 主같은영상/265주요정답에서구입력251→새입력253(94.72→95.47%), 대기/수행충돌7영상→0. 원판정255/265, 역할처리2정답손실남음. 순서상태2/9→4/9이나4중2는주요오분류/관측보류가남아완료통과아님(깨끗한2/9). 대기48/48. J추가CTC이득은이입력에서도없음. 과거완료점수/모델/웹불변. 기존속도등정상16조건은public_seated_crop에서생성된반면FIT입력은전체센서계보임을확인. 전체goal active.'
    nxt='K12추출24진단/J2학습176비교/F/E/D완료재실행금지. K CAUSE_AND_NEXT를근거로 FIT01~09의공개36원본에기존공통shoulder-center+one-width 좌석화면proxy계약을적용할입력을준비·검증한다. 실제사용자영상영역을새로제한하는것이아니라공개학습자료에서웹캠시야변이를학습하는것. source_axis/수어주요손모양·단계가남는지확인하고잘린근거는정답강제금지. FIT에서표현/수행자별분리·원본재생유지·하위가림/짧은동작회귀를동결후필요한기존신경망미세조정비교로이어간다. 원자료/H14·15/사용자/멘토/C/T3F/운영웹보존. 성능통과후안정성·전체지연·별도데모;현재활성프로세스없음.'
    times=[q['extraction_seconds'] for q in manifest['rows']]
    txt=['같은 입력 처리 조건으로 맞춘 공개 영상 비교',at,'',
        '이번에 실제로 달라진 점',
        '모델 가중치를 바꾸지 않고, 공개 원본12개의 화면·관절·RGB 처리 조건을 학습 자료와 맞췄다. 예전 결과를 지우거나 점수를 덮어쓰지 않고 별도 입력24비교를 만들었다.',
        '비교 항목 | 이전 입력 | 동일 조건 새 입력',
        '주요 동작 정답 시점 | 251/265 (94.72%) | 253/265 (95.47%)',
        '대기/주요 동작 판단이 충돌한 영상 | 7/12 | 0/12',
        '순서 상태가 확인된 영상 | 2/9 | 4/9',
        '새 입력의 대기 정답 | — | 48/48',
        '',
        '순서4개 중2개는 다른 주요 시점의 오분류 또는 기존 주요 관측 보류가 남는다. 따라서4개가 완전히 올바른 수어로 검증됐다는 뜻은 아니다. 순서 확인과 추가 문제 없음이 함께 만족되는 것은2/9이다. 반복 표현 조상3영상은 현재 단일 행동가족으로는 반복순서/횟수 규칙을 검증할 수 없어 순서 분모에서 제외한다.',
        '265는 손 모양·행동 구분 정답이 있는 시점 수이며 TCN 단어 인식률이 아니다. 학습ID01~09와 다른 공개개발ID11~13이지만, 이미 여러 번 분석한 개발 자료이므로 독립 최종평가라고 부르지 않는다.','',
        '왜 이 비교를 했나',
        '앞선2학습176비교에서 대기 정보는 회복했지만 동작/순서 개선은 없었다. 자료 경로를 추적하니 학습은 전체센서 처리였고 개발원본12개는 과거 상반신영역 처리 결과였다. 이미 옛 입력감사에 있던 차이가 현재 비교에 남아 있었다.',
        '이번에 같은 원본12개를 학습과 동일 경로로 새로 처리하니7개 충돌이 사라졌다. 입력 처리 차이가 실제 결과에 영향을 준다는 근거다. RGB만 바꾼 시험은 아니므로 RGB자르기 하나가 모든 원인이라는 뜻은 아니다.',
        '시점별학습/확인시점CTC보조 두 고정모델 모두 새입력의 주요판정253/265, 대기48/48, 순서4/9였다. 이번에도 CTC추가이득은 확인되지 않았다.','',
        '남은 문제',
        '1) 주요 원판정255/265 중2개를 수행/비수행 해석이 잘못 제외했다. 최종253/265는 이 손실도 포함한 결과다.',
        '2) 팔을 따라 움직이는 중과 도착한 모양, 양손의 합쳐짐/분리 사이 등 일부 행동 구분이 아직 헷갈린다. 예상 정답순서를 강제로 넣어 맞춘 것으로 처리하지 않는다.',
        '3) 기존 재인코딩/속도/대기잘림16개 정상조건은 책상앞 웹캠을 흉내 낸 공개 crop영상에서 만들어졌다. 원본 전체센서 조건만 해결했다고 그 조건까지 해결됐다고 할 수 없다.',
        f"4) 이번 입력 추출 경로는GPU.0을사용했으며 영상별약{min(times):.1f}~{max(times):.1f}초였다. 운영최종피드백전과정을새로측정한것은아니지만속도도여전히개선할부분이다.",
        '',
        '다음 작업',
        '공개 학습ID01~09의36영상에도 이미 정의한 몸 기준 웹캠시야 변이를 준비한다. 사용자 촬영을 학습하는 것이 아니며 실제 사용자 화면에 새 제한선을 긋는 것도 아니다. 원본 전체화면 사례를 함께 유지한다.',
        '시간/좌표/손 관측을 먼저 검증하고, 잘려서 안 보이는 특징은 확정 정답으로 강제하지 않는다. 같은 조건 안에서만 기존 신경망을 필요한 만큼 미세조정하고 정상 속도·짧은 중요동작·가림을 함께 검증한다. 학습 손실 종류를 계속 바꾸는 접근을 먼저 반복하지 않는다.',
        '',
        '보존·검증',
        '12개 원본SHA256, 모든원래프레임/시간,24새출력의산술·확률·동일T관측·고정모델해시를별도검산했다. 기존완료결과는불변이다.',
        '동일Codex 구현/별도검산이며 독립수어전문가정답은아니다. C·T3-F·웹기본/사용자·멘토자료변경0. 아직최종데모통과/전체목표완료가아니며 새HTML은생략했다.',
        '실행전role필드명오류와generic model.py import충돌을기록하고복구했다. 첫동결과정정동결을모두보존했고추출·진단은각1회완료했다.']
    (K/'RESULT_KO.txt').write_text('\n'.join(txt)+'\n',encoding='utf8')
    exclusive(K/'FINAL.json',dict(at=at,status='INPUT_CONSISTENCY_BENEFIT_NOT_FULL_DEMO_READY',new_sources=12,new_comparisons=24,new_training=0,
        result=str(K/'RESULT_KO.txt'),verified=str(K/'VERIFIED.json'),next=nxt,web_changed=False,whole_goal_complete=False))
    write(K/'CURRENT_EXECUTION.json',dict(at=at,status='COMPLETE_DIAGNOSTIC',active=None,prepare_session=57498,prepare_exit=0,diagnose_exit=0,verify_exit=0,next=nxt,whole_goal_complete=False))
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf8')
    for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','동일입력12영상·24진단 완료/충돌7→0',summary),('## 9. 다음 단일 작업','공개 학습자료의 웹캠시야 변이 보완',nxt)]:
        h='### 2026-11-02 '+title;assert h not in text;text=text.replace(marker,marker+'\n\n'+h+'\n\n'+body+'\n',1)
    p.write_text(text,encoding='utf8');state=read(O/'STATE.json');state['current_stage']='COHERENT_SENSOR_DIAGNOSTIC_COMPLETE_SEATED_FIT_NEXT';state['updated_at']=at;state['next_single_action']=nxt
    state[K.name]=dict(at=at,status='INPUT_CONSISTENCY_BENEFIT_NOT_FULL_DEMO_READY',active=None,result=str(K/'RESULT_KO.txt'),verified=str(K/'VERIFIED.json'),summary=summary,whole_goal_complete=False);write(O/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n'+str(K/'RESULT_KO.txt')+'\n'+str(K/'VERIFIED.json')+'\n'+str(K/'CAUSE_AND_NEXT.json')+'\n'+str(J/'FINAL.json')+'\n\nK준비57498/진단/검산exit0,J학습38902/평가3915/검산exit0. 현재active=None. 같은Codex검산≠독립전문가정답. 새HTML없음.\n',encoding='utf8')
    for filename in ['FINAL.json','CURRENT_EXECUTION.json','VERIFIED.json','CAUSE_AND_NEXT.json']:read(K/filename)
    assert summary in p.read_text(encoding='utf8') and summary in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8') and read(O/'STATE.json')[K.name]['summary']==summary
    exclusive(K/'FINAL_VERIFICATION.json',dict(at=at,status='PASS',documents_agree=True,json_valid=True,whole_goal_complete=False))
    print('K finalized; goal remains active; clean order',len(supported)-len(caution),'/',9)
if __name__=='__main__':main()
