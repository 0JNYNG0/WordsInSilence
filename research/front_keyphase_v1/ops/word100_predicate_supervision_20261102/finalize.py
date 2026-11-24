from pathlib import Path
import sys,json
from compile_supervision import U,O,read,sha,save

def main():
    assert read(U/'VERIFIED.json')['status']=='DIRECT_PREDICATE_SOURCE_AND_MASKS_VERIFIED'
    b=read(U/'BINDING_VERIFIED.json');assert b['clips']==36 and b['claims']==117
    review=read(U/'DIRECT_BINDING_REVIEW.json');assert review['structural_checks']['claims']==117
    typed=read(U/'TYPED_TARGETS_v2.json');old=read(U/'TYPED_TARGETS.json')
    assert len(typed['rows'])==532
    assert [{k:v for k,v in r.items() if k!='dependency_group'} for r in typed['rows']]==old['rows']
    for p,h in typed['pins'].items():assert sha(p)==h
    for p,h in review['pins'].items():assert sha(p)==h
    for r in typed['rows']:
        assert r['kind'] in typed['types'][r['target']]['allowed_kinds']
        assert r['dependency_group']==r['clip_id'] and r['person'] in range(1,10)
        assert r['y'] in [0,1] if r['mask'] else r['y'] is None
    plan=dict(next_single_action='Complete directly scoped negative/control and DEV source-component support targets under the same rubric, then freeze and run a matched learned source-component connector comparison.',
        current_training_data=str(U/'TYPED_TARGETS_v2.json'),current_training_data_sha256=sha(U/'TYPED_TARGETS_v2.json'),
        objective_unchanged='Learned KSL handshape/body relation/movement/order feedback linked to guide phrase and native user time; actual small demo, not merely data audit.',
        steps=[
          'Join only same-definition existing direct negatives at their own native point/interval. Preserve source-component relevance separately from event ownership.',
          'For elbow reorientation, bilateral thumb configuration, alternation and oblique pose, prepare necessary directly inspected counterexamples or precisely justified temporal controls. A reversed alternation remains possible alternation; no universal reversed-is-wrong rule.',
          'Prepare same-rubric DEV11-13 binding witnesses before new candidate predictions, disclosing their repeated development exposure; old exclusive-family labels stay untouched.',
          'Freeze a matched learned connector comparison using the existing SignRep/physical observer, old16physical outputs protected. No role-epoch/KL-only search. Record point and interval losses separately, mask unknowns, group correlated records by clip/event.',
          'Test native feedback time/source phrase, normal speed/rest/cross, short omission/reversal, image/pose gaps; all-held or fewer claims cannot count as improvement. Preserve old event metrics as separate diagnostics.',
          'If useful, integrate a separate full demo and measure actual video-to-feedback time. Audit future public recorded IDs/duplicates/use history before independent evaluation; H14/15 stay locked.'],
        unresolved_source_requirements=['Egypt dictionary-variant lowering connection','all fingertips gathered when individual tips/hidden thumb are not established'],
        safety_scope=['C/T3F/oldweb unchanged','private/mentor never training tuning selection','no new download/upload/Git','W originals read-only','no independent-human/expert-gold claim'],
        no_rerun_completed=['retention176','FITdiagnostic72','crossover176','nonmain440','directFITsourceReview36'],
        whole_goal_complete=False)
    save('NEXT_PLAN.json',plan)
    report='''이번 작업: 관측한 행동과 수어 설명을 연결할 학습 근거 준비

왜 했나
기존 판정부는 손모양·관계·이동 일부를 이미 배웠지만, 내부 단계 번호와 실제 설명의 행동이 같은 것은 아니다. 팔 근처 손이 보인다는 사실만으로 손바닥으로 팔을 쓸었다고 말해서는 안 된다. 이전 실패 점수는 그대로 남겼다.

실제 완료
1. 기존 48개 공개 영상의 144중앙점(10물리항목)과108움직임구간(6항목)을 원본해시·시간·값·미확인마스크까지 별도 검산했다. FIT는36영상/9기록ID, DEV는12영상/3기록ID이며 DEV는 이전부터 사용한 개발자료다.
2. 공개 FIT36영상(4표현×9기록ID)의 시간순 시트1102장면을 root와두Codex에이전트가 직접 확인했다. 필요한18개 큰장면 재확인도 있었다. 이는 새로운36사람이나 사람 전문가 검토가 아니다.
3. 출처 문구와 직접 보인 부분 근거117건을 연결했다: 지지72, 해당 특징과 반대인 장면18, 미확인27. 반대18건도 정상적인 앞/뒤 동작일 수 있으므로 사용자 수어 오답으로 쓰지 않는다.
4. 예: 순종의 팔 내림은 손목 하강 하나가 아니라 팔꿈치·전완 관계 변화로 확인했다. 잘하다의 팔 따라 이동과 나중 손목 근처 배치를 나눴다. 조상은 양손 높이가 번갈아 바뀌는 여러 시점을 함께 기록했다. 이집트의 양손 벌어짐을 사전의 내린다와 임의로 같은 뜻으로 만들지 않았다.
5. 학습용 항목532기록으로 편성했다. 같은36영상에서 나온 시점/구간 기록이라 독립532사례가 아니다. 기존 동일정의 정답을 연결한 결과4항목은 양성/음성 근거가 있고,4항목은 양성만 있으며2항목은 출처 연결/손끝 형태가 미확인이다. 전체손모양을 새로운 엄지정답으로 자동확장하지 않았다.

검증과 수정
출처·원본시점·마스크를 구현자와별도검토자가 확인했다. 학습읽기규격이sequence와interval을혼동할수있다는지적을고쳐allowed_kinds와영상단위묶음을명시했다. 첫TYPED_TARGETS는보존하고_v2에규격만보완했으며값/마스크/시간은동일함을검산했다. 같은동작의두표현을독립사례로세지않도록했다.

현재 한계
새 연결부 학습이나 새 성능비교는 이번 작업에서 실행하지 않았다. 성능이 올라갔다거나 피드백 데모가 완성됐다는 뜻은 아니다. 직접검토는Codex의영상관찰이며독립농인강사/전문가정답이아니다. 손끝접촉·사전표현변이·엄지의모든시점·필수반복횟수등은아직전체판정하지않는다.

다음
새 항목에 필요한 반대 사례와 같은 기준의 개발 평가 정답을 준비한 뒤, 기존 물리 관측은 지우지 않는 작은 학습 연결부를 비교한다. 정상 속도·손내림·교차와 짧은 누락/반대 이동을 함께 확인하고, 통과한 것을 별도 실제 촬영 데모 및 전체 처리시간 검증으로 연결한다. 작은 준비 완료를 전체 목표 완료로 바꾸지 않는다.

C·T3-F·기존웹은 그대로. 사용자·멘토촬영 접근/학습/튜닝/선택없음. 새 다운로드/외부전송/Git없음. HTML은생략했다.
'''
    with (U/'RESULT_KO.txt').open('x',encoding='utf8') as f:f.write(report)
    save('FINAL_VERIFIED.json',dict(status='PREDICATE_AND_DIRECT_SOURCE_COMPONENT_SUPERVISION_READY_FOR_TARGET_COMPLETION',
        points=144,motion_windows=108,fit_clips_reviewed=36,sampled_images=1102,partial_claims=117,
        typed_records=532,typed_independent_clips=36,schema_revision_labels_unchanged=True,
        current_training_file='TYPED_TARGETS_v2.json',new_training=0,new_inference=0,whole_goal_complete=False))
    sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'));import sync_flow;sync_flow.U=U
    summary='기존144중앙점/108움직임구간의값·원시간·mask검산완료. 공개FIT36영상1102샘플장면을root/2agent직접검토해출처부분근거117(지지72/특징반례18/미확인27)편성. typed532기록은같은36영상으로중복독립계산금지. v2규격의point/sequence/interval구분보완,기존값동일. 새학습/성능향상/웹적용아님. 전체goal active.'
    nxt='NEXT_PLAN에따라부분행동target의부족한음성/정당한시간대조와DEV11-13동일rubric연결정답준비후SignRep기존16물리출력보존하는작은source-component연결부고정비교. TYPED_TARGETS_v2만사용;117부분근거를문장전체정오나dense정답으로승격금지. 반례정상단계≠사용자오류,reverse≠일괄negative. 완료retention176/FIT72/교차176/합산440/36시각감사재실행금지. H14/15·개인·멘토·C/T3F·기존웹보존. 현재프로세스없음/전체goal미완료.'
    sync_flow.sync('관측 항목과 출처 동작 연결 정답 편성 완료',summary,nxt,'DIRECT_COMPONENT_BINDING_SUPERVISION_COMPILED')
    print('Result/NEXT/STATE/context/handoff synced; whole goal active')
if __name__=='__main__':main()
