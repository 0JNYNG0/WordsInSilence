"""Close this milestone, not the persistent full goal."""
import json,hashlib,urllib.request
from collections import Counter
from common import G,H,R,P,O,read,sha,write,exclusive,clock

def main():
    v=read(G/'DIAGNOSTIC_VERIFICATION.json');ui=read(G/'DEMO_UI_VERIFICATION.json');render=read(G/'DEMO_RENDER_VERIFICATION.json')
    assert ui['status']==render['status']=='PASS';assert abs(ui['fresh_public_analysis_seconds']-8.347108700007084)<1e-6
    launch=read(G/'DEMO_LAUNCH.json');state=json.load(urllib.request.urlopen(launch['url']+'api/state',timeout=5));assert state['ready'] and not state['busy']
    public=list((G/'public_runtime_runs').glob('*/INTENT.json'));assert len(public)==1
    intent=read(public[0]);out=read(public[0].parent/'analysis/FEEDBACK.json')
    assert intent['public_only'] and sha(intent['source'])==out['source_sha256']
    assert out['training'] is False and out['tuning'] is False and out['individual_reference_comparisons']==0 and out['learned_prototype_comparisons']==1
    assert out['whole_sign_correctness'] is None and out['model_version']=='learned-guide-rgb-motion-20261102-fixed42'
    for p,h in read(G/'FROZEN.json')['pins'].items():assert sha(p)==h
    for p,h in read(P/'INPUT_MANIFEST.json')['source_pins'].items():assert sha(p)==h
    exclusive(G/'INTEGRATED_PUBLIC_VERIFICATION.json',dict(status='PASS_PUBLIC_FRESH_RUNTIME_AND_MODEL_PRESERVATION',
        public_source=str(intent['source']),source_sha256=out['source_sha256'],result=str(public[0].parent/'analysis/FEEDBACK.json'),
        actual_seconds=ui['fresh_public_analysis_seconds'],reference_video_comparisons=0,learned_prototype_comparisons=1,
        model_pins_preserved=True,no_private_source_access=True,camera_opened=False,verifier='Same Codex, separate saved-result and browser paths'))
    # Inventory prior normal stage messages without scoring unlabelled fingers as truth.
    public_stages={}
    for kind in ['original','speed_0.8','speed_1.25','scale0.9_shift0.03']:
        rr=[r for r in read(H/'INPUT_MANIFEST.json')['rows'] if r['kind']==kind]
        public_stages[kind]=dict(Counter(s['status'] for r in rr for s in read(r['result'])['stages']))
    exclusive(G/'REMAINING_SCOPE.json',dict(whole_goal_complete=False,public_normal_stage_messages=public_stages,
        not_framewise_accuracy='These normal stage messages are not expert labels or independently graded handshape errors.',
        causal_findings=[
            'Short3-frame pose loss: exact reviewed feature centers60/63 unchanged. Pose7:59/63, one case loses2 features while another gains1.',
            'Image+pose unavailable centers40/63 raw estimates; raw estimates not observable truth; all missing times explicitly retained.',
            'Main-order reversal changes the physical source order; neural stage configuration accuracy29/30 to25/30. Some same-direction motion or splice jumps remain.',
            'Monotonic DTW always finds a path. Direction occurrence must not be promoted to guide-order or full-sign correctness.',
            'Stage-wide expected feature averages may mix hold and changing handshape; this is a general code-level issue to audit on FIT, not a word exception.'
        ],
        next_unit='word100_learned_phase_sequence_20261102',
        next_actions=[
            'Audit FIT01-09 exact physical labels, guide clauses, and transition spans; distinguish reviewed labels from DTW-propagated weak labels. Unknown is not a negative.',
            'Use the existing neural stage/feature representations. Compare current fixed pipeline against a small ordered-phase/clause head with fixed parts preserved. Actual learning only where FIT physical order labels are supported.',
            'Supervise normal multi-person sequences, normal speed/gap variation, and physically verified changed order/missing events together. Repeated equivalent states are not wrong by default. No automatic required repeat count.',
            'Predeclare conditions before any new scores. Use person-separated FIT validation and known exposed DEV11-13 for development. Do not tune on locked14/15 or rerun the finished H comparison. Report limited independence honestly.',
            'Require reduced important-order misses without more normal false correction, protected shape/relation outputs, explicit missing evidence, and whole-runtime timing before optional demo update.'
        ],
        no_new_external_model=True,no_new_download=True,no_private_training_tuning_selection=True))
    summary=('학습된 손모양·관계·이동 신경망을 새 연구 데모8807에 연결하고 공개 영상 실제 분석8.35초를 확인했다. '
        '참고 영상 개별 비교0회, 학습된 흐름 DTW1회. 관절만3프레임 누락시 손모양 정답60/63 유지,7프레임은59/63으로 한 사례의 약화가 남았다. '
        '영상까지없는7프레임은원시추정40/63으로저하했지만 모든누락시간을보존했다. 주요구간역순4개에서 신경망상태구분29/30→25/30; '
        '이동이보임과그이동을올바른순서로완료함은다르므로 두근거를화면에서분리했다. 12저장시험의영상·뼈대·구간재생·글자대비와 새공개분석을검증. '
        '전체순서·엄지·정확한접촉·필수반복횟수미완료,목표active. C·T3F/8804~8806보존,사용자·멘토촬영 학습·튜닝·선택 없음.')
    nxt=('word100_learned_phase_sequence_20261102에서 FIT01~09의 기존 물리적 주석·가이드 연결을 먼저 확인하여 유지상태/변화중/순서의 학습정답 범위를 고정한다. '
        '현재 모양·관계·이동 신경망은 보존하고, 같은 이동이 있다는 것과 설명의 중요한 순서대로 이뤄졌다는 것을 나누는 작은 시간 판정부를 검토·학습한다. '
        '올바른 반복·속도·짧은관절누락은정상으로함께학습하고, 물리적으로확인한순서변경/주요단계누락만대조한다. '
        'DTW가주입한가짜정답이나 단어별예외/반복횟수강제금지. FIT사람분리검증과이미노출된DEV11~13비교,후속실험조건은점수전에고정한다. '
        'H14/15자료를후속학습/튜닝/선택에쓰지않고완료한H평가재실행금지. 새다운로드/사용자자료학습없음. 데모8807 PID'+str(launch['pid'])+' 실행중이며카메라는사용자클릭만. 전체목표완료아님.')
    report=('이번에 만든 것과 남은 것\n\n'
        '목표: 여러 사람의 공개 영상과 동작 설명을 배운 신경망이 새 사용자의 주요 동작을 이해하고, 설명의 어느 부분을 어떻게 수행했는지 알려주는 연구용 데모.\n\n'
        '실제 학습 구조\n'
        'SignRep의 사전학습 영상 표현 + 관절 관측 → 학습된 손모양/신체관계/시간 흐름/이동 판정부 → 학습한 단어 흐름과 DTW1회 → 원래 영상 시간과 가이드 문구에 연결된 피드백. '
        '지도학습이며 이전 단위에서 SignRep 마지막 투영층 일부도 미세조정했다. 이번 후보는 기존 영상 표현을 직접 읽는 작은 이동 신경망의 실제 가중치를 학습했다. '
        '목표 단어는 배운 설명을 고르는 데 쓰고, 관측한 손모양이나 관계를 정답처럼 바꾸는 데 쓰지 않는다. 4표현에서 목표를 바꿔도 물리적 관측 출력은 동일함을 검증했다.\n\n'
        '확인한 개선\n'
        '- 개발 자료 정상 속도·대기 차이의 주요 이동 확인47/60→57/60, 원래 개발 자료40/45유지. 중요한 팔이동을 삭제한 시험의 충분확인2/3→0/3.\n'
        '- 현재 후보 학습/선택에 쓰지 않은 공개 기록ID14/15: 이전104/120→현재109/120. 단, 직접 구간을 지정한5방향 균형정확도96.67%→94.58%이므로 모든지표개선은아니다.\n'
        '- 관절 검출만3프레임(30fps에서약0.1초)빠진 경우 실제RGB를유지하면12장면의확인가능한63특징중60개를맞춰원래성적을유지.7프레임(약0.23초)에서는59/63. 이는인공관측누락시험이며모든실제가림의해결을뜻하지않는다.\n'
        '- 영상도없는경우기억/추정출력이정답처럼나오지않도록누락시간을남겼다. 원시분류40/63은사용자에게맞았다고채택한점수가아니다.\n\n'
        '실제 화면: http://127.0.0.1:8807/\n'
        '1) 정상 공개 영상 / 주요 구간을 뺀 시험 / 순서를 바꾼 시험을 볼 수 있다.\n'
        '2) “공개 영상 새로 분석”은 저장 결과 표시가 아니라 공개 영상을 실제 처리한다. 확인한1회는8.35초였다.\n'
        '3) 사용자가 직접 카메라를 켜거나 파일을 선택하면 새 촬영도 추론만 할 수 있다. 이번 검증은카메라와사용자촬영을쓰지않았다.\n'
        '4) 확인한손모양·관계·이동, 추정/보류한항목, 설명문구와원래영상의재생시간을따로보여준다. 참고영상1편은따라하기예시이며분석때9개영상을따로대조하지않는다.\n'
        '5) 이 화면은 새 연구 후보이며 C·T3-F나 기존8804~8806을대체하지않았다.\n\n'
        '아직 완성하지 못한 핵심\n'
        '손이 내려간 장면을 찾는 것만으로 “정해진 순서에서 그 동작을 했다”고 말할 수 없다. 주요 구간을 거꾸로 재생한 영상에서도 일부 이동 자체나 편집 경계의 이동이 남는다. '
        '현재 신경망은 확인한30개손상태표본에서원래29개,순서를바꾼후25개를구분했지만, 전체순서의정오를안정적으로설명하는판정부는아직완료하지못했다. '
        '그래서“이동이보임”과“설명의이구간과연결이확실함”을분리했고단어전체를합격처리하지않는다.\n'
        '정상 참고 영상에서도 일부 특징 차이 후보가 남는다. 이는 검증된 사용자 오류가 아니며, 유지 상태와 모양이 바뀌는 과정에 같은 기준을 적용하는지 FIT자료로 점검해야 한다.\n\n'
        '다음: '+nxt+'\n\n'
        '검증담당: Codex구현/실행,별도저장결과산술검증과로컬브라우저검증. 독립된한국수어전문가검토아님. '
        '내장브라우저도구와제한환경의브라우저시작실패는기록하고,프로젝트전용숨김브라우저로공개영상만검증해통과했다.\n')
    (G/'RESULT_KO.txt').write_text(report,encoding='utf8')
    now=clock();cur=dict(status='ISOLATED_NEURAL_DEMO_VERIFIED_SEQUENCE_GOAL_REMAINS',at=now,summary=summary,next=nxt,
        active_training_or_evaluation=None,demo_pid=launch['pid'],demo_url=launch['url'],goal_status='active',whole_goal_complete=False)
    write(G/'CURRENT_EXECUTION.json',cur);s=read(O/'STATE.json');s.update(current_stage=cur['status'],next_single_action=nxt,next_single_task=nxt,updated_date_kst='2026-11-02');s[G.name]=cur;write(O/'STATE.json',s)
    root=O.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf8')
    for h,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','학습피드백 연구데모8807 검증·순서판정은진행목표',summary),('## 9. 다음 단일 작업','중요한상태·변화·순서 신경망 감독 보완',nxt)]:t=t.replace(h,h+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
    p.write_text(t,encoding='utf8');(root/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n현재:'+str(G)+'\n완료 학습/추출/평가 재실행 금지. 미완료 전체목표를 작은 데모 완료로 대체하지 않는다.\n',encoding='utf8')
    p=root/'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt';t=p.read_text(encoding='utf8');lo=t.index('최신 실행 현황');hi=t.index('1. 사용자가 확정한 목표',lo)
    p.write_text(t[:lo]+'최신 실행 현황 ('+now+')\n'+summary+'\n다음: '+nxt+'\n\n'+t[hi:],encoding='utf8')
    p=root/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
    with p.open('a',encoding='utf8') as f:f.write('\n\n2026-11-02: 학습된 신경망 피드백의 실제 통합 및 목표 보존\n'+summary+'\n'
        '인식률(TCN)을 올렸다는 보고가 아니라 세부 피드백의 움직임/관측/흐름 실험이다. 다인참고를매번9회대조하는방식에서학습된모델+기준흐름1회정렬로실제전환했다. '
        '학습은감독신호가있는부분을지도학습하고기존SignRep투영층일부와추가신경망을미세조정. 이번단위는가중치변경없이통합검증. '
        '정상변형확인향상과정밀방향점수하락을함께보존했다. 순서이해의빈틈을좌표검출완벽화문제로돌리지않고상태/변화/순서감독의다음과제로정했다.\n'+nxt+'\n')
    exclusive(G/'FINAL_ARTIFACTS.json',dict(at=now,status=cur['status'],goal_complete=False,
        code_and_results={str(p):sha(p) for p in G.iterdir() if p.is_file() and p.suffix in ['.py','.js','.html','.json','.txt'] and p.name not in ['CURRENT_EXECUTION.json','demo_stdout.txt','demo_stderr.txt']},
        original_models_preserved=True,old_demos_preserved=True,private_training_tuning_selection=False))
    print('Verified demo',launch['url'],'PID',launch['pid'],'goal remains active')

if __name__=='__main__':main()
