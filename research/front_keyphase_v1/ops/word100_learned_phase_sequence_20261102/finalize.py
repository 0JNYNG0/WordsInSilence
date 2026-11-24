"""Close the phase comparison, retain the active larger neural-feedback goal."""
import shutil
from phase_model import *

def main():
    assert not (S/'FINAL_ARTIFACTS.json').exists()
    v=read(S/'SAVED_VERIFICATION.json');iv=read(S/'INPUT_VERIFICATION.json');diag=read(S/'SEQUENCE_DIAGNOSIS.json');training=read(S/'TRAINING_RESULT.json');assert len(training['runs'])==6 and v['status']=='DO_NOT_REPLACE_CURRENT_DEMO'
    assert not (S/'RUNTIME_RESULTS.json').exists()
    for p,h in read(P/'INPUT_MANIFEST.json')['source_pins'].items():assert sha(p)==h
    for p,h in read(S/'FROZEN_TRAINING_V2.json')['pins'].items():assert sha(p)==h
    T=O/'word100_learned_state_context_balance_20261102';T.mkdir(exist_ok=True)
    nxt=('word100_learned_state_context_balance_20261102에서 공개 FIT01~09만 이용해 전체역순뿐 아니라 주요부분만 순서가 바뀐 실제 영상 입력을 준비한다. '
         '현재 실제 자세를 앞뒤 문맥이 덮어쓰지 않도록 현재관측과문맥의 감독을 함께 검토하며, 손내림/배경의 직접확인 주석을 더해 DTW가고른창만으로비교범위를제한하지않는방법을 검증한다. '
         '같은9개물리상태·사람분리·기존모양/관계/이동보존·정상속도/교차/반복/짧은누락·상태삭제/국소순서변경을 함께비교한다. '
         'S의주후보는정상일부상태회귀로미채택. 학습데이터/정답/후보/기준을먼저동결하며DEV기반문턱반복조정·단어예외·H14/15재튜닝·사용자/멘토학습금지. 전체목표active.')
    exclusive(T/'PLAN.json',dict(at=clock(),status='AUTHORIZED_SOURCE_PLAN_PREPARED_NOT_TRAINED',source_diagnosis=str(S/'SEQUENCE_DIAGNOSIS.json'),
        goal='Learn the visible current physical configuration and its supported order without temporal guesses or a restricted DTW window hiding an important event.',
        steps=[
          'Prepare actual FIT local-main reorders from direct-reviewed source frame maps; review sparse visible hands-down/background evidence before assigning labels. No new external models.',
          'Freeze a current-state/context neural comparison which protects supervised current observations, masks uncertain transitions, and preserves existing handshape/relations/movement.',
          'Train only on public FIT, use explicitly disclosed head-level person-separated validation, and evaluate fixed candidates on exposed development. Local edits are physical contrasts, not automatically all linguistic errors.',
          'Compare native state/order and whole relevant range, normal variation, observation gaps, and actual full latency. Integrate only validated scope in an isolated research demo.'
        ],fit_recorded_ids=list(range(1,10)),development_recorded_ids=[11,12,13],locked14_15='No reuse for fitting, selection or repeat evaluation',private_inputs='Forbidden',preserve=['C','T3-F','8804-8807','old results','raw/native time'],whole_goal_complete=False))
    # Source/index ledger only; not a newly trained or evaluated candidate.
    labels={r['clip_id']:r for r in read(S/'PHYSICAL_STATE_TARGETS.json')['rows'] if r['role']=='fit'};records=[]
    for r in read(P/'INPUT_MANIFEST.json')['rows']:
        if r['role']!='fit':continue
        points=[p['frame'] for p in labels[r['clip_id']]['points'] if p['mask']];n=r['frames'];lo=max(0,min(points)-3);hi=min(n-1,max(points)+3);mapping=list(range(n));mapping[lo:hi+1]=reversed(mapping[lo:hi+1])
        assert len(mapping)==n and sorted(mapping)==list(range(n))
        records.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],source=r['source'],source_sha256=r['source_sha256'],frames=n,local_reverse_range=[lo,hi],source_frame_indices=mapping,
            background_review_frames=[0,3,n-4,n-1],background_labels_assigned=False))
    exclusive(T/'SOURCE_PLAN.json',dict(rows=records,actual_extraction_started=False,new_training=False,derived_from_reviewed_fit_only=True,local_reverse_is_not_automatic_linguistic_error=True))
    summary=('실제영상에서확인한9종넓은자세358시점(FIT262/DEV96)을정리하고 공개FIT36원본의180입력으로 작은자세판정부6회+사전점검1회지도미세조정을완료했다. '
        '고정시간문맥후보는원래DEV자세85/96→95/96,전체역순74/96→93/96,주요부분역순25/30→26/30. '
        '다만정상변형의한팔관계시점을문맥이너무일찍다음자세로분류해같은원본의여러변형에서회귀했다. 정상전체201/219→211/219지만상태별보존기준실패로8807미교체. '
        '88입력×7조건616저장비교별도검산. 정상28조건의순서충돌경고0이나국소역순1개를순서근거있음으로남겨,자세오류와비교창/반복상태집계문제가함께남음을확인. '
        '새판정부약0.007초는판정부만의시간이며전체처리시간아님. 새실행화면/전체속도시험미실행. C·T3F/기존데모와개인자료보존,전체목표active.')
    report=('새 목표 아래 진행한 실제 학습 비교\n\n'
      '목표는 여러 사람의 공개 한국수어 영상과 동작 설명을 배운 신경망이 새 사용자의 손모양·신체관계·움직임·순서를 분석하고, 해당 설명과 사용자 시간대에 연결한 피드백을 주는 것이다. 기존 단어인식 C·T3-F는 별도로 보존한다.\n\n'
      '이번에 한 일\n'
      '1. 기존 동작 구간 중 직접 확인한 구간과 DTW로 옮긴 임시 구간을 구분했다. FIT36개 중4개는기준영상직접구간,32개는DTW에서옮긴약한구간이었다. 이 임시구간 전체를 새정답으로 쓰지않았다.\n'
      '2. 공개48영상의404표본을 직접 보고9종의넓은자세를확인했다. 확실한358시점만사용했고 나머지46시점은변화중/겹침/마무리등으로확정정답을주지않았다. FIT9기록ID262시점, DEV3기록ID96시점이다. 이는Codex의물리적주석이며독립수어전문가정답은아니다.\n'
      '3. 공개FIT36원본의원래영상·0.8배·1.25배·실제전체역순144입력과RGB를유지한3프레임관절누락36입력을사용했다. 실제역순영상은픽셀을바꾼뒤다시추출했으며시간특징배열만뒤집지않았다.\n'
      '4. 기존SignRep학습표현을보존하고자세판정가중치를지도학습으로미세조정했다. 시점별판정/추가시간문맥판정을각3seed,총6회비교했다. 앞선SignRep최종투영층미세조정은유지했으며이번에본체전체를추가학습한것은아니다.\n\n'
      '주후보 결과 (학습에 넣지 않은 개발기록ID11~13, 이미 여러 실험에 쓰인개발자료)\n'
      '- 원래 영상의 확인 가능한 자세:85/96→95/96.\n'
      '- 전체를 거꾸로 재생한 실제 영상:74/96→93/96.\n'
      '- 주요 부분만 거꾸로 바꾼 실제 영상:25/30→26/30.\n'
      '- 정상 속도·대기 변형 포함:201/219→211/219. 변형들은 같은 원본에서 나왔으므로219개의독립영상이아니다.\n'
      '- 새판정부중앙처리시간약0.007초. 영상추출을포함한전체시간은이번에새로측정하지않았다.\n\n'
      '왜 기존 화면을 바로 바꾸지 않았나\n'
      '전체 성적은 좋아졌지만, 한 정상 장면에서는 오른손이 아직 왼팔 위쪽에 있는데 앞뒤 문맥을 너무 강하게 따라 이미 손목 쪽에 도착했다고 분류했다. 같은 원본의 속도·대기변형에서도이오류가반복돼중요상태별보존기준을통과하지못했다. '
      '시점별판정대조군은해당원래장면을맞췄지만, 전체역순에서는시간문맥후보가더좋았다. 어느한쪽의좋은결과만골라전체성공이라고보고하지않는다.\n\n'
      '순서 검사에서 확인한 남은 원인\n'
      'DTW는 항상 어떤 대응을 만들기 때문에, 그것만으로 순서가 맞았다고 할 수 없다. 새 판정은 원래 영상 시간에서 상태가 나타난 순서를 별도로 보았다. 정상28조건에서순서충돌경고는0이었다. '
      '그러나 주요부분역순시험1개에서 앞에잠깐보인준비자세와뒤의이동을조합해충분한순서로남겼다. 정확한구간을별도로주어도일부자세오분류는남았고,다른사례는구간범위만바꿔도순서차이를찾았다. '
      '즉 현재자세의분류와비교구간·반복상태집계가함께관련된다. 반복횟수나처음/마지막손을임의규칙으로강제하지않았다.\n\n'
      '다음 작업\n'+nxt+'\n\n'
      '이번6학습·616비교·별도산술검산은완료했다. 새후속단위에는36공개FIT자료의국소역순원본대응계획만준비했으며추출/학습은아직실행하지않았다. '
      '현재사용가능한연구화면8807은그대로이다. 새HTML은만들지않았다. 사용자·멘토자료학습/튜닝/선택,새외부다운로드,운영모델교체없음. 전체목표완료아님.\n'
      '검증기록: 초기평가입력확인에서누락시험의시간축이별도원본배열에저장되어있음을발견해연결을정정했고최초오류로그를보존했다. 새예측전에해결했다. '
      '원래동결계획은보존하고학습전V2에예정된짧은관절누락조건을명시했다. 점수를본뒤문턱이나정답을수정하지않았다.\n')
    (S/'RESULT_KO.txt').write_text(report,encoding='utf8');now=clock();cur=dict(status='STATE_LEARNING_IMPROVED_CONTEXT_REGRESSION_RETAINED',at=now,summary=summary,next=nxt,active_training_or_evaluation=None,goal_status='active',whole_goal_complete=False,report=str(S/'RESULT_KO.txt'),existing_demo='http://127.0.0.1:8807/',new_web_replacement=False)
    write(S/'CURRENT_EXECUTION.json',cur)
    root=O.parents[2];files=[root/'WordsInSilence_PROJECT_CONTEXT.md',root/'NEW_CHAT_HANDOFF.md',O/'STATE.json',root/'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt',root/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']
    (S/'before').mkdir(exist_ok=True)
    for path in files:shutil.copy2(path,S/'before'/path.name)
    state=read(O/'STATE.json');state.update(current_stage=cur['status'],next_single_action=nxt,next_single_task=nxt,updated_date_kst='2026-11-02');state[S.name]=cur;state[T.name]=dict(status='AUTHORIZED_SOURCE_PLAN_PREPARED_NOT_TRAINED',plan=str(T/'PLAN.json'),source_plan=str(T/'SOURCE_PLAN.json'),active=None);write(O/'STATE.json',state)
    path=root/'WordsInSilence_PROJECT_CONTEXT.md';text=path.read_text(encoding='utf8')
    for heading,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','신경망자세감독6학습·문맥회귀와구간문제분리',summary),('## 9. 다음 단일 작업','현재자세근거와국소순서·비교범위공동보완',nxt)]:
        assert heading in text;text=text.replace(heading,heading+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
    path.write_text(text,encoding='utf8')
    (root/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n현재완료:'+str(S)+'\n다음:'+str(T)+'\n진행중학습/추출/평가없음. 완료한S616비교/H잠금평가재실행금지. 기존8807은미교체. 현재신경망지도학습목표active.\n',encoding='utf8')
    path=root/'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt';text=path.read_text(encoding='utf8');lo=text.index('최신 실행 현황');hi=text.index('1. 사용자가 확정한 목표',lo);path.write_text(text[:lo]+'최신 실행 현황 ('+now+')\n'+summary+'\n다음: '+nxt+'\n\n'+text[hi:],encoding='utf8')
    with (root/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:f.write('\n\n2026-11-02: 사용자 새지도학습·미세조정신경망목표의 지속 확인과 실제상태감독\n'+summary+'\nDTW에서옮긴구간을전체프레임정답으로여기지않고직접보이는상태를학습했다. 새모델탐색이나좌표완벽화대신현재자세와앞뒤흐름의충돌을분리했다. 전체순서학습은계속해야하며작은검증을전체완료로간주하지않는다.\n'+nxt+'\n')
    # Separate read-back checks after the authoritative documents were written.
    ss=read(O/'STATE.json');assert ss[S.name]['status']==cur['status'] and ss[T.name]['active'] is None
    assert summary in (root/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8') and nxt in (root/'WordsInSilence_PROJECT_CONTEXT.md').read_text(encoding='utf8')
    exclusive(S/'FINAL_ARTIFACTS.json',dict(at=now,status=cur['status'],goal_complete=False,old_models_and_demos_preserved=True,
        files={str(p):sha(p) for p in S.iterdir() if p.is_file() and p.suffix in ['.json','.py','.txt'] and p.name!='CURRENT_EXECUTION.json'},root_documents_checked=True,next_unit=str(T)))
    print('Phase unit verified, original demo kept, active goal continues; next source plan36FIT clips prepared.')

if __name__=='__main__':main()
