from pathlib import Path
import json,datetime,hashlib,shutil,re,sys
sys.stdout.reconfigure(encoding='utf-8')
R=Path('C:/MyUnityProjects/WordsInSilence');O=R/'research/front_keyphase_v1/ops'
I=O/'word100_wrist_only_inputs_20261026';D=O/'word100_wrist_only_dev_inputs_20261026';J=O/'word100_wrist_only_train_20261026'
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    review=read(J/'interim_visual_v2/REVIEW.json');trace=read(J/'interim_visual_v2/COORDINATE_TRACE.json')
    assert trace['selected_wrist_frames']==40 and trace['feature_insertions_verified']==71
    now=datetime.datetime.now().astimezone().isoformat();backup=J/'before_interim_review_sync';assert not backup.exists();backup.mkdir()
    paths=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
    prior={str(p):sha(p) for p in paths}
    for p in paths:shutil.copy2(p,backup/p.name)
    events=[json.loads(s) for s in (I/'EVENTS.jsonl').read_text('utf8').splitlines() if s.strip()];ev=events[-1]
    n=len(list((I/'records').glob('*.json')))
    summary=f'공개 TRAIN 입력 {n}/1798 준비 중({ev["at"]},exec98610/PID4948). DEV직렬후속exec83056/PID3468대기. 중간 위험40장면·5시트와4사례 앞뒤장면을 확인했고, 저장검출점0→출력손목40건/분류입력71삽입의 좌표변환 검산PASS. 일부회전·원근장면에서 손목점이 손바닥/손등쪽에 치우치는 검출위치 불확실성 확인. 합치기/스케일변환 오류는40표본에서 없었고 기존MP의 인접프레임에서도 유사오차가 있음. 다른pose손목도정답은아님. 손의대략위치와 정밀교정근거를분리: 현재W는불확실한위치입력의인식효과를시험할연구후보이며 정확한손목복원으로인정하지않는다. 전체검산·최종위험검토전학습READY아님;W학습0. C/T3F/E3/운영·사용자자료보존.'
    nxt='먼저 I exec98610와 D 직렬후속exec83056의실제세션/RUNNING/EVENTS/CONTINUATION_STATUS확인,중복금지. I1798완료후직렬후속이 I verify.py all→D prepare.py all600→D verify.py all을실행한다. 완료검산재실행금지. J/interim_visual_v2/INDEX·COORDINATE_TRACE·REVIEW와4temporal그림은완료근거재사용. 손목의해부학적정밀오차를전부해결했다고말하지않으며 pose손목으로일괄치환/특정단어필터금지. 전체ALL_VERIFIED뒤J review_full.py <I또는D>로모든YOLO+기록ID/view+위험순위표본검토. 기존좌표보존·새손의동일성/영상근거가충분하고명백한반대손/배경추가없으면정밀교정과분리한위치입력분류시험으로FULL_REVIEW판정;불확실성과검토범위명시,숫자PASS만으로READY금지. 중대한오배정/계약위반은먼저해결,조용한제외금지. 양쪽ready후J control.py freeze→run3seed→verify_results.py verify→cross_inputs.py run. 원래C저장점수재사용,모든899/300·원래학습일정보존,고정gate로개선/회귀보고. 추가손목bodyXY/valid만사용,손가락/크기/구간/dt불변,운영DTW에불확실좌표도입금지. 사용자/멘토/추가199/200금지. 작은후보완료로전체goal완료처리금지. SPEED972원인진단/합성348배치검산은완료재사용.'
    text=paths[0].read_text('utf-8-sig')
    for number,heading,body in [(8,'공개 손목 입력 중간 검토·위치 오차 출처 확인',summary),(9,'전수 검산·위험 검토 후 위치 입력 분류시험 계속',nxt)]:
        m=re.search(r'(?m)^## '+str(number)+r'\.[^\n]*\n',text);assert m
        text=text[:m.end()]+'\n### 2026-10-26 '+heading+'\n'+body+'\n근거: research/front_keyphase_v1/ops/word100_wrist_only_train_20261026/interim_visual_v2/REVIEW.json, COORDINATE_TRACE.json.\n\n'+text[m.end():]
    paths[0].write_text(text,encoding='utf8')
    state=read(paths[2]);state['updated_date_kst']=now;state['current_stage']='AUTONOMOUS_WRIST_INPUT_PREP_LOCALIZATION_UNCERTAINTY_REVIEW'
    state[I.name].update(at=now,last_records_count=n,latest_event=ev,next=nxt,interim_review=str(J/'interim_visual_v2/REVIEW.json'))
    state[I.name]['execution']['records_at_sync']=n
    state[J.name].update(at=now,status='INPUTS_RUNNING_INTERIM_COORDINATE_TRACE_PASS_PRECISION_UNCERTAIN',next=nxt,
        interim_review=str(J/'interim_visual_v2/REVIEW.json'),interim_trace=str(J/'interim_visual_v2/COORDINATE_TRACE.json'),learning_runs=0)
    paths[2].write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    handoff='# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음 단일 작업: '+nxt+'\n\n'
    handoff+='I: '+str(I)+'\nD: '+str(D)+'\nJ: '+str(J)+'\n\n'
    handoff+='실행: exec98610/PID4948(prepare.py all),exec83056/PID3468(serial_next.py),이번턴두실제세션연결확인. PID없음만으로잠금삭제금지. I에는STATUS.json없음. 현재새추론은입력준비만,학습0. 중간review v1은sha(str).read_bytes오류exit1보존;v2새폴더성공,원본재검출없음. interim_trace.py40점/71채널검산과4시퀀스출력완료exit0. preview5장/temporal4장모두view_image확인. 같은Codex시각·산술검사이며독립사람정답아님.\n\n'
    handoff+='재사용할완료근거: J EXECUTION_PREFLIGHT.json 합성348배치재구성/손상7유형거절/평가배치차이2.98e-8; 실제학습검증아님. SPEED=word100_speed_warning_decomposition_20261026의DONE/VERIFIED/RESULT_KO: 정상속도972대조저장결과의시간역대응진단,24940440항목검산. 경고감소는실제정렬개선/인식률아님. 기존단순위치/방향가중추가후보무효과를그대로반복하지않음.\n\n'
    handoff+='현재goal active/진행중. 작은실험이나문서완료만으로complete금지. 별도자동화없음. 기존브라우저file정책차단우회금지. 연구원본W:/sources읽기전용,사용자·멘토촬영접근/학습/튜닝/선택금지,추가199·200평가도튜닝금지,C/T3F/E3·운영웹/DTW·Git변경금지.\n'
    paths[1].write_text(handoff,encoding='utf8')
    after={str(p):sha(p) for p in paths}
    for p in paths:assert 'interim_visual_v2' in p.read_text('utf-8-sig')
    result={'at':now,'status':'SYNC_PASS','before':prior,'after':after,'records':n,'latest_event':ev,'training':0,'protected_models_changed':False}
    with (J/'INTERIM_REVIEW_SYNC.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'SYNC_PASS','records':n,'latest_event':ev,'training':0}))
if __name__=='__main__':main()
