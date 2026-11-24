from prepare_holdout import H,R,O,read,write,clock

def main():
    root=O.parents[2]; now=clock(); r=read(R/'SAVED_VERIFICATION.json'); assert r['counts']==dict(normal=57,native=40,blackout=0,removed_arm=0)
    summary=('SignRep 영상특징을 움직임 신경망에 직접 연결한 후보의 6회 학습과 288개 비교·별도 검산을 완료했다. '
        '정상 속도·대기 차이 확인은 초기47/60→현재57/60, 원래 개발 영상은40/45로 유지했고 주요 팔 이동 삭제의 충분확인2/3→0/3, 검은 구간15/15보류. '
        'RGB 없이 동일하게 추가 학습한 대조군도 정상57/60이며 원래39/45이므로 직접 RGB의 추가 이득은 제한적이다. 기존 손모양·관계·DTW 출력 보존. '
        '고정한 공개 기록 ID14/15의8원본·32조건을 준비 중(exec57188). 이 자료는 과거 TCN 평가 이력은 있으나 현재 후보 학습·선택에는 쓰지 않았다. '
        '기존 C·T3-F와8804~8806미교체. 사용자·멘토 영상/결과 내용 학습·튜닝·선택 없음. 전체 목표는 아직 미완료다.')
    nxt=('word100_learned_feedback_holdout_20261102의exec57188/PREP_PROGRESS와 실제 종료를 확인한다. 중복 추출 금지. '
        '완료 후 verify_inputs.py→evaluate_holdout.py→verify_results.py 순서로 고정2조건×3seed의192비교를 한 번 수행한다. '
        'ID14/15점수로 모델/seed/문턱을 선택하거나 튜닝하지 않는다. 원본27/30·전체108/120이상, 정상 새 반대방향 경고 없음·이전출력 보존 기준. '
        '통과하더라도 관절만 잠시 빠진 경우, 중요한 순서 차이, 새 후보 실제 데모와 전체 처리시간이 남는다. 완료된 학습·기존288비교 재실행 금지.')
    report=('영상 특징 직접 연결 개발 결과\n\n'+summary+'\n\n'
        '학습: 공개 FIT 기록 ID01~09, 4표현의 기존 입력144개, 지도학습된 움직임 신경망2조건×3seed. SignRep 본체를 이번에 새로 학습한 것은 아니다. 이전 단위의 최종 투영층 일부 미세조정은 유지한다.\n'
        '57/60은20동작 항목×3seed, 40/45는15항목×3seed이며 독립적인60/45개 영상이 아니다.\n'
        '영상 특징 없이 추가학습: 정상57/60, 원본39/45. 영상 특징 추가: 정상57/60, 원본40/45. 제한된5방향 균형정확도 평균은 두 조건 모두94.44%. RGB가 모든 성능을 크게 높였다는 결론은 아니다.\n'
        '추가 움직임 신경망 계산 중앙값3.01ms는 전체 분석 시간이 아니다. 공개 영상 전체 처리시간은 다음 평가에서 별도 측정한다.\n'
        '기존 손모양/신체관계/DTW 출력은288비교 모두 보존. 구현과 별도 산술검증은 Codex가 수행했으며 독립된 사람의 정답 검증과 다르다.\n'
        '한계: 제한된4표현·5방향, 반복횟수와 접촉 깊이·모든 수형·전체 언어 정오 미검증. 사용자자료 내용은 열지 않았으며 파일 경로의 우연한 목록 노출은 SOURCE_ACCESS_NOTE에 별도 기록.\n\n다음: '+nxt+'\n')
    (R/'RESULT_KO.txt').write_text(report,encoding='utf8')
    write(R/'CURRENT_EXECUTION.json',dict(status='DEV_GATE_PASSED_LOCKED_EVALUATION_PENDING',at=now,active_exec_session=None,train_session86124_exit=0,evaluate_session76832_exit=0,goal_status='active',next_unit=H.name))
    current=dict(status='LOCKED_PUBLIC_INPUT_PREPARATION',at=now,active_exec_session=57188,summary=summary,next=nxt,goal_status='active')
    write(H/'CURRENT_EXECUTION.json',current); state=read(O/'STATE.json');state.update(current_stage=current['status'],next_single_action=nxt,next_single_task=nxt,updated_date_kst='2026-11-02')
    state[R.name]=read(R/'CURRENT_EXECUTION.json');state[H.name]=current;write(O/'STATE.json',state)
    p=root/'WordsInSilence_PROJECT_CONTEXT.md'; text=p.read_text(encoding='utf8')
    for h,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','RGB개발기준 통과·별도기록ID 평가 준비',summary),('## 9. 다음 단일 작업','고정 공개32입력·192비교',nxt)]:
        text=text.replace(h,h+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
    p.write_text(text,encoding='utf8')
    (root/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n현재 단위: '+str(H)+'\n',encoding='utf8')
    print('Synchronized completed RGB development and locked32 preparation')

if __name__=='__main__':main()
