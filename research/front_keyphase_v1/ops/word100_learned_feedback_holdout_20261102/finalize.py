from prepare_holdout import H,R,Q,O,read,sha,exclusive,write,clock
import numpy as np

def main():
    v=read(H/'SAVED_VERIFICATION.json');g=read(H/'EVALUATION_GATE.json'); assert v['gate_status']==g['status']
    summary=('公開記録ID14/15'.replace('公開記録','공개 기록')+'의8원본과 속도/화면크기 변형32입력·192비교를 완료했다. '
        '고정 RGB후보의 원본 주요이동 확인27/30, 전체109/120으로 사전기준을 통과했다. 이전 후보는26/30·104/120. '
        '다만 직접 지정한 물리적 구간의5방향 균형정확도 평균은96.67%→94.58%로 낮아져 모든 지표의 개선은 아니다. '
        '기존 손모양·관계·DTW192비교 보존, 정상의 새로운 반대방향 경고0. 전체 전처리·기존분석 중앙7.87초, 새 움직임판정부는약0.003초 추가. '
        '검증은8원본·2기록ID의 제한된 시험이며 독립 한국수어 전문가 평가가 아니다. 학습/튜닝/seed선택 없이 결과를 보존했다. C·T3-F와 기존 데모 미교체.')
    next_step=('다음은 word100_learned_feedback_integration_20261102에서 공개 DEV의 정확한 손모양 정답 중앙시점에 '
        '3/7프레임 관절만 누락시키고 RGB·앞뒤흐름이 남는 조건을 검증한다. 영상도없는 대조조건은 별도로 보류해야 한다. '
        '목표 단어를 바꿔도 관측된 손모양·관계 자체가 바뀌지 않는지 검사하고 실제 순서변형의 근거를 확인한다. '
        '통과 범위만 새 연구 데모에 연결하여 사용자 원래시간·가이드문구·추정/보류·전체처리시간을 표시한다. '
        'H잠금평가 재추론/재선택 금지. 14/15를 후속학습으로 돌리지 않는다. 사용자·멘토자료 학습·튜닝·선택 금지. 전체목표 active.')
    text=('다른 공개 수행자 검증 결과\n\n'+summary+'\n\n'
        '무엇을 확인했나\n'
        '같은4표현을 현재 후보 학습에 쓰지 않은 기록 ID14·15가 수행한8원본으로 확인했다. 원본,0.8배속,1.25배속,화면축소0.9+가로이동0.03의4조건이다. '
        '이들은 과거 TCN 평가에는 사용된 이력이 있어 프로젝트 전체에서 처음 본 자료라고 부르지 않는다. 사람의 실제 동일인 여부를 외형으로 판정하지 않았다.\n\n'
        '이전 → 현재(주요 이동을 충분히 관측했다고 확인한 횟수)\n'
        '원본26/30 →27/30\n조금 느린 영상21/30 →24/30\n조금 빠른 영상28/30 →29/30\n조금 작고 옆으로 보이는 영상29/30 →29/30\n합계104/120 →109/120\n'
        '30은8영상의10이동항목×3seed. 120은이4변형을 포함하며 독립120영상이 아니다. 보류된 나머지를 수어 오답이라고 판정하지 않는다.\n\n'
        '반대 방향 또는 정지 여부까지 정답이 있는18물리적 구간을 직접 지정했을 때에는5방향 균형정확도가96.67%→94.58%였다. '
        '정상 동작의 불필요한 보류는 줄었지만 기본 방향구분이 모든 조건에서 좋아진 것은 아니다. 주석은성긴시점의Codex시각검토이며 정밀경계/언어정오 정답이 아니다.\n'
        '처리시간:32입력의 전체기존분석5.05~10.38초, 중앙7.87초. 현재움직임판정부 중앙0.00294초 추가. 모델준비·업로드·화면표시까지 포함한 새 데모시간은 별도검증 대상.\n'
        '속도변형은같은영상을재인코딩한통제시험이다. 실제사람이빨리움직일때발생하는모션블러까지재현하지않는다.\n\n'
        '남은 핵심:짧은관절누락에대한이미지/시간판단,주요순서차이,새모델실제데모검증. 정확한접촉·엄지·손바닥방향·필수반복횟수·모든수어의정오판정은아직지원하지않는다.\n\n다음: '+next_step+'\n')
    (H/'RESULT_KO.txt').write_text(text,encoding='utf8')
    now=clock(); cur=dict(status=g['status'],at=now,summary=summary,next=next_step,active_exec_session=None,prepare57188_exit=0,evaluate70082_exit=0,goal_status='active',no_training=True)
    write(H/'CURRENT_EXECUTION.json',cur);s=read(O/'STATE.json');s.update(current_stage='LOCKED_EVALUATION_DONE_INTEGRATION_NEXT',next_single_action=next_step,next_single_task=next_step,updated_date_kst='2026-11-02');s[H.name]=cur;write(O/'STATE.json',s)
    root=O.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf8')
    for h,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','다른 공개기록ID 고정비교109/120·통합검증으로',summary),('## 9. 다음 단일 작업','이미지·관절누락·순서와 새데모 통합검증',next_step)]:t=t.replace(h,h+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
    p.write_text(t,encoding='utf8');(root/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_step+'\n\n현재:'+str(H)+'\n진행 중 실행 없음. 완료 추출/학습/평가 중복 실행 금지.\n',encoding='utf8')
    docs=root/'Docs';p=docs/'FEEDBACK_GOAL_AND_PLAN_KO.txt';t=p.read_text(encoding='utf8')
    lo=t.find('최신 실행 현황');hi=t.find('1. 사용자가 확정한 목표',lo)
    assert lo>=0 and hi>lo;t=t[:lo]+'최신 실행 현황 ('+now+')\n'+summary+'\n다음: '+next_step+'\n\n'+t[hi:];p.write_text(t,encoding='utf8')
    exclusive(H/'FINAL_ARTIFACTS.json',dict(at=now,status=g['status'],hashes={str(p):sha(p) for p in H.glob('*.json') if p.name!='CURRENT_EXECUTION.json'},
        goal_complete=False,training_on_locked_data=False,model_or_seed_selected_on_locked_results=False,verifier='Same Codex separate arithmetic; not independent human truth'))
    print('Finalized locked evaluation with mixed metrics and continuing full goal')

if __name__=='__main__':main()
