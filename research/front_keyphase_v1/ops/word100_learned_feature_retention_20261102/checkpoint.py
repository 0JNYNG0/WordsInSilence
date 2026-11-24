from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def main():
 now=datetime.now(timezone(timedelta(hours=9))).isoformat()
 summary=('직전 목표 턴은6학습·8805새입력 데모·정확한 병렬화 검증까지 실제진행. 이번 저장진단에서 새모델이 기존정답4~6항목을 회차별로 놓친 것을 확인했다. '
  '기존 목표단어 비의존 신경망 관측분기를 보존하고 새 학습된 단계/기준흐름 분기와 연결했다. 같은 RGB 특징1회, 개별참고0회. '
  '저장DEV36 clip-seed 비교에서 기존특징출력과 새DTW경로 보존. 성능향상 아닌 기존특징퇴행방지다. '
  '이제 실제 공개ID12영상에 속도·초기대기자르기·관찰동작자르기·검은화면 변형24개를 만들고 새 추출 검증 중이다. '
  '특징배열만 잘라 주변16프레임 RGB문맥에 정보가 남는 문제를 실제영상 편집 후 재추출로 분리한다.')
 next_action=('word100_learned_feature_retention_20261102의 run_video_controls.py exec21224를 먼저 확인. CONTROL_PROGRESS/VIDEO_CONTROL_RESULTS와 실제세션을 함께 확인하고 중복실행하지 않는다. '
  '완료 후 정상속도/손내림/관측부족/행동삭제 결과를 분리검산한다. 삭제는물리적편집이며 모든항목의수어오답정답이아니다. '
  '남는실패가있으면 FIT에서만 감독신호를준비해 기존특징을보존하는 분리신경망의 단계/차이판정부를 학습. '
  'DEV점수로문턱반복조정/seed선택금지. 기존C·T3F/8804/8805/사용자·멘토자료보존. 목표active유지.')
 rec=dict(status='RETAINED_OBSERVER_REAL_VIDEO_CONTROLS_RUNNING',unit=str(U),at=now,summary=summary,next_action=next_action,
  active_exec_session=21224,goal_status='active',new_training=0,user_mentor_access=False,previous_goal_turn='progress')
 (U/'CURRENT_EXECUTION.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf8')
 p=O/'STATE.json';d=json.loads(p.read_text('utf8'));d['current_stage']=rec['status'];d['next_single_action']=next_action;d[U.name]=rec;p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
 p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig')
 for h,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','기존손모양 보존 분기·실제영상 편집 대조 실행',summary),('## 9. 다음 단일 작업','실제 공개영상 변형24개 진행 확인',next_action)]:
  assert h in t;t=t.replace(h,h+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
 p.write_text(t,encoding='utf8')
 (R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_action+'\n\n단위: '+str(U)+'\n구현/검증 Codex. 기존8805는원래후보그대로. 새분리후보아직웹적용아님.\n',encoding='utf8')
 print(rec['status'])
if __name__=='__main__':main()
