from pathlib import Path
from rgb_model import R,O,read
from prepare_real_fit import write,clock

def main():
 root=O.parents[2];now=clock();summary=('이전단위의정상속도56/60개선과원래DEV39/45회귀를보존한채, 기존SignRep768차원영상특징을움직임신경망에직접연결하는후속비교를시작했다. '
  '초기출력일치·영상없음차단·패딩·방향/연속성분리기울기·공개FIT144파일해시검증통과. '
  'no_rgb_replay/rgb_residual 각3seed·120epoch의6미세조정 exec86124실행중이며 새DEV결과는아직없다. '
  '새SignRep본체학습/다운로드/새영상추출없음. 기존C·T3F/8804~8806·손모양/관계/학습DTW보존. 사용자·멘토자료접근0.')
 next_step=('exec86124와 TRAIN_PROGRESS/TRAINING_RESULT를확인해중복학습하지않는다. 종료확인후evaluate_rgb.py→verify_rgb.py로고정288비교와별도검산. '
  '主후보rgb_residual/seed42는이미고정. 원래40/45이상·정상변형54/60이상·주요팔이동삭제충분확인0·검은구간미통과·기존출력보존을모두충족해야잠금ID14/15평가/별도연구데모로간다. '
  '새DEV로문턱/seed선택금지. completed이전추출/학습/평가재실행금지. 전체목표active.')
 current=dict(status='DIRECT_RGB_MOTION_SIX_FINETUNES_RUNNING',at=now,summary=summary,next=next_step,active_exec_session=86124,
  previous_goal_turn='progress:108 real public inputs,12 actual finetunes,576 preserved-output comparisons,normal gain47to56of60,remaining original regression documented',
  goal_status='active',new_backbone_update=False,private_access=False)
 write(R/'CURRENT_EXECUTION.json',current);s=read(O/'STATE.json');s.update(current_stage=current['status'],next_single_action=next_step,next_single_task=next_step,updated_date_kst='2026-11-02');s[R.name]=current;write(O/'STATE.json',s)
 p=root/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf8')
 for h,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','영상특징 직접연결6학습 진행',summary),('## 9. 다음 단일 작업','RGB미세조정 종료확인·고정비교',next_step)]:text=text.replace(h,h+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
 p.write_text(text,encoding='utf8')
 (root/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_step+'\n\n현재단위: '+str(R)+'\n실제실행 exec86124. 기존PID/이전요약만보고재시작하지않는다. 독립사람정답아닌Codex시각주석/별도산술검증.\n',encoding='utf8')
 print('Recorded live exec86124 and frozen evaluation conditions')

if __name__=='__main__':main()
