from common import *
import argparse

def sync(title,body,next_title,next_body,stage,details):
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf-8-sig')
 for heading,t,b in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',title,body),('## 9. 다음 단일 작업',next_title,next_body)]:
  assert text.count(heading)==1
  text=text.replace(heading,heading+'\n\n### '+t+'\n'+b+'\n',1)
 p.write_text(text,encoding='utf-8')
 s=read(O/'STATE.json');s['current_stage']=stage;s['next_single_action']=next_body;s['updated_date_kst']=now()
 s['word100_hand_evidence_ablation_20261018']={'at':now(),**details}
 write(O/'STATE.json',s)
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n갱신 '+now()+'\n\n### '+title+'\n'+body+'\n\n### '+next_title+'\n'+next_body+'\n\n기존 C/M0/DTW/웹 유지. 사용자·멘토·보류14~18 읽기/학습/튜닝 금지. 원본·과거동결결과 읽기전용. 자동화 st-gcn 삭제 상태 유지.\n',encoding='utf-8')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--exec',type=int,required=True);args=ap.parse_args()
 assert (U/'before').exists()
 ctx=U/'TRAIN_CONTEXT.json';prepared=ctx.exists()
 write(U/'CURRENT_EXECUTION.json',{'at':now(),'exec':args.exec,'stage':'TRAINING' if prepared else 'PREPARING','context_sha256':sha(ctx) if prepared else None})
 sync('2026-10-18 네 추가 정보 묶음 제외 12회 비교 사용자 승인·진행',
      f'ops/{U.name}. 사용자 직접 승인으로 위치/영역경계/현재교차/최근근접 한 묶음만 제외하는 4조건×3seed 실행. 기존899/DEV300·310구조·원초기 상태·C시간/보기일정·80epoch/patience12 고정. '+('1199입력·초기출력·제외채널기울기 검산 PASS. ' if prepared else '입력 불변/배치제외 검증 중. ')+f'exec{args.exec}. 기존 C/C0/C1은 저장점수 재사용. 사용자자료/운영변경 없음. 아직 성능 판정 전.',
      '2026-10-18 승인된 12회 완료 후 독립 검산·원인 정리',
      f'{U.name}/CURRENT_EXECUTION·STATUS·TRAIN_RUNNING과 exec{args.exec}를 먼저 확인하여 중복 실행 금지. 준비 통과 후 train.py의 12회 완료까지 진행하고 verify.py로 저장점수·배치/선택/체크포인트 검산. 결과에 맞춘 조건변경·추가학습·웹교체 금지. 모든조건의 평균/범위/회귀/한계 보고 후 대기.',
      'WORD100_HAND_EVIDENCE_ABLATION_RUNNING',
      {'status':'TRAINING' if prepared else 'PREPARING','exec':args.exec,'training_runs_planned':12,'context':str(ctx),'running':True,'user_mentor_heldout_used':False,'production_changed':False})
 print('Current approval and execution synchronized.')
