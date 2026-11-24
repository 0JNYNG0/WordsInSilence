from core import *
ROOT=O.parents[2]

def sync(status,summary,next_action):
 title='2026-11-09 공개100단어 공통 학습과 비교 재생 보완'
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8')
 for name in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md']:
  src=ROOT/name;dst=U/'before'/name
  if not dst.exists():dst.write_bytes(src.read_bytes())
 for section,text in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
  start=s.index(section)+len(section);marker='\n\n### '+title+'\n\n'
  if s[start:].startswith(marker):
   end=s.find('\n### ',start+len(marker));end=len(s) if end<0 else end;s=s[:start]+s[end:]
  s=s[:start]+marker+text+'\n'+s[start:]
 p.write_text(s,encoding='utf-8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n### '+title+'\n\n'+summary+'\n\n다음 단일 작업: '+next_action+'\n\n상세 실행·재개: '+str(U/'CURRENT_EXECUTION.json')+' 및 EVENTS.jsonl. 기존 완료 실행을 중복하지 않는다. 보호: C·T3F/현재 TCN/모델 유지, 사용자·멘토 자료 접근·학습·튜닝·선택 금지, sources/원본 읽기 전용, 새 다운로드·Git·자동 카메라 없음.\n',encoding='utf-8')
 p=O/'STATE.json';d=read(p);d={U.name:dict(status=status,report=str(U/'RESULT_KO.txt'),execution=str(U/'CURRENT_EXECUTION.json'),summary=summary,next=next_action),**{k:v for k,v in d.items() if k!=U.name}};d['current_stage']=U.name;d['next_single_action']=next_action;write(p,d)

if __name__=='__main__':
 sync('TRAINING_FIXED_COMPARISON_IN_PROGRESS','사용자 승인으로 기존 RTMW 공개100단어899FIT/300DEV 입력을 재사용하여 실제 Conv+GRU 추가학습 중이다. 8추가표현의 공개RGB72시점에서 명확한 공통 손모양166항목을 보강했다. 동일 추가정답만 학습 / 여기에100단어 보조학습을 더함의2조건×3seed를 사전고정했다. 기록ID01~09와11~13 및 원본SHA를 분리하며 실제 인물 동일성은 외형으로 추정하지 않는다. 새 평가점수 아직 미확인. 8814 비교재생은 구8812와 달리 두 영상의 독립재생+반복seek가 겹쳐 있었으며 사용자영상만 재생하고 참고는 정렬위치를 따라가도록 수정했다. 로직8/공개실제영상 브라우저8검사 통과; 모델/정렬계산/판정은 변경하지 않았다. 추가공개24영상72시점의 평가정답133항목/미확정83은 새모델 출력을 보기 전에 별도기록했다. 기존모델·사용자자료·운영교체 없음. 구현/시각검토는Codex, 전문가 독립정답 아님.','exec56042 훈련실행상태와 CURRENT_EXECUTION을 확인하고 중복 없이6학습을 마친 뒤 evaluate.py 고정비교1회, 저장예측별도검산, RESULT_KO·문서동기화. 완성된 run은 재학습하지 않는다. 공개추가꼬리 시계열진단 완료: 주요구간매칭 변화0/12지만 끝부분정렬 영향은 존재. 검증된경계없이뒤를임의제거하지 않는다.')
