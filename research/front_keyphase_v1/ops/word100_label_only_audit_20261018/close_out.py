"""Synchronize completed evidence; no training/production changes."""
import json, hashlib, re
from pathlib import Path
from datetime import datetime, timezone, timedelta
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(U/'SUMMARY.json');v=read(U/'INDEPENDENT.json')
assert v['status']=='PASS' and s['counts']==v['counts']
at=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
status=('ops/word100_label_only_audit_20261018. 이전 단위 guard 이후 남은 라벨 단독 배정456건을 공개 TRAIN353영상/100단어에서 전수 원본 검토했다. '
        '정상441(96.71%)·명확한 오류13(2.85%)·판정 보류2(0.44%). 단위는 프레임별 손 후보이며 전체 검출/수어 인식률이 아니다. '
        '오류는 첫 등장 반대 배정7, 기존 추적에서 반대 손으로 넘어감2, 같은 실제 손의 중복 가설4. '
        '38시트/456장면 확인, 의심23건 앞뒤 장면 추가 검토. 정상 교차 사례 보존 필요, 정상 좌우가 손가락21점 정확성 인증은 아님. '
        '별도 검산: TRAIN899 원추적의457건−기존보류1=456 일치, 집계·근거·원본353해시·보호12602핀 PASS. '
        'Codex 단일 시각검토와 별도 계산 검산이며 독립 사람 정답 검증은 아님. ID36 초벌 L 해독을 실제 저장 R로 정정했고 초벌 기록 보존. '
        '관측/좌우확정 분리·추적 근거출처·손전체 중복/정상교차 구분을 다음 공통 보완안으로 제안했다. '
        '새 배정규칙·학습·추론0, 기존 C/TCN/DTW/웹 유지, 사용자/멘토/DEV/보류 자료 사용0.')
next_action=('word100_label_only_audit_20261018/RESULT_KO.html·RESULT_KO.txt와 NEXT_PLAN.json을 사용자와 확인한다. '
             '후속 제안은 공개 자료에서 좌우 임시/확정 상태와 추적 출처를 보존하고, 실제 중복과 정상 교차를 함께 검증하는 격리 보완 후보이다. '
             '라벨 단독456건 일괄 제거·화면 좌우 반으로 배정·특정 단어 예외 금지. '
             '현재456진단 및 이전 정상교차315 회귀검사와 TRAIN899 전수 재생/변경장면 원본 검토 후 결과를 보고한다. '
             '새 규칙 구현/학습은 이번 감사에 포함되지 않았으며 학습은 사용자 결과 확인 후 별도 조건 동결·승인 범위에서 진행. 기존 C 유지, 사용자 자료 학습/튜닝/선택 금지.')
for name in ('WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md'):
 # At this point no concurrent root edits are expected; preserve any newly arrived history by insertion.
 assert (R/name).exists()
ctx=R/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text('utf-8-sig')
heading8='### 2026-10-18 라벨 단독 456건 전수 원본 감사 완료'
heading9='### 2026-10-18 라벨 단독 감사 결과 검토·공통 보완안 확인'
assert heading8 not in text and heading9 not in text
text=re.sub(r'(^## 8\.[^\n]*\n)',lambda m:m.group(1)+'\n'+heading8+'\n'+status+'\n\n',text,count=1,flags=re.M)
text=re.sub(r'(^## 9\.[^\n]*\n)',lambda m:m.group(1)+'\n'+heading9+'\n'+next_action+'\n\n',text,count=1,flags=re.M)
assert heading8 in text and heading9 in text
ctx.write_text(text,encoding='utf-8')
state_path=O/'STATE.json';state=read(state_path)
state['updated_date_kst']=at;state['current_stage']='WORD100_LABEL_ONLY_AUDIT_COMPLETE_AWAITING_REVIEW';state['next_single_action']=next_action
state['word100_label_only_audit_20261018']={**s,'at':at,'result':str(U/'RESULT_KO.html'),'independent':v['status'],'protected_hashes':v['protected_hashes'],'review_operator':'Codex visual; separate arithmetic/source/hash verification by same operator','next_plan':str(U/'NEXT_PLAN.json'),'training_authorized_now':False}
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {at}

### 현재 목표·마지막 완료
{status}

### 다음 단일 작업
{next_action}

### 근거·실행 상태
- 단위: research/front_keyphase_v1/ops/word100_label_only_audit_20261018
- CONTEXT/EVENTS/VISUAL_REVIEW/SUMMARY/INDEPENDENT/RESULT_KO/NEXT_PLAN에 대상·판정·계산·보완안 기록.
- 구현 및 시각검토: Codex 로컬. 감사 도구만 추가. 기존 assigner/모델/웹 변경 없음.
- 원본 렌더 exec98721, 상세장면 exec46018, 별도검산 exec5767 모두 종료0. 현재 단위 실행·잠금 없음. st-gcn 자동화 삭제 유지.
- 판정 보류 #293·295는 양손 겹침에서 혼합된 관절 가설로 확정하지 않음. 가림·손가락 형상·후속 전파의 전체 오류율 미측정.
- 원본 W:/sources/과거 동결 결과/C 읽기 전용. 사용자·멘토·DEV·보류 자료 미사용. 학습은 결과 함께 확인 후.
'''
(R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
assert read(state_path)['word100_label_only_audit_20261018']['counts']==s['counts']
assert '441' in ctx.read_text('utf-8') and '441' in (R/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
report=(U/'RESULT_KO.html').read_text('utf-8')
for target in re.findall(r'(?:href|src)="([^"]+)"',report):
 if not target.startswith(('http:','https:','#')):assert (U/target).exists(),target
evidence={str(p):sha(p) for p in U.iterdir() if p.is_file() and p.suffix in ('.json','.py','.txt','.html')}
final={'status':'PASS','at':at,'independent_reused':str(U/'INDEPENDENT.json'),'protected_pin_check':'12602/12602 PASS before documentation-only synchronization','audit_artifact_hashes':evidence,'synchronized':{str(p):sha(p) for p in (ctx,state_path,R/'NEW_CHAT_HANDOFF.md')},'training':0,'production_changed':False,'remaining':'User review of results/proposal; no training launched.'}
with (U/'FINAL_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(final,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'PASS','at':at,'counts':s['counts'],'docs_synced':3},ensure_ascii=False))
