from pathlib import Path
import json,hashlib,datetime,shutil
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def rd(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def wr(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
result=rd(U/'FINAL_RESULT.json');assert result['verification']['passed']
title='2026-11-11 설명 요구·실제 행동 직접 연결 학습 완료 — 순서부 개선, 자동 구간 연결 잔존'
summary='공개 FIT54영상/10표현의 기존 RGB 정답479항목(손모양268·방향136·순서75쌍)을 연결표/3004대조입력으로 준비. 직접 설명에 연결된 손모양·방향은97항목. 100단어899 공통부 동결. 작은 요구비교부3회 학습은 방향32→31/32/31로 미채택. FIT의118순서 입력이 기존 점수 잔차제한 때문에 수정불가능함을 확인, 앞뒤 관찰·설명 요구를 직접 짝짓는 공통순서부3회 학습. 지정구간 역순12쌍은 기존9정답(확실8)→11확실정답+1보류, 정상12·속도24 유지. 반대설명 질문12는0→12. 자동구간 역순9쌍은 여전히5거짓지지/4연결보류로 웹 미채택. 손모양227/228 유지, 원래 모양/방향·C·T3-F·웹8814 불변. 짧음/유지/반대방향/관측부족 분리. 7920저장검산PASS(동일Codex 별도코드), 개인자료0, 역사적DEV·합성시험이며 실제학습자 정오검증 아님.'
next_='설명 순서를 강제로 따르지 않는 후보-설명 대응 정답/연결부 보완. FIT의 준비·복귀·서로비슷한다른행동·미관측을 구분하고, 연결 후 실제 시각 순서를 검사. 좋은구간 순서부 능력은 보존하되 자동구간 역순9의5거짓지지를 우선 줄인다. 상세 NEXT_PLAN_KO.txt. 후속 미착수, 완료학습/평가반복금지.'
paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt',ROOT/'Docs/RESEARCH_REPORT_INDEX_KO.txt']
backup=U/'before';backup.mkdir(exist_ok=True)
for p in paths:
 b=backup/p.name
 if not b.exists():shutil.copy2(p,b)
p=paths[0];t=p.read_text(encoding='utf-8-sig')
for header,text in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_)]:
 assert header in t
 entry=f'\n\n### {title}\n\n{text}\n근거 {U / "RESULT_KO.txt"}\n계획 {U / "NEXT_PLAN_KO.txt"}\n'
 if f'### {title}' not in t[t.index(header):t.index(header)+len(entry)+100]:t=t.replace(header,header+entry,1)
p.write_text(t,encoding='utf-8')
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
s=rd(O/'STATE.json');s['current_stage']=U.name;s['next_single_action']=next_;s['last_updated']=stamp;s['updated_at']=stamp
s[U.name]=dict(status=result['status'],summary=summary,result=str(U/'FINAL_RESULT.json'),report=str(U/'RESULT_KO.txt'),next_plan=str(U/'NEXT_PLAN_KO.txt'),verification=str(U/'VERIFICATION.json'),training_running=False,evaluation_running=False,web_changed=False,private_used=False)
wr(O/'STATE.json',s)
handoff=f'''# 최신 인수인계

{title}

{summary}

현재 {U}
RESULT_KO.txt / FINAL_RESULT.json / VERIFICATION.json / NEXT_PLAN_KO.txt.
다음 {next_}

실행: 학습·평가 종료, PID 없음. 첫 요구비교부3개와 보정 순서부3개 완료; 재학습/재추론하지 않는다.
새 순서부 OrderedBinding은 관찰앞뒤와 구조화된 설명앞뒤의 대응을 학습. 단어ID예외없음. 지정구간에서 개선됐으나 자동 연결은 아직 실패. 현재 웹8814는 그대로이며 이번 서버/카메라 실행확인은 하지 않았다.
주요 오류/정정: 첫 비교에서 낮은 원하는방향 확률을 오방향으로 해석했던 consumer 오류는 SEMANTIC_CORRECTION.json에 기록, 첫 출력 보존. 유지구간은 반대방향이 아니며 관측부족은 오답이 아님. 첫 잔차후보 학습과 순서부 FIT 원인수정은 구분해 보고할 것.
공개54영상은100단어전체영상이 아님. 공통100단어899학습부는 고정, 이번세부지도10표현/웹검증4단어. 실제사람오류정답/전문가허용범위는 없음. 이집트 끝손모양2요구의 직접FIT연결 근거부족.
사용자·멘토 자료 접근/학습/튜닝/선택금지. 원본/기존동결읽기전용. C·T3-F/웹기본 불변. 이번검산은 동일Codex의다른코드이지 독립사람정답검증이아님.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with history.open('a',encoding='utf-8') as f:f.write(f'\n\n{title}\n{summary}\n연구 판단 변화: 물리이동의 완전함만 평가하는 구조에서, 설명의 핵심요구와 보이는행동을 직접 연결하는 구조로 이동. 단순전체/부분 라벨이 수어정오 라벨이 아님을 명시. 이번에는 지정구간에서 배우는 순서판별 능력과 자동구간의 실패를 분리하여 다음 병목을 구체화했다. 완성된웹피드백개선으로 과장하지않음.\n근거 {U / "RESULT_KO.txt"}\n')
index=ROOT/'Docs/RESEARCH_REPORT_INDEX_KO.txt'
with index.open('a',encoding='utf-8') as f:f.write(f'\n{U.name} | {title}\n{U / "RESULT_KO.txt"}\nSHA256 {hashlib.sha256((U/"RESULT_KO.txt").read_bytes()).hexdigest()}\n')
checks=dict(state_current=rd(O/'STATE.json')['current_stage']==U.name,context_both=paths[0].read_text(encoding='utf-8').count('### '+title)==2,handoff=U.name in paths[1].read_text(encoding='utf-8'),history=U.name in history.read_text(encoding='utf-8'),reports_exist=all((U/n).exists() for n in ['RESULT_KO.txt','NEXT_PLAN_KO.txt','FINAL_RESULT.json','VERIFICATION.json']),no_running=not rd(U/'CURRENT_EXECUTION.json')['training_running'])
assert all(checks.values());wr(U/'DOCUMENT_CHECK.json',dict(passed=True,checks=checks,updated=stamp))
print(checks)
