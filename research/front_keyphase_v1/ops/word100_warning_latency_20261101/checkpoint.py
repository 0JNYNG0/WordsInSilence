from pathlib import Path
import json,datetime,hashlib
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
summary='정상 공개12영상39구간 중 차이경고7 원인 RGB감사완료: 손모양전환1, 2D주먹방향불안정2, 팔높이/벌림4. 실제 수어오류정답7개 아님. 새독립단위에서 중복decode4→3 및 입력준비/GPU연산겹침 후보검증중; 모든frame/window/모델/얼굴유지. 원경고를지우지않고근거종류/지속시간을추가. 공개2편 ABBA각2회 새속도시험진행, 기존결과재실행아님. 기존8802서버 PID26000·C/T3F·운영·사용자자료보존.'
next_action='benchmark.py exec81798 및 benchmark.log를 확인하여 중복실행하지 않는다. BENCHMARK완료후 encoder_gate.py를한번만실행하여동일ROI/PTS/입력tensor및전체출력bitexact검증. agent semantic_feedback detail_enrichment96저장검증, semantic_data_audit 별도리뷰진행. 비교결과확인후분리연구데모적용/최종보고·문서동기화. 사용자자료접근/튜닝/모델교체금지.'
pins={str(U/n):hashlib.sha256((U/n).read_bytes()).hexdigest() for n in ['benchmark.py','candidate_adapter.py','encoder_gate.py']}
with (U/'EXECUTION_CHECKPOINT.json').open('x',encoding='utf8') as f:json.dump(dict(updated=now,summary=summary,next_action=next_action,pins=pins),f,ensure_ascii=False,indent=2)
p=O/'STATE.json';d=json.loads(p.read_text(encoding='utf-8-sig'));d['current_stage']='WARNING_LATENCY_VALIDATION_IN_PROGRESS';d['next_single_action']=next_action;d[U.name]=dict(status=d['current_stage'],summary=summary,next_action=next_action,updated_at=now,unit=str(U));p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n'+next_action+'\n\n현재단위 '+U.name+' PLAN_KO/FROZEN/VALIDATION_ADDENDUM 참조. 이전완료단위 word100_multipeople_live_feedback_20261101 read-only. root만GPU실측. 별도Codex검산은독립사람정답아님.\n',encoding='utf8')
p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig');t=t.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n### 2026-11-01 경고 원인·무손실 속도 보완 검증 진행\n'+summary,1);t=t.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n### 2026-11-01 다음 단일 작업 — 경고/속도 후보 검증 마무리\n'+next_action,1);p.write_text(t,encoding='utf8')
print('checkpoint')
