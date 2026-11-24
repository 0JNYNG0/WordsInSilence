from common import *
from datetime import datetime, timezone, timedelta
import shutil
repo=ROOT.parents[3]
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
status=sys.argv[1] if len(sys.argv)>1 else 'APPROVED_SIGNREP_CPU_INTEL_GPU_FEASIBILITY_RUNNING'
summary=(ROOT/'STATUS_KO.txt').read_text(encoding='utf-8') if (ROOT/'STATUS_KO.txt').exists() else '사용자가 SignRep 확보·실행시간 선검증·조건부 공개 한국수어 적응학습을 승인. 공식 checkpoint SHA검증과 55,644,329개 파라미터 strict로드 완료. 공개 개발3영상 원본368프레임/164창을 준비하고 공식 출력/특징전용 경로9창 동일값 확인. 전용 환경에 OpenVINO를 설치했으며 Arc140V GPU 인식. 현재 GPU 변환·CPU/GPU 전체시간 비교 중; 한국수어 추가학습·성능개선·운영적용 아직 없음.'
next_action=(ROOT/'NEXT_KO.txt').read_text(encoding='utf-8') if (ROOT/'NEXT_KO.txt').exists() else '진행 실행·EVENTS 확인 후 OpenVINO 변환과 동일 공개3영상의 CPU/GPU 시간·수치 검증을 완료한다. 결과 수치와 입력/정답 준비성을 근거로 공개 KSL 학습 진입 여부 판단. C/T3F·웹·DTW와 사용자/멘토자료 보존. 승인한 다운로드/실행에 재승인 질문하지 않는다.'
paths=[repo/'WordsInSilence_PROJECT_CONTEXT.md',repo/'NEW_CHAT_HANDOFF.md',ROOT.parent/'STATE.json']
before=ROOT/'before';before.mkdir(exist_ok=True)
for p in paths:
    dst=before/p.name
    if not dst.exists():shutil.copy2(p,dst)
ctx=paths[0].read_text(encoding='utf-8-sig')
title='### 2026-10-29 SignRep 승인 실행 — CPU·Intel GPU 검증'
for heading,text in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
    start=ctx.index(heading)+len(heading)
    tail=ctx[start:]
    if tail.lstrip().startswith(title):
        old_start=start+tail.index(title)
        old_end=ctx.find('\n### ',old_start+len(title))
        if old_end<0:old_end=len(ctx)
        ctx=ctx[:start]+'\n\n'+title+'\n'+stamp+'\n'+text+'\n근거: '+str(ROOT.relative_to(repo)).replace('\\','/')+'/AUTHORIZATION.json, FROZEN.json 및 실행 결과.\n\n'+ctx[old_end:]
    else:
        ctx=ctx[:start]+'\n\n'+title+'\n'+stamp+'\n'+text+'\n근거: '+str(ROOT.relative_to(repo)).replace('\\','/')+'/AUTHORIZATION.json, FROZEN.json 및 실행 결과.\n'+ctx[start:]
paths[0].write_text(ctx,encoding='utf-8')
state=read(paths[2]);state['current_stage']=status;state['updated_date_kst']=stamp;state['next_single_action']=next_action
state[ROOT.name]=dict(status=status,at=stamp,unit=str(ROOT),summary=summary,next_single_action=next_action,operator='Codex local implementation and separately scripted verification; not independent human ground truth',private_data_access=0,production_changes=0)
paths[2].write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

{stamp}

목표: 한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 배워, 사용자가 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고, 중요한 차이와 허용 가능한 차이를 구분해 알려준다.

현재 단위: {ROOT.relative_to(repo)}
현재 상태: {status}

{summary}

다음 단일 작업: {next_action}

최신 승인: SignRep 공식 모델·코드 확보, 격리 실행환경 설치, Intel GPU 사용 가능성과 속도/정확성 검증. 조건 통과시 기존 공개 KSL 적응학습. GPU가 정확도를 자동으로 높이지 않는다. 기존 연구의 다운로드 승인대기는 최신 승인으로 해제됐다. 다른 모델·데이터 확대, 드라이버 교체, 운영 배포 승인 아님.

실행은 CURRENT_EXECUTION.json과 EVENTS.jsonl, 실제 exec 세션을 먼저 확인한다. 완료 산출물 재실행/덮어쓰기 금지. 최초 DNS 실패 등 오류 보존. inputs/원본3영상368프레임·164창(끝1프레임 누락 막는 마지막창2개 추가). 공식16프레임/stride2와224 fullRGB 사용, 원30fps 유지. pooled768은 의미 단계 정답이 아니다. 12영상은 이미 살펴본 개발자료이며 새 독립평가 아님. 기존 AI합의 주석은 사람 전문가 정답이 아니다.

노트북: Samsung960XHA, CoreUltra7 258V/RAM32GB/IntelArc140V 공유메모리/NPU. 현재torch CPU 재사용, 전용OpenVINO로GPU확인. 가용C약7GB로 시작했고 기존파일삭제 없음. 전역패키지·드라이버 유지.

보호: C·T3F/운영웹·DTW 그대로. 사용자·멘토자료 접근/학습/튜닝/선택 금지, W:/sources/과거동결결과 읽기전용. 로컬영상 외부전송/카메라/Git 없음. 담당 Codex, 별도 검산은 독립 사람정답과 다르다.
'''
paths[1].write_text(handoff,encoding='utf-8')
print(status)
