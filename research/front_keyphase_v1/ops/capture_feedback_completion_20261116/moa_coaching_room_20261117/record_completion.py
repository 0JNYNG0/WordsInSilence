from pathlib import Path
import json, shutil
ROOT=Path.cwd();O=ROOT/'research/front_keyphase_v1/ops';D=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    deployed=read(D/'web/DEPLOYMENT_CHECK.json');assert deployed['passed']
    now=deployed['at'];pid=deployed['server']['pid']
    backup=D/'before_applied_docs';backup.mkdir(exist_ok=True)
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',D/'CHECKPOINT.json']:
        if not (backup/p.name).exists():shutil.copy2(p,backup/p.name)
    next_action='새 8821 화면에서 사용자가 가능할 때 사과·감사 촬영의 자동 종료와 결과·교정 재시도를 확인한다. 이미 확인한 공개 검사는 반복하지 않고, 새 문제가 확인되면 공개 정면자료로 재현한 뒤 보완한다. 사용자촬영은 진단만, 학습·튜닝·후보선택 금지. 실제 시연 확인 전 전체 완료로 보고하지 않는다.'
    summary='사용자가 MOA_BOTTOM_UP_PREVIEW.gif를 명시적으로 선택해 구름·연기 질감 없는 아래→위 페이드와 MOA 위 퇴장·오른쪽 재등장을 기존8821에 적용. 전환 중 대화·메뉴 숨김, 좌우45도 시범·중앙 하단 실시간 웹캠·단일촬영/Space·별도 결과창 적용. 브라우저14검사 및 실제HTTP 녹화보존/권한13검사 통과. 공개 검증된 관측 역할 분리 보완도 적용:88조건 중 속도변형4개 개선, 정상32·직접149프레임143정답·부분16 유지, 공개영상 캡처4 자동종료. 새 학습0·개인자료접근0·이전122핀 및 C·T3-F 유지. 실제 웹캠 확인 전이며 전체 프로젝트 완료는 아님.'
    c=read(D/'CHECKPOINT.json');c.update(at=now,status='MOA_SMOOTH_UI_AND_PUBLIC_ENDPOINT_ROLE_WEB_APPLIED',production_ui_changed=True,production_backend_changed=True,active_entrypoint=str(D/'serve.py'),runtime_pid=pid,preview_server=None,next_single_action=next_action,
        pending=['Actual seated webcam automatic-stop and correction/retry/quest verification when user is available','Final demo/packaging verification; not represented as completed'])
    selected=read(D/'BOTTOM_UP_REVIEW.json')
    c.update(storyboard=selected['storyboard'],animation=selected['animation'],ui_review=dict(status='USER_SELECTED_BOTTOM_UP_APPLIED_NO_SMOKE_TEXTURE',user_request='그냥 이 느낌으로 해줘. 구름연기 질감도 넣지 마 그냥.',reference=selected['animation'],files=selected['files']),deployment=str(D/'web/DEPLOYMENT_CHECK.json'),admission=str(D/'web/ADMISSION.json'))
    write(D/'CHECKPOINT.json',c)
    graph=dict(goal='정면 웹캠 사과·감사 시범 → 촬영 → 자동 종료 → 교정/통과/관측 부족 → 재시도·퀘스트',status=c['status'],previous=str(D.parent/'rest_stability_endpoint_20261117/GRAPH.json'),nodes=[
        dict(id='R1',label='자동 종료 흔들림 보완',status='public_verified_web_applied_actual_recheck_pending',evidence=str(D.parent/'rest_stability_endpoint_20261117/checks/STOP_CHECK.json')),
        dict(id='R2',label='반복 동작의 끝 확인',status='public_candidate_verified_web_applied',evidence=str(D/'checks/ROLE_CHECK.json')),
        dict(id='R3',label='정상·오류·속도 및 웹 연결',status='public_checks_passed',evidence=str(D/'web/HTTP_INTEGRATION_CHECK.json')),
        dict(id='UI',label='MOA 비행·매끈한 페이드·연습 화면',status='user_approved_browser_verified_web_applied',evidence=str(D/'qa/BROWSER_CHECK.json')),
        dict(id='R4',label='실제 촬영·교정 재시도·퀘스트',status='user_check_pending_not_complete')],edges=[['R1','R3'],['R2','R3'],['R3','R4'],['UI','R4']],next_single_action=next_action)
    write(D/'GRAPH.json',graph)
    report=summary+f'\n실행: http://127.0.0.1:8821/ ; PID {pid}; {D / "serve.py"}. 8822 종료.\n'
    report+='선택한 GIF의 scene.js/hologram.js 해시와 실제 제공 파일이 일치한다. 기존 START_WEB_DEMO.cmd 실행 경로도 최신 서버로 연결.\n'
    report+='끝 상태 보완은 이미 학습된 신체 상대 위치 관측과 손모양 상태의 역할을 분리한 코드 연결이다. 새 가중치 학습은 아니다. 움직임 근거와 연속 관측이 있어야 도달로 인정하며 손모양 오류를 성공으로 바꾸지 않는다. 해당 과제 공개자료에서 확인한 결과이며 모든 단어·새사용자 성능을 보장하지 않는다.\n'
    report+='공개 캡처4회 마지막 프레임→결과 계산: 정상0.266/0.368초, 누락0.323/0.835초. 자동종료 대기 및 브라우저 렌더 시간 제외. 이 수치를 실제 사용자 전체 대기시간으로 표현하지 않는다.\n'
    report+='구현과 별도 검산: 같은 Codex. 독립 전문가 검증 아님. 실제 카메라 호출0, 사용자자료 열람0, 학습/튜닝0.\n'
    report+='검증 중 제한환경 HTTP 연결이 차단돼 일반실행으로 공개 가상카메라 검사 수행. 첫 서버교체의 PowerShell 경로 배열 오류로 이전서버가 남아 포트중복 발생; 경로·대기상태를 다시 확인해 올바른 서버트리만 종료 후 실제8821 ready/카메라off 확인. 상세 web/DEPLOYMENT_CHECK.json.\n'
    report+='근거: checks/ROLE_CHECK.json, checks/DIRECT_TRUTH_CHECK.json, checks/REVIEWED_PARTIAL_CHECK.json, web/CONSUMER_CHECK.json, web/CARD_CHECK.json, web/CAPTURE_CHECK.json, web/HTTP_RECORDING_CHECK.json, web/HTTP_INTEGRATION_CHECK.json, qa/BROWSER_CHECK.json, web/ADMISSION.json, web/DEPLOYMENT_CHECK.json.\n다음: '+next_action+'\n'
    (D/'RESULT_KO.txt').write_text(report,encoding='utf-8')
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig')
    for n,title,body in [(8,'2026-11-17 선택한 매끈한 MOA 전환·공개 끝 상태 보완 웹 적용',summary+f' 현재8821 PID{pid}, {D / "serve.py"}. 별도8822 종료.'),(9,'2026-11-17 적용된 연습 화면의 실제 촬영 확인',next_action)]:
        lines=t.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if l.startswith(f'## {n}.'))
        lines.insert(i+1,f'\n### {title}\n{body}\n근거 {D / "CHECKPOINT.json"}, {D / "GRAPH.json"}.\n\n');t=''.join(lines)
    p.write_text(t,encoding='utf-8')
    s=read(O/'STATE.json');s['current_stage']=c['status'];s['next_single_action']=next_action
    k=s['capture_feedback_completion_20261116'];k.update(status=c['status'],updated_at=now,next_single_action=next_action,moa_coaching_room_20261117=c,current_graph=str(D/'GRAPH.json'),current_work_graph=str(D/'GRAPH.json'),current_checkpoint=str(D/'CHECKPOINT.json'),active_feedback_runtime=str(D/'serve.py'),runtime_pid=pid)
    write(O/'STATE.json',s)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {now}
목표: 정면 웹캠 사과·감사 시범→촬영→자동종료→교정/관측부족→재시도·퀘스트. 5방향은 아바타 전용.
{summary}

현재 서버: {D / 'serve.py'} --port 8821, 실제PID {pid}. 시작 시 카메라off, 사용자가 연습/실전에 들어오면 미리보기. 촬영은 버튼/Space로 시작. 별도8822 종료. 재개 시 실제상태 먼저 확인할 것.
작업폴더 D={D}
사용자 최신 선택은 MOA_BOTTOM_UP_PREVIEW.gif 그대로, 구름/연기 질감 없음. 앞선 세모/ㅗ 연기 시안과 승인대기는 이 선택으로 대체. 선택한 scene.js/hologram.js 해시 일치 검증 및 웹 적용 완료, 재승인 불필요.
UI: 좌측 동기화45도 두 시범, 중앙캐릭터·하단웹캠, 우측MOA 교정, 전환중대화숨김. 촬영/Space·3초·수동종료·로딩·촬영후웹캠유지·분석결과버튼. 실제 카메라 조작감 미확인.
판정: probe_endpoint_roles.py/role_runtime.py/thanks_runtime/worker.py 연결. 기존 학습 신체관측의 역할 분리, 새 가중치학습0. 공개88조건중4속도조건 개선, 직접149프레임143정답 유지, 부분16/정상32/카드7/공개캡처4 통과. 정상·교정기준과C·T3-F 유지,137핀 중이전122핀 보존. 자연 사용자 정확도로 해석금지.
검증: qa/BROWSER_CHECK.json 14, web/HTTP_RECORDING_CHECK.json 13, HTTP_INTEGRATION_CHECK.json 공개정상2 비교, DEPLOYMENT_CHECK.json 실제8821/선택효과/모델핀 확인. 사용자 녹화 접근0. 공개 캡처 결과계산 정상0.266/0.368초, 재관측포함최대0.835초; 종료대기/브라우저 제외.
{next_action}
근거: D/CHECKPOINT.json, D/GRAPH.json, D/RESULT_KO.txt, D/web/ADMISSION.json. 구현·별도검산 동일Codex, 독립전문가검증 아님. 학습진행없음, 현재서버만 유지. 전체목표완료 아님.
''',encoding='utf-8')
    for p in [D/'CHECKPOINT.json',D/'GRAPH.json',O/'STATE.json']:read(p)
    assert all(Path(n['evidence']).is_file() for n in graph['nodes'] if 'evidence' in n)
    print('CURRENT DOCS SYNCED; UI and admitted endpoint applied; actual webcam check pending')
if __name__=='__main__':main()
