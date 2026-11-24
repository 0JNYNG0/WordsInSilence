from pathlib import Path
import json,hashlib,urllib.request,datetime,time,shutil,socket
ROOT=Path.cwd();O=ROOT/'research/front_keyphase_v1/ops';D=Path(__file__).resolve().parent;P=D.parent/'layout_refinement_20261117'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    url='http://127.0.0.1:8821'
    with urllib.request.urlopen(url+'/api/meta') as r:meta=json.load(r)
    for _ in range(150):
        req=urllib.request.Request(url+'/api/status',headers={'X-Session-Token':meta['token']})
        with urllib.request.urlopen(req) as r:status=json.load(r)
        if status['state']!='preparing':break
        time.sleep(.25)
    assert status['state']=='ready' and not status['camera_thread_active'],status.get('error')
    assert status['layout_admitted'] and status['companion_ui_version']=='moa_coaching_room_v3'
    for key in ['frontal_scope_active','evidence_repair_active','selective_retry_active','rest_stability_active','endpoint_role_active']:assert status[key],key
    admission=read(D/'web/ADMISSION.json');assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in admission['pins'].items())
    for name in ['scene.js','live.js','dock.css','hologram.js']:
        with urllib.request.urlopen(url+'/'+name) as r:assert r.read()==(D/'site'/name).read_bytes(),name
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',8822))!=0
    motion=read(D/'qa/MOTION_DOCK_CHECK.json');browser=read(D/'qa/BROWSER_CHECK.json');assert motion['passed'] and browser['passed']
    at=datetime.datetime.now().astimezone().isoformat();pid=read(D/'SERVER.json')['pid']
    result=dict(status='MOA_FLIGHT_GAZE_BOTTOM_DOCK_WEB_APPLIED',at=at,passed=True,url=url+'/',server=str(D/'serve.py'),pid=pid,
        implementation='Codex',verification='Same Codex separate mock-camera browser checks; not independent expert review',
        departure_ms=dict(before=2365,after=2000),right_reentry_ms=dict(before=817,after=950),transition_total_ms=dict(before=4300,after=4200),
        gaze='Continuous side-to-front during return; front-to-inward during quest move. Direct practice/quest navigation returns through center.',
        webcam='Bottom-center natural aspect ratio, object-fit contain; adjacent capture button on desktop; 4:3,16:9,portrait tested',
        motion_layout_checks=len(motion['checks']),browser_checks=len(browser['checks']),prior_pins_preserved=admission['prior_pins_preserved'],total_pins=len(admission['pins']),
        inference_changed=False,weights_changed=False,training_runs=0,private_data_read=False,physical_camera_verified=False,preview_server_stopped=True)
    write(D/'web/DEPLOYMENT_CHECK.json',result);write(D/'CHECKPOINT.json',result)
    summary='MOA 출발 구간2365→2000ms, 오른쪽 재등장817→950ms로 조정. 가상공간에서 중앙으로 돌아올 때 옆보기→정면, 실전으로 이동할 때 정면→안쪽보기를 이동과 함께 연속 보간. 연습/실전 직접 전환도 중앙 복귀를 거쳐 순간 위치변화를 방지. 웹캠은 중앙 맨 아래에 원본 비율대로 표시하고 촬영버튼은 왼쪽으로 이동. 4해상도에서 NPC 상반신(골반 관절 위) 가림 없음,4:3/16:9/세로 비율 보존 확인. 실제렌더 이동·배치28검사와 기존촬영흐름14검사 통과. 판정·자동종료·모델 변경0, 이전148핀 유지, 실제촬영 미실시.'
    next_action='같은8821을 새로고침해 이동 속도·고개 회전·낮춘 웹캠 배치를 확인할 수 있다. 사용자 실제촬영의 자동 종료/교정 재시도/퀘스트 확인은 가능할 때 이어간다. 새 문제는 공개 정면자료로 재현해 보완하며 사용자촬영은 진단만, 학습·튜닝·후보선택 금지. 새 근거 없이 기존 공개검사 반복 금지.'
    graph=read(P/'GRAPH.json');graph.update(status=result['status'],previous=str(P/'GRAPH.json'),next_single_action=next_action)
    for node in graph['nodes']:
        if node['id']=='UI':node.update(label='MOA 전환 속도·시선 연결·하단 웹캠 배치',status='user_requested_refinement_verified_web_applied',evidence=str(D/'qa/MOTION_DOCK_CHECK.json'))
    write(D/'GRAPH.json',graph)
    (D/'RESULT_KO.txt').write_text(summary+f'\n현재 {url}/ PID{pid}, {D / "serve.py"}. 별도8822종료.\n'+next_action+'\n근거: qa/MOTION_DOCK_CHECK.json, qa/BROWSER_CHECK.json, web/ADMISSION.json, web/DEPLOYMENT_CHECK.json.\n구현/별도 검산 동일Codex. 실제카메라·사용자자료 접근 없음. 가상 카메라/공개fixture와제어된애니메이션시계검사이며 실제사용자체감/전체판정성능검증은아님.\n',encoding='utf-8')
    backup=D/'before_docs';backup.mkdir(exist_ok=True)
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
        if not (backup/p.name).exists():shutil.copy2(p,backup/p.name)
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig')
    for n,title,body in [(8,'2026-11-17 MOA 이동·시선·하단 웹캠 배치 적용',summary+f' 현재8821 PID{pid}.'),(9,'2026-11-17 새 전환과 낮춘 웹캠의 실제 사용 확인',next_action)]:
        lines=text.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if l.startswith(f'## {n}.'))
        lines.insert(i+1,f'\n### {title}\n{body}\n근거 {D / "CHECKPOINT.json"}.\n\n');text=''.join(lines)
    p.write_text(text,encoding='utf-8')
    s=read(O/'STATE.json');s.update(current_stage=result['status'],next_single_action=next_action)
    k=s['capture_feedback_completion_20261116'];k.update(status=result['status'],updated_at=at,next_single_action=next_action,current_graph=str(D/'GRAPH.json'),current_work_graph=str(D/'GRAPH.json'),current_checkpoint=str(D/'CHECKPOINT.json'),active_feedback_runtime=str(D/'serve.py'),runtime_pid=pid,motion_dock_20261117=result)
    write(O/'STATE.json',s)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {at}
목표: 정면 웹캠 사과·감사 시범→촬영→자동종료→교정/관측부족→재시도·퀘스트. 5방향은 아바타 전용.
{summary}

현재 서버: {D / 'serve.py'}, 실제PID{pid}, 8821. START_WEB_DEMO.cmd 경로 갱신. 시작 시카메라off. 사용자가연습/실전진입하면미리보기,촬영은버튼/Space. 별도8822 검사서버 종료. 재개 시실제상태 확인.
현재 작업폴더 D={D}
이전 UI폴더 P={P}; 판정 원본은 그 상위 moa_coaching_room_20261117.
D는 UI만 변경. 기존 RoleCapture/Handler 재사용, 이전148핀 모두 보존·새160핀 admission. C·T3-F·모델/자동종료/재검출/학습 변경0. 사용자촬영 접근0. 공개 끝상태 역할분리 보완 및 이전정상판정은 그대로 유지.
UI: 기존 매끈한 아래→위 페이드 질감 그대로. 출발15%빠르게/재등장16%느리게(소요시간 기준),중앙복귀와실전이동중고개연속회전. 연습↔실전 직접이동도 중앙복귀후연결. 웹캠하단중앙·contain·native비율,촬영버튼왼쪽. 원본영상/아바타동작자료변경없음. 이전 MOA말풍선앵커·확대45도두시범·30fps재생·촬영/분석중3D정지 유지.
검증: 실제WebGL좌표를 제어시계로 확인한 이동·배치28검사,기존카운트다운/Space/결과/퀘스트14검사,실제8821자산·160핀·카메라off/ready 확인.4해상도NPC골반위영역가림없음;4:3/16:9/세로카메라비율보존. 실제카메라체감/사용자인식성능검증은이번범위아님.
{next_action}
근거 D/CHECKPOINT.json, D/GRAPH.json, D/RESULT_KO.txt, D/web/ADMISSION.json. 구현·별도검산동일Codex,독립전문가검증아님. 전체실제시연완료아님. 진행중학습없음.
''',encoding='utf-8')
    for p in [O/'STATE.json',D/'CHECKPOINT.json',D/'GRAPH.json']:read(p)
    print(json.dumps(dict(passed=True,url=url+'/',pid=pid,previous_pins=result['prior_pins_preserved'],checks=result['motion_layout_checks']),ensure_ascii=False))
if __name__=='__main__':main()
