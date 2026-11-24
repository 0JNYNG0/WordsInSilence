from pathlib import Path
import json,hashlib,urllib.request,datetime,time,shutil,socket
ROOT=Path.cwd();O=ROOT/'research/front_keyphase_v1/ops';D=Path(__file__).resolve().parent;P=D.parent
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
    assert status['layout_admitted'] and status['companion_ui_version']=='moa_coaching_room_v2'
    for key in ['frontal_scope_active','evidence_repair_active','selective_retry_active','rest_stability_active','endpoint_role_active']:assert status[key],key
    a=read(D/'web/ADMISSION.json');assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
    for name in ['scene.js','renderer.js','layout.css','hologram.js']:
        with urllib.request.urlopen(url+'/'+name) as r:body=r.read()
        assert body==(D/'site'/name).read_bytes(),name
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',8822))!=0
    before=read(D/'qa/MEASURE_BEFORE.json');after=read(D/'qa/MEASURE_AFTER.json');browser=read(D/'qa/BROWSER_CHECK.json')
    assert after['passed'] and browser['passed']
    at=datetime.datetime.now().astimezone().isoformat();pid=read(D/'SERVER.json')['pid']
    result=dict(status='MOA_UI_LAYOUT_PLAYBACK_REFINED_WEB_APPLIED',at=at,passed=True,url=url+'/',server=str(D/'serve.py'),pid=pid,
        implementation='Codex',verification='Same Codex separate mocked browser checks; not independent review',
        changes=['Dialogue anchored below projected MOA, tail points at robot, stable while robot bobs','Both angle panes wider/taller; controls follow pane bottom','Demonstration 30 fps, idle preview 12 fps; 3D remains paused during countdown/recording/analysis','Skin matrices reused for three views; animation poses and transition unchanged'],
        before_fps=before['performance']['fps'],after_fps=after['performance']['fps'],fps_scope='3-second headless browser measurement with mocked webcam, not physical webcam performance',
        old_pane=before['sizes'][0]['left'],new_pane=after['sizes'][0]['left'],viewport='1366x768',layout_checks=len(after['checks']),browser_checks=len(browser['checks']),
        prior_pins_preserved=a['prior_pins_preserved'],total_pins=len(a['pins']),backend_changed=False,weights_changed=False,training_runs=0,private_data_read=False,physical_camera_verified=False,preview_server_stopped=True,
        corrected_checks=['The pause test initially dispatched a lock while mock status still reported preview; corrected mock to recording and verified zero draw calls. Application lock logic unchanged.','Startup initially exposed ready briefly during the final UI hash check; now status remains preparing until admission completes.'])
    write(D/'web/DEPLOYMENT_CHECK.json',result);write(D/'CHECKPOINT.json',result)
    summary='MOA 말풍선을 로봇의 실제 화면 위치 바로 아래에 연결하고 꼬리 위치도 맞춤. 좌우45도 예시는 1366×768에서 각179×238→213×276px로 확대. 예시 재생을12fps 제한에서30fps로 변경하고 동일 관절행렬의3화면 중복계산을 줄임. 모의 웹캠 브라우저 측정10.51→30.09fps,4해상도 배치·촬영정지 등24검사와 기존촬영흐름14검사 통과. 촬영·분석 시3D정지는 유지. 모델/관측/판정/자동종료 코드 변경0, 사용자자료접근0, 이전137핀 모두 유지. 실제 웹캠 성능은 이번검사에 포함하지 않음.'
    next_action='사용자는 같은8821을 새로고침해 MOA 말풍선·확대 시범·재생감을 확인할 수 있다. 새 실제 촬영의 자동 종료/교정 재시도/퀘스트 확인은 가능할 때 진행하며, 문제는 공개 정면자료로 재현해 보완한다. 기존 공개 검사는 새 근거 없이 반복하지 않는다. 사용자자료는 진단만, 학습·튜닝·후보선택 금지.'
    graph=read(P/'GRAPH.json');graph.update(status=result['status'],previous=str(P/'GRAPH.json'),next_single_action=next_action)
    for node in graph['nodes']:
        if node['id']=='UI':node.update(label='MOA 전환·말풍선 연결·확대 시범·부드러운 재생',status='user_requested_refinement_verified_web_applied',evidence=str(D/'qa/MEASURE_AFTER.json'))
    write(D/'GRAPH.json',graph)
    (D/'RESULT_KO.txt').write_text(summary+f'\n현재 {url}/ PID{pid}, {D / "serve.py"}. 별도8822종료.\n'+next_action+'\n근거: qa/MEASURE_BEFORE.json, qa/MEASURE_AFTER.json, qa/BROWSER_CHECK.json, web/ADMISSION.json, web/DEPLOYMENT_CHECK.json.\n구현/검증은 같은Codex. 실제카메라 호출없음. 최초검사 fixture와 시작상태 수정 이력은 DEPLOYMENT_CHECK.json.\n',encoding='utf-8')
    backup=D/'before_docs';backup.mkdir(exist_ok=True)
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
        if not (backup/p.name).exists():shutil.copy2(p,backup/p.name)
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig')
    for n,title,body in [(8,'2026-11-17 MOA 말풍선·시범 확대·재생 개선 적용',summary+f' 현재8821 PID{pid}.'),(9,'2026-11-17 수정 UI의 실제 사용감·촬영 확인',next_action)]:
        lines=text.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if l.startswith(f'## {n}.'))
        lines.insert(i+1,f'\n### {title}\n{body}\n근거 {D / "CHECKPOINT.json"}.\n\n');text=''.join(lines)
    p.write_text(text,encoding='utf-8')
    s=read(O/'STATE.json');s.update(current_stage=result['status'],next_single_action=next_action)
    k=s['capture_feedback_completion_20261116'];k.update(status=result['status'],updated_at=at,next_single_action=next_action,current_graph=str(D/'GRAPH.json'),current_work_graph=str(D/'GRAPH.json'),current_checkpoint=str(D/'CHECKPOINT.json'),active_feedback_runtime=str(D/'serve.py'),runtime_pid=pid,layout_refinement_20261117=result)
    write(O/'STATE.json',s)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {at}
목표: 정면 웹캠 사과·감사 시범→촬영→자동종료→교정/관측부족→재시도·퀘스트. 5방향은 아바타 전용.
{summary}

현재 서버: {D / 'serve.py'}, 실제PID{pid}, 8821. START_WEB_DEMO.cmd 경로 갱신. 시작 시 카메라off. 사용자가 연습/실전에 들어오면 미리보기, 촬영은 버튼/Space. 별도8822 검사 서버 종료. 재개 시 실제상태 확인.
현재 작업폴더 D={D}
이전 판정폴더 P={P}
D는 UI만 보완. P/serve.py Handler와 RoleCapture 판정부를 재사용하며 P 및 그 이전137핀 모두 보존, 새148핀 admission. 모델/학습/자동종료/재검출 변경0. 이전 공개 끝상태 역할분리 보완(88조건중4속도조건 개선, 직접149프레임143정답/부분16/정상32/카드7/공개캡처4)은 그대로 유지. 모든단어/사용자 정확도를 의미하지 않음.
UI: MOA의 투영된 위치 아래 말풍선·꼬리, 예시 두 화면 약18%넓고16%높게, 시범 재생30fps·대기12fps·촬영/분석중3D정지. 세 시점의 관절행렬 재사용. 사용자 선택 아래→위 매끈한 페이드/비행 코드 및 구름질감없음 유지. 영상 재생·아바타 자료변경없음.
검증:4해상도 배치와재생·촬영정지24검사, 기존카운트다운/Space/결과/퀘스트14검사, 실제8821 HTML/JS·모든핀·카메라off·준비완료 확인. Headless 모의카메라 재생10.51→30.09fps; 실제 웹캠/다른프로그램부하 검증아님. 사용자자료열람0.
{next_action}
근거 D/CHECKPOINT.json, D/GRAPH.json, D/RESULT_KO.txt, D/web/ADMISSION.json. UI검사는가상카메라/공개fixture사용. 실제사용자시연 전체완료아님. 구현·별도검산 같은Codex이며독립전문가검증아님. 진행중학습없음.
''',encoding='utf-8')
    for p in [O/'STATE.json',D/'CHECKPOINT.json',D/'GRAPH.json']:read(p)
    print(json.dumps(dict(passed=True,url=url+'/',pid=pid,before_fps=result['before_fps'],after_fps=result['after_fps'],previous_pins=result['prior_pins_preserved']),ensure_ascii=False))
if __name__=='__main__':main()
