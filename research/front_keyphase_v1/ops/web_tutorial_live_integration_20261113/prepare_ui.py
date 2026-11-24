from pathlib import Path
import json, hashlib
U=Path(__file__).resolve().parent;P=U.parent/'web_tutorial_stage_preview_20261113'/'site';S=U/'site';S.mkdir(exist_ok=True)
base=(P/'app-v2.js').read_text('utf-8')
base=base.replace("import {Quest} from './quest.js';",'')
a=base.index('const quest=new Quest(');b=base.index('function say(',a);base=base[:a]+base[b:]
base=base.replace('quest.reset();','').replace('resetQuest();','')
a=base.index('function renderQuest()');b=base.index('async function startQuest()',a);base=base[:a]+base[b:]
a=base.index('async function simulateSuccess()');b=base.index('function next()',a);base=base[:a]+base[b:]
base=base.replace("$('#simulate-success').addEventListener('click',simulateSuccess);$('#quest-reset').addEventListener('click',()=>{if(scene==='quest'&&!quest.pending)resetQuest();});",'')
base='\n'.join(line for line in base.splitlines() if not line.startswith("$('#simulate-success').addEventListener"))
a=base.index("Object.defineProperty(window,'previewState'");b=base.index('\nstart();',a);base=base[:a]+base[b:]
base=base.replace("'preview-controls'","'live-controls'")
base=base.replace('mix(1.48,1.31,','mix(1.48,1.53,').replace('robotY+.28:robotY','robotY+.06:robotY')
base=base.replace("record(name);}","record(name);window.dispatchEvent(new CustomEvent('stage:scene',{detail:{scene:name}}));}")
base=base.replace("let scene='intro'","let captureLocked=false;\nlet scene='intro'")
base=base.replace('if(busy)return;','if(busy||captureLocked)return;').replace('if(busy||!loaded)return;','if(busy||captureLocked||!loaded)return;').replace('if(!b||busy||!loaded)return;','if(!b||busy||captureLocked||!loaded)return;')
base=base.replace("연출 미리보기 · 실제 분석 미연결","로컬 웹캠 테스트 · 사용자 촬영 학습 제외")
base=base.replace("캐릭터 동작 시험 · 카메라/분석 연결 전","기존 관절 모델 연결 · 교정 안내는 관측 근거 범위")
base=base.replace("실전 UI 미리보기 · 성공은 시험 버튼으로 확인","TCN 단어 인식 기준 · 자세 전체 정답 판정과 구분")
base=base.replace("인사하고, 필요한 것을 부탁해요.","배운 단어를 순서대로 표현해요.")
base=base.replace("이번에는 실제 대화처럼 해 보자. 인사하고, 사과 다섯 개를 부탁하는 거야.","이번에는 도움 없이 단어를 차례로 표현해 보자. 인식되면 다음 단어로 넘어갈 거야.")
base=base.replace("인사, 사과, 다섯. 위에 있는 순서를 따라 천천히 이야기해 봐.","위에 있는 단어부터 하나씩 표현해 봐. 이번에는 실제 촬영 결과로 진행할 거야.")
base=base.replace("가운데 캐릭터를 함께 살펴보자. 네 모습은 아래 작은 화면에서 볼 수 있어.","참고 동작을 보고 따라 해 봐. 촬영을 마치면 확인할 부분을 함께 살펴볼게.")
base=base.replace("start();", "window.addEventListener('coach:say',e=>say(e.detail.text,e.detail.speaker||'MOA'));\nwindow.addEventListener('coach:lock',e=>{captureLocked=!!e.detail;lock(busy);for(const el of document.querySelectorAll('#next,.steps button,#reset,#gesture,#rotate-left,#rotate-right,#zoom'))el.disabled=busy||captureLocked;});\nwindow.addEventListener('coach:stop-gesture',stopGesture);\nstart();")
(S/'scene.js').write_text(base,encoding='utf-8')
html=(P/'index.html').read_text('utf-8')
html=html.replace('<link rel="stylesheet" href="style-v2.css">','<link rel="stylesheet" href="style-v2.css"><link rel="stylesheet" href="live.css">')
html=html.replace('연출 미리보기','실제 촬영 테스트').replace('가운데 캐릭터를 보며 동작을 살펴보세요.','<label>연습 단어 <select id="word"></select></label>')
a=html.index('    <div class="quest-eyebrow">');b=html.index('\n  </section>',a)
html=html[:a]+'''    <div class="quest-eyebrow">단어 인식 연결 시험 <span>시연 단어 확장 전 · 현재 지원 4단어</span></div>
    <ol id="quest-track"></ol><p id="quest-prompt" aria-live="polite"></p>
    <div id="quest-success" hidden role="status"><span>✓</span><strong>퀘스트 성공</strong><small>네 단어의 인식을 마쳤어요.</small></div>'''+html[b:]
a=html.index('  <section id="self-view"');b=html.index('  <section class="dialogue"',a)
html=html[:a]+'''  <section id="self-view" hidden aria-label="중앙 아래 내 촬영 화면"><div class="camera-top"><span><i></i> 내 모습</span><small id="camera-label">카메라 꺼짐</small></div><div class="camera-placeholder" id="camera-placeholder"><p>손을 내린 자세로 준비해요</p><small>촬영은 이 PC에서만 분석해요</small></div><img id="live-camera" hidden alt="내 실시간 웹캠"><video id="self-replay" hidden playsinline muted controls></video><strong id="countdown" hidden></strong></section>
  <aside id="live-controls" hidden><span class="test-label" id="capture-state">분석기 연결 중</span>
  <div class="capture-buttons"><button id="camera-open">카메라 켜기</button><button id="capture-start" disabled>3초 뒤 촬영</button><button id="capture-finish" disabled>수동 종료</button><button id="camera-close" disabled>카메라 닫기 / 취소</button></div>
  <p id="capture-help">동작 뒤 손을 내리면 자동 종료돼요.</p><div id="ranks" hidden></div><button id="show-feedback" hidden>확인할 순간 보기 →</button>
  <button id="quest-reset" hidden>퀘스트 처음부터</button><button id="reference-open">공개 참고 동작 보기</button>
  <details id="plan"><summary>추가할 시연 단어</summary><p>안녕하세요 → 사과 → 감사</p><small>AIHub 안녕하세요 기준. 학습·세부 피드백 검증 전에는 성공 처리하지 않아요.</small></details>
  <details id="public-tools"><summary>공개 자료로 화면 확인</summary><div id="public-examples"></div><small>저장된 공개 결과예요. 실전 진행에는 반영하지 않아요.</small></details>
  </aside>
  <div id="analysis-busy" hidden role="status"><span class="spinner"></span><h2 id="busy-label">동작을 살펴보고 있어요</h2><p>촬영 중 뽑은 관절을 재사용하고 있어요.</p></div>
  <dialog id="feedback-dialog"><button id="feedback-close" class="dialog-close">닫기 ×</button><h2 id="feedback-title">확인할 순간</h2><p id="feedback-source"></p><div id="feedback-cards"></div><div class="comparison"><figure><figcaption>공개 참고 · 표시용 대표 영상</figcaption><div><video id="ref-video" playsinline muted></video><canvas id="ref-skeleton"></canvas></div></figure><figure><figcaption id="query-caption">내 촬영</figcaption><div><video id="query-video" playsinline muted></video><canvas id="query-skeleton"></canvas></div></figure></div><p id="feedback-message"></p><p id="feedback-instruction"></p><div id="feedback-bands"></div><input id="feedback-seek" type="range" min="0" max="1000" value="0" aria-label="비교 시점"><div class="play-buttons"><button id="compare-play">전체 비교 재생</button><button id="compare-stop">멈춤</button><span id="feedback-time"></span></div><small>초록: 확인한 특징이 비슷함 · 주황: 차이 관측 · 회색: 확인 부족(오답 아님). 단어 전체 정답률 점수는 아닙니다.</small></dialog>
  <dialog id="reference-dialog"><button id="reference-close" class="dialog-close">닫기 ×</button><h2 id="reference-title"></h2><video id="reference-video" controls playsinline muted></video><p>공개 참고 영상 · 캐릭터 동작은 아직 ‘조상’ 적용 시험만 준비됐어요.</p></dialog>
'''+html[b:]
html=html.replace('<script type="module" src="app-v2.js"></script>','<script type="module" src="scene.js"></script><script type="module" src="live.js"></script>')
(S/'index.html').write_text(html,encoding='utf-8')
protected=[P/'app-v2.js',P/'index.html',P/'renderer.js']
protected += [U.parent/'word100_action_feedback_demo_20261108'/n for n in ['capture.py','consumer.py','auto_end.py','server.py','runs/42/model.pt']]
(U/'PROTECTED.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},indent=2),encoding='utf-8')
print('Prepared isolated site; original preview and capture/model code unchanged')
