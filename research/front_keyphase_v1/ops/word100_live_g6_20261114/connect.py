from pathlib import Path
import shutil,json,hashlib
U=Path(__file__).resolve().parent;W=U.parent/'web_tutorial_live_integration_20261113'
files=['server.py','site/index.html','site/live.js','site/feedback.js','site/scene.js']
(U/'before/site').mkdir(parents=True,exist_ok=True)
for f in files:
 if not(U/'before'/f).exists():shutil.copy2(W/f,U/'before'/f)
server=(U/'before/server.py').read_text('utf-8-sig')
server=server.replace("PORT=8821;TOKEN=", "sys.path.insert(0,str(O/'word100_live_g6_20261114'))\nfrom g6_runtime_adapter import CaptureService,attach\nfrom g6_routes import handle_get as handle_g6_get,catalog as g6_catalog\nACCEPTANCE_PATH=O/'word100_live_g6_20261114/WEB_ACCEPTANCE.json'\nPORT=8821;TOKEN=")
server=server.replace("if ATTEMPT and SERVICE.state=='done' and SERVICE.result is not None and RESULT is None:","if ATTEMPT and SERVICE.state=='done' and SERVICE.result is not None and RESULT is None and not (SERVICE.camera_thread and SERVICE.camera_thread.is_alive()):")
server=server.replace('if handle_g5_get(self,route):return','if handle_g5_get(self,route) or handle_g6_get(self,route):return')
server=server.replace("examples=read(OLD/'EXAMPLES.json')","examples=g6_catalog()")
server=server.replace("model='기존 RTMW TCN + 관절 전용 시간합성곱·GRU 연구 모델'","model='기존 RTMW TCN + G4 통합 판정 / G6 촬영 연결'")
server=server.replace("elif route=='/api/public-replay':",'''elif route=='/api/rendered':
                    if OWNER!=client or not RESULT or b.get('attempt')!=ATTEMPT['id']:raise ValueError('현재 촬영의 표시 확인이 아니에요.')
                    r=RESULT['result'];stop_clock=r.get('_stop_clock')
                    if stop_clock is None:raise ValueError('촬영 종료 시간 정보가 없어요.')
                    seconds=time.perf_counter()-stop_clock
                    audit=dict(attempt_id=ATTEMPT['id'],feedback_version=r.get('feedback_version'),stop_reason=r.get('stop_reason'),recorded_frames=r.get('recorded_frames'),observed_frames=r.get('observed_frames'),stop_to_result_seconds=r.get('stop_to_result_seconds'),stop_to_browser_ack_seconds=seconds,training_use=False,scope='User-started inference only; browser acknowledgement after two animation frames, includes network return; no correctness grading')
                    if getattr(SERVICE,'public_fixture',False):audit['scope']='Public-camera fixture through browser, not physical webcam';audit['public_fixture']=True
                    ACCEPTANCE_PATH.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
                    self.reply(audit);return
                elif route=='/api/public-replay':''')
(W/'server.py').write_text(server,encoding='utf-8')
html=(U/'before/site/index.html').read_text('utf-8-sig')
html=html.replace('<div id="ranks" hidden></div>','<div id="ranks" hidden></div><p id="result-timing" hidden></p>')
html=html.replace('저장된 공개 결과예요. 실전 진행에는 반영하지 않아요.','G6 촬영 연결 경로로 계산한 공개 결과예요. 편집 시험은 따로 표시하며 실전 진행에는 반영하지 않아요.')
(W/'site/index.html').write_text(html,encoding='utf-8')
live=(U/'before/site/live.js').read_text('utf-8-sig')
live=live.replace("$('ranks').hidden=true;","$('ranks').hidden=true;$('result-timing').hidden=true;")
live=live.replace("if(publicMode){say('공개 자료의 저장된 결과야. 실전 성공에는 반영하지 않아.');return}","if(publicMode){$('result-timing').hidden=false;$('result-timing').textContent='공개 관절 입력의 판정 계산: '+data.compute_seconds.toFixed(2)+'초 · 촬영·화면 대기시간과는 달라요.';say((data.public_example_name||'공개 자료')+' 결과야. 실전 성공에는 반영하지 않아.');return}")
live=live.replace("await displayResult(d.result,d.quest,d.outcome)","await displayResult(d.result,d.quest,d.outcome);await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));const measured=await command('/api/rendered',{attempt});$('result-timing').hidden=false;$('result-timing').textContent='촬영 종료 → 화면 표시 확인 약 '+measured.stop_to_browser_ack_seconds.toFixed(2)+'초 (계산 '+measured.stop_to_result_seconds.toFixed(2)+'초)';")
(W/'site/live.js').write_text(live,encoding='utf-8')
feedback=(U/'before/site/feedback.js').read_text('utf-8-sig')
feedback=feedback.replace("let ink=color(card);", "let ink=color(card);const visibleSpans=focusCue?.spans||card?.query_spans;const hands=focusCue?.hands||card?.hands;if(visibleSpans?.length&&!visibleSpans.some(([a,b])=>i>=a&&i<=b))ink='#96a4b8';")
feedback=feedback.replace("ink,card?.hands)","ink,hands)")
feedback=feedback.replace("'#508cd8',card?.hands)","'#508cd8',hands)")
feedback=feedback.replace("(cue.kind==='direction'?'이동 방향':'손모양')", "({direction:'이동 방향',shape:'손모양',order:'행동 순서'}[cue.kind]||'행동')")
feedback=feedback.replace("'공개 자료의 저장 분석 결과 · 사용자 촬영 결과가 아닙니다'","(data.public_example_name||'공개 자료')+' · G6 연결 확인 결과 · 사용자 촬영이 아닙니다'")
feedback=feedback.replace("$('query-video').pause();clipEnd=null;", "$('query-video').pause();$('query-video').currentTime=clipEnd;clipEnd=null;")
feedback=feedback.replace("$('query-video').currentTime=Math.max(0,cue.start_seconds-.15)","$('query-video').currentTime=cue.start_seconds")
(W/'site/feedback.js').write_text(feedback,encoding='utf-8')
scene=(U/'before/site/scene.js').read_text('utf-8-sig')
scene=scene.replace("if(disposed)return;const dt=(now-last)/1000;last=now;const t=now/1000;", """if(disposed)return;
      // RTMW shares the integrated GPU with WebGL. Keep the last background
      // image during capture instead of competing with observation frames.
      const pause3D=captureLocked||document.hidden||$('#feedback-dialog').open;
      document.body.dataset.renderMode=pause3D?'observation-priority':'animated';
      if(pause3D){last=now;raf=requestAnimationFrame(frame);return;}
      const dt=(now-last)/1000;last=now;const t=now/1000;""")
(W/'site/scene.js').write_text(scene,encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(U/'CONNECTION.json').write_text(json.dumps(dict(files={f:dict(before=sha(U/'before'/f),after=sha(W/f)) for f in files},scope='Current research web adapter; C·T3-F, RTMW, TCN/DTW weights unchanged',camera_started=False),ensure_ascii=False,indent=2),encoding='utf-8')
print('G6 connection prepared; server restart follows after checks')
