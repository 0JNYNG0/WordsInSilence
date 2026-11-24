from pathlib import Path
import json,struct,hashlib,datetime
import numpy as np
U=Path(__file__).resolve().parent;R=U.parents[3]
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
checks=[]
def check(name,ok,detail=None): checks.append(dict(name=name,passed=bool(ok),detail=detail))
for name in ['robot','npc']:
    b=(U/'site/assets'/f'{name}.glb').read_bytes();length=struct.unpack_from('<I',b,12)[0];j=json.loads(b[20:20+length]);start=28+length
    check(name+'_valid_glb',b[:4]==b'glTF' and struct.unpack_from('<I',b,8)[0]==len(b))
    check(name+'_self_contained',all('uri' not in x for x in j.get('buffers',[])+j.get('images',[])))
    if name=='npc':
        check('original_65_bones',len(j['skins'][0]['joints'])==65)
        at=j['meshes'][0]['primitives'][0]['attributes'];check('eight_weight_attributes',all(k in at for k in ['JOINTS_0','JOINTS_1','WEIGHTS_0','WEIGHTS_1']))
        values=[]
        for k in ['WEIGHTS_0','WEIGHTS_1']:
            a=j['accessors'][at[k]];v=j['bufferViews'][a['bufferView']]
            x=np.frombuffer(b,dtype='<f4',count=a['count']*4,offset=start+v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,4);values.append(x)
        weights=np.concatenate(values,axis=1)
        check('finite_normalized_skin_weights',np.isfinite(weights).all() and np.max(np.abs(weights.sum(1)-1))<1e-4,float(np.max(np.abs(weights.sum(1)-1))))
    check(name+'_no_camera_nodes',not j.get('cameras'))
code='\n'.join((U/'site'/n).read_text(encoding='utf-8') for n in ['app.js','renderer.js','index.html','style.css'])
check('no_camera_or_inference_calls',all(x not in code for x in ['getUserMedia','mediaDevices','/api/','WebSocket(','XMLHttpRequest']))
check('no_remote_urls','https://' not in code and 'http://' not in code)
check('preview_disclosure_present','연출 미리보기' in code and '아직 연결하지 않았어요' in code)
check('concept_image_saved',(U/'STORYBOARD_PROPOSAL.png').is_file())
check('proof_screenshots_saved',all((U/n).is_file() for n in ['ROOM_PREVIEW.jpg','NPC_PREVIEW.jpg','PRACTICE_PREVIEW.jpg']))
verification=dict(time=now,checks=checks,passed=all(x['passed'] for x in checks),browser_observed=['MOA render','typewriter','practice transition','return to white','NPC appear','rotation','public gesture play','reduced effects'],browser_console_error_warn_observed=[],implementation='root Codex',verification='root separate file and CUA checks; not independent human validation',not_tested=['AI integration','sign accuracy','runtime inference latency','mobile and cross-browser performance'],status='CONCEPT_APPROVAL_PENDING')
(U/'VERIFIED.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
assert verification['passed'],verification
summary='사용자 제안에 따라 기존 MOA 로봇·Ch29를 재사용한 흰 3D 격자 가상 연습실의 독립 로컬 미리보기를 제작했다. 타이핑/기본 홀로그램 전환/복귀/NPC 생성/회전/공개 동작 재생을 브라우저에서 확인했다. 사용자가 연출 그림을 먼저 보고 승인하기로 요청하여 추가 구현을 멈추고 6컷 STORYBOARD_PROPOSAL.png를 생성했다. 이 그림은 콘셉트이며 실제 구현 캡처가 아니다. 카메라·분석 API 미연결, 새학습/검출/다운로드/원본·운영 변경 없음. 본선 G2와 기존 C·T3-F 보호 유지.'
next_action='연출은 web_tutorial_stage_preview_20261113/STORYBOARD_PROPOSAL.png의 A 홀로그램 전환·B NPC 생성에 대한 사용자 승인/수정을 기다린다. 승인 전 추가 연출/분석 연결 금지. 피드백 본선은 word100_feedback_recovery_20261113/NEXT_PLAN_KO.txt의 G2 재개점을 보존하며 완료6+18학습을 다시 실행하지 않는다. 아바타 교육용 정확도/얼굴관계 미해결도 그대로 유지한다.'
ctx=R/'WordsInSilence_PROJECT_CONTEXT.md';txt=ctx.read_text(encoding='utf-8')
entry='### 2026-11-13 웹 가상 연습실 연출 미리보기 · 그림 승인 대기\n'
for header,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
    at=txt.index(header)+len(header)
    txt=txt[:at]+'\n\n'+entry+body+'\n근거: `research/front_keyphase_v1/ops/web_tutorial_stage_preview_20261113/RESULT_KO.txt`, `VERIFIED.json`, `STORYBOARD_PROPOSAL.png`.\n'+txt[at:]
ctx.write_text(txt,encoding='utf-8')
sp=R/'research/front_keyphase_v1/ops/STATE.json';state=json.loads(sp.read_text(encoding='utf-8-sig'))
state['web_tutorial_stage_preview_20261113']={'status':'CONCEPT_APPROVAL_PENDING','updated_at':now,'result':str(U/'RESULT_KO.txt'),'storyboard':str(U/'STORYBOARD_PROPOSAL.png'),'server':json.loads((U/'SERVER.json').read_text(encoding='utf-8-sig')),'ai_connected':False,'training_resumed':False,'next':next_action}
state['next_single_action']=next_action;state['last_updated']=now
sp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 2026-11-13 웹 연출 그림 승인 대기\n\n'+summary+'\n\n다음: '+next_action+'\n\n현재 연출 단위: '+str(U)+'\n로컬 미리보기: http://127.0.0.1:8820/ (정적서버, 사용자 카메라 열지 않음, PID는 SERVER.json 확인)\n일반 환경 로컬 서버만 실행 중. 추가 모델 학습/추출/분석은 없음. 기존 웹을 교체하지 않음.\n\n사용자 선호: 기존 MOA 로봇, 기존 여자 NPC Ch29와 원래 Mixamo 뼈대. 흰 입체 격자 공간, 검정 한글 도트 타이핑, 눈빛 홀로그램 전환, 우측 뒤로 로봇 이동, 회색 글리치로 NPC 등장. 연출 콘셉트를 사용자에게 먼저 보여주고 확인받은 뒤 적용.\n\n아바타 직전 결과: avatar_motion_smoothing_20261113/RESULT_KO.txt. 공개 조상 REAL02 5뷰 추정3D·65뼈 유지·손목 제한 보완, 교육용완성 아님. 정확 얼굴관계·큰손가락회전·접촉/충돌 미해결.\n피드백 본선: word100_feedback_recovery_20261113. 손모양6+물리이동18 비교 검산 완료/미채택, G2보완→G3통합→G4평가 순서. 기존 C·T3-F/운영/공통100 보존, 사용자·멘토자료 학습·튜닝·후보선택 금지.\n\n구현/파일검산/브라우저 확인: root Codex. 사람 또는 독립 수어 전문가 검수와 다름.\n',encoding='utf-8')
print(json.dumps({'passed':verification['passed'],'checks':len(checks),'status':'CONCEPT_APPROVAL_PENDING'},ensure_ascii=False))
