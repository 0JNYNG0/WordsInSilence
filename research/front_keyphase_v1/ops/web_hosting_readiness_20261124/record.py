from pathlib import Path
import json

D=Path(__file__).resolve().parent
ROOT=D.parents[3]
audit=json.loads((D/'AUDIT.json').read_text('utf-8'))
assert audit['original_pins_all_match'] and audit['original_pin_count']==160
rel=D.relative_to(ROOT).as_posix()
summary=('다른 Windows 강의실 PC에서 URL 시연·HDD/개발PC 독립·ChatGPT 호스팅 제외 요청. '
         '실제 코드 확인: 현재 OpenCV는 호스트 카메라, localhost 단일 세션, MP4/NPZ/진단JSON 로컬 보관, '
         '선택 재검출도 MP4 의존. 현재는 외부 AI 서버 임대가 아님. OpenVINO Intel GPU 경로이며 '
         '관절추출 ONNX 두 개 합계315.4MiB(전체 배포크기 아님). 기존 공개2영상 C: 의존성 감사는 있으나 '
         'HDD 실제분리/타PC 실행/브라우저RTMW는 미검증. 사용자 관절값 사용 허용·영상잔류 우려를 반영해 '
         '기본 후보는 외부RGB전송 없이 사용자측 관절추출. URL-only 브라우저 이식과 로컬Windows 패키지 '
         '발표 대안을 분리. Netlify만 올려서는 현재 Python 추론 미작동. 소스7개 행/해시와 운영160핀 보존 확인. '
         '학습/추론/카메라/개인기록열람/운영변경/외부업로드/결제/배포0. Codex 확인, 독립검증아님. '
         '근거 '+rel+'/AUDIT.json 및 RESULT_KO.txt.')
next_action=('서버 임대는 필수가 아님을 설명하고 실행 위치를 확정. URL만 이용하면서 영상 외부전송을 피하려면 '
             '브라우저 RTMW 호환·공개영상 좌표/속도 검증이 선행되어야 함. 발표 대안은 HDD 없는 Windows 로컬 '
             '실행 패키지이며 강의실 GPU/실행권한 확인 필요. 개인정보 미보관은 파일저장·선택재검출·비교재생 '
             '전체 경로를 함께 바꿔 검증. 비용·계정·외부배포 미승인. 검토만 완료; 배포 완료로 취급하지 않음.')
state_path=D.parent/'STATE.json'
state=json.loads(state_path.read_text('utf-8-sig'))
stage=state['current_stage']
state['web_hosting_readiness_20261124']={
    'status':audit['status'],'summary':summary,'next_action':next_action,
    'audit':rel+'/AUDIT.json','report':rel+'/RESULT_KO.txt',
    'runtime_changed':False,'deployment_completed':False,
    'public_upload':False,'camera_opened':False,'training_runs':0,
    'original_pins_preserved':160,'pid':None,
    'verification_actor':'Codex source assertions and hash verification; no hardware/deployment test'}
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8')
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
text=context.read_text('utf-8')
heading='### 2026-11-24 외부 웹 시연·HDD 독립 실행 구조 검토'
if heading not in text:
    h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
    h9='## 9. 다음 단일 작업'
    assert text.count(h8)==1 and text.count(h9)==1
    text=text.replace(h8,h8+'\n\n'+heading+'\n\n'+summary,1)
    text=text.replace(h9,h9+'\n\n### 2026-11-24 영상 미보관 시연의 실행 위치 검토\n\n'+next_action,1)
    context.write_text(text,'utf-8')
handoff=ROOT/'NEW_CHAT_HANDOFF.md'
text=handoff.read_text('utf-8')
header='# 최신 검토 — 2026-11-24 외부 시연 실행·개인정보 조건 (미배포)'
if not text.startswith(header):
    handoff.write_text(header+'\n\n'+summary+'\n\n다음: '+next_action+'\n\n'+text,'utf-8')
after=json.loads(state_path.read_text('utf-8'))
assert after['current_stage']==stage
assert (ROOT/after['web_hosting_readiness_20261124']['audit']).is_file()
assert (ROOT/after['web_hosting_readiness_20261124']['report']).is_file()
print(json.dumps({'recorded':True,'operational_stage_preserved':stage,'deployed':False},ensure_ascii=False))
