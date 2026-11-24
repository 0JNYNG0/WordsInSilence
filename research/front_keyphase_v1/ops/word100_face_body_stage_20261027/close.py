from pathlib import Path
import json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[4];LAB=ROOT/'research/front_keyphase_v1';U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
at=datetime.datetime.now().astimezone().isoformat();unit=U.name
f=read(U/'FROZEN.json');v=read(U/'VERIFIED.json');scope=read(U/'FIXED_SCOPE_VERIFIED.json');ui=read(U/'HTML_RENDER_VERIFIED.json');rg=read(U/'REGION_CHECKS.json')
assert v['status']=='PASS_SAVED_DATA_AND_INVARIANTS' and scope['status'].startswith('PASS') and ui['status']=='PASS'
for p,h in f['protected'].items():assert sha(p)==h,p
assert sha(ROOT/'Tools/models/face_landmarker.task')==f['face_model']['sha256']
visual=dict(status='REVIEWED_BY_CODEX_NOT_HUMAN_GROUND_TRUTH',
 viewed=['VISUAL_CONTACT_SHEET.jpg (all36 previews)','images/NIA_SL_WORD0004_REAL01_F_0.jpg',
         'images/NIA_SL_WORD0949_REAL02_F_1.jpg','images/NIA_SL_WORD1139_REAL03_F_0.jpg',
         'images/paired_WORD0949.jpg','QA_DESKTOP.png','QA_FACE.png'],
 findings=['36 selected face overlays show plausible surface-region locations, not measured landmark accuracy',
           'Face points remain predicted behind hand overlaps; no occluded finger/face truth claim',
           'Paired middle frame of public WORD0949 shows B matching the forward finger stage more plausibly than A; one illustrative inspection only',
           'Original body/hand points unchanged inside ROI; no claim earlier detector mistakes fixed'],
 limitations=['Not all2285frames visually annotated','Cheek patch is approximate surface vicinity, not exact bone/contact location',
              'Hand hull warning misses occluders if hand detector itself misses them','No real webcam test or certified sign feedback'])
write(U/'VISUAL_REVIEW.json',visual)
summary=("얼굴 부위 연결 및 공개 DTW 관측범위 검증 완료(연구 후보, 운영 미적용). MediaPipe FaceLandmarker478점을 기존75점 몸/손 입력과 분리 저장하고 이마·좌우 광대부근·입·턱·코·목·가슴과 각5손끝 관계를 계산. 공개6표현×TRAIN ID01/02/03=18영상2285프레임. 기본전체화면얼굴0/2285→현재Pose머리영역재확인2285/2285; 전역좌표복원,몸/손검출범위변경없음. 검출률이지정확도아님. 얼굴추론추가 중앙값의중앙값9.76ms/프레임(디코딩/TCN/DTW제외). 손-얼굴투영겹침84프레임-부위경고,실제접촉확정0/표정평가0. 얼굴정보는정렬후부위비교에연결,정렬비용의얼굴가중치추가는하지않음. 시간만늘린공개동일영상18개 A원본참고/상반신query 평균대응358.34ms→B양쪽상반신0.23ms;실제속도/다른사람정확도아님. 다른공개사람12쌍 동일주석528동작프레임에서참고준비마무리연결96→45(B),창밖누락0;내부세부단계정답아님. 추가시작끝제한C는별REAL01주석49중6제외로미채택(기존TCN C와별개). 손소실분모유지/이웃손끝보존/좌표이동크기회전불변/PTS불일치거부18영상검산,216부위중심·작은이동합성점검통과,36장면시각검토,40이미지데스크톱모바일검증. 32운영보호핀/원본유지. 기존C/T3F/웹/DTW운영변경·학습·사용자/멘토접근0. 같은Codex별도검산,독립사람정답아님. 최초기본얼굴실패보존·공통머리ROI복구규칙별도기록.")
next_task=("word100_face_body_stage_20261027의완료얼굴/정렬/검산을재사용. 얼굴부위기준은연구진단이며practice웹에는아직미연결. B(전체몸손검출뒤양쪽동일ROI)를기반으로공개사람간남은45주석동작-준비마무리연결을직접검토하여관측누락/동작시점문제를분리하고,정상내림·교차·짧은동작보존범위내시간정렬보완. 추가시작끝자르기C는주석잘림으로미채택. 얼굴은손끝과부위관계의정렬후진단으로사용하고,가림추정/부위경계/2D접촉불확실성을정답으로승격하지않음. 정상부위내차이와부위변경을검증한후일관된실사용입력및별도얼굴표시웹후보연결을계획. 표정/실제접촉/수어정오·교정문장아직미구현. 후보측정과실제언어적허용범위구분. 기존C/T3F/운영보존·사용자자료학습튜닝선택금지;이번요청으로TCN재학습없음. 활성실행없음.")
st=read(LAB/'ops/STATE.json');st['updated_date_kst']=at;st['current_stage']='FACE_BODY_REGION_DIAGNOSTICS_AND_PUBLIC_SCOPE_ALIGNMENT_VERIFIED_RESEARCH_ONLY'
st[unit]=dict(status='IMPLEMENTED_PUBLICLY_VERIFIED_RESEARCH_ONLY',at=at,summary=summary,
 result=str(U/'RESULT_KO.html'),next=next_task,
 face_counts={'public_clips':18,'frames':2285,'full_frame_found':0,'head_roi_recovery_found':2285},
 coarse_fixed_scope=read(U/'FIXED_SCOPE.json')['summary'],source_preservation=v['checks'],
 operating_changed=False,trained=False,user_or_mentor_access=False,
 implementation='Codex',verification='Same Codex separate saved-data/scalar checks, not independent human annotations',active_execution=None)
write(LAB/'ops/STATE.json',st)
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf-8-sig')
title='### 2026-10-27 얼굴 부위 관측 추가 및 공개 동작 시점 비교'
if title in text:raise RuntimeError('Context already closed')
for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_task)]:
 assert heading in text
 text=text.replace(heading,heading+'\n\n'+title+'\n'+at+'\n'+body+'\n근거: research/front_keyphase_v1/ops/'+unit+'/RESULT_KO.html, VERIFIED.json, FIXED_SCOPE_VERIFIED.json, REGION_CHECKS.json, VISUAL_REVIEW.json.\n',1)
p.write_text(text,encoding='utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음 단일 작업: '+next_task+'\n\n구현: src/wis_lab/capture/face_landmarks.py, core/face_body_relations.py(별도연구입력). 공식모델 Tools/models/face_landmarker.task/source.json. 근거: research/front_keyphase_v1/ops/'+unit+'/RESULT_KO.html 및 FROZEN/RECOVERY_PROTOCOL/VERIFIED/FIXED_SCOPE_VERIFIED/REGION_CHECKS/VISUAL_REVIEW/HTML_RENDER_VERIFIED. 모든실행종료,활성세션없음.\n',encoding='utf-8')
note='\n보완 설명: 얼굴 특징은 DTW 정렬 후 부위 비교에 연결했으며 이번 정렬 개선은 양쪽 관측범위 일치 효과다. 다른 사람 12쌍의 동일528주석동작프레임 중 참고 준비·마무리 연결96→45, 추가누락0. 216개 부위중심/작은이동 통제점 구분은 기하 점검일 뿐 언어적 허용범위 검증이 아니다. 손 검출 자체가 없으면 얼굴 가림 경고도 놓칠 수 있다.\n'
with (U/'RESULT_KO.txt').open('a',encoding='utf-8') as fh:fh.write(note)
outputs={str(p):sha(p) for p in U.glob('*.json')}
codepaths=[U/x for x in ('run.py','verify.py','verify_regions.py','report.py','pair_view.py','check_report.cjs')]+[LAB/'src/wis_lab/capture/face_landmarks.py',LAB/'src/wis_lab/core/face_body_relations.py']
write(U/'CLOSURE.json',dict(at=at,status='COMPLETE_BOUNDED_PUBLIC_VALIDATION_NOT_WEB_DEPLOYED',outputs=outputs,
 code={str(p):sha(p) for p in codepaths},docs={str(p):sha(p) for p in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',LAB/'ops/STATE.json')},
 protected_unchanged=len(f['protected']),active_execution=None,training=False))
print('CLOSED',unit)
