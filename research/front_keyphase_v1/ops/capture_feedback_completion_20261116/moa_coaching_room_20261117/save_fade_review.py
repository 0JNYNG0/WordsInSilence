from pathlib import Path
import datetime,json,hashlib
R=Path.cwd();O=R/'research/front_keyphase_v1/ops';D=O/'capture_feedback_completion_20261116/moa_coaching_room_20261117'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
q=json.loads((D/'qa/FADE_CAPTURE.json').read_text(encoding='utf-8'));assert q['passed'] and not q['physical_camera']
text='사용자 최신 수정 반영: 가느다란 연기꼬리/구름 입자를 제거하고, MOA가 고개를 숙이기 시작할 때부터 아랫부분 주변의 넓은 네이비 페이드가 번지도록 별도 시안 수정. 아래→앞→위 비행 후 화면 밖 대기, 공간 전환 완료 뒤 오른쪽에서 재등장. 대화/메뉴는 전환 중 숨김. 실제 렌더 이미지/GIF 제시, 기존8821 UI·판정 미변경, 적용 승인 대기.'
next_action='넓은 홀로그램 페이드 시안의 사용자 확인을 반영한다. UI 승인 전8821 교체 금지. 기존 공개 판정 후보의 admission/동결해시 및 녹화 보존 HTTP 검증을 이어서 마무리하고 통과분만 연결한다. 사용자촬영은 학습·튜닝·후보선택 금지.'
rec=dict(at=now,status='HOLOGRAM_FADE_PREVIEW_AWAITING_USER_APPROVAL',summary=text,storyboard=str(D/'qa/MOA_HOLOGRAM_FADE_STORYBOARD.png'),animation=str(D/'qa/MOA_HOLOGRAM_FADE_PREVIEW.gif'),capture=str(D/'qa/FADE_CAPTURE.json'),camera_opened=False,production_changed=False,training_runs=0,private_data_read=False,files={f:hashlib.sha256((D/'site'/f).read_bytes()).hexdigest()for f in ['scene.js','hologram.js']},next_action=next_action)
(D/'HOLOGRAM_FADE_REVIEW.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
c=json.loads((D/'CHECKPOINT.json').read_text(encoding='utf-8'));c['ui_review']=rec;c['storyboard']=rec['storyboard'];c['animation']=rec['animation'];c['pending'][0]='User approval of broad holographic fade starting with bow and right-side return';(D/'CHECKPOINT.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
p=R/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8');lines=s.splitlines(keepends=True)
for n,heading,body in [(8,'2026-11-17 넓은 홀로그램 페이드 시안 수정·확인 대기',text),(9,'2026-11-17 수정 시안 확인과 공개 판정 통합',next_action)]:
 i=next(i for i,l in enumerate(lines)if l.startswith(f'## {n}.'));lines.insert(i+1,f'\n### {heading}\n{body}\n근거 {D / "HOLOGRAM_FADE_REVIEW.json"}.\n\n')
p.write_text(''.join(lines),encoding='utf-8')
s=json.loads((O/'STATE.json').read_text(encoding='utf-8'));s['next_single_action']=next_action;k=s['capture_feedback_completion_20261116'];k['next_single_action']=next_action;k['updated_at']=now;k['moa_coaching_room_20261117']=c;(O/'STATE.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
p=R/'NEW_CHAT_HANDOFF.md';s=p.read_text(encoding='utf-8');lines=s.splitlines();lines[0]='# 최신 인수인계 — '+now;lines.insert(1,text+'\n다음 한 작업: '+next_action+'\n최신 시안 근거: '+str(D/'HOLOGRAM_FADE_REVIEW.json'))
s='\n'.join(lines)+'\n';s=s.replace('MOA_FLIGHT_TRAIL_STORYBOARD.png','MOA_HOLOGRAM_FADE_STORYBOARD.png').replace('MOA_FLIGHT_TRAIL_PREVIEW.gif','MOA_HOLOGRAM_FADE_PREVIEW.gif')
s=s.replace('아랫부분 연기가 궤적에 남다가 빠르게 커질 것.','고개숙임 시작부터 넓은 홀로그램 페이드. 가는 연기꼬리는 제거. 화면 밖 퇴장 후 전환 완료 뒤 오른쪽 재등장.')
p.write_text(s,encoding='utf-8')
print('FADE_REVIEW_RECORDED; production unchanged; public candidate unchanged')
