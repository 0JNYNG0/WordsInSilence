from pathlib import Path
import json, hashlib, shutil
from datetime import datetime
from zoneinfo import ZoneInfo
U=Path(__file__).resolve().parent
O=U.parent
ROOT=O.parents[2]
R=O/'word100_feedback_recovery_20261113'
def read(p):return json.loads(p.read_text('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
for filename,status in [('ROOT_ARRAY_VERIFIED.json','PASS_SAVED_ARRAY_PROTECTION_AND_METRICS'),('ROOT_BLEND_VERIFIED.json','PASS_BLEND_REREAD'),('ROOT_PACKAGE_VERIFIED.json','PASS_PACKAGE_TIMING_AND_PANEL_CONTENT')]:
    assert read(U/filename)['status']==status
assert read(U/'CURRENT_EXECUTION.json')['all_sessions_exited']
protected=[]
for item in read(R/'PROTECTED_VERIFIED.json')['files']:
    p=Path(item['path']);assert sha(p)==item['sha256'],p
    protected.append(dict(path=str(p),sha256=sha(p),unchanged=True))
write(U/'PROTECTED_MODELS_VERIFIED.json',dict(status='PASS',files=protected,no_inference=True))
write(U/'ROOT_VISUAL_REVIEW.json',dict(status='LIMITED_VISUAL_REVIEW_COMPLETE',
    actor='root Codex; no human KSL expert validation',
    inspected=['CANDIDATE_TEXTURE_CHECK.png','SOURCE_BASELINE_CANDIDATE_KO.png','ROOT_REVIEW_MOTION_SHEET.png','ROOT_VIDEO_FIRST_FRAME.png'],
    frame_samples=[0,16,32,48,64,80,88,96,104,110,118,126],
    observed=['Original Ch29 appearance is restored.','Broad raise/alternate/lower sequence visible.','Native before/after pose differences small as expected from 2 degree limit.'],
    unresolved=['Source hand at forehead appears lower near character nose/mouth.','Sparse still review cannot validate all motion or anatomical finger accuracy.','Closest-tip timings are not physical contact ground truth.']))
title='2026-11-13 기존 Ch29 Mixamo 뼈대·외형 적용 및 제한된 손목 회전 보완'
status='사용자가 가운데 기존 여자 NPC(Ch29)와 Mixamo 원래 뼈대 사용을 선택했다. 공개 조상 REAL02 1클립127프레임의 기존 5뷰 추정 움직임을 그대로 재사용했다. 원래65뼈·스킨·텍스처를 유지하고 양손목 회전만 최대2도 범위의3프레임 평균화, 회전 보간 LINEAR 후보1개를 시험했다. native 팔·손목 궤적/양손 높이 순서/손가락 local 굽힘 보존, 88~96프레임 손목 회전변화 평균 왼1.27→0.49도/오른1.84→0.53도. 전체 프레임 사이 최대회전28.48→20.71도지만15도초과칸83→84 및 손끝2쌍 최소거리 시점변경이 있어 전체동작통과/교육용완성/운영채택으로 판정하지 않는다. root 저장배열30·Blend재열기20·영상32검사완료, FBX재반입 native8255위치와505중간시점 확인. 60fps 비교MP4/PNG/FBX 완료. 얼굴부위 대응·큰손가락회전·접촉/충돌 미해결. 기존 피드백 학습6+18 미채택/G2보완 재개점, C·T3-F/운영웹/사용자자료 보호 유지.'
next_action='현재 아바타 시험은 완료되어 재렌더/재검출/동일후보 재실행하지 않는다. Ch29 Mixamo 뼈대를 유지한 채 손과 얼굴·가슴의 부위 관계 및 큰 손가락 회전 원인을 공개 원본 여러 방향과 비교하는 공통 리타게팅 설계를 다음으로 둔다. 접촉/정확3D 검증 없이 교육용 시범으로 승격하지 않는다. 피드백 본선은 word100_feedback_recovery_20261113/NEXT_PLAN_KO.txt의 공개 FIT 일반화·작은 실제이동 대조 감사부터 재개하며 완료6+18학습 재실행 금지. G2→G3→G4 순서, 사용자/멘토/봉인최종 자료·기존C/T3F/운영보존.'
stamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
targets=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',ROOT/'docs/PROJECT_DIRECTION_HISTORY_KO.txt',O/'STATE.json']
before=U/'before';before.mkdir(exist_ok=True)
for p in targets:
    dst=before/p.name
    if not dst.exists():shutil.copy2(p,dst)
s=read(O/'STATE.json')
s['next_single_action']=next_action;s['last_updated']=stamp
s[U.name]=dict(status='TECHNICAL_PREVIEW_COMPLETE_LIMITED_WRIST_GAIN_NOT_TEACHING_READY',
    at=stamp,summary=status,report=str(U/'RESULT_KO.txt'),source_clip='NIA_SL_WORD1567_REAL02',
    source_frames=127,candidate_count=1,rig='Ch29 original Mixamo65',current_execution=str(U/'CURRENT_EXECUTION.json'),
    verification=[str(U/p)for p in ['ROOT_ARRAY_VERIFIED.json','ROOT_BLEND_VERIFIED.json','ROOT_PACKAGE_VERIFIED.json','PROTECTED_MODELS_VERIFIED.json']],
    video=str(U/'SOURCE_BASELINE_CANDIDATE_60FPS.mp4'),whole_project_complete=False,training_running=False,production_changed=False)
write(O/'STATE.json',s)
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';v=p.read_text('utf-8')
for head,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',status),('## 9. 다음 단일 작업',next_action)]:
    at=v.index(head)+len(head)
    if ('### '+title)not in v[at:at+3500]:v=v[:at]+'\n\n### '+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/avatar_motion_smoothing_20261113/RESULT_KO.txt` 및 `ROOT_*VERIFIED.json`.\n'+v[at:]
p.write_text(v,'utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — '+title+'\n\n'+status+'\n\n다음: '+next_action+'\n\n현재 아바타 단위: '+str(U)+'\n보고: '+str(U/'RESULT_KO.txt')+'\n구현 avatar_multiview_review_1113, 실제파일/배열/화면 별도검산 root. 같은 Codex 검산이며 독립 사람 정답 검증 아님.\n현재 실행 없음. 렌더37091/25121 종료. 새학습/다운로드/개인자료접근/카메라/Git/운영교체 없음.\n사용자가 선택한 캐릭터는 원래 여자 NPC Ch29. 다른 모델 선호나 뼈대 재교체를 임의로 재적용하지 않는다.\n피드백 본선: '+str(R)+'; 기존 손모양6학습과 물리이동18학습은 검산완료·미채택. G2/실제G3통합 미완료. 재추론·재평가 금지.\n','utf-8')
p=ROOT/'docs/PROJECT_DIRECTION_HISTORY_KO.txt';v=p.read_text('utf-8')
if title not in v:v+='\n\n### '+title+'\n'+status+'\n방향 결정: 사용자 선택으로 새 마네킹 뼈대 교체 대신 보유 Mixamo 뼈대·스킨을 유지했다. 공개 추정 동작의 전송, 눈에 보이는 부드러움, 정확한 수어의 부위·접촉 관계를 분리해 검증했다. 제한된 손목 보완을 전체 정확도 개선으로 포장하지 않는다.\n'+next_action+'\n'
p.write_text(v,'utf-8')
execution=read(U/'CURRENT_EXECUTION.json');execution.update(status='COMPLETE_VERIFIED_LIMITED_PREVIEW',root_verification_complete=True,root_report=str(U/'RESULT_KO.txt'),next='No active process. Refer to project next_single_action; no rerun of this candidate.')
write(U/'CURRENT_EXECUTION.json',execution)
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',ROOT/'docs/PROJECT_DIRECTION_HISTORY_KO.txt']:assert title in p.read_text('utf-8')
evidence_files=['RESULT_KO.txt','IMPLEMENTATION_RESULT_KO.txt','FROZEN.json','ROOT_ARRAY_VERIFIED.json','ROOT_BLEND_VERIFIED.json','ROOT_PACKAGE_VERIFIED.json','FBX_TRANSPORT_VERIFIED.json','FBX_FRACTIONAL_DIAGNOSTIC.json','SOURCE_BASELINE_CANDIDATE_60FPS.mp4','SOURCE_BASELINE_CANDIDATE_KO.png','CANDIDATE_ORIGINAL_MATERIAL.fbx']
write(U/'CLOSEOUT_VERIFIED.json',dict(status='REPORT_STATE_HANDOFF_HISTORY_SYNCHRONIZED',at=stamp,all_sessions_exited=True,production_changed=False,whole_project_complete=False,pins={name:sha(U/name)for name in evidence_files}))
print('Avatar limited preview verified and checkpoint synchronized. No training or inference.')
