"""Freeze the completed technical pilot; does not promote it as a teaching animation."""
from pathlib import Path
import json,hashlib,re,numpy as np
U=Path(__file__).resolve().parent;V=U/'video_only';D=U/'public_approximate_preview_body_axis'
read=lambda p:json.loads(p.read_text('utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=read(V/'EXTRACTION_COMPLETE.json')['rows'];source=[]
for r in rows:
    assert sha(Path(r['video']))==r['video_sha256'];assert sha(Path(r['raw']))==r['raw_sha256']
    source.append({k:r[k]for k in ['view','source_member','source_crc32','source_member_bytes','video','video_sha256','raw','raw_sha256','reused_existing','frames']})
pin=read(U/'STATIC_FBX_PIN.json');assert sha(Path(pin['source']))==pin['sha256']
html=U/'PREVIEW_KO_BODY_AXIS.html';urls=re.findall(r'(?:src|href)="([^"]+)"',html.read_text('utf-8'))
for url in urls:assert(U/url).is_file(),url
geo=read(V/'GEOMETRY_DIAGNOSTIC.json');bake=read(D/'BAKE_RECORD.json');fbx=read(D/'FBX_TRANSPORT_VERIFIED.json');pkg=read(D/'PACKAGE_VERIFIED.json');body=read(U/'BODY_AXIS_MAPPING_VERIFIED.json')
files=[html,U/'BODY_AXIS_BEFORE_AFTER_KO.png',D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.blend',D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.fbx',D/'APPROXIMATE_PUBLIC_WORD1567_REAL02.mp4',D/'SOURCE_AVATAR_SEQUENCE_REVIEW.jpg',V/'estimated_cameras.npz',V/'approximate_xyz.npz']
files+=list(U.glob('*.py'))+list(U.glob('*CONTRACT.json'))
ev=dict(status='TECHNICAL_PUBLIC_ONE_CLIP_PREVIEW_COMPLETE_NOT_TEACHING_READY',real_public_source_used=True,synthetic_only=False,
         clip='NIA_SL_WORD1567_REAL02',word='조상',source_video_views=5,source_frames_per_view=127,total_frame_views=635,new_rtmw_frame_views=508,front_reused_frame_views=127,
         source=source,source_archive_pin=read(V/'SOURCE_PIN.json'),static_asset=pin,static_asset_unchanged=True,
         camera_assumptions=read(V/'ESTIMATED_CAMERAS.json')['intrinsics_assumptions'],official_camera_calibration=False,
         hand_3d_official_cache_used_for_animation=False,pose_groundtruth_or_sign_correctness_claim=False,
         final_body_axis='Shoulder/hip torso axis; prior nose-axis variant preserved',
         body_axis_error_check={k:{kk:vv for kk,vv in val.items()if kk!='rows'}for k,val in body['variants'].items()},
         original_body_axis_angle_deg=body['original_nose_vs_torso_axis_angle_deg'],
         reprojection_check={v:{k:x for k,x in row.items()if k in ['arms_shoulders_wrists','hands']}for v,row in geo['left_out_view_errors_px'].items()},
         rig=dict(bones=65,driven_finger_bones=30,driven_upperarm_forearm_wrist_bones=6,terminal_finger_extensions_undriven=10,
                  fixed_torso_legs_face=True,source_required_points_missing=0,manual_source_pose_corrections=0,
                  max_finger_direction_error_deg=bake['max_hand_direction_error_deg'],max_rig_bone_length_error=bake['max_bone_length_error'],
                  wrist_unreachable_frames=bake['wrist_unreachable_frames'],synthetic_skin_test=read(U/'SYNTHETIC_CHARACTER_VERIFIED.json')),
         fbx_transport=fbx,encoded_video=pkg,
         visualization_review=dict(source_frames=[32,48,56,64,72,80,88,96,104,126],oblique_frames=[32,64,96,126],
             observed='Left/right upward alternation retained. Nose-axis height bug corrected. Cartoon-head proportion leaves forehead-related upper hand lower relative to the face.',
             thumb_orientation='Directions mapped mathematically; all-frame thumb anatomical correctness not independently annotated or established.',
             self_collision='Selected front and oblique stills inspected, no systematic all-frame mesh collision/contact certification.',
             limitations=['Unknown true calibration/absolute scale','Handshape errors from occlusion and2D detection may persist','Character proportions and contact semantics are unresolved','No expert evaluation']),
         preview_hdd_independent=True,preview_reason='HTML, copied front video, encoded avatar MP4 and figures all reside under this C drive pilot folder',
         browser_status='Static asset references and decoded MP4 checked; actual browser interaction not independently exercised by this child',
         original_assets_or_operating_models_modified=False,new_downloads=False,private_media_used=False,
         next_action='Await user choice: relation-preserving retarget for existing stylized character versus review a character with human-like proportions. No further character or semantic-IK changes performed.',
         first_attempts_preserved=['FROZEN.json original source-camera plan','initial_synthetic_diagnostic','FIRST_SYNTHETIC_FBX_ERROR.json','video_only/calibration_v1_all_body','public_approximate_preview'],
         artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))for p in files],
         primary_reference='https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html')
(U/'EVIDENCE.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2)+'\n','utf-8')
result='''공개5뷰 → 캐릭터 근사 동작: 기술 시연 1개 완료

1. 실제로 만든 것
공개 한국수어 ‘조상’의 동일 기록 REAL02를 정면·좌·우·위·아래5방향에서 사용했다. 5영상은 각각127프레임,30fps이다. 정면 관절은 원본/프레임 SHA가 일치하는 기존 RTMW를 재사용했고, 나머지4방향508프레임만 새로 추출했다.
영상의 관절 대응으로 카메라와3D를 추정하고, 기존 NPC 캐릭터에 양팔·손목·양손30개 손가락 뼈를 적용했다. 원본과 같은127프레임30fps의 MP4, Blender, FBX를 만들었다. 실제 공개 영상에서 나온 동작이며 합성 손가락 시험만 보여주는 결과가 아니다.

2. 정확한3D 복원이 확인된 것은 아니다
원본 카메라 파일은 없었다. 공통 초점거리, 영상 중앙의 주점, 왜곡0이라는 가정을 명시한 추정이다. 화면에 없는 하체의 관절 추정이 카메라 계산을 흔드는 원인을 확인해, 첫 결과를 보존하고 실제 보이는 상체·팔·손으로 카메라 추정을 제한했다.
카메라를 맞추지 않은 시점에서 한 방향을 빼고 재구성한 뒤 그 방향으로 다시 그려 비교했다. 어깨·팔·손목의 중앙 차이는 방향별 약4~9픽셀, 손은 약3~6픽셀이다. 이 값은 관측끼리의 일관성이며 실제3D 정답률이 아니다. 일부 손의 개별점은 여전히 크게 어긋난다.

3. 만들면서 고친 실제 문제
첫 합성 시험에서 왼손의 굽힘 부호가 반대였던 것을 확인해 수정했다. 양손 손바닥 측면을 별도로 그려 확인했고, 첫 결과도 보존했다.
FBX를 다시 읽을 때의 기본1프레임 오프셋 때문에 다른 순간을 비교하던 문제를 수정했다. 최종 파일은 같은 순간260개 뼈 위치 비교를 통과했다.
첫 실제 아바타에서는 손이 너무 높았다. 코가 어깨보다 앞쪽에 있다는 점을 빼먹고 코→어깨를 몸의 세로축으로 써서, 손의 앞쪽 이동이 위쪽 이동으로 섞였다. 축이 몸통축보다 중앙39.4도 기울어 있었다. 양 어깨·골반 중앙선으로 바꿨고, 단어별 위치 보정은 넣지 않았다. 기존 시도는 별도로 보존했다.

4. 현재 결과를 어떻게 봐야 하나
아래 손이 가슴 부근에 남고 양손이 번갈아 올라가는 순서는 보인다. 뼈 길이를 늘려 맞추지 않았고, 누락된 관절을 임의의 정답 자세로 채우지 않았다.
다만 이 캐릭터는 사람보다 머리가 크고 목이 짧다. 어깨 기준 위치를 정확히 옮겨도 사람의 이마 앞 동작이 캐릭터에서는 코·입 앞에 가깝게 보이는 차이가 남는다. 따라서 지금 결과를 수어 교육용 정답 애니메이션으로 사용하면 안 된다. 모든 프레임의 엄지 모양, 손 접촉, 메시 관통도 검증 완료가 아니다.

5. 바로 볼 파일
PREVIEW_KO_BODY_AXIS.html: 실제 참고 영상과 수정 아바타를 같은 프레임으로 재생/이동
BODY_AXIS_BEFORE_AFTER_KO.png: 원본·첫 시도·몸통축 수정, 실제64/96프레임
public_approximate_preview_body_axis/APPROXIMATE_PUBLIC_WORD1567_REAL02.mp4: 수정된127프레임 동작
같은 폴더의.blend/.fbx: 동작을 저장한 캐릭터 파일
이 미리보기 재생 파일은 모두 C:에 있어 HDD 없이 재생할 수 있다. 새 원본 처리에는 원본 HDD가 다시 필요하다.

6. 남은 선택
기존 캐릭터를 유지하며 얼굴·가슴 등 신체부위 관계를 보존하는 리타게팅을 추가할지, 사람과 비율이 비슷한 캐릭터를 먼저 비교할지 사용자 답변을 기다린다. 답변 전에 캐릭터를 바꾸거나 새 의미기반 IK를 적용하지 않았다.
기존 C·T3-F, 운영 웹, Unity 설정, 원본 데이터는 변경하지 않았다. 개인/멘토 영상과 봉인 기록은 사용하지 않았다. 모델을 새로 내려받거나 학습하지 않았다.

근거: EVIDENCE.json 및 각 단계 검산 JSON. 구현·검산은 Codex의 로컬 도구로 수행했고 부모 에이전트도 주요 이미지를 별도 검토했다. 전문가의 수어 정답 검토와는 다르다.
OpenCV 공식 카메라/삼각측량 문서: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
'''
(U/'RESULT_KO.txt').write_text(result,'utf-8')
(U/'CURRENT_STATUS.json').write_text(json.dumps(dict(status='TECHNICAL_ONE_CLIP_COMPLETE_AWAITING_CHARACTER_DIRECTION',running=False,report=str(U/'RESULT_KO.txt'),preview=str(html),teaching_ready=False),ensure_ascii=False,indent=2)+'\n','utf-8')
print('AVATAR_EVIDENCE_FROZEN',len(source),len(files))
