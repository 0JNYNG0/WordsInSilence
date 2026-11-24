from pathlib import Path
import json,hashlib,datetime
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
media=load(HERE/'PUBLIC_MEDIA_AUDIT.json')
cache=load(HERE/'EXISTING_3D_CACHE_CHECK.json')
docs=load(HERE/'DOCUMENT_AUDIT.json')
assert len(media['clips'])==5
assert {Path(c['member']).stem[-1] for c in media['clips']}==set('FLRUD')
assert all(c['decoded_frames']==142 and c['fps_reported']==30 for c in media['clips'])
assert cache['sha256_matches_previous_record'] and cache['finite_points']==5334
files=[ROOT/'Tools/blender_retarget_and_export.py',ROOT/'Tools/blend_animate_multiview_v2.py',ROOT/'Tools/process_multiview.py',ROOT/'Tools/s04_multiview_v2.py',ROOT/'_multiview_rebuild_v3/04_triangulation/s4_triangulation.py',ROOT/'_multiview_rebuild_v3/03_camera_geometry/s3_camera_geometry.py',ROOT/'Assets/_Project/Animations/Multiview/SIGN001_GREETING.fbx.meta',ROOT/'Packages/manifest.json']
sources=[
 {'title':'AIHub103 dataset','url':'https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103','supports':'studio videos filmed simultaneously in five views; avatar videos exist, not proof of released FBX'},
 {'title':'AIHub usage policy','url':'https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105','supports':'R&D use, attribution, training-purpose and onward-sharing conditions; derived animation distribution scope not settled here'},
 {'title':'OpenCV calibration and 3D reconstruction','url':'https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html','supports':'triangulation requires projection matrices and corresponding 2D points; undistortion/reprojection'},
 {'title':'Unity Humanoid retargeting','url':'https://docs.unity3d.com/6000.3/Documentation/Manual/Retargeting.html','supports':'configured Avatars required for Humanoid retargeting'},
 {'title':'Blender inverse kinematics','url':'https://docs.blender.org/manual/en/5.0/animation/armatures/posing/bone_constraints/inverse_kinematics/introduction.html','supports':'end effector targets drive bone chains'},
 {'title':'Mixamo FAQ','url':'https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html','supports':'Mixamo characters and animations may be used royalty-free in games; does not license AIHub inputs'},
 {'title':'EasyMocap calibration','url':'https://chingswy.github.io/easymocap-public-doc/quickstart/calibration.html','supports':'K/dist and R/T needed per camera; not installed'},
 {'title':'EasyMocap repository','url':'https://github.com/zju3dv/EasyMocap','supports':'multiview reconstruction/fitting tools; some paths depend on SMPL/MANO, not installed'},
 {'title':'EasyMocap license','url':'https://github.com/zju3dv/EasyMocap/blob/master/LICENSE','supports':'license observed during review is PRL v1.0; research/personal vs registered project-use distinction; no code adopted'}]
evidence={
 'status':'REVIEW_COMPLETE_GEOMETRIC_AND_AVATAR_EXECUTION_NOT_STARTED',
 'at_local':datetime.datetime.now().astimezone().isoformat(),
 'scope':{'public_decoded':'WORD1567 REAL01 F/L/R/U/D only','public_cached':'WORD1567 REAL02 F official_xyz only','morpheme_metadata':'WORD1133 and WORD1567 REAL01 all five views','private_media_read':False,'private_landmark_payload_read':False,'sealed_id_payload_read':False,'new_downloads':0,'new_training':0,'new_detector_inference':0,'production_edits':0,'root_context_state_edits':0,'camera_or_external_upload':False},
 'findings':[
 {'id':'synchronized_capture','state':'documented_confirmed','evidence':['AIHub103 website','construction_usage_guide.pdf PDFpage11']},
 {'id':'calibration_and_3d_provided','state':'documented_and_historical_payload_support','evidence':['guide PDFpages18,22,23,40,42','historical REAL02 record reports camparam127/127','verified preserved REAL02 hand3d cache5334/5334 finite'],'limit':'fresh camera matrix payload unavailable because historical raw ZIP path missing; no reprojection validation this turn'},
 {'id':'five_view_timing','state':'small_sample_observed','evidence':{'record_id':'REAL01','word':'WORD1567','clips':5,'resolution':[1920,1080],'fps':30,'frames_each':142,'timestamp_step_ms':1000/30,'sampled_visual_frames':[30,60,90,120]},'limit':'decoder timestamp equality and coarse visual correspondence do not establish subframe exposure synchronization'},
 {'id':'calibration_source_missing','state':'current_location_unresolved','path':'C:/Users/local-user/Downloads/수어 영상/1.Training/[라벨]02_real_word_keypoint.zip','also_searched':'W:/WordsInSilence/data/AIHub_103 filenames; no keypoint ZIP found','no_download':True},
 {'id':'old_three_view_triangulation','state':'intent_and_code_confirmed_not_metric_validation','evidence':['Tools/process_multiview.py reconstruct_3d uses body-local frontXY and side horizontal/sin45 depth','_multiview_rebuild_v3/04_triangulation/s4_triangulation.py method2.5D_weak_perspective','_multiview_rebuild_v3/03_camera_geometry/s3_camera_geometry.py custom weak-perspective reprojection','Tools/s04_multiview_v2.py not_calibrated=True'],'limit':'old scripts inspected but never executed; personal media/results not reopened; old calibrated metric triangulation success not established'},
 {'id':'rig_code_reusable_not_ready','state':'source_audited','reuse':['arm IK','finger FK','palm normal orientation','FBX baking'],'issues':['single direction lacks twist constraint in unified script','hardcoded mirror/palm flip/depth scales in v2','word-specific contact profile','current rig import mapping not validated']}],
 'recommended':{'name':'validate supplied3D/camera then offline retarget and bake','order':['recover exact public keypoint source and pair exact clip/views','validate frames,units,camera convention and reprojection on all views','use validated supplied3D; RTMW triangulation only where needed with calibrated matching views','retarget with rest-pose/local-axis/scale/IK/FK/contact constraints','review handshape,palm direction,contact,order,repetitions against public source','bake FBX/AnimationClip and play from packaged assets'],'fallback':'if calibrated evidence unavailable or fingers poor, author small vetted keypose animation with five-view reference; no invented geometric truth','first_pilot':'one public clip with correctly matching full 3D+camparam+fiveviews, preferably recovered REAL02 source; do not mix REAL01 media with REAL02 cameras','not_requested_or_done':['large3Dmodel installation','full-dataset reconstruction','avatar release','personal recording processing']},
 'license':{'mixamo':'officialFAQ allows game use of their characters/animations','aihub':'animation-asset distribution permission not established by training/R&D permission; local study and product distribution separate','model_dependencies':'no new MANO/SMPL dependency chosen','easymocap':'method documentation only; no installation/copying'},
 'primary_web_sources':sources,
 'source_code_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
 'pdf_documents':[{'path':d['path'],'sha256':d['sha256'],'pages':d['pages']} for d in docs],
 'artifacts':{'media_audit':'PUBLIC_MEDIA_AUDIT.json','document_audit':'DOCUMENT_AUDIT.json','cache_recheck':'EXISTING_3D_CACHE_CHECK.json','five_view_visual':'WORD1567_five_view_contact_sheet.jpg','guide_render':'official_guide_page23.png','first_failure':'FIRST_FAILURE.json'},
 'reviewer':'Codex subagent source/data audit; not independent human sign-language validation'
}
(HERE/'EVIDENCE.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
report='''공개 5방향 영상으로 아바타 수어 만들기 — 검토 결과
2026-11-13 / 별도 조사 / 학습·운영 변경 없음

결론
가능성을 뒷받침하는 자료가 실제로 있습니다. 우선 추천은 ‘AIHub가 제공한 3D 관절·카메라 정보를 검증한 뒤, 기존 캐릭터에 맞춰 애니메이션으로 저장’하는 방식입니다. 5개 영상의 좌표를 캐릭터에 바로 복사하는 방식은 아닙니다. 만들어 둔 애니메이션을 게임에서 재생하면 매번 5뷰 분석을 할 필요도 없습니다.

실제로 확인한 것
• AIHub 공식 설명에는 스튜디오 단어·문장을 5각도에서 동시 촬영했다고 명시돼 있습니다.
• 기존 구축활용가이드의 PDF11쪽은 카메라5대 동기화 확인,18쪽은 캘리브레이션 유,22~23·40·42쪽은 3D 관절과 카메라 파라미터 제공을 설명합니다. 따라서 ‘보정정보가 원래 없다’고 말하면 틀립니다.
• 공개 ‘조상’ WORD1567 REAL01의 정면·좌우·상하 영상을 실제 읽었습니다. 다섯 파일 모두1920×1080,30fps,142프레임이며 디코더 타임스탬프도 같습니다. 프레임30·60·90·120을 보면 같은 동작의 흐름입니다. 이것만으로 촬영 노출시점의 미세한 동기 오차까지0이라고 보장하지는 않습니다.
• 예전 공개 REAL02 ‘조상’ 좌표감사는127프레임 모두 camparam이 있다고 기록했습니다. 지금 남아있는 양손3D 캐시의 SHA256을 당시 기록과 비교했고,127×2손×21관절=5,334점의 실제3D 값이 남아있는 것도 확인했습니다. 좌표 존재와 정확도 검증은 다릅니다.
• 다만 keypoint 원본ZIP의 옛 Downloads 경로는 현재 없습니다. W: AIHub103 하위에서도 찾지 못했습니다. 해당 경로 오류를 FIRST_FAILURE.json에 남겼습니다. 새 다운로드는 하지 않았습니다.
• 이번5뷰 영상은REAL01, 남은3D 캐시는REAL02입니다. 다른 사람·촬영 세션의 영상과 카메라 값을 섞지 않습니다.

예전 3방향 시도와의 관계
사용자 말씀처럼 예전에도 삼각측량으로3D를 만들려는 시도가 있었습니다. 이번에 개인 영상은 열지 않고 계산 코드만 다시 확인했습니다. 실제 저장된 코드의 방식은 정면 좌표와 옆 영상의 가로 이동을 어깨폭으로 정규화하고45도 가정으로 깊이를 합치는 ‘2.5D 약한 원근’ 방식입니다. 다른 코드에는 not_calibrated=True가 명시돼 있습니다. 재투영 코드는 있지만 같은 단순화된 가정으로 되돌리는 검사입니다. 정확한 내부·외부 카메라 보정값을 사용한 삼각측량이 성공했다고 확인된 것은 아닙니다.

이번에는 무엇을 다르게 해야 하나
삼각측량은 ‘같은 순간의 같은 관절을 여러 카메라가 본 위치’와 ‘각 카메라의 렌즈·자세·위치’를 함께 써야 합니다. 카메라가5개라는 사실이나 대략±30도 배치만으로 정확한3D가 나오지는 않습니다. 영상끼리DTW로 임의 변형해서 맞추는 것과, 동시에 촬영된 물리적인 같은 순간을 연결하는 것도 다릅니다.

추천 경로
3D·카메라 원본 확인 → 같은5뷰에 다시 그려 검증 → 캐릭터에 맞게 변환 → 애니메이션 저장 → 게임에서 재생

1. 공개1개 클립의 keypoint 원본 위치부터 복구합니다.
같은 단어·기록ID·시점·프레임의 영상과JSON을 연결합니다. 좌표 단위, 좌우, 프레임 수, 카메라 행렬을 검사합니다. 원본을 찾지 못하면 필요한 소량을 추가 확보할지 별도 판단합니다.

2. 제공3D를 먼저 검증하고, 부족한 곳만 재계산합니다.
3D 관절을 각 카메라 화면에 다시 그려 손목·손끝·팔꿈치와 맞는지 확인합니다. 한 방향에서만 맞거나 가린 손가락의 값이 그럴듯하게 채워져 있는 것은 통과가 아닙니다. 제공3D가 쓸 만하면 전부RTMW로 다시 추출할 필요가 없습니다. 불량 구간은 검증된 카메라 정보와5뷰의RTMW 2D 관측으로 삼각측량하는 대안을 씁니다. 짧은 손모양 전환을 부드럽게 만든다는 이유로 지우지 않습니다.

3. 기존 Mixamo 캐릭터에 맞춰 변환합니다.
몸통 기준으로 위치·크기를 맞추고, 팔은 손목 목표와 팔꿈치 방향을 함께 씁니다. 손은 손바닥 방향, 손가락은 각 뼈의 기준자세와 회전축을 맞춥니다. 캐릭터 뼈 길이는 유지하면서 손-손·손-팔·손-얼굴 접촉을 보존합니다. 마지막에FBX 또는AnimationClip으로 저장합니다.

4. 수어 특징을 유지했는지 확인합니다.
원본과 아바타의 손모양·방향·접촉·순서·반복을 같은 순간에 나란히 확인합니다. 움직임이 매끈해도 수어가 달라지면 통과시키지 않습니다. 아바타 결과를 피드백AI의 정답으로 자동 사용하지 않습니다. 실제 학습용 교재로 확정할 때에는 수어 사용자 검토를 별도로 받습니다.

기존 코드를 그대로 돌려도 되나
Blender용 팔IK·손가락FK·손바닥 법선·FBX 저장 코드가 있어 활용할 수 있습니다. 다만 고정된 좌우반전, 손바닥뒤집기, 깊이배율과 특정 표현의 접촉 예외가 들어 있어 새 자료에 그대로 실행하면 안 됩니다. 또 손목에서 중지뿌리까지 한 방향만으로 손을 돌리는 코드로는 손바닥의 비틀림을 충분히 알 수 없습니다.
옛 Mixamo FBX는 손가락뼈 이름이 있으나 Unity import의 humanoid mapping은 비어 있고 animationType:2입니다. 현재Meshy 플레이어 예시는 animationType:3입니다. 실제 캐릭터의 손가락 매핑·기준자세·스킨변형을 먼저 확인해야 합니다. 이번에는FBX 애니메이션이나 개인 원본을 열거나 재생하지 않았습니다.

Mixamo를 바꿔야 하나
지금 증거로는 Mixamo 자체가 문제라는 근거가 없습니다. 손가락뼈와 스킨이 정상인 기존 캐릭터부터 검사하는 편이 효율적입니다. 손가락뼈가 부족하거나 손이 뭉개지는 스킨이면 그때 캐릭터를 바꿉니다. 캐릭터 교체가 영상의3D 추정 오류를 해결해 주지는 않습니다.

대안
카메라·3D 원본을 확보하지 못하거나 손가락 복원 품질이 낮으면, 시연용 몇 표현은5방향 영상을 참고해 핵심 손모양과 접촉 자세를Blender에서 직접 만들고 중간 움직임을 연결하는 방법이 더 예측 가능합니다. 일반 영상→모션AI를 새로 들여오는 것을 우선하지 않습니다. EasyMocap도 보정 정보가 필요하며 전체 몸·손 복원 경로의 일부는SMPL/MANO에 의존합니다.

속도와HDD
애니메이션 제작은 개발 중 한 번 하는 오프라인 작업입니다. 게임에는 완성된 캐릭터·애니메이션을 포함해 재생하므로 아바타 재생 자체에는HDD나5뷰 분석이 필요 없습니다. 피드백 모델과 참고 자산 전체의HDD 독립성은 별도 패키징 시험으로 확인해야 합니다.

이용 범위
Adobe 공식FAQ는 Mixamo 캐릭터·애니메이션의 게임 사용을 허용한다고 안내합니다. AIHub 원천 데이터의 이용 조건은 별개입니다. AIHub는 연구개발 활용·출처표시와 함께 학습 목적 및 제3자 제공 제한을 명시합니다. 수어 애니메이션 자산으로 전환해 게임에 배포하는 범위까지 이번에 확인됐다고 말하지 않습니다. 로컬 연구 검토와 실제 배포 허용을 구분합니다. 이번에 외부 전송·업로드·새 모델 설치·출시는 없습니다.

완료한 범위와 남은 범위
자료의 존재, 문서 근거,5뷰의 기본 대응, 남아있는3D 캐시, 기존 리타게팅 코드까지 검토했습니다. 같은 클립의3D+카메라+5뷰 재투영, 캐릭터 손가락 품질, 수어 전달성 검증은 아직 하지 않았습니다. 이번 결과는 새 애니메이션 완성이 아닙니다.
C·T3-F, 웹, 기존 학습 결과, Unity 설정, 원본ZIP를 수정하지 않았습니다. 사용자·멘토 촬영은 읽지 않았습니다. 담당은Codex 하위 에이전트이며 독립 사람 검수는 아닙니다.

근거
• AIHub103: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103
• 기존공식PDF: W:/WordsInSilence/data/AIHub_103/docs/construction_usage_guide.pdf (PDF11,18,22,23,40,42쪽), data_description.pdf (6~10쪽)
• OpenCV: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
• Unity: https://docs.unity3d.com/6000.3/Documentation/Manual/Retargeting.html
• Blender: https://docs.blender.org/manual/en/5.0/animation/armatures/posing/bone_constraints/inverse_kinematics/introduction.html
• Adobe: https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html
• AIHub정책: https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105
• EasyMocap: https://chingswy.github.io/easymocap-public-doc/quickstart/calibration.html

세부 소스경로·해시·확인/미확인 구분: EVIDENCE.json
5뷰 비교사진: WORD1567_five_view_contact_sheet.jpg
'''
(HERE/'RESULT_KO.txt').write_text(report,encoding='utf-8')
# Remove only this audit's unused blank contact sheet; no data or other run output.
unused=HERE/'WORD1133_five_view_contact_sheet.jpg'
if unused.exists(): unused.unlink()
review={'status':'LOCAL_CHECKS_PASS_WITH_UNRESOLVED_RAW_CAMERA_SOURCE','checks':['five actual public FIT01 videos decoded','each source copy hash verified againstZIP payload','all five framecounts142/FPS30/PTSconsistent','official PDFcamparam page23 rendered and visually reviewed','preserved3D cache matches previousSHA256','REAL01 media and REAL02 cached3D identity scopes explicitly separated','no animation/model training or rig execution'],'not_independent_human_validation':True,'limitations':['raw camera payload missing at historical location','exact subframe sync and3D reprojection not measured','UnityAvatar/surface deformations not executed']}
(HERE/'VERIFIED.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'result':str(HERE/'RESULT_KO.txt'),'evidence':str(HERE/'EVIDENCE.json'),'status':evidence['status']},ensure_ascii=False))
