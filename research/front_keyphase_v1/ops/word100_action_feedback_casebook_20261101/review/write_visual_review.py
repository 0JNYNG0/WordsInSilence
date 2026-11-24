"""Serialize same-Codex qualitative observations; inspect stored masks only.

No image/model inference, feature extraction, alignment, score or label changes.
The observations below were made by viewing all 30 fixed chronological pages,
plus five explicitly listed original full RGB frames, before this script ran.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

selection = read(HERE/"WINDOW_SELECTION.json")
manifest = read(HERE/"RGB_MANIFEST.json")
ack = read(HERE/"WINDOW_ACK.json")
inputs = read(HERE.parent/"INPUTS.json")
assert sha(HERE/"WINDOW_SELECTION.json") == "e4a406079828ba530808ae19b00634e2b5d495eb1c6cf5596a47c4e8754e7d67"
assert manifest["selection_sha256"] == sha(HERE/"WINDOW_SELECTION.json")
assert manifest["ack_sha256"] == sha(HERE/"WINDOW_ACK.json")
assert len(selection["windows"]) == 10

# All intervals below describe visible projected RGB events, not new GT labels.
# Frames are original native zero-based frame IDs; intervals are inclusive.
notes = {
"W01": {
 "q": [([64,67],"오른손의 편 긴 손가락 윤곽이 턱 부근에서 코·입 앞으로 올라온다."),
       ([68,87],"오른손의 긴 손가락이 모여 펴진 윤곽을 얼굴 앞에 둔다. 손 일부가 얼굴을 가리며 왼손은 몸 옆 아래에 보인다."),
       ([88,88],"오른손이 내려가며 닫히는 다음 동작으로 넘어간다.")],
 "r": [([43,46],"오른손의 편 긴 손가락 윤곽이 가슴 위쪽에서 코·입 앞으로 올라온다."),
       ([47,60],"오른손의 긴 손가락을 모아 편 윤곽이 얼굴 앞에 있다. 왼손은 몸 옆 아래에 보인다."),
       ([61,61],"오른손이 내려가며 닫히기 시작한다.")],
 "visible": "두 원본에서 얼굴 앞에 오른손의 긴 손가락을 모아 펴는 큰 구조가 보인다. 화면상 손 기울기와 유지 길이는 같지 않지만 손가락별 정답 일치를 확인한 것은 아니다.",
 "missing": "두 영상의 몸 옆 왼손 윤곽은 보이지만 해당 창의 저장 왼손 14각도와 2D축은 모두 missing이다. 오른손 저장 지원은 있어도 정확한 관절 위치의 참값을 뜻하지 않는다.",
 "correspondence": "저장 경로는 q70–76을 r49에 연결한다. 양쪽 큰 자세는 비슷하지만 여러 질의 시점을 한 참고 시점에 연결하므로 유지시간 차이와 자세 차이를 분리해야 한다.",
 "pairs": [(70,49),(76,49),(85,58)],
 "causes": ["coarse_shape_similarity_observed", "hold_duration_or_projection_difference_possible", "left_stored_observation_missing_confirmed", "fine_finger_difference_unresolved"],
 "limits": "왼손 세부, 각 손가락 관절각의 정확한 일치, 실제 손바닥 방향은 확정하지 않는다."
},
"W02": {
 "q": [([85,87],"얼굴 앞에 편 오른손이 남아 있다."),
       ([88,94],"오른손을 닫으며 얼굴 앞에서 가슴 앞으로 내린다."),
       ([95,103],"닫힌 오른손이 가슴 앞에 있고 팔꿈치가 바깥에 놓여 전완이 대체로 가로로 보인다."),
       ([104,106],"닫힌 오른손과 전완이 다시 어깨 쪽으로 올라가며 자세를 바꾸기 시작한다.")],
 "r": [([58,60],"얼굴 앞에 편 오른손이 남아 있다."),
       ([61,63],"오른손을 닫으며 가슴 쪽으로 내린다."),
       ([64,71],"닫힌 오른손이 가슴 앞에 있고 팔꿈치가 바깥에 놓여 전완이 대체로 가로로 보인다."),
       ([72,73],"다음 전완 방향 변화로 넘어가는 경계가 보인다.")],
 "visible": "두 원본 모두 편 손에서 닫힌 손으로 바꾸면서 아래로 옮기는 흐름이 보인다. 전환 속도와 가슴 앞 자세가 다르며, 이 창에서 큰 순서의 삭제는 보이지 않는다.",
 "missing": "왼손 윤곽은 아래에 보이지만 저장 왼손 각도/축이 전 구간 missing이다. 닫힌 오른손의 손가락 내부는 자기 가림 때문에 모두 확인할 수 없다.",
 "correspondence": "q88은 r61과 r62에, q95–97은 r69에 연결되어 있다. 시작 전환과 후반 유지가 한 부모 안에 있으므로 수치 차이를 동작 오류 하나로 귀속할 수 없다.",
 "pairs": [(88,61),(88,62),(95,69),(97,69)],
 "causes": ["closing_and_descent_sequence_observed_in_both", "tempo_and_projected_posture_difference_possible", "left_stored_observation_missing_confirmed", "closed_finger_self_occlusion"],
 "limits": "주먹 세부, 접촉, 교육적으로 필수인 속도나 순서는 확정하지 않는다."
},
"W03": {
 "q": [([99,103],"닫힌 오른손이 가슴 앞에 있고 전완이 대체로 가로로 놓인다."),
       ([104,110],"닫힌 오른손이 어깨 쪽으로 올라가며 전완의 화면 방향이 바뀐다."),
       ([111,129],"닫힌 손을 어깨 옆 높이에 두고 전완을 세운 자세가 이어진다."),
       ([130,132],"손이 조금 내려가는 후속 경계가 보인다.")],
 "r": [([71,73],"닫힌 오른손을 가슴 앞에 두고 다음 변화로 넘어간다."),
       ([74,78],"닫힌 오른손이 어깨 쪽으로 올라가며 전완 방향이 바뀐다."),
       ([79,91],"어깨 옆의 닫힌 손 자세가 이어진다."),
       ([92,94],"조금 내려가는 후속 경계가 보인다.")],
 "visible": "가슴 앞 가로 전완에서 어깨 쪽 세운 전완으로 바뀌는 큰 흐름은 양쪽에 있다. 닫힌 손 윤곽과 팔의 화면 기울기에는 차이가 보인다.",
 "missing": "몸 옆 왼손은 보이지만 각도와 축의 저장 지원은 없다. 오른손 내부 손가락 및 3D 표면 방향은 윤곽만으로 분리할 수 없다.",
 "correspondence": "경로가 q110–113을 r76에 연결한다. q111 이후의 비교적 머무르는 모습과 r76의 변화 중 모습이 함께 연결될 수 있어 시간 대응과 자세 차이가 공존할 가능성이 있다.",
 "pairs": [(106,73),(110,76),(113,76),(126,89)],
 "causes": ["projected_forearm_reorientation_observed", "transition_vs_hold_correspondence_possible", "left_stored_observation_missing_confirmed", "3d_rotation_and_wrist_bend_unresolved"],
 "limits": "이 화면 변화가 실제 주먹 3D 회전인지 손목 꺾임인지 분해하지 않으며 미세 흔들림으로 삭제하지도 않는다."
},
"W04": {
 "q": [([71,74],"편 오른손의 모인 긴 손가락이 반대팔 어깨·윗부분 쪽으로 이동한다. 왼손은 앞쪽에 닫힌 윤곽으로 보인다."),
       ([75,86],"오른손이 반대팔 위쪽 근처에서 팔을 따라 손목 쪽으로 진행한다. 왼전완은 화면에서 비교적 짧고 앞으로 뻗은 투영으로 보인다."),
       ([87,95],"오른손이 반대팔의 더 먼 쪽에 놓인다. 팔 높이와 기울기가 참고와 다르게 보인다.")],
 "r": [([44,47],"편 오른손이 반대팔 위쪽으로 올라가고 닫힌 왼손이 앞에 보인다."),
       ([48,59],"오른손이 반대팔 위쪽에서 손목 쪽으로 진행한다. 왼전완은 화면에 더 길게 가로·사선으로 보인다."),
       ([60,65],"오른손이 반대팔의 더 먼 쪽 가까이에 놓인다.")],
 "visible": "양쪽 모두 편 오른손과 닫힌 왼손의 큰 윤곽, 반대팔을 따라 이동하는 흐름이 보인다. 왼팔의 화면 높이·기울기·길이 투영에는 실제 영상상의 차이가 있다.",
 "missing": "질의 f72–74의 왼손 닫힌 윤곽은 보이지만 저장 왼손 14각도와 2D축은 모두 missing이다. f73 원본 전체 RGB도 별도로 확인했다. 따라서 이 누락을 손이 없거나 무의미한 동작으로 해석할 수 없다.",
 "correspondence": "q74–77은 r49, q78–80은 r50에 연결된다. 둘 다 같은 큰 팔 위 진행이지만 접근·유지·미끄러지는 시점과 팔의 투영 차이가 동시에 남을 수 있다.",
 "pairs": [(72,47),(73,48),(74,49),(77,49),(80,50),(93,63)],
 "causes": ["projected_forearm_posture_difference_observed", "left_rgb_visible_but_stored_missing_confirmed", "temporal_correspondence_contribution_possible", "contact_and_fine_finger_state_unresolved"],
 "limits": "오른손이 팔 가까이 겹쳐 보이는 것과 실제 접촉은 구별한다. 팔 차이가 허용되는 표현 변이인지 교육 오류인지는 미확정이다."
},
"W05": {
 "q": [([56,59],"오른손이 낮은 곳에서 올라오며 닫힌 윤곽과 엄지가 드러난다. 왼손은 몸 앞 아래쪽에 있다."),
       ([60,64],"엄지가 위로 드러난 닫힌 오른손이 올라가며 왼손보다 높아진다."),
       ([65,67],"오른손이 얼굴 높이 위쪽에 머물고 왼손은 아래에 있다.")],
 "r": [([48,51],"오른손이 올라오며 엄지를 드러낸 닫힌 윤곽이 보인다. 왼손은 낮은 곳에 있다."),
       ([52,54],"오른손은 얼굴 아래쪽 높이에 있고 왼손은 아래에서 올라오기 시작한다."),
       ([55,57],"왼손이 올라와 두 손의 화면상 높이가 가까워지고 겹치는 경계가 보인다.")],
 "visible": "두 원본 모두 엄지가 드러난 닫힌 손의 큰 모양이 보인다. 참고의 왼손 시작 높이와 질의의 왼손 시작 높이, 오른손이 도달하는 얼굴 기준 높이는 같지 않다.",
 "missing": "질의 오른손 f61과 참고 오른손 f51은 윤곽이 보이는데 저장 14각도/축이 없다. 참고 왼손은 낮은 위치·교차 근처에서 누락 또는 일부 각도 지원만 있다.",
 "correspondence": "저장 core는 q61–64와 r51–54를 순서대로 연결한다. 오른손 상승이라는 큰 흐름은 대응 가능하지만 도달 높이와 왼손 진행은 다르다.",
 "pairs": [(61,51),(64,54)],
 "causes": ["thumb_visible_closed_outline_in_both", "height_and_timing_difference_observed", "intermittent_stored_observation_missing", "finger_internal_state_unresolved"],
 "limits": "엄지 윤곽 확인이 주먹의 모든 관절이나 사전 표현형 확정을 뜻하지 않는다."
},
"W06": {
 "q": [([64,69],"오른손이 얼굴 앞 높은 위치, 왼손이 아래에 있다. f68 원본 전체 RGB에서 오른손이 위임을 확인했다."),
       ([70,71],"양손 높이가 가까워져 겹쳐 보이며 위아래가 바뀌는 경계가 나타난다."),
       ([72,77],"왼손이 높고 오른손은 아래로 간다."),
       ([78,79],"다음 위아래 교환을 향해 움직이기 시작한다.")],
 "r": [([54,56],"오른손이 높은 상태에서 왼손이 올라와 두 손이 가까워진다."),
       ([57,57],"두 손이 가까이 겹치는 화면상 경계가 보인다."),
       ([58,62],"왼손이 위, 오른손이 아래에 있다. f58 원본 전체 RGB에서 확인했다."),
       ([63,63],"다음 위아래 교환으로 넘어가기 시작한다.")],
 "visible": "저장 경로의 대표 (질의68, 참고58)는 질의 오른손이 위인 모습과 참고 왼손이 위인 모습을 연결한다. 이는 선택된 원본에서 보이는 서로 다른 진행 시점이다.",
 "missing": "양쪽 대표 f68/f58의 손 각도·축은 저장되어 있어 이 대표 대응 차이를 단순히 missing 하나로 설명할 수 없다. 주변 참고 f54/56/57 왼손과 질의 f71 오른손 등에는 저장 누락이 공존한다.",
 "correspondence": "고정 경로의 (68,58),(69,59),(69,60)는 서로 다른 위아래 관계의 시점을 연결한다. 양 원본의 전체 교환 흐름은 존재하므로 이 대응 차이를 잘못 배운 동작이나 반대 순서 수행으로 승격하지 않는다.",
 "pairs": [(67,57),(68,58),(69,59),(69,60)],
 "causes": ["different_visible_subphase_connected_by_saved_path", "surrounding_stored_observation_missing", "crossing_self_occlusion_and_timing_difference", "educational_correctness_unresolved"],
 "limits": "여기서 입증되는 것은 저장 대응의 시점 차이이다. 교육 오류 탐지 성능이나 자연 교차 전체의 처리 성공은 아니다."
},
"W07": {
 "q": [([69,71],"오른손 위에서 양손이 가까워지고 왼손 위로 넘어간다."),
       ([72,78],"왼손 위·오른손 아래 자세가 이어진다."),
       ([79,81],"두 손이 다시 가까워지며 화면상 위아래가 교환된다."),
       ([82,89],"오른손이 다시 위에 놓인다."),
       ([90,92],"다음 교환을 향해 손들이 가까워진다.")],
 "r": [([60,62],"왼손이 위, 오른손이 아래에 있다."),
       ([63,65],"손들이 가까워지며 오른손이 위로 바뀐다."),
       ([66,70],"오른손 위·왼손 아래 자세가 이어진다."),
       ([71,72],"다음 교환을 향해 가까워진다.")],
 "visible": "양쪽 모두 양손의 위아래가 교대로 바뀌는 흐름이 보이며 유지 길이가 다르다. 좁은 한 참고 시점에 긴 질의 구간이 연결되면 이동과 유지가 함께 포함된다.",
 "missing": "참고 f63/f66과 질의 f71/f82의 오른손 각도·축 누락이 있다. 교차 근처에는 일부 손 윤곽이 다른 손·팔과 겹쳐 모든 관절을 분리할 수 없다.",
 "correspondence": "q71–79 전체가 r63 하나에 연결된다. q71의 전환, q72–78의 왼손 위 유지, q79의 다음 변화가 한 참고 시점에 모여 있으므로 세부 시점 대응은 미확정이며 큰 교대 흐름만 확인된다.",
 "pairs": [(71,63),(75,63),(79,63),(80,64),(86,69)],
 "causes": ["alternating_height_flow_observed_in_both", "many_query_times_one_reference_time_confirmed", "intermittent_stored_observation_missing", "exact_phase_correspondence_not_resolved"],
 "limits": "반복 횟수의 언어적 필수성, 특정 멈춤 길이의 정답 여부를 새로 정하지 않는다."
},
"W08": {
 "q": [([69,74],"양손이 낮은 곳에서 올라오며 서로 가까워진다."),
       ([75,87],"손가락이 비스듬히 모여 얼굴 아래·앞에 삼각형 같은 빈 공간을 만든 윤곽이 보인다."),
       ([88,92],"가까운 자세를 풀고 양손이 벌어지며 낮아진다.")],
 "r": [([27,31],"양손이 올라오며 서로 가까워진다."),
       ([32,39],"손가락이 비스듬히 모여 삼각형 같은 빈 공간을 만든 윤곽이 보인다."),
       ([40,41],"가까운 자세를 풀며 양손이 벌어지기 시작한다.")],
 "visible": "두 원본에서 양손을 모아 삼각형 같은 공간을 만들었다가 푸는 큰 흐름이 보인다. 손의 얼굴 기준 높이·간격과 유지 길이가 동일하다는 뜻은 아니다.",
 "missing": "질의 f70의 양손과 f72 오른손, 참고 f28–29 오른손은 올라오는 윤곽이 보이나 저장 각도·축이 누락된다. core 후반에 저장 지원이 있어도 접촉은 확인되지 않는다.",
 "correspondence": "q71–72가 r29에, q80–83이 r36에 연결된다. core에 접근과 유지가 모두 포함되므로 한 부모 전체를 하나의 정지 모양으로 볼 수 없다.",
 "pairs": [(71,29),(72,29),(80,36),(83,36),(89,39)],
 "causes": ["coarse_two_hand_geometry_observed_in_both", "approach_hold_boundary_and_duration_difference", "stored_observation_missing_during_approach", "contact_unresolved"],
 "limits": "끝끼리 가까워 보이는 것은 실제 손가락 접촉의 확정이 아니다."
},
"W09": {
 "q": [([86,88],"중앙에서 가까운 양손 자세가 남아 있다."),
       ([89,94],"양손을 벌리고 조금 낮추며 손의 투영 방향이 평평해진다."),
       ([95,96],"분리된 양손 자세로 머무른다.")],
 "r": [([39,40],"중앙에서 가까운 자세를 풀기 시작한다."),
       ([41,43],"양손을 벌리고 낮추며 손의 투영 방향이 평평해진다."),
       ([44,45],"분리된 양손 자세로 머무른다.")],
 "visible": "양쪽 모두 가까운 양손을 벌리고 낮추는 전환이 있다. 세부 시작 시점·속도·팔 자세 차이는 보이지만 전환 자체가 빠졌다고 볼 근거는 없다.",
 "missing": "이 고정 context에서는 양손 14각도와 2D축 마스크가 모두 지원된다. 이는 측정치의 정확도나 동작의 정답 여부를 보증하지 않는다.",
 "correspondence": "q91–92가 r41에, q93이 r42에, q94가 r43에 연결된다. 큰 벌어짐 흐름은 대응 가능하나 정확히 같은 진행률인지는 정하지 않는다.",
 "pairs": [(91,41),(92,41),(93,42),(94,43)],
 "causes": ["separation_and_lowering_flow_observed_in_both", "tempo_or_projected_pose_difference_possible", "no_missing_for_selected_angle_axis_masks", "fine_correspondence_unresolved"],
 "limits": "기존 물리 주석의 준비·core 구별과 사전 필수 단계는 별개다."
},
"W10": {
 "q": [([94,108],"양손을 떨어뜨린 채 얼굴 아래 높이에 둔다. 긴 손가락들이 모인 평평한 윤곽이 측면·끝 방향으로 보인다."),
       ([109,110],"조금 내려가는 후속 경계가 보인다.")],
 "r": [([43,55],"양손을 떨어뜨린 채 얼굴 아래 높이에 둔다. 모인 긴 손가락의 평평한 윤곽이 측면·끝 방향으로 보인다."),
       ([56,57],"조금 내려가는 후속 경계가 보인다.")],
 "visible": "분리된 양손의 납작한 윤곽은 양쪽에서 확인된다. q100과 r50 원본 전체 RGB도 확인했지만 겹친 손가락·엄지 안쪽 세부는 여전히 확정할 수 없다.",
 "missing": "이 창에서는 양손 각도와 축 마스크가 모두 지원된다. 영상에서 안 보이는 세부를 저장 각도만으로 참값으로 채우지 않는다.",
 "correspondence": "q97–98이 r46, q107–108이 r55에 연결되는 유지 구간이다. 큰 자세의 대응 가능성은 있으나 손가락별 일치나 사전의 끝부분 동작 완수는 미확정이다.",
 "pairs": [(96,45),(100,48),(102,50),(108,55)],
 "causes": ["coarse_flat_grouped_finger_outline_observed", "edge_on_finger_occlusion", "no_missing_for_selected_angle_axis_masks", "linguistic_suffix_unverified"],
 "limits": "이 관측으로 이전 unverified_edge_on 언어 설명을 확정 상태로 승격하지 않는다."
}
}

cases_text = {
"P036": ("순종은 얼굴 앞의 편 손→가슴 쪽 닫힌 손→어깨 옆 자세라는 큰 흐름이 두 영상에 보입니다. 시간 연결과 팔 기울기 차이, 몸 옆 왼손의 저장 누락을 따로 남겼습니다.",
 "주먹 내부·실제 3D 회전·손목 꺾임과 교육적 옳고그름은 확정하지 않았습니다."),
"P048": ("잘하다는 편 오른손이 반대팔을 따라 가는 흐름은 양쪽에 있지만 팔 높이와 화면 기울기가 다릅니다. 질의 f72–74는 왼손이 보이는데 저장 각도·축이 빠져 있어 관측 누락을 별도로 확인했습니다.",
 "실제 접촉 여부, 세부 손가락 모양과 표현 변이의 허용 여부는 미확정입니다."),
"P066": ("조상은 양손 높이를 교대로 바꾸는 흐름이 두 영상에 있습니다. 저장 대응 (질의68, 참고58)은 오른손이 위인 시점과 왼손이 위인 시점을 연결하므로, 그 차이를 학습자의 동작 오류로 바로 읽으면 안 됩니다.",
 "교차 주변 관측 누락도 함께 있으며 반복 횟수나 멈춤 길이의 언어적 정답은 정하지 않았습니다."),
"P072": ("이집트는 양손을 모아 삼각형 같은 공간을 만들고 벌려 낮추는 흐름이 양쪽에 보입니다. 끝의 납작한 손 윤곽은 보이지만 측면으로 겹친 손가락 세부는 여전히 불확실합니다.",
 "접촉과 사전 끝부분 동작의 충족 여부는 확정하지 않았고 이전 미확정 설명을 승격하지 않았습니다.")
}

full_viewed = {("P048",73),("P066",68),("P063",58),("P072",100),("P069",50)}
pins = dict(selection["source_pins"])
for name in ["WINDOW_SELECTION.json","WINDOW_ACK.json","RGB_MANIFEST.json","prepare_rgb_review.py"]:
    pins[str(HERE/name)] = sha(HERE/name)
pins[str(HERE.parent/"INPUTS.json")] = sha(HERE.parent/"INPUTS.json")
for x in manifest["full_frames"]:
    pins[x["image"]] = x["image_sha256"]
for x in manifest["pages"]:
    pins[x["file"]] = x["sha256"]
bad_pins = [{"path":p,"expected":h,"actual":sha(p)} for p,h in pins.items() if sha(p)!=h]
assert not bad_pins, bad_pins
npzs = {}
def mask_records(axis):
    p = axis["input"]
    if p not in npzs:
        z = np.load(p)
        npzs[p] = {k:z[k] for k in ["x","mask","frame_id","timestamp_us"]}
    z = npzs[p]
    assert z["x"].shape[1] == 69
    result = []
    for f,t in zip(axis["context_frame_ids"],axis["context_timestamp_us"]):
        idx = int(np.where(z["frame_id"]==f)[0][0])
        assert int(z["timestamp_us"][idx])==t
        mask = z["mask"][idx]>0
        hands = {}
        for side,off in [("L",0),("R",28)]:
            hands[side] = {
                "finger_angle_known_count":int(mask[off+11:off+25].sum()),
                "finger_angle_mask":mask[off+11:off+25].astype(int).tolist(),
                "hand_axis_xy_known":bool(mask[[off+25,off+26]].all()),
                "hand_axis_xy_mask":mask[[off+25,off+26]].astype(int).tolist(),
                "hand_wrist_xy_known":bool(mask[[off,off+1]].all()),
                "pose_wrist_elbow_xy_known":bool(mask[off+2:off+6].all())
            }
        result.append({"frame_id":f,"timestamp_us":t,"input_row":idx,"hands":hands})
    return result

windows=[]
for w in selection["windows"]:
    n=notes[w["window_id"]]
    mw=next(v for v in manifest["windows"] if v["window_id"]==w["window_id"])
    out={k:w[k] for k in ["window_id","query","reference","parent_id","categories","word_id"]}
    out["pairparent"]={"query":w["query"],"reference":w["reference"],"parent_id":w["parent_id"]}
    for key,tag in [("query_axis","q"),("reference_axis","r")]:
        a=w[key]
        frames=mw["axes"][key]
        assert [f["frame_id"] for f in frames]==a["context_frame_ids"]
        pages=[p for p in manifest["pages"] if p["alias"]==a["alias"] and set(p["frame_ids"])&set(a["context_frame_ids"])]
        out[key]={k:a[k] for k in ["alias","clip_id","source","source_sha256","input","input_sha256","core_frame_range_inclusive","context_frame_range_inclusive"]}
        out[key].update({
            "reviewed_frame_ids":a["context_frame_ids"],"reviewed_timestamp_us":a["context_timestamp_us"],
            "full_rgb_frames":frames,"contact_pages":pages,
            "review_mode":"Every fixed context frame seen on chronological RGB contact pages; original full RGB separately opened only when listed.",
            "separately_opened_full_rgb_frame_ids":[f["frame_id"] for f in frames if (a["alias"],f["frame_id"]) in full_viewed],
            "observations":[{"frame_range_inclusive":r,"text_ko":t} for r,t in n[tag]],
            "stored_mask_observations":mask_records(a),
            "mask_interpretation":"Stored support only, not RGB visibility or coordinate truth. No mask or coordinate changed."
        })
        for r,t in n[tag]:
            assert a["context_frame_ids"][0]<=r[0]<=r[1]<=a["context_frame_ids"][-1]
    evidence_pairs=[]
    for q,r in n["pairs"]:
        assert [q,r] in w["saved_core_correspondence"]
        qrgb=next(f for f in mw["axes"]["query_axis"] if f["frame_id"]==q)
        rrgb=next(f for f in mw["axes"]["reference_axis"] if f["frame_id"]==r)
        evidence_pairs.append({"query_frame":qrgb,"reference_frame":rrgb,
                               "selection_path_index":w["saved_core_correspondence"].index([q,r])})
    out.update({
        "source_evidence":{"selection_file":str(HERE/"WINDOW_SELECTION.json"),"selection_window_id":w["window_id"],
          "baseline_path_source":w["baseline_path_source"],"baseline_sha256":w["baseline_sha256"],
          "saved_core_correspondence":w["saved_core_correspondence"],"representative_existing_pairs":evidence_pairs,
          "previous_physical_annotations_source":w["previous_physical_annotations_source"],
          "previous_source_links":w["previous_source_links"]},
        "visible_difference_review":{"status":"QUALITATIVE_PROJECTED_RGB_OBSERVATION_ONLY","summary_ko":n["visible"]},
        "observation_missing":{"summary_ko":n["missing"],"support_evidence":"query_axis/reference_axis.stored_mask_observations"},
        "correspondence_review":{"summary_ko":n["correspondence"],"new_alignment":False,"query_gt_used_to_change_path_or_window":False},
        "coexisting_causes":n["causes"],
        "uncertainty_ko":n["limits"],
        "educational_verdict":None,"actual_contact_verdict":None,"actual_3d_rotation_verdict":None,"anatomical_wrist_bend_verdict":None,
        "reviewer":"Same Codex semantic_data_audit qualitative review; not an independent human/KSL expert or blind evaluation."
    })
    windows.append(out)

cases=[]
for q,(summary,limits) in cases_text.items():
    ws=[w for w in windows if w["query"]==q]
    cases.append({"query":q,"label":inputs["rows"][q]["label"],"word_id":ws[0]["word_id"],
                  "window_ids":[w["window_id"] for w in ws],"reviewed_reference":ws[0]["reference"],
                  "summary_ko":summary,"limits_ko":limits})
result={
 "schema":"fixed_public_rgb_case_review_v1",
 "status":"COMPLETED_10_FIXED_WINDOWS_QUALITATIVE_ONLY_NO_EDUCATIONAL_VERDICT",
 "at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
 "scope":{"selected_windows":10,"category_slots":12,"unique_public_clips":8,
          "unique_native_frames":303,"window_frame_opportunities":361,"chronological_contact_pages_viewed":30,
          "full_original_rgb_frames_separately_viewed":5,
          "full_casebook_pairs_preserved":12,"full_casebook_parents_preserved":39,
          "parents_with_this_new_visual_review":10,"other_parents_without_this_new_review":29,
          "other_reference_or_parent_review_claim":None},
 "provenance":{"selection_sha256":sha(HERE/"WINDOW_SELECTION.json"),"root_ack_sha256":sha(HERE/"WINDOW_ACK.json"),
               "rgb_manifest_sha256":sha(HERE/"RGB_MANIFEST.json"),"script_sha256":sha(__file__),
               "selection_was_fixed_before_new_rgb_opening":True,"selection_based_on_new_residual_scores":False,
               "all_contact_pages_actually_viewed":[p["file"] for p in manifest["pages"]],
               "full_original_frames_separately_viewed":[{"alias":a,"frame_id":f} for a,f in sorted(full_viewed)],
               "image_scope":"Original public RGB, presentation crop only. No new learned extraction; reused frames or local public decoder.",
               "source_pins":pins,"source_pin_checks":len(pins),"source_pin_failures":[]},
 "mask_schema":{"source":"Stored NPZ mask[T,69], frame_id[T], timestamp_us[T].",
                "left_finger_columns_zero_based":list(range(11,25)),"right_finger_columns_zero_based":list(range(39,53)),
                "left_axis_columns":[25,26],"right_axis_columns":[53,54],
                "known_rule":"mask > 0; no replacement/interpolation","angle_units":"Projected 2D internal angles/pi, not 3D rotation.",
                "axis_units":"Projected wrist-to-middle-MCP unit direction, not palm normal."},
 "windows":windows,"cases":cases,
 "limits":["Only the 10 fixed pair-parent contexts were reviewed; all other references remain unreviewed by this new RGB activity.",
           "303 unique frames were seen on 30 chronological sheets; five full original RGB images additionally opened. Fine features can remain occluded at either presentation size.",
           "Same Codex reviewer with prior exposure: qualitative cause investigation, not blind or independent expert truth.",
           "Visible projected differences, stored missing, and temporal pairing can coexist; no exclusive causal attribution from residual size.",
           "Physical temporal correspondence mismatch is not educational error. Natural valid variation was not labeled incorrect.",
           "No guide form, mandatory linguistic stage, contact, 3D fist rotation, or anatomical wrist bend was certified.",
           "No source label, saved path, probability, coordinate, mask, old result, or threshold was changed.",
           "No image outside the fixed contexts was opened for this review; no user/private/mentor material was accessed."],
 "actions":{"training":0,"model_inference":0,"landmark_extraction":0,"new_alignment":0,"old_score_recalculation":0,
            "label_changes":0,"source_video_writes":0,"threshold_tuning":0}
}
assert sum(len(w[a]["reviewed_frame_ids"]) for w in windows for a in ["query_axis","reference_axis"])==361
assert len({(w[a]["alias"],f) for w in windows for a in ["query_axis","reference_axis"] for f in w[a]["reviewed_frame_ids"]})==303
assert len(manifest["pages"])==30
outfile=HERE/"VISUAL_REVIEW.json"
assert not outfile.exists(), "Preserve first output; do not overwrite review."
outfile.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"status":result["status"],"path":str(outfile),"sha256":sha(outfile),"pins":len(pins),"windows":10},ensure_ascii=False))
