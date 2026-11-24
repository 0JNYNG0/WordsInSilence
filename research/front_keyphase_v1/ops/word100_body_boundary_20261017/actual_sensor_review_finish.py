"""Record actual visual inspection; read saved TRAIN captures only."""
from pathlib import Path
import json
import hashlib
import datetime
import collections
import numpy as np

U = Path(__file__).resolve().parent
NOTES = {
    'NIA_SL_WORD0920_REAL01_F': ('overlap_single_candidate_opposite_assignment', '양손이 가슴 앞에서 겹친다. native/masked의 한 손은 L, sensor의 한 손은 R로 배정되며 센서 원후보는 구간 전체1개다. 실제 손이 선 아래로 사라진 것이 아니다. 겹친 후보가 어느 물리손인지 이 표본만으로 확정하지 않는다.'),
    'NIA_SL_WORD1545_REAL01_F': ('crossing_occlusion_extra_sensor_detection_loss', '양팔이 가슴 앞에서 교차하고 한 손이 반대 팔에 일부 가려진다. sensor 원후보1개에 비해 native/masked는 양손이 남는다. f70에는 센서도 회복한다. 손은 경계보다 훨씬 위다.'),
    'NIA_SL_WORD2348_REAL02_F': ('entire_core_right_missing_duplicate_same_hand_candidates', '표본에서 양손이 상반신 안에 분리되어 보이는데 sensor 오른손이 핵심48~82 전체에서 누락된다. 원후보는35프레임 중32프레임2개지만 샘플상 두 후보 손목이 같은 왼손 근방이며 trace same_side_alternative로 하나만 유지한다. 단순 ROI 잘림이 아니라 한 실제손 중복검출과 다른 손 미검출이다. 원본/masked에는 오른손이 있어 해당 학습보기 전체 생성불가는 아니다. 주요 잔존 위험.'),
    'NIA_SL_WORD0185_REAL02_F': ('overlapping_contact_hands_one_sensor_candidate', '양손이 가슴 앞에서 접촉·겹치는 자세다. native/masked의 양손 근거는 있고 sensor 원후보는11프레임 모두1개다. 경계선은 양손 아래 멀리 있어 물리적 잘림 근거는 없다.'),
    'NIA_SL_WORD0185_REAL05_F': ('overlap_opposite_side_and_transition_assignment_hold', '양손 접촉 중 native는 주로R, sensor는L 단일 배정으로 어긋난다. 원후보13프레임 중12프레임1개, f71 두 후보 중 하나는 ambiguous_or_no_evidence로 보류한다. 경계 위 겹침·배정 불확실성으로 보이며 정확한 손 정체는 영상 표본만으로 확정하지 않는다.'),
    'NIA_SL_WORD0795_REAL02_F': ('near_face_in_frame_sensor_candidate_loss', '얼굴 앞에서 양손이 가까워지는 동작이며 둘 다 경계보다 높다. native/masked 양손에 비해 sensor는11프레임 모두 원후보1개다. 한 손의 추가검출 누락으로 확인하며 정확한 검출 실패 원인은 이 자료만으로 확정하지 않는다.'),
    'NIA_SL_WORD0866_REAL03_F': ('support_palm_sensor_loss_during_contact', '받치는 손바닥과 그 위로 움직이는 다른 손이 보인다. sensor는25프레임 중24프레임 후보1개로 받치는 손의 좌표가 누락된다. 단어 구별에 중요한 지지손 관계가 손실될 위험이 있으나 손의 물리적 영역잘림이 아니다. 관측학습보기는 존재한다.'),
    'NIA_SL_WORD1139_REAL06_F': ('contact_hand_occlusion_one_sensor_candidate', '가슴 앞에서 손끼리 접촉·겹치는 부분이고12프레임 모두 sensor 원후보1개다. f59 손이 분리되자 양손회복한다. 손 전체가 경계 위에 있어 잘림과 구분한다.'),
    'NIA_SL_WORD1998_REAL05_F': ('crossing_detector_loss_then_assignment_hold', '양손이 올라오며 교차·겹친다.16프레임 중3프레임 원후보0,13프레임1. f84는 검출0, f88은 후보1개지만 pose와track근거 불일치로 ambiguous_or_no_evidence 보류, f90 L회복. 짧은 양손 동시누락도 포함된 기존검출/배정 한계다. 영역 밖 잘림은 아니다.'),
}


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    meta = read(U/'actual_sensor_review_MONTAGES.json')
    pins = {str(U/n): sha(U/n) for n in ['actual_sensor_review_MONTAGES.json', 'actual_sensor_review_finish.py']}
    rows = []
    for c in meta['cases']:
        rec = read(c['record'])
        cap = rec['sensor_capture']
        rp, cp = Path(cap['raw']), Path(cap['candidate'])
        tp = cp.parent/'trace.json'
        for p in [rp, cp, tp]:
            pins[str(p)] = sha(p)
        assert pins[str(rp)] == cap['pins'][str(rp)]
        assert pins[str(cp)] == cap['pins'][str(cp)]
        raw = np.load(rp, allow_pickle=False)
        tr = read(tp)
        a, e = c['flag_frame_range']
        ids = list(range(a, e+1))
        b = 33 if c['side'] == 'left' else 54
        obs = np.load(rec['views']['sensor']['observation'], allow_pickle=False)
        missing = ~(obs['observed'][ids, b:b+21] & obs['usable'][ids, b:b+21]).any(1)
        assert bool(missing.all())
        rtype, text = NOTES[c['clip_id']]
        frames = [r['frame_id'] for r in c['frames']]
        rows.append({
            'clip_id': c['clip_id'], 'label': c['label'], 'role': 'train_candidate',
            'side': c['side'], 'flag_frame_range': [a, e], 'flag_frames_count': len(ids),
            'flag_duration_seconds': round((int(obs['timestamp_us'][e])-int(obs['timestamp_us'][a])+33333)/1e6, 6),
            'actual_viewed_frames': frames, 'continuous_interval_reviewed': False,
            'image': c['image'], 'image_sha256': c['image_sha256'],
            'record': c['record'], 'record_sha256': sha(c['record']),
            'source': c['source'], 'box_source_pixels': rec['calibration']['box_source_pixels'],
            'pts_origin_us': c['source_pts_origin_us'], 'source_and_relative_pts_verified': True,
            'finding_type': rtype, 'finding_ko': text,
            'sensor_raw_candidate_count_in_flag': dict(collections.Counter(str(int(raw['raw_count'][i])) for i in ids)),
            'trace_reason_counts_in_flag': dict(collections.Counter(x for i in ids for x in tr[i]['reasons'])),
            'trace_snapshots_at_viewed_frames': [tr[i] for i in frames],
            'new_geometric_essential_cut_confirmed_in_viewed_frames': False,
            'all_three_required_learning_views_available': c['all_required_view_files_present'],
            'feature_info': c['feature_info'], 'prior_visual_lineage': c['prior_visual_lineage'],
            'gate_interpretation': 'Observed detector/assignment information loss; record as residual sensor stress risk. No automatic semantic PASS or automatic failure. Complete interval and overall READY remain root-owned.'
        })
    result = {
        'status': 'NINE_TRAIN_SPARSE_VISUAL_REVIEW_COMPLETE_WITH_RESIDUAL_SENSOR_RISKS',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reviewer': 'roi_training_protocol independent actual view_image review', 'cases': rows,
        'actual_viewed_frame_count': sum(len(r['actual_viewed_frames']) for r in rows),
        'continuous_full_interval_coverage': False,
        'all_required_learning_views_available': all(r['all_three_required_learning_views_available'] for r in rows),
        'new_geometric_cut_confirmed_in_viewed_frames': 0,
        'important_residual_risks': [
            '전멸 REAL02 핵심35프레임(약1.167초) 오른손 전체미관측: 동일손 중복후보/다른손 미검출.',
            '초대권 REAL03 받치는 손 관계가 sensor에서 누락될 위험.',
            '추출·후원 REAL05 겹친손 단일좌우배정 차이.',
            '칠곡군지부 REAL05 교차중 검출0 및 후속배정보류.'
        ],
        'recommendation': '9사례는 센서열화가 있는 입력조건을 연구하는 근거로 유지하되 C 개선/의미보존을 보장하지 않는다. native/masked counterpart와 센서누락마스크를 그대로 보존한다. 전멸같은 전구간한손누락은 최종보고의 중요위험으로 공개하고 별도재검출·임계변경·단어예외없이 비교한다. 이63프레임 확인만으로631영상/765flag조건 전체READY를 발행할수없다. 나머지flag구간검토와원래gate를root가통합판정한다.',
        'source_pins': pins,
        'first_visualization_error': 'actual_sensor_review_FIRST_ERROR.json; raw absolutePTS를 저장 첫프레임상대PTS와 비교한 시각화assert만복구. 원본3이미지해시불변.',
        'report_build_error': 'Inline report draft had a parse-only SyntaxError (else54); no script executed and no result written. New explicit script corrected spacing.',
        'input_changes': 0, 'new_detector_runs': 0, 'model_inference': 0, 'training': 0,
        'dev_observations_read': 0, 'no_sign_language_semantic_expert_guarantee': True,
    }
    for p, h in pins.items():
        assert sha(p) == h
    with (U/'actual_sensor_review_RESULT.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (U/'actual_sensor_review_RESULT_KO.txt').open('x', encoding='utf-8') as f:
        f.write('9개 TRAIN 영역내 센서손누락 독립 검토\n\n')
        f.write('9개 원본RGB+저장관측 몽타주, 각7프레임 총63프레임을 실제 확인했다. 연속구간 전체 검토나 전체READY 판정이 아니다.\n\n')
        for r in rows:
            f.write(f"{r['label']} {r['clip_id']}: {r['finding_ko']}\n실제 확인 프레임 {r['actual_viewed_frames']}\n\n")
        f.write(result['recommendation']+'\n새검출/모델추론/학습/입력변경은 모두0.\n')
    print(json.dumps({'status': result['status'], 'frames': result['actual_viewed_frame_count'], 'all_views_available': result['all_required_learning_views_available']}, ensure_ascii=False))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
