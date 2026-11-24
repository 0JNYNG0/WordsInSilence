import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

folder = Path(__file__).resolve().parent
request_path = folder / 'SECOND_IMAGE_REQUEST.json'
request = json.loads(request_path.read_text(encoding='utf-8-sig'))
observations = [
    ('T01', 'K03', None, 0, '지정 오른손의 세 손가락은 접힌 덩어리로 보인다. 검지 축은 정면으로 짧아지고 반대 손과 겹쳐 펴짐 정도를 확정하기 어려우므로 index_only는 보류한다.'),
    ('T04', 'K09', 0, None, '검지 외의 긴 손가락 윤곽도 보이므로 검지 단독은 아니다. 손의 측면에서 손가락들이 겹쳐 네 손가락 각각이 충분히 펴졌는지는 보류한다.'),
    ('T05', 'K12', 1, 0, '검지는 대각선 위로 길게 펴져 있고 나머지 세 손가락은 손바닥 쪽으로 접혀 있다. 엄지는 별도로 튀어나와 있으나 index_only 정의에서는 자유롭다.'),
    ('T07', 'K10', None, None, '낮게 놓인 왼손은 보이지만 손가락이 측면에서 겹치고 자연스럽게 중간 굽힘 상태여서 두 모양 특징 모두 확정하지 않는다.'),
    ('T10', 'K09', 0, 1, '오른손 네 손가락의 대체로 펴진 긴 윤곽이 보인다. 반대 손이 아래에 있지만 지정 손 네 손가락의 펼침을 막지는 않는다.'),
    ('T11', 'K12', 0, 1, '오른손의 네 손가락이 나란히 대체로 펴져 있다. 검지만 따로 편 모양은 아니다.'),
    ('T12', 'K12', 0, 1, '두 손의 영상 영역은 일부 겹치지만 지정 오른손의 네 손가락이 나란히 길게 펴진 윤곽은 보인다.'),
    ('T13', 'K06', 0, 1, '오른손 네 손가락이 벌어져 대체로 길게 펴져 있다. 검지 외 손가락도 펴져 있으므로 index_only는 아니다.'),
    ('T13', 'K07', 0, 1, '네 손가락이 벌어져 있고 길게 펴진 각 손가락 윤곽이 보인다. 약한 자연 굽힘은 있으나 대체로 펼친 모양이다.'),
    ('T15', 'K04', 0, 0, '손가락들이 안쪽으로 굽혀져 가까운 덩어리를 이룬다. 네 손가락이 길게 펼쳐져 있지 않고 검지 단독으로 편 형태도 아니다.'),
    ('T15', 'K03', 0, 1, '네 손가락의 긴 윤곽이 서로 벌어져 있다. 끝부분의 약간의 굽힘은 있지만 대체로 펼쳐진 상태로 판단한다.'),
    ('T15', 'K13', 0, None, '여러 손가락이 벌어져 있어 검지 단독은 아니다. 특히 위쪽 손가락이 눈에 띄게 굽고 나머지는 더 펴진 중간 형태라 open_four를 강제하지 않는다.'),
    ('T15', 'K05', 0, 0, '여러 손가락이 함께 안쪽으로 굽혀져 묶인 모양이다. 길게 펼친 네 손가락 또는 검지 단독 펼침은 보이지 않는다.'),
    ('T17', 'K08', 0, 0, '손가락이 손바닥 쪽으로 접힌 주먹 모양이며 펴진 검지는 보이지 않는다.'),
]
assert len(request['rows']) == len(observations) == 14
rows = []
for r, obs in zip(request['rows'], observations):
    case, key, index_only, open_four, note = obs
    assert (r['case_id'], r['image_key']) == (case, key)
    p = Path(r['path'])
    rows.append({**r, 'image_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                 'index_only': index_only, 'open_four': open_four, 'note': note,
                 'actually_viewed_with': 'tools.view_image'})
out = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'reviewer': 'Codex ksl_source_audit',
    'status': 'ADDITIONAL_CURRENT_IMAGE_DRAFT_FROZEN',
    'request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
    'scope': request['scope'],
    'limitations': [
        '동일 AI의 관찰 초안이며 실제 손모양 정답/수어 전문가 판정이 아니다.',
        '실패 후보와 통제 장면을 선택한 표본이며 무작위 표본이 아니다.',
        '판정 중 기존 라벨 파일, 시간순 시트, 시간 맵, 비교 결과를 읽지 않았다.',
        '과거 중심 이미지와 사전/영상 맥락의 기억은 있어 완전 맹검은 아니다.',
        '현재 표시된 정적 이미지의 직접 보이는 모양만 기록했고 앞뒤 순서로 메우지 않았다.',
    ],
    'initial_center_file_modified': False,
    'user_material_accessed': False,
    'training_performed': False,
    'operating_models_changed': False,
    'rows': rows,
    'summary': {
        'images_actually_viewed': len(rows), 'feature_cells': 28,
        'known_cells': sum(r[k] is not None for r in rows for k in ('index_only','open_four')),
        'unknown_cells': sum(r[k] is None for r in rows for k in ('index_only','open_four')),
    },
}
dest = folder / 'SECOND_IMAGE_LABELS.json'
with dest.open('x', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
digest = hashlib.sha256(dest.read_bytes()).hexdigest()
with dest.with_suffix('.sha256').open('x', encoding='ascii') as f:
    f.write(digest + '\n')
print(json.dumps({'path': str(dest), 'sha256': digest, 'summary': out['summary']}, ensure_ascii=False))
