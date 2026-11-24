import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

u = Path(__file__).resolve().parent.parent
frozen_path = u / 'FROZEN.json'
frozen = json.loads(frozen_path.read_text(encoding='utf-8-sig'))
vectors = [
    (None, 0), (None, 0), (None, 0),
    (0, None), (0, None), (0, None),
    (None, None), (None, None), (None, None),
    (0, 1), (0, 1), (0, 1),
    (0, 1), (0, None), (0, None),
    (0, 1), (0, 1), (0, 1),
]
notes = [
    '오른손의 중지·약지·소지는 접혀 보이므로 네 손가락 펼침은 아니다. 검지는 정면 축약과 왼손 겹침으로 끝과 굽힘 상태를 이 중심 이미지 하나만으로 확정하기 어려워 보류한다.',
    '오른손의 세 손가락은 접힌 윤곽이 보인다. 검지 축이 정면으로 짧아지고 반대 손과 겹쳐 전체 펴짐 여부는 보류한다.',
    '오른손의 접힌 손가락 덩어리는 보인다. 앞으로 나온 검지의 일부가 반대 손에 겹치고 축약되어 단독 펼침을 확정하지 않는다.',
    '검지 외에도 여러 손가락이 위로 나와 검지 단독은 아니다. 손의 측면과 손가락끼리 가림 때문에 네 손가락 모두 충분히 펴졌는지는 보류한다.',
    '두 개 이상 손가락의 긴 윤곽이 있으므로 검지 단독은 아니다. 손가락을 옆에서 본 각도와 흐림으로 네 손가락 전체의 펴짐은 보류한다.',
    '여러 손가락이 나란히 얼굴 앞에 있어 검지 단독은 아니다. 측면 겹침 때문에 네 손가락 각각의 펴짐을 다 확인할 수 없다.',
    '왼손 자체는 낮은 위치에 보인다. 자연스러운 중간 굽힘과 옆면 겹침 때문에 검지 단독/네 손가락 펼침 모두 확정하지 않는다. 안 보이거나 불확실한 모양을 0으로 바꾸지 않았다.',
    '왼손은 보이지만 일부 손가락이 옆면에서 겹치고 자연스럽게 굽혀져 있다. 두 특징 모두 근거 부족으로 보류한다.',
    '왼손은 몸 옆에서 보인다. 손가락들이 측면에서 겹쳐 정확한 묶임과 펴짐 상태를 확정할 수 없어 두 특징을 보류한다.',
    '오른손등과 네 손가락의 긴 윤곽 및 끝이 보인다. 네 손가락은 대체로 펴져 있고 검지 단독 모양은 아니다.',
    '오른손 네 손가락이 나란히 길게 펴진 모습이 보인다. 검지 외 세 손가락도 펴져 있다.',
    '오른손 네 손가락이 대체로 펴져 있으며 작은 자연 굽힘은 허용한 기준에 부합한다. 검지 단독은 아니다.',
    '오른손 네 손가락이 벌어진 채 대체로 펴져 보인다. 자연스러운 약간의 굽힘은 있으나 검지 단독 형태는 아니다.',
    '여러 손가락이 펼침과 굽힘 사이의 중간 형태로 보인다. 검지 단독은 아니지만 네 손가락 펼침을 1 또는 0으로 강제하지 않는다.',
    '여러 손가락이 굽혀진 중간 형태로 보인다. 검지 단독은 아니다. 완전 묶임/펼침 어느 쪽도 분명하지 않아 open_four는 보류한다.',
    '오른손등과 네 손가락이 얼굴 앞에서 나란히 길게 펴진 모습이 보인다. 검지 단독은 아니다.',
    '오른손 네 손가락이 함께 펴져 있고 손끝들이 확인된다. 검지 외 손가락도 펴져 있다.',
    '오른손 네 손가락의 길게 펴진 윤곽이 보인다. 손가락 간 약간의 굽힘 차이와 무관하게 검지 단독은 아니다.',
]
rows = []
viewed = []
for i, meta in enumerate(frozen['rows']):
    p = u / 'visual_inputs' / meta['case_id'] / 'center_only.png'
    if not p.is_file():
        raise FileNotFoundError(p)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    rows.append({
        'case_id': meta['case_id'], 'side': meta['side'],
        'center_frame_id': meta['center_frame_id'],
        'index_only': vectors[i][0], 'open_four': vectors[i][1],
        'note': notes[i], 'image_path': str(p), 'image_sha256': h,
    })
    viewed.append({'path': str(p), 'sha256': h, 'method': 'tools.view_image'})
assert len(rows) == 18 and len({r['case_id'] for r in rows}) == 18
out = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'reviewer': 'Codex ksl_source_audit; independent center-image draft',
    'status': 'FROZEN_FIRST_CENTER_ONLY_DRAFT',
    'frozen_sha256': hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
    'rubric_sha256': frozen['pins'][frozen['rubric_source']],
    'scope': '18 specified performer-side hands; each center_only.png actually viewed; index_only/open_four only',
    'label_status': '동일 AI 계열의 이미지 관찰 초안. 독립 사람 또는 수어 전문가 정답이 아님.',
    'blinding_limits': 'Previously saw dictionary/source context and some earlier images in prior tasks. FROZEN metadata includes lexical labels. Therefore not fully blinded. This first draft uses the current center images only; no ordered/shuffled sheets, INDEX, other reviewers\' labels or current root labels were inspected before freezing.',
    'temporal_context_inspected_before_freeze': False,
    'other_reviewer_labels_inspected_before_freeze': False,
    'word_name_used_to_fill_shape': False,
    'user_material_accessed': False,
    'training_performed': False,
    'operating_models_changed': False,
    'rows': rows,
    'actual_viewed_images': viewed,
    'summary': {
        'cases': len(rows), 'feature_cells': 36,
        'known_cells': sum(r[k] is not None for r in rows for k in ('index_only', 'open_four')),
        'unknown_cells': sum(r[k] is None for r in rows for k in ('index_only', 'open_four')),
    },
}
destination = Path(__file__).with_name('CENTER_LABELS.json')
with destination.open('x', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
digest = hashlib.sha256(destination.read_bytes()).hexdigest()
with destination.with_suffix('.sha256').open('x', encoding='ascii') as f:
    f.write(digest + '\n')
print(json.dumps({'path': str(destination), 'sha256': digest, 'summary': out['summary']}, ensure_ascii=False))
