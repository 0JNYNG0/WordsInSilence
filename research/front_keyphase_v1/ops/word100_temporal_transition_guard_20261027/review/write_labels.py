import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

u = Path(__file__).resolve().parent.parent
frozen_path = u / 'FROZEN.json'
frozen = json.loads(frozen_path.read_text(encoding='utf-8-sig'))
states = {'I': (1,0), 'O': (0,1), 'Z': (0,0), 'U': (0,None), 'V': (None,0), 'X': (None,None)}
# Each listed K image was visually inspected before this transcription.
# These defaults summarize repeated observed states, not source-word expectations.
specs = {
    'N01': ('I', {6:'V'}),
    'N02': ('I', {}),
    'N03': ('I', {}),
    'N04': ('I', {1:'V'}),
    'N05': ('I', {9:'O',10:'U',13:'U'}),
    'N06': ('O', {5:'I',8:'U',10:'X'}),
    'N07': ('I', {}),
    'N08': ('I', {}),
    'N09': ('I', {}),
    'N10': ('O', {}),
    'N11': ('U', {}),
    'N12': ('U', {}),
    'N13': ('Z', {6:'X'}),
    'N14': ('Z', {8:'X'}),
    'N15': ('Z', {11:'X'}),
    'N16': ('O', {2:'U',6:'U',7:'U',8:'U',12:'U'}),
    'N17': ('O', {}),
    'N18': ('U', {}),
}
notes = {
    'I': '지정 오른손의 검지가 길게 펴져 있고 중지·약지·소지는 접힌 모습이 보인다. 엄지 자세는 이 특징에서 제한하지 않는다.',
    'O': '지정 오른손의 네 비엄지 손가락이 대체로 펴진 긴 윤곽과 손끝이 보인다. 약한 자연 굽힘은 허용하며 검지 단독 모양은 아니다.',
    'Z': '지정 오른손의 비엄지 손가락들이 주먹처럼 접혀 있는 모습이 보인다. 검지만 펴거나 네 손가락을 편 상태는 아니다.',
    'U': '여러 손가락의 긴 윤곽이 보여 검지 단독은 아니다. 측면 가림 또는 중간 굽힘 때문에 네 손가락 전체의 펼침을 확정하지 않는다.',
    'V': '접힌 다른 손가락들이 보여 네 손가락 펼침은 아니다. 검지는 정면 축약·겹침/굽힘 때문에 단독 펼침을 확정하기 어렵다.',
    'X': '지정 오른손의 핵심 손가락 영역이 반대 손에 가려 현재 RGB 하나만으로 두 모양 특징을 확정할 수 없다. 보류를 0으로 바꾸지 않는다.',
}
special = {
    ('N01',6): '오른 검지 끝과 축이 반대 손에 겹치고 정면으로 짧아져 펴짐 판단을 보류한다. 다른 세 손가락은 접혀 보여 open_four는 0이다.',
    ('N04',1): '오른 검지가 카메라 쪽으로 축약되고 부분 굽힘 가능성이 있어 단독 펼침은 보류한다. 나머지 손가락은 접혀 보인다.',
    ('N05',10): '검지 외 손가락들도 부분적으로 펴져 있으므로 검지 단독은 아니다. 네 손가락은 자연 굽힘과 중간 펼침 사이여서 open_four는 보류한다.',
    ('N05',13): '검지 외 손가락들이 부분적으로 펴져 있다. 모두 펴짐 또는 모두 묶임으로 단정하기 어려운 중간 모양이라 open_four는 보류한다.',
    ('N06',8): '오른손 여러 손가락의 길쭉한 윤곽이 있으나 일부 끝과 굽힘이 반대 팔/손과 겹쳐 네 손가락 전체 펼침은 보류한다.',
    ('N06',10): '반대 손이 지정 오른손의 손가락 대부분을 가려 오른손 모양을 추측하지 않는다.',
    ('N13',6): '지정 오른손의 손가락이 앞의 왼손에 가려지고 엄지·손 옆면만 일부 보인다. 기대되는 주먹 모양으로 채우지 않는다.',
    ('N14',8): '두 손이 겹치면서 지정 오른손의 손가락이 앞손에 가려진다. 엄지가 보이는 것만으로 나머지 손가락 상태를 확정하지 않는다.',
    ('N15',11): '지정 오른손은 반대 손 뒤에 있어 일부 윤곽만 보인다. 주변 장면의 주먹 모양을 이 이미지에 복사하지 않고 보류한다.',
}
extra = [
    ('N01','K06'), ('N04','K01'), ('N05','K10'), ('N05','K13'), ('N05','K09'), ('N05','K01'),
    ('N06','K05'), ('N06','K11'), ('N06','K08'), ('N06','K12'), ('N06','K10'),
    ('N11','K01'), ('N11','K02'), ('N12','K01'), ('N13','K06'),
    ('N14','K08'), ('N14','K07'), ('N15','K11'),
    ('N16','K01'), ('N16','K06'), ('N17','K01'), ('N18','K01'), ('N18','K02'),
]
files_viewed = []
for n in specs:
    p = u/'visual_inputs'/n/'shuffled_contact.png'
    files_viewed.append({'path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                         'method':'tools.view_image', 'panels_inspected':[f'K{k:02d}' for k in range(1,14)]})
for n,k in extra:
    p=u/'visual_inputs'/n/f'{k}_focus.png'
    files_viewed.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'method':'tools.view_image'})
rows=[]
for n,(base,overrides) in specs.items():
    for k in range(1,14):
        code=overrides.get(k,base)
        a,b=states[code]
        note=special.get((n,k),notes[code])
        if n in ('N11','N12'):
            note='팔 연결로 지정 오른손을 확인했다. 여러 손가락이 길게 펴진 측면 윤곽은 보이므로 검지 단독은 아니다. 손날 방향으로 겹쳐 네 손가락 각각의 펼침은 이 이미지에서 확인하지 못해 보류한다.'
        if n=='N18':
            note='오른손이 손날 쪽에서 보여 여러 손가락의 길게 뻗은 윤곽은 있으나 네 손가락 각각은 겹친다. 검지 단독은 아니지만 open_four는 불확실로 남긴다.'
        rows.append({'case_id':n,'image_key':f'K{k:02d}','side':'R','index_only':a,'open_four':b,'note':note})
assert len(rows)==234 and len({(r['case_id'],r['image_key']) for r in rows})==234
out={
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'reviewer':'Codex ksl_source_audit',
    'status':'FROZEN_FIRST_PASS_CURRENT_RGB_ONLY_DRAFT',
    'frozen_sha256':hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
    'rubric_sha256':frozen['pins'][frozen['rubric_source']],
    'scope':'18 shuffled public-image sheets,13 K images each; performer right hand only;234 image rows/468 coarse feature cells',
    'label_status':'동일 AI 계열의 이미지 관찰 초안. 독립 사람/수어 전문가 정답 또는 관절의 실제 3D 진실이 아님.',
    'blinding_limits':'FROZEN의 출처 단어/ID 메타와 과거 사전·이미지 맥락의 기억이 있으므로 완전 맹검이 아니다. 현재 K 번호는 시간순을 제공하지 않았고 최초 라벨 전 INDEX/ordered/center_only/다른 리뷰 라벨/계산 결과를 읽지 않았다.',
    'ordered_or_time_map_read':False,'other_labels_or_metrics_read':False,
    'word_expectation_used_to_fill_unknown':False,
    'unknown_rule':'미관측·가림·중간 굽힘은 null; 보이지 않는 손가락을 이전/다음 모양이나 단어 예상으로 채우지 않음.',
    'review_method':'Each shuffled panel was visually inspected;23 ambiguous focus images inspected separately with tools.view_image. No temporal-order inference or joint estimates used as ground truth.',
    'user_material_accessed':False,'training_performed':False,'operating_models_changed':False,
    'rows':rows,'files_viewed':files_viewed,
    'summary':{'image_rows':234,'feature_cells':468,'known_cells':sum(r[k] is not None for r in rows for k in ('index_only','open_four')),
               'unknown_cells':sum(r[k] is None for r in rows for k in ('index_only','open_four')),
               'actually_viewed_files':len(files_viewed)},
}
payload=json.dumps(out,ensure_ascii=False,indent=2).encode('utf-8')
for name in ['IMAGE_LABELS_FIRST_PASS.json','IMAGE_LABELS.json']:
    dest=Path(__file__).with_name(name)
    with dest.open('xb') as f:f.write(payload)
    with dest.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(hashlib.sha256(payload).hexdigest()+'\n')
print(json.dumps({'sha256':hashlib.sha256(payload).hexdigest(),'summary':out['summary']},ensure_ascii=False))
