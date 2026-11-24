"""Distribution of already verified WORD metadata; no ZIP/video/network/model access."""
import hashlib
import html
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

U = Path(__file__).resolve().parent
SOURCE = U.parent/'word_acquisition_inventory_20261011/FRONT_MANIFEST.jsonl'
assert not (U/'DISTRIBUTION.json').exists(), 'Reuse the completed distribution'
raw = SOURCE.read_bytes()
source_sha = hashlib.sha256(raw).hexdigest()
rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
assert len(rows) == 54000 and len({r['clip_id'] for r in rows}) == 54000
by_code, by_label = defaultdict(list), defaultdict(list)
for row in rows:
    by_code[row['word_code']].append(row)
    if row['metadata_clean']:
        assert len(row['labels']) == 1
        by_label[row['labels'][0]].append(row)
families = []
for code, group in sorted(by_code.items()):
    clean = [r for r in group if r['metadata_clean']]
    names = Counter(r['labels'][0] for r in clean)
    ids = sorted({r['filename_signer_id'] for r in clean})
    train = [r for r in clean if r['role'] == 'train_candidate']
    assert len(ids) == len(clean), 'Unexpected multiple front performances per signer/code'
    families.append(dict(code=code, raw_count=len(group), count=len(clean), signer_ids=ids,
                         people=len(ids), labels=dict(names), label_agreement=len(names)==1,
                         quarantined=len(group)-len(clean), training_count=len(train),
                         role_counts=dict(Counter(r['role'] for r in clean))))
family_map = {x['code']:x for x in families}
labels = []
for label, group in by_label.items():
    train = [r for r in group if r['role']=='train_candidate']
    codes = sorted({r['word_code'] for r in group})
    labels.append(dict(label=label, people=len({r['filename_signer_id'] for r in group}),
                       count=len(group), codes=codes, code_count=len(codes),
                       training_people=len({r['filename_signer_id'] for r in train}),
                       training_count=len(train), role_counts=dict(Counter(r['role'] for r in group)),
                       local_video_count=sum(r['video'] is not None for r in group),
                       mixed_label_codes=[c for c in codes if not family_map[c]['label_agreement']]))
labels.sort(key=lambda x:(-x['people'],-x['count'],x['label']))
previous = None
for pos, row in enumerate(labels,1):
    key=(row['people'],row['count'])
    if key != previous: rank=pos
    row['rank']=rank; previous=key

def histogram(values):
    return {str(k):v for k,v in sorted(Counter(values).items(),reverse=True)}

result = dict(
    status='COMPLETE_METADATA_DISTRIBUTION', at_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
    source=str(SOURCE), source_sha256=source_sha,
    count_unit='Unique front-view clip ID from existing annotation; not downloaded-video availability or validated distinct semantic classes.',
    raw_front_count=len(rows), metadata_clean_count=sum(r['metadata_clean'] for r in rows),
    quarantined_count=sum(not r['metadata_clean'] for r in rows), code_count=len(families), raw_label_count=len(labels),
    raw_code_count_distribution=histogram(x['raw_count'] for x in families),
    clean_code_count_distribution=histogram(x['count'] for x in families),
    label_people_distribution=histogram(x['people'] for x in labels),
    label_clip_distribution=histogram(x['count'] for x in labels),
    single_label_codes=sum(x['label_agreement'] for x in families),
    mixed_label_codes=sum(not x['label_agreement'] for x in families),
    code18_single_label_count=sum(x['count']==18 and x['label_agreement'] for x in families),
    existing_training_roles_unchanged=True,
    ranking_rule='Competition rank: distinct metadata signer IDs desc, clean front clips desc. Alphabetic order only displays tied rows; no automatic top100.',
    semantics='Preserve exact raw labels and original codes. Counts across different codes do not prove identical forms. Low exact-label counts may reflect spelling/suffix inconsistency rather than lack of a performed word.',
    original_claim_correction='AI Hub103 includes separately filmed WORD and SEN data. Claiming sentence-only for the entire dataset was incorrect; previously used SEN-derived inputs do not describe the entire source.',
    download_status='No new downloads. Existing20:00 timing restriction not overridden. Full WORD acquisition remains staged planning, not started in this counting task.',
    label_ranking=labels, codes=families,
    training=0, inference=0, camera=0, new_video_decode=0,
)
(U/'DISTRIBUTION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

report = '''AI Hub 독립 단어 자료 분포 — 2026-10-11

설명 정정
AI Hub 전체에 문장 데이터밖에 없다는 설명은 잘못이다. 단어 직접 촬영 WORD와 문장 촬영 SEN이 별도로 있다.
이전에 사용한 SEN 기반 단어 구간과 제공 자료 전체를 구별해야 한다. 이번 집계는 REAL/WORD 정면 주석만 대상으로 한다.

전체 규모와 단위
- 원본 단어 코드 3,000개, 코드마다 정면 주석 18건씩: 총 54,000건.
- 기존 점검에서 보류한 83건을 제외하면 53,917건.
- 코드별: 2,917개 코드는 18건, 83개 코드는 17건이다. 유효한 한 코드 안에서는 서로 다른 촬영자 ID와 수행 건수가 같다.
- 같은 수행의 5개 촬영 각도를 5명의 자료나 독립 수행 5회로 세지 않았다.
- 주석에서 확인한 자료 수다. 정면 영상 54,000개를 다운로드했다는 뜻이 아니다.
- 사람 ID는 파일·폴더 등의 내부 일치 기준이며 생체 신원 검증은 아니다.

이름으로 묶으면
정확한 원문 표기는 2,965종이다. 숫자 접미어·공백·의미와 표현 차이를 자동 정리하거나 병합하지 않았다.
같은 이름이 여러 WORD 코드에 쓰일 수 있어 코드 수와 표기 수가 다르다.
코드 2,733개는 점검 통과 자료의 라벨이 모두 같고, 267개는 라벨이 서로 달라 추가 확인이 필요하다.
그중 18개 ID와 단일 라벨이 모두 확보된 코드는 2,659개다. 영상 품질이나 수형 일치까지 검증했다는 뜻은 아니다.
표기별 1~2건인 경우에도 같은 코드의 다른 수행에 다른 표기가 붙었을 수 있으므로 곧바로 그 동작 자체의 자료 부족으로 단정하지 않는다.

정확한 원문 표기별 촬영자 ID 분포
'''
report += '\n'.join(f'- {n}개 ID: {v:,}개 표기' for n,v in result['label_people_distribution'].items())
report += '''

표기별 상위 순위 (촬영자 ID 수 우선, 다음 정면 수행 건수; 같은 조건은 공동 순위)
순위 | 표기 | 전체 ID 수 | 정면 수행 | WORD 코드 수 | 기존 학습 ID 수 | 기존 학습 수행
'''
report += '\n'.join(f"{r['rank']} | {r['label']!r} | {r['people']} | {r['count']} | {r['code_count']} | {r['training_people']} | {r['training_count']}" for r in labels[:30])
report += '''

해석과 다음 작업
'꺼지다' 108건은 18명 × 6코드의 집계다. 같은 수형 108회 또는 108명으로 해석하지 않는다.
전체18명에는 기존 개발/보류/공식Validation 역할의 사람이 포함된다. 모두 학습에 넣지 않는다. 별도 학습 열은 기존01~09 역할만 표시한다.
단어당 다양한 사람의 표본을 늘리는 방향은 합리적이다. 하지만 특정 개수만으로 일반화가 보장되지 않으며 같은 사람·동일 수행의 각도 중복과 라벨 혼합을 피해야 한다.
코드별 표본 수가 대부분 동률이므로 빈도 TOP100만으로 최종 어휘를 정하지 않는다. 교육적 용도, 표현 일관성, 검출 가능성, 사람 분할을 함께 점검한다.
3,000코드/2,965표기 전체 목록은 단어별_전체분포.html에 있다. 기존 통과/보류 조건을 변경하거나 모델 점수로 순위를 정하지 않았다.
이 집계는 기존 검증된54,000행을 재사용했다. ZIP 전수검사·기존 실험·영상 디코딩을 반복하지 않았다.
추가 다인 영상 원천 ZIP 연결과 획득이 필요하다. 현재20시 이후이므로 대용량 수신은 시작하지 않았다. 기존 소용량·시각 제한을 이번 분포 조사로 자동 해제하지 않았다.
다음 입력 준비는 기존12개 전체시간열·주석구간 별도 인덱스·MediaPipe/RGB 대응 확인을 유지한다.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')

esc=lambda v:html.escape(str(v),quote=True)
def table(headers, body):
    return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in r)+'</tr>' for r in body)+'</tbody></table>'
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>AI Hub 단어별 전체 분포</title>
<style>body{font-family:'Malgun Gothic',sans-serif;max-width:1250px;margin:30px auto;padding:0 22px;color:#152439;line-height:1.7}h1{font-size:28px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #d0d9e3;padding:8px 12px;text-align:left;vertical-align:top}th{background:#e5edf6;position:sticky;top:0}tbody tr:nth-child(even){background:#f7f9fc}.box{background:#edf6ee;border-left:5px solid #227c46;padding:16px}.note{background:#fff3df;padding:14px}nav a{margin-right:18px}a{color:#0759aa}.chart{max-width:700px}.barrow{display:flex;align-items:center;gap:12px;margin:8px 0}.barlabel{width:80px}.track{flex:1;background:#eef1f5;height:22px}.bar{height:100%;background:#2667a5}.number{width:100px}</style>
<h1>AI Hub 독립 단어 자료: 전체 분포와 순위</h1><p>2026-10-11 · 검증된 REAL/WORD 정면 주석 집계 · 대용량 영상 확보 전 현황</p>
<div class="box"><b>3,000개 단어 코드 × 정면18건 = 주석54,000건</b><br>83건 보류 후 53,917건. 원문 표기2,965종. 동일 수행의 다른4각도는 제외했습니다.</div>
<p><b>설명 정정:</b> AI Hub에는 독립 단어 WORD와 문장 SEN이 모두 있습니다. 전체 자료가 문장으로만 구성됐다는 이전 설명은 잘못입니다.</p>
<nav><a href="#distribution">분포</a><a href="#labels">표기별 전체 순위</a><a href="#codes">코드별3,000개</a><a href="RESULT_KO.txt">설명 보기</a></nav>
<p>Ctrl+F로 단어나 WORD 코드를 찾을 수 있습니다. 아래 수치는 <b>주석상 자료 수</b>이며 다운로드된 영상 수·수형 정확도·최종 학습 어휘 수가 아닙니다.</p>
<h2 id="distribution">코드별 분포</h2>'''
page+=table(['점검 통과 정면 수행 / ID 수','단어 코드 수'],[[18,2917],[17,83]])
page+='<p>코드2,733개는 라벨이 일치하고267개는 표기가 혼재합니다. 18건과 단일 표기를 모두 갖춘 코드는2,659개입니다. 모든 코드에서 실제 수형까지 일치하는지 확인한 것은 아닙니다.</p><h2>표기별 촬영자 ID 분포</h2><div class="chart">'
for n,v in result['label_people_distribution'].items():
    width=v/max(result['label_people_distribution'].values())*100
    page+=f'<div class="barrow"><span class="barlabel">{n}개 ID</span><div class="track"><div class="bar" style="width:{width:.4f}%"></div></div><span class="number">{v:,}개 표기</span></div>'
page+='</div><div class="note">이름별 집계는 여러 WORD 코드의 자료를 합칠 수 있습니다. 예: 꺼지다108건=18명×6코드. 같은 수형의 반복108회나108명을 뜻하지 않습니다. 표기1~2건도 코드 내 표기 차이 때문일 수 있습니다.</div><h2 id="labels">표기별 전체 순위: 2,965종</h2><p>전체 촬영자 ID 수↓ → 정면 수행 수↓. 동률은 공동 순위입니다. 전체 ID에는 평가·보류 사람이 포함되며, 기존 학습 열은01~09 역할만 보여 줍니다. 표기 차이·다중 코드를 자동 병합하지 않습니다.</p>'
page+=table(['순위','원문 표기','전체 ID','정면 수행','코드 수','기존 학습 ID','기존 학습 수행','코드 / 점검'],
 [[r['rank'],repr(r['label']) if r['label']!=r['label'].strip() else r['label'],r['people'],r['count'],r['code_count'],r['training_people'],r['training_count'],', '.join(r['codes'])+(' / 코드 내 표기 혼재 있음' if r['mixed_label_codes'] else '')] for r in labels])
page+='<h2 id="codes">코드별 전체 목록: 3,000개</h2><p>자료 번호를 유지한 목록입니다. 원문18건씩이며 통과 건수·원문 표기별 수를 표시합니다. 라벨이 다른 경우 이름만으로 합치지 않았습니다.</p>'
page+=table(['코드','통과 수행','통과 ID','기존 학습 수행','보류','원문 표기: 통과 건수'],
 [[r['code'],r['count'],r['people'],r['training_count'],r['quarantined'],'; '.join(f'{k!r}: {v}' for k,v in r['labels'].items())] for r in families])
page+='<p>구현: Codex. 기존 원문 검증 결과 재사용, 별도 SQL 집계 검산. 영상 품질·의미·수형·새 사용자 일반화는 별도 검증 대상입니다.</p></html>'
(U/'단어별_전체분포.html').write_text(page,encoding='utf-8')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_sha
print(json.dumps({k:result[k] for k in ['status','code_count','raw_label_count','clean_code_count_distribution','label_people_distribution','code18_single_label_count']},ensure_ascii=False))
