"""Readable, local-only evidence viewer. Uses saved results, no inference."""
from pathlib import Path
import json, html, hashlib
from collections import Counter
import numpy as np

U=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def esc(x): return html.escape(str(x))
def uri(p): return Path(p).resolve().as_uri()

NAMES={'R_index_only':'오른손 검지 펴기','R_open_four':'오른손의 엄지 외 네 손가락 펴기','L_open_four':'왼손의 엄지 외 네 손가락 펴기','R_face_overlap':'오른손과 얼굴의 화면상 겹침','hands_overlap':'두 손의 화면상 겹침','R_motion_up':'오른손의 화면 위쪽 이동','R_motion_down':'오른손의 화면 아래쪽 이동','R_motion_left':'오른손의 화면 왼쪽 이동','R_motion_right':'오른손의 화면 오른쪽 이동','L_motion_down':'왼손의 화면 아래쪽 이동','unsupported':'현재 관측 기능 밖'}
STATUS={'GUIDE_RELEVANT_DIFFERENCE_CANDIDATE':'설명 관련 차이 후보 · 재확인 필요','DESCRIBED_FEATURE_OBSERVED':'대응 장면에서 일부 특징 관측','INSUFFICIENT_LINKED_EVIDENCE':'비교 근거 부족','SOURCE_LINK_WITHHELD':'설명 연결 또는 관측 기능 미확인','OBSERVER_RELATION_CONFLICT':'관측 근거 충돌 · 재확인 필요'}
CELL={'SUPPORTED_STATE':'설명과 같은 특징 예측','OPPOSITE_STATE':'설명과 반대 특징 예측','UNOBSERVED':'현재 영상 관측 부족','UNCERTAIN_MODEL':'모델 판단 불확실','ALIGNMENT_UNAVAILABLE':'대응 시점 확인 불가','REFERENCE_UNOBSERVED':'참고 관측 부족','REFERENCE_OBSERVER_CONFLICT':'참고 RGB 주석과 모델 충돌'}

TEXT='''수어 설명과 실제 관측을 연결한 첫 비교 기능

현재 구조는 전체 영상에서 MediaPipe로 관측하고, 고정 상반신 범위의 관절로 특징을 계산한 뒤, 각 참고 영상과 DTW로 시간 대응을 잡는 방식이다. 이번에는 그 뒤에 한국수어 사전 설명과 관측 특징을 연결하는 공통 비교 기능을 구현했다. 새 거대 모델이나 새 학습을 추가하지 않았다. 이미 학습한 10개 물리 특징 판정부의 저장 출력과 기존 DTW 경로를 사용했다.

목표: 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행하는지 파악하고, 중요한 차이와 허용 가능한 차이를 구분해 설명한다. 이번 완료는 그 목표의 첫 연결 기능이다. 전체 수어의 교육적 정오·허용 범위를 판단하는 AI의 완성을 뜻하지 않는다.

무엇을 했나
1. 공개 12표현의 사전 설명을 원문 출처와 연결했다. 같은 이름의 다른 뜻·다른 표현형은 한 정답으로 섞지 않았다. '1지'는 검지, '4지'는 새끼손가락, '5지'는 엄지라는 사전 표기 근거도 확인했다. '4지 옆면'을 '네 손가락'으로 읽지 않게 했다.
2. 설명에서 현재 관측할 수 있는 손모양·이동과 아직 측정하지 못하는 접촉·손바닥 방향·정확한 반복 수를 분리했다. 가려진 손을 설명에 맞춰 채우지 않는다.
3. 각 표현의 참고 3개와 다른 공개 촬영자 영상 1개를 연결했다. 참고는 ID1/2/3의36영상, 개발 비교는 ID4의12영상이다. 사용자·멘토 촬영은 열거나 학습·튜닝에 사용하지 않았다. 이전 보류·신규평가 점수도 다시 튜닝에 쓰지 않았다.
4. DTW로 대응 장면을 찾되, 동작 순서는 원래 영상 시간으로 따로 확인했다. 일부 장면이 안 보이면 비교 대상에서 지우지 않고 관측 부족으로 남겼다.
5. 화면 위·아래 이동과 '반대팔을 따라 움직임'을 구분하기 위해 양손 각각의 손바닥 중심을 반대쪽 팔꿈치→손목 축에 투영하는 공통 측정을 추가했다. 모든48영상·양쪽96궤적을 계산했다. 실제 접촉이나 올바른 수행을 증명하는 값은 아니다.

실제 결과
- 설명41항목 중 대응 장면에서 부분 특징 관측5, 차이 후보1, 근거 부족14, 설명 연결/관측 기능 미확인21이다. 사전의 대체 표현을 따로 센 항목 수이며 독립 영상41개나 정확도41건이 아니다.
- 수어·마음에들다의 오른손 검지, 감사의 양손 펼침, 별의 오른손 펼침이 일부 대응 시점에서 관측됐다. 해당 단어 전체를 잘했다고 판정한 것은 아니다.
- 순서5항목 중 순종에서 오른손 위쪽 이동2.300~2.900초→아래쪽 이동2.967~3.100초라는 일부 흐름1개가 나왔다. 나머지4개는 필요한 접촉/모음/자세 특징이 없어 판정하지 않았다. 실제 영상에는 추가 흐름도 있어 이것이 전체 의미 단계나 반복 수의 해석은 아니다.
- '멀다'는 현재 한 손 영상에 맞지 않는 두 손 설명을 강제로 적용하지 않았다.
- '잘하다'에서 원래 이동 판정부는2.967초를 참고와 다른 후보로 냈다. 영상과 손목 위치에는 내려가며 반대팔을 따라 움직이는 흐름이 보였다. 추가 팔 상대좌표는 짧은 창에서 흔들려 사전에 정한 일관 방향 조건을 통과하지 못했다. 따라서 자동 해결0건이고 후보1건은 그대로 보존했다. 이를 사용자 오류나 보완 성공으로 보고하지 않는다.

검증과 한계
현재 손·팔 관측 배열과 DTW 경로는 그대로 보존했다. 참고의 알려진 양성 표본224셀은 누락돼도 삭제하지 않았다. 이는 전체 동작시간이 아니라 사전에 주석된 특정 표본의 분모다. 원래 차이 후보420셀 중 설명 범위 밖351셀도 별도로 남겼다. 420셀과 새1항목은 단위가 달라 오류 감소율로 비교할 수 없다.
별도 Codex가 저장 비교1125검사·팔 상대좌표1265검사를 독립 계산해 불일치0을 확인했다. 엔진 반례48검사, 자체 계약39검사 통과. 다른 Codex가 모델 출력 수치를 보지 않고 공개5영상의10장/65표본을 확인했다. 같은 AI 계열의 별도 검토이며 수어 교육자 정답 검증은 아니다.
관측이 없는데 확신하기, 없는 수치를 확률처럼 쓰기, 확인되지 않은 출처 연결을 통과시키기, 확률0.2 경계 처리 오류를 수정했다. 첫 실패 증거와 초기 결과를 보존했다. 팔 상대좌표의 자동 충돌 판정은0건이며 기능 효과 미확인이다. 일부 시점에만 충돌할 때 다른 차이 후보까지 가려질 수 있는 분기는 별도 수정했다. 충돌한 시점과 남은 후보 시점을 각각 보존하는18통제를 통과했고, 현재 실제41항목 상태는 그대로다. 운영 기능으로 적용한 것은 아니다.
새 학습0·새 모델 추론0·운영 연결0. C·T3-F와 운영 웹/DTW는 유지했다. 비교 후단 계산시간만 측정했으므로 이를 전체 피드백 대기시간으로 말하지 않는다. 이번 시험으로 인식률이나 교정 정확도가 높아졌다고 주장할 수 없다.

다음 핵심 작업
새 검출기를 더하지 않고, 이번 사전 연결에서 빠진 공통 특징의 시간별 근거를 채운다. 먼저 '펼침→접근→팔을 따라 이동→분리'처럼 관측 가능한 단계와 실제 영상의 시간을 연결하고, 화면 방향 판정부와 원래 궤적이 충돌하는 구간을 일반적인 기준으로 구분한다. 손목 위치/팔 축 자체의 오차와 시간 대응 오차를 먼저 나눠야 한다.
그다음 이미 확보한 fit/dev의 시간 주석으로 작은 단계·관계 판정부를 학습할 근거가 충분한지 판단한다. 전체 장면을 단어 설명으로 정답 처리하거나, 비어 있는 단계에 가짜 정답을 넣지 않는다. 확인된 단계만 학습하고, 기존 방식과 동일한 공개 영상에서 비교한다.
평가의 핵심은 단어를 맞힌 횟수가 아니라 정상 동작을 잘못 지적한 횟수, 중요한 손모양·단계 차이를 찾은 횟수, 불충분한 관측을 정답으로 넘긴 횟수다. 실제 정상/오류의 수어학적 기준이 있는 사례와 단순한 시간·누락 통제시험을 구분한다. 확인되지 않은 허용폭과 의미상 필수 반복은 사람의 확인 없이 임의로 만들지 않는다.
현재 바로 보여줄 수 있는 피드백은 '이 시점에서 검지를 편 특징이 보입니다', '설명과 연결한 이동 예측이 참고와 다릅니다. 장면 재확인이 필요합니다', '이 부분은 손 관측 근거가 부족해 비교할 수 없습니다'이다. '틀렸으니 이렇게 고치세요'와 '이 정도 차이는 한국수어에서 허용됩니다'는 아직 검증이 부족하다.
'''

def main():
    r=read(U/'DELIVERY_RESULT.json');f=read(U/'FROZEN.json');s=read(U/'source_audit/SOURCE_CONTRACTS.json')
    contracts={x['word_id']:x for x in s['expressions']};data={};tables=[]
    for row in r['rows']:
        c=contracts[row['word_id']];clauses=[];anchors=[]
        for item in row['clauses']:
            original=next(x for x in c['clauses'] if x['id']==item['id'])
            counts=Counter(cell['state'] for ref in item['references'] for cell in ref['cells'])
            clauses.append(dict(id=item['id'],name=NAMES.get(item.get('feature'),'현재 관측 기능 밖'),status=STATUS[item['status']],quote=item['source_span'],url=item['source_url'],source_entry=original['source_entry_id'],link=item['link_status'],reason=original['rationale'],observed=item.get('observed_anchor_count',0),expected=item.get('expected_anchor_count',0),counts={CELL[k]:v for k,v in counts.items()}))
            for ref in item['references']:
                for cell in ref['cells']:
                    if cell.get('query_seconds') is None: continue
                    anchors.append(dict(ref=ref['reference'],q=cell['query_seconds'],r=cell['reference_seconds'],frame=cell['query_frame'],feature=NAMES.get(item.get('feature'),'未'),state=CELL[cell['state']],qp=cell.get('query_probability'),rp=cell.get('reference_probability')))
        counts=Counter(x['status'] for x in row['clauses'])
        tables.append('<tr><td>'+esc(row['label'])+'</td><td>'+str(counts['DESCRIBED_FEATURE_OBSERVED'])+'</td><td>'+str(counts['GUIDE_RELEVANT_DIFFERENCE_CANDIDATE'])+'</td><td>'+str(counts['INSUFFICIENT_LINKED_EVIDENCE'])+'</td><td>'+str(counts['SOURCE_LINK_WITHHELD'])+'</td></tr>')
        sources={ref['alias']:uri(ref['source']) for ref in row['reference_sources']}
        plot=None
        if row['query']=='P048':
            plot=[]
            for alias in ['P045','P046','P047','P048']:
                z=np.load(U/'relations'/(alias+'.npz'));old=next(x for x in f['rows'] if x['alias']==alias);inp=np.load(old['input']);names=old['feature_names'];j=names.index('R_hand_wrist_y')
                ts=z['timestamp_us']/1e6; mask=z['R_valid'];a,b=old['intervals'][0]
                vals=[dict(t=float(ts[i]),u=float(z['R_progress'][i]) if mask[i] else None,y=float(inp['x'][i,j]) if inp['mask'][i,j] else None) for i in range(len(ts)) if a<=ts[i]<=b]
                plot.append(dict(alias=alias,a=a,b=b,values=vals))
        visual=U.parent/'word100_stage_feedback_cycle_20261031'/'visuals'/row['query']/'sequence.jpg'
        data[row['query']]=dict(label=row['label'],query=uri(row['source']),refs=sources,clauses=clauses,anchors=anchors,orders=row['orders'],plot=plot,unsupported=row['unsupported_claims'],interval=row['known_token_interval'],sequence=uri(visual) if visual.exists() else None)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>수어 설명과 관측 연결 — 2026-10-31</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#182532;font:17px/1.7 system-ui,'Malgun Gothic',sans-serif}main{max-width:1140px;margin:auto;padding:30px 24px 90px}h1{font-size:32px;line-height:1.35}h2{font-size:24px;margin-top:36px}h3{font-size:19px;margin:10px 0}.lead{font-size:21px}.card{background:white;border:1px solid #d9e1e9;border-radius:12px;padding:22px;margin:16px 0}.note{border-left:5px solid #b87815;background:#fff8e9;padding:16px}.good{border-left:5px solid #1c7666}.small{font-size:14px;color:#48596d}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}video{width:100%;max-height:390px;background:#15212c}select,button{font:inherit;padding:8px;border:1px solid #91a1b3;border-radius:6px;background:white;max-width:100%}button{cursor:pointer}button:hover{background:#e7eef9}a{color:#165d9e}p{margin:.5em 0 1em}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border-bottom:1px solid #dce3e9;text-align:left;padding:9px;vertical-align:top;overflow-wrap:anywhere}th{background:#edf2f7}.wrap{overflow:auto}.badge{display:inline-block;background:#edf2f7;padding:3px 9px;border-radius:6px;font-size:14px}.clause{padding:14px 0;border-bottom:1px solid #ddd}.quote{font-weight:600}.anchorbox{max-height:310px;overflow:auto}svg{width:100%;height:auto}img{max-width:100%;height:auto}details{margin:15px 0}code{font-size:13px;overflow-wrap:anywhere}@media(max-width:760px){main{padding:16px 12px 50px}.grid{grid-template-columns:1fr}h1{font-size:27px}.card{padding:14px}table{font-size:13px}}
    </style><main><span class="badge">격리 연구 결과 · 공개자료 · 추가 학습 없음</span><h1>수어 설명을 영상의 관측과 연결했습니다</h1>
    <p class="lead">이제 단순한 좌표 차이 옆에 <b>“어느 설명과 관련된 차이인지”</b>를 붙일 수 있습니다.<br>아직 그 차이가 한국수어에서 틀렸는지·허용되는지까지 자동 확정할 단계는 아닙니다.</p>
    <div class="card good"><h3>이번에 실제로 추가한 기능</h3><p>전체 영상 관측 → 기존 특징 예측 → 참고별 DTW 대응 → <b>설명에 나온 특징과 연결 → 원래 시간 순서 확인 → 근거 있는 관측과 판단 불가 구분</b></p><p>기존에 학습한 관측 판정부를 사용하고, 설명을 비교에 연결하는 부분을 구현했습니다. 설명을 읽고 없던 손모양을 만들어 내거나, 단어를 맞혔다는 이유로 세부 동작도 정답이라고 하지 않습니다.</p></div>
    <div class="note"><b>발견한 중요한 문제</b><p>‘잘하다’에서 영상에는 팔을 따라 움직이는 흐름이 보이는데, 기존 이동 판정부가 “아래로 이동하지 않음”에 가까운 값을 냈습니다. 팔을 기준으로 다시 측정한 값도 짧은 구간에서 흔들렸습니다. <b>자동 해결은 0건</b>이며 차이 후보 1건은 그대로 남겼습니다. 사용자에게 고치라고 안내할 근거는 부족합니다.</p></div>
    <h2>같은 공개 영상에서 무엇이 나왔나</h2><p>참고: ID1·2·3의36영상 / 개발 비교: 다른 기록 ID4의12영상. 사용자 촬영은 쓰지 않았습니다.</p><div class="wrap"><table><thead><tr><th>표현</th><th>일부 특징 관측</th><th>차이 후보</th><th>비교 근거 부족</th><th>설명/기능 미확인</th></tr></thead><tbody>__TABLE__</tbody></table></div>
    <p class="small">단위는 사전 설명 항목(총41개)입니다. 대체 표현을 따로 센 항목이 포함됩니다. 정확도나 전체 단어 정답 수가 아닙니다. 참고의 알려진 특징 표본224셀은 보이지 않는 경우에도 남겼지만, 전체 동작 구간을 촘촘히 주석한 것은 아닙니다.</p>
    <h2>직접 확인하기</h2><p>표현을 고르면 설명·관측 결과를 함께 볼 수 있습니다. 아래 대응 시점 버튼은 기존 DTW가 맞춘 두 순간을 동시에 보여줍니다.</p><label>공개 표현 <select id="choose"></select></label><div class="card"><div class="grid"><div><h3 id="qt"></h3><video id="qv" controls preload="metadata"></video></div><div><h3>개별 참고 영상 <select id="ref"></select></h3><video id="rv" controls preload="metadata"></video></div></div><p id="scope" class="small"></p><p id="seekinfo"></p><div class="anchorbox"><table><thead><tr><th>대응 장면</th><th>특징</th><th>저장된 판단</th></tr></thead><tbody id="anchors"></tbody></table></div><p class="small">각 버튼은 원래 표본 중 DTW 대응이 있는 시점입니다. 버튼 사이의 간격은 원본1프레임 간격이 아닙니다. 영상 재생은 각각 독립적이며 버튼 클릭으로 두 시점만 맞춥니다.</p></div>
    <div id="orders" class="card"></div><div id="clauses" class="card"></div><div id="plots" class="card" hidden></div><details id="seqdetail"><summary>이 영상의 원래 시간순13개 표본 보기</summary><img id="sequence" alt="공개 영상의 원래 시간순 표본"><p class="small">기존에 저장한 공개 영상 표본입니다. 연속 영상 전체나 수어 정답 주석은 아닙니다.</p></details>
    <h2>‘중요한 차이’와 ‘허용되는 차이’까지 얼마나 왔나</h2><div class="grid"><div class="card"><h3>지금 가능한 것</h3><p>“설명에 나온 검지 펴기와 관련된 장면입니다.”<br>“이 시점에는 같은 특징이 예측됩니다.”<br>“참고와 다른 예측이지만 원본 장면 확인이 필요합니다.”<br>“손 관측 근거가 부족해 비교할 수 없습니다.”</p></div><div class="card"><h3>아직 근거가 부족한 것</h3><p>“이 단어는 완전히 맞았습니다.”<br>“반복이 한 번 부족하니 틀렸습니다.”<br>“이만큼 위치가 달라도 한국수어에서 허용됩니다.”<br>“손이 닿았고 손바닥 방향도 정확합니다.”</p></div></div>
    <p>관절 차이가 작다는 이유만으로 허용되는 수어라고 정하지 않았습니다. 관측되지 않은 부분도 잘했다고 넘기지 않습니다. 반대로 몸 크기·속도·정상 손내림 차이만으로 오류를 만들지 않습니다. 이번 좌표 이동·회전·크기 통제는 계산의 성질을 확인한 것이며 언어적 허용폭 검증이 아닙니다.</p>
    <h2>검증에서 확인한 것</h2><p>저장 비교1125검사, 팔 상대좌표1265검사를 별도 코드로 검산해 불일치0. 자체 계약39개와 별도 반례48개 통과. 다른 Codex가 모델 출력값을 보지 않고5개 공개 영상의10장·65표본을 확인했습니다. 수어 교육자의 정답 검증은 아닙니다.</p><p>원래420개 차이 후보 셀과 새1개 설명 항목은 단위가 다릅니다. <b>420→1의 오류 감소로 해석하면 안 됩니다.</b> 설명 밖351셀도 삭제하지 않고 보존했습니다.</p>
    <h2>다음 작업의 초점</h2><ol><li><b>설명과 실제 동작 단계를 연결:</b> 펼침·접근·팔을 따른 이동·분리 중 영상에서 확인되는 시점을 정리하고, 팔 축/손목 관측 오류와 DTW 대응 오류를 나눕니다.</li><li><b>작은 단계·관계 판정부의 학습 근거 확보:</b> 기존fit/dev의 검증된 시간 주석만 쓰고, 미관측과 의미가 불명확한 단계는 정답을 만들지 않습니다. 새 검출기를 추가하는 일이 우선이 아닙니다.</li><li><b>피드백 효과 비교:</b> 정상 동작에 잘못 경고하는지, 중요한 차이를 놓치는지, 관측 부족을 정답으로 넘기는지를 같은 조건에서 비교합니다. 실제 수어 정오 사례와 인위적인 누락·속도 통제를 구분합니다.</li></ol>
    <p>일부 시점의 근거가 충돌할 때 다른 후보 시점까지 숨기지 않도록 보존하는18통제도 추가했습니다. 정상 손과 교차를 포함한 기존 관측 배열·DTW 경로는 바꾸지 않았습니다. 이 사실만으로 새 피드백의 오경고가 없다고 보장할 수는 없습니다.</p><p>새 모델 학습·추론·운영 연결은 이번에 하지 않았습니다. 기존 C·T3-F와 운영 웹·DTW는 그대로입니다.</p><p><a href="RESULT_KO.txt">설명 전문</a> · <a href="review/INDEPENDENT_REVIEW_FINAL.json">독립 검산</a> · <a href="review/VISUAL_RELATION_REVIEW.json">영상 검토</a> · <a href="DELIVERY_RESULT.json">저장 결과</a> · <a href="source_audit/SOURCE_CONTRACTS.json">출처 연결</a></p>
    </main><script>const D=__DATA__;const $=s=>document.querySelector(s);const E=s=>String(s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));let current;
    function settime(v,t){if(v.readyState>=1)v.currentTime=t;else v.addEventListener('loadedmetadata',()=>v.currentTime=t,{once:true})}
    function jump(i){const a=current.anchors[i];if($('#ref').value!==a.ref){$('#ref').value=a.ref;$('#rv').src=current.refs[a.ref]}$('#qv').pause();$('#rv').pause();settime($('#qv'),a.q);settime($('#rv'),a.r);$('#seekinfo').textContent=`비교 ${a.q.toFixed(3)}초 ↔ ${a.ref} ${a.r.toFixed(3)}초 / ${a.feature} / ${a.state}`}
    function lines(vals,key,x,y){let s='',up=false;for(const v of vals){if(v[key]===null){up=false;continue}s+=(up?'L':'M')+x(v.t).toFixed(2)+','+y(v[key]).toFixed(2)+' ';up=true}return s}
    function chart(track,key,title){const vals=track.values.filter(v=>v[key]!==null);if(!vals.length)return '';let lo=Math.min(...vals.map(v=>v[key])),hi=Math.max(...vals.map(v=>v[key]));if(key==='u'){lo=Math.min(lo,0);hi=Math.max(hi,1)}if(hi-lo<.1)hi=lo+.1;const x=t=>65+(t-track.a)/(track.b-track.a)*590,y=v=>155-(v-lo)/(hi-lo)*110;let ticks='';for(let i=0;i<4;i++){let v=lo+(hi-lo)*i/3;ticks+=`<line x1="65" y1="${y(v)}" x2="655" y2="${y(v)}" stroke="#dde4e9"/><text x="58" y="${y(v)+4}" text-anchor="end" font-size="13">${v.toFixed(2)}</text>`}return `<svg viewBox="0 0 700 195" role="img" aria-label="${title}"><text x="65" y="24" font-size="16">${E(title)}</text>${ticks}<path d="${lines(track.values,key,x,y)}" fill="none" stroke="${key==='u'?'#8c4caa':'#2163a2'}" stroke-width="2.2"/><text x="65" y="182" font-size="13">${track.a.toFixed(3)}초</text><text x="655" y="182" text-anchor="end" font-size="13">${track.b.toFixed(3)}초</text></svg>`}
    function choose(){const alias=$('#choose').value;current=D[alias];$('#qt').textContent=`${current.label} · 개발 비교 ${alias}`;$('#qv').src=current.query;$('#ref').innerHTML=Object.keys(current.refs).map(x=>`<option>${x}</option>`).join('');$('#rv').src=current.refs[$('#ref').value];$('#scope').textContent='개별 참고3개 유지 · 전체 단어 정오 판정 없음 · 의미상 허용폭 미확인';$('#seekinfo').textContent='';$('#anchors').innerHTML=current.anchors.length?current.anchors.map((a,i)=>`<tr><td><button onclick="jump(${i})">${a.q.toFixed(3)}초 ↔ ${E(a.ref)} ${a.r.toFixed(3)}초</button></td><td>${E(a.feature)}</td><td>${E(a.state)}</td></tr>`).join(''):'<tr><td colspan="3">현재 설명과 연결해 비교할 알려진 참고 표본이 없습니다. 정상 수행 판정이 아닙니다.</td></tr>';
    $('#clauses').innerHTML='<h3>사전 설명과 연결된 항목</h3><p class="small">출처가 다른 항목은 대체 표현일 수 있습니다. 전부 지켜야 하는 하나의 규칙으로 합치지 않았습니다. 아래 관측 수는 알려진 참고 표본 중 양쪽 근거가 있는 수이며 전체 시간 비율이 아닙니다.</p>'+current.clauses.map(c=>`<div class="clause"><span class="badge">${E(c.status)}</span><p class="quote">“${E(c.quote)}” <a href="${E(c.url)}" target="_blank" rel="noreferrer">사전 ${E(c.source_entry)}</a></p><p>계산한 특징: ${E(c.name)} · 양쪽 관측 ${c.observed}/${c.expected} 표본</p><p>${E(c.reason)}</p><p class="small">${Object.entries(c.counts).map(([k,v])=>E(k)+': '+v).join(' · ')}</p></div>`).join('');
    $('#orders').innerHTML='<h3>원래 시간으로 확인한 순서</h3>'+(current.orders.length?current.orders.map(o=>{let w=o.witness;return `<p>설명: “${E(o.source_span)}”</p><p>${w?'일부 이동 순서 관측: '+w.before.start_seconds.toFixed(3)+'~'+w.before.end_seconds.toFixed(3)+'초 위쪽 이동 → '+w.after.start_seconds.toFixed(3)+'~'+w.after.end_seconds.toFixed(3)+'초 아래쪽 이동':'필요한 구성 특징을 관측하지 못하므로 순서를 판정하지 않음'}</p><p class="small">전체 단계·정확한 반복 수·필수 동작 누락의 판정은 아닙니다. 순서 확인에 DTW로 재배열한 시간을 쓰지 않았습니다.</p>`}).join(''):'<p>현재 확인된 설명과 관측 기능으로 실행할 순서 항목이 없습니다.</p>');
    $('#plots').hidden=!current.plot;if(current.plot)$('#plots').innerHTML='<h3>잘하다: 영상 판단과 좌표 계산이 왜 갈리는가</h3><p>왼쪽/첫 그래프의 세로축은 오른손목 높이로, 값이 커지면 화면에서 아래로 내려갑니다. 둘째 그래프는 오른손 중심을 반대팔 축에 투영한 위치입니다. 0은 관측된 팔꿈치, 1은 관측된 손목이며 실제 접촉점이 아닙니다. 선이 끊긴 곳은 미관측입니다.</p><p>전체 이동은 보이지만 짧은 창의 상대 위치는 흔들립니다. 팔 축 자체의 오차와 깊이·투영 영향도 남습니다. 곡선이 그럴듯하다는 이유로 손바닥 접촉이나 정상 수행을 확정하지 않았습니다.</p>'+current.plot.map(t=>`<h3>${t.alias} ${t.alias==='P048'?'개발 비교':'개별 참고'}</h3><div class="grid">${chart(t,'y','오른손목 높이 · 신체 크기 기준')}${chart(t,'u','반대팔 축 위의 손 중심 위치')}</div>`).join('');$('#seqdetail').hidden=!current.sequence;$('#sequence').src=current.sequence||'';}
    $('#choose').innerHTML=Object.entries(D).map(([a,d])=>`<option value="${a}">${E(d.label)} · ${a}</option>`).join('');$('#choose').value='P048';$('#choose').addEventListener('change',choose);$('#ref').addEventListener('change',()=>$('#rv').src=current.refs[$('#ref').value]);choose();</script></html>'''
    page=page.replace('__TABLE__',''.join(tables)).replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    (U/'RESULT_KO.html').write_text(page,encoding='utf8');(U/'RESULT_KO.txt').write_text(TEXT,encoding='utf8')
    (U/'REPORT_DATA.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf8')
    print('Report written: 12 public comparisons, saved-only, no inference')
if __name__=='__main__':main()
