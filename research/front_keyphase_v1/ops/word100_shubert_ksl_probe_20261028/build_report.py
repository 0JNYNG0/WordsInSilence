"""Render saved, already verified results. No inference, training or score changes."""
from pathlib import Path
import json
import html
from datetime import datetime, timezone, timedelta
from PIL import Image

U = Path(__file__).resolve().parent
def read(name):
    return json.loads((U/name).read_text(encoding='utf-8'))
E, C, F, S = [read(x) for x in ('EVALUATION.json','CORRESPONDENCE.json','FROZEN.json','CONSENSUS.json')]
A = read('REFERENCE_SCOPE_AUDIT.json')
V = read('INDEPENDENT_VERIFY.json')
assert V['status'].startswith('PASS') and A['status'].startswith('PASS')
stamp = datetime.now(timezone(timedelta(hours=9))).isoformat()

report = '''SHuBERT 한국수어 비교 결과와 추가 학습 방향

결론
이번 시험은 끝났다. 공개 한국수어에서 SHuBERT의 저장된 특징을 그대로 비교하는 방식이 기존 관절 비교보다 더 유용하다는 근거를 얻지 못했다. SHuBERT를 한국수어에 맞춰 추가 학습한 결과가 아니다. 추가 학습은 0회이며, 한국수어에 맞춘 판정부도 아직 학습하지 않았다. 따라서 이 결과로 미세조정의 가능성까지 실패로 판단해서는 안 된다. 현재 조건을 운영에 채택하지 않는다.

무엇을 했나
멀다·믿다·감사·별, 기록상 수행자 ID 01·02·03의 공개 영상 12개를 사용했다. 01·02는 참고, 03의 4영상은 비교 대상이다. 모두 이미 사용한 개발 자료이며 새로운 독립 성능 평가가 아니다. 영상당 미리 정한 9시점과 각 시점 앞뒤 0.2초를 확인했다. 원영상 30fps의 전체 화면 관측을 재사용했으며 웹 상반신 범위와 같은 조건은 아니다.

기존 관절 기준은 실험용 69특징 비교이고, C·T3-F의 단어 인식률이나 현재 웹 DTW 전체와 비교한 시험이 아니다. 이번에 새로 신경망을 돌리지 않고 앞서 저장한 관절·DINO·SHuBERT 특징을 재사용했다. 학습·새 다운로드·운영 교체·사용자 및 멘토 촬영 접근은 하지 않았다.

눈으로 확인할 기준
서로의 답을 모르는 두 Codex 검토자가 단어명과 모델 결과를 보지 않고 36장의 배열 이미지, 총108시점을 각각 읽었다. 확인 항목은 검지만 편 모양, 네 손가락을 편 모양, 손과 얼굴의 화면상 겹침, 양손의 겹침, 위아래 이동, 양손이 가까워지거나 멀어짐이다. 1080항목 중711개는 두 검토자의 확인값이 같았고,5개는 달랐으며364개는 한쪽 이상이 불확실했다. 불확실한 값은 정답으로 채우지 않았다. 이들은 AI의 잠정 관측이며 한국수어 전문가나 독립 사람의 정답이 아니다. 화면에서 겹친다는 판단도 실제 접촉의 증명이 아니다.

동작 시점 비교 결과
대상 활동 구간의 전체240항목 중 대상 관측 근거가 부족한96항목은 제외 사유와 함께 남겼다. 남은144항목에는 대응 참고의 근거가 부족한 경우도 포함된다. 그중 세 방법 모두 판독할 수 있었던 같은132항목을 비교했다. 기존 관절+DTW는120항목, SHuBERT+DTW는118항목, 단순 시간비율 맞춤은118항목에서 잠정 관측이 일치했다. 움직임만 보면 같은38항목에서34,32,32개였다. 정적인 특징94항목에서는 세 방법 모두86개였다. 이 숫자는 영상 수나 단어 정답률이 아니라 시점·특징·참고의 조합 수다. 대상 미관측 때문에 분모가 작아지는 주평가의 한계를 보완하기 위해 다음의 참고 기준 감사를 별도로 수행했다.

쉬고 있는 처음·끝까지 포함한 전체 공통348항목에서는 기존326개, SHuBERT327개, 시간비율321개가 일치했다. 이 작은 전체 차이를 실제 수어 활동 구간의 개선이라고 해석하지 않았다. 활동 구간의 동일 비교 범위에서는 우위를 확인하지 못했다. 특정 단어만 골라 결론을 내리지 않았고 모든8참고쌍·9시점을 아래 화면에 남겼다.

비교 범위가 사라지지 않는지 추가 확인
DTW 경로 자체는9참고시점을 모두 통과하지만, 비교 대상 시점마다 대표 참고 하나를 고르면24경로 중7개에서는 참고1시점이 대표에서 빠진다. 이를 숨기지 않고 모든 참고 시점을 기준으로 하는 보충 감사를 추가했다. 기존 주평가는 고치지 않았다. 참고의 활동 구간 중 근거 있는173항목을 세 방법에서 고정하자, 기존은 일치126·차이11·대상 관측부족36, SHuBERT는126·13·34, 시간비율은116·14·43이었다. 대상 손이 안 보인다고173항목을 줄이지 않았다. 대상 관측을 모두 없애는 통제에서도 기대 범위가 그대로 남았다. 이는 범위 보존 검산이며 교육적 정확도 검증은 아니다.

손 모양 비교 결과
같은 쪽 손의 다른 단어 참고에서 가장 가까운 장면을 하나 찾았다. 총144항목 중117개는 두 AI 관측으로 대상 손 모양을 확정하지 못했고,4개는 다른 단어 참고에 필요한 정답 종류가 없었고,2개는 현재 관측 근거가 없었다. 사진 자체를 사람이 판독할 수 없다는 증명은 아니다. 남은21항목에서 기존 관절은 일치17·차이0·보류4, 손 DINO는4·12·5, SHuBERT는2·1·18이었다. SHuBERT는 손 모양을 확정하기 어려운 참고 장면을 자주 골랐다. 세 방법 모두 판단 가능한 공통 항목은 겨우2개였다. 따라서 이 값들을 손 모양 인식 정확도로 부르거나 모델 전체의 우열로 일반화할 수 없다. 한 손 특징과 전체 문맥 특징은 입력 정보도 다르다. 여기서 보류는 교육용으로 검증된 모델의 자신감 출력이 아니라 잠정 관측·현재 검출 근거가 부족하다는 뜻이다.

정상 손과 교차
원래 관절과 RGB, 기존 좌우 배정, 모델을 수정하지 않았다. 겹침 장면도 평가에서 유지하고 불확실한 손 모양을 억지로 정하지 않았다. 이는 입력 보존이며, 정상 교차의 검출 정확도가 좋아졌다는 증거는 아니다. 손 복구 모델을 이번에 새로 학습한 것도 아니다.

왜 추가 학습을 안 했나
처음 비교는 사전학습 특징 자체에 이득이 있는지 보는 단계였다. 현재 자료는 단어 ID·전체 활동 구간·일부 설명이 있으나, 모든 영상의 손 모양과 단계별 실제 상태를 확정한 학습 정답은 아니다. 사전 설명을 모든 순간에 복사하거나 AI 관측을 독립 정답처럼 쓰면 잘못된 정보를 학습할 수 있다. 이 문제 때문에 학습 단계까지 완료했다고 하지 않는다. 다만 학습 전에 이미 우수해야만 학습할 수 있다는 뜻은 아니다. 동결 특징 비교의 부진은 미세조정을 배제할 이유가 아니다.

Fine-tuning은 가능한가
가능하며 목적에 맞는 후속이다. 공식 SHuBERT 논문 부록B·표9는 ASL 자료에서 모델 전체와16개 속성 분류기를 추가 학습한 결과다. 단순한 특징 거리 비교와 다른 조건이다. 영상 전체의 손 모양·위치·움직임 속성 결과이며 한국수어의 특정 시점 교정 정확도는 아니다.
공식 논문: https://aclanthology.org/2025.acl-long.1397.pdf
추가로2026년 터키수어 번역 연구는 동결 SHuBERT와 미세조정을 비교했고 미세조정 조건의 이득을 보고한다. 이는 다른 수어에 적응시킬 가능성의 근거이지만 우리 한국수어 손가락·단계 피드백 성능을 보장하지 않는다. 이번에는 해당 연구의 공식 초록만 확인했으며 PDF 요청은 실패했다.
공식 초록: https://aclanthology.org/2026.wslp-1.16/

다음 작업 제안
1. 모델을 또 바꾸기보다, 보유 한국수어에서 학습과 평가에 쓸 작은 속성·단계 묶음을 먼저 확정한다. 손별 모양·이동 방향·얼굴/반대손과의 관계·변화 구간을 분리하고 사람별로 자료를 나눈다. 현재 사진에서 직접 보인 값과 앞뒤 흐름으로 추정한 값은 별도다. 손이 가려진 경우는 틀린 동작으로 가르치지 않는다. 모델 오답만 고르거나 불확실 장면을 삭제하지 않으며, 필요한 종류가 참고와 평가 양쪽에 있는지 먼저 확인한다.
2. 확인된 자료로 SHuBERT를 고정한 작은 시간별 판정부를 먼저 학습한다. 기존 관절만 사용하는 동일 목적 판정부와 비교한다. 이것은 출력 부분 학습이며 SHuBERT 본체의 미세조정과 구분한다. 원시 특징 거리가 못 구분한 정보를 판정부가 배울 수 있는지 확인하는 시험이다.
3. 자료와 계산 자원이 감당되면 SHuBERT 일부 층 미세조정을 별도 후보로 비교한다. LoRA는 가능한 구현 선택지이지 자동 성능 향상 기능이 아니다. 현재 노트북에서 학습 가능 시간·메모리는 미측정이므로 전체 모델 미세조정부터 시작하지 않는다. 사용자 촬영은 어떤 경우에도 학습·튜닝·선택에 쓰지 않는다.
4. 사람별 미사용 평가에서 손 모양·시점·순서·관측부족과 정상 차이 보존을 검증한 후보만 DTW 피드백에 연결한다. 단어 이름만 맞추는 학습으로 교육적 오류 교정까지 검증했다고 하지 않는다. 중요 특징과 허용 경계의 교육적 정오에는 신뢰할 별도 확인이 필요하다. 모든 단어 모든 프레임을 수동 검토하자는 계획은 아니다.

눈·입을 제외하면 빨라질까
얼굴의 위치 좌표를 찾는 기능과 얼굴 RGB를 DINO로 계산하는 기능은 다르다. 표정 판정은 목표에서 제외하고, 코·이마·턱 등과 손의 관계를 확인할 위치 근거는 유지한다. 현재 SHuBERT 시험은 얼굴 사진도 특징으로 바꾸지만 별도의 표정 피드백을 출력하지 않는다. 이전 CPU3영상 측정에서 얼굴 DINO만4.71~6.04초였다. 얼굴 좌표 검출은 캐시를 재사용했으므로 그 검출 시간은 측정하지 않았다. 얼굴 RGB 계산을 빼는 것은 절약 후보이지만 실제 절약과 품질 유지 실험은 아직 하지 않았다. 사전학습 모델의 입력을 그냥0으로 바꾸면 입력 조건이 달라진다. 품질 개선 가능성을 확인하면서 별도 조건으로 검증해야 한다.

GPU와 출시 환경
이 노트북에는 Intel Arc140V GPU가 있지만 현재 시험은 CPU용 PyTorch에서 실행했다. GPU가속 효과를 아직 측정하지 않았다. 로컬에서 실행하는 제품이면 고객의CPU/GPU·실행환경·전원상태 등에 따라 처리시간이 달라진다. 동일 노트북이 꼭 필요한 것은 아니며 비슷한 사양도 같은 시간을 보장하지 않는다. 서버 분석이라면 서버와 네트워크가 영향을 주지만 이번에는 서버 전송·출시를 진행하지 않았다. 이전14.47~17.87초는 저장된 사진부터 특징 생성까지이며 촬영부터 피드백까지의 전체 시간이 아니다.

검산과 보존
별도 Codex 에이전트가 저장 입력·마스크·거리·DTW·결과 합계를 검산했다. 주요 거리 차이는 부동소수점 수준이며 같은 매칭과 합계를 확인했다. 두 이미지 검토자, 구현자, 산술 검증자를 구분했다. 같은 AI 도구의 별도 검산은 독립 사람 정답과 다르다. 최초 JSON 저장 실패는 FIRST_FAILURE.json에 보존했고 numpy 정수를 표준 정수로 바꾼 뒤 같은 조건으로 완료했다. 기존 동결 결과는 덮어쓰지 않았다.

이번 완료 범위: 고정 특징 비교·잠정 RGB 관측 대조·범위 보존 감사·보고.
아직 미완료: 한국수어 맞춤 학습, 교육용 단계·허용범위 정답 검증, 얼굴 RGB 생략 비교, GPU 가속, 실제 웹 피드백 효과.
'''
(U/'RESULT_KO.txt').write_text(stamp+'\n\n'+report,encoding='utf-8')

# Derivative images for presentation only; preserve all original review images.
out=U/'display';out.mkdir(exist_ok=True)
for row in C['sample_metadata']:
    sid=row['sample_id']; clip,idx=sid.split('_'); idx=int(idx)-1
    sheet=U/'blind'/f'{clip}_{idx//3+1}.jpg'
    with Image.open(sheet) as im:
        w,h=im.size; rh=h//3; cw=w//3
        im.crop((0,(idx%3)*rh,w,(idx%3+1)*rh)).save(out/f'{sid}_context.jpg',quality=92)
        im.crop((cw,(idx%3)*rh,2*cw,(idx%3+1)*rh)).save(out/f'{sid}.jpg',quality=92)

names={'geometry':'기존 관절 비교','shubert':'SHuBERT 특징 비교','dino':'손 DINO 특징 비교','time_fraction':'단순 시간비율'}
def table_stats(stats):
    rows=''.join(f'<tr><th>{names[k]}</th><td>{v["agree"]}</td><td>{v["different"]}</td><td>{v["unknown"]}</td><td>{v["total"]}</td></tr>' for k,v in stats.items())
    return '<table><thead><tr><th>방법</th><th>관측 일치</th><th>관측 차이</th><th>근거 부족</th><th>범위</th></tr></thead><tbody>'+rows+'</tbody></table>'
def bars(stats):
    ans=''
    for k,v in stats.items():
        spans=''.join(f'<span class="{field}" style="width:{100*v[field]/v["total"]}%" title="{field}: {v[field]}">{v[field] if v[field] else ""}</span>' for field in ['agree','different','unknown'])
        ans+=f'<div class="barrow"><label>{names[k]}</label><div class="bar">{spans}</div></div>'
    return ans
def section(title,body):return f'<section><h2>{title}</h2>{body}</section>'
active=E['stage']['annotation_active']['all_eligible']
shape=E['shape']['all']['all_eligible']
refstats={k:{'agree':v['agree'],'different':v['different'],'unknown':v['query_unknown_in_fixed_expected_scope'],'total':v['expected_reference_cells']} for k,v in A['reference_annotation_active'].items()}
content=section('이번 결론', '<p class="lead">아직 추가 학습은 하지 않았고,<br>그대로 가져온 SHuBERT의 개선 효과는 확인하지 못했습니다.</p><p>공개 한국수어 4표현·기록 수행자 ID3개·12영상을 사용한 작은 비교입니다. 미국수어로 사전학습한 특징을 그대로 맞춰 보는 시험이며, 한국수어에 맞춰 미세조정한 모델의 결과가 아닙니다. C·T3-F와 웹 기능은 그대로입니다.</p><div class="callout">이 결과는 “추가 학습해도 안 된다”는 결론이 아닙니다. 지금의 특징 거리 비교를 바로 제품에 넣을 근거가 부족하다는 뜻입니다.</div>')
content+=section('1. 실제 동작 구간에서 더 잘 맞췄나', '<p>활동 구간의 공통 판독 가능 <b>132항목</b>에서 기존 <b>120</b>, SHuBERT <b>118</b>, 시간비율 <b>118</b>개가 일치했습니다. 움직임만 보면 같은38항목에서34·32·32개였습니다.</p><p>대상 활동 구간 전체240항목 중 대상 근거 부족96항목은 따로 기록했습니다. 아래144항목에는 대응 참고의 근거가 부족한 경우도 포함합니다. 파랑=관측 일치, 빨강=관측 차이, 회색=근거 부족. <b>단어 인식률이나 교육적 정답률이 아닙니다.</b></p>'+bars(active)+table_stats(active)+'<details><summary>처음·끝의 정지 구간까지 포함하면?</summary><p>세 방법 공통348항목에서326·327·321개가 일치했습니다. SHuBERT의 전체 +1항목을 실제 활동 구간의 개선이라고 해석하지 않았습니다.</p></details>')
content+=section('2. 손이 안 보인다고 비교 범위가 줄었나', '<p>대표 장면 선택만 사용하면 일부 참고 시점이 빠질 수 있어, <b>참고 시점 전체를 다시 기준</b>으로 삼는 보충 감사를 했습니다. 대상 관측이 부족해도173항목을 유지했습니다. 원래 평가는 변경하지 않았습니다.</p>'+table_stats(refstats)+'<p>활동 구간의 일치 수는 기존과 SHuBERT 모두126개입니다. 모든 대상 관측을 없애는 통제에서도 참고의 기대 범위는 유지됐습니다.</p>')
content+=section('3. 손 모양을 더 잘 구별했나', '<p>144항목 중 두 AI 관측으로 확정하지 못한117개, 참고의 필요한 종류 부재4개, 현재 센서 근거 부족2개를 각각 기록했습니다. 남은21항목에서 다른 단어의 가장 가까운 손 장면을 찾은 결과입니다.</p>'+table_stats(shape)+'<p><b>세 방법 모두 판독 가능한 공통 항목은 2개뿐</b>입니다. 이 수치를 손 모양 인식 정확도로 일반화할 수 없습니다. 현재는 SHuBERT가 고른 참고의 손 모양을 확인하기 어려운 경우가 많습니다. 한 손 특징과 전체 문맥 특징은 사용하는 정보도 다릅니다.</p>')
content+=section('4. 직접 장면 비교하기', '<p>네 표현의 모든8참고쌍과9시점을 확인할 수 있습니다. 이 슬라이더의 한 칸은 <b>원영상1프레임이 아니라 미리 고른 다음 시점</b>입니다. 움직임은 “앞뒤0.2초” 보기에서 확인하세요. 화면의 자르기는 보기용이며 모델은 전체 화면 관측을 사용했습니다.</p><div class="controls"><select id="pair" aria-label="참고쌍"></select><button id="prev">이전 시점</button><input id="time" aria-label="시점" type="range" min="0" max="8" value="5"><button id="next">다음 시점</button><label><input type="checkbox" id="context"> 앞뒤0.2초 함께 보기</label></div><p id="position"></p><div id="cards" class="cards"></div><p id="detail"></p><p>두 AI 검토자의 관측은 잠정값입니다. 특히 손·얼굴의 화면상 겹침은 실제 접촉이나 잘못된 수어라는 뜻이 아닙니다.</p>')
content+=section('5. 눈·입과 처리시간', '<p><b>얼굴 부위 좌표는 유지하고, 얼굴 사진의 무거운 특징 계산은 별도로 줄일 수 있는지 시험</b>하는 방향입니다. 현재 표정 피드백은 없습니다. 기존 CPU 측정에서 얼굴 DINO 계산은4.71~6.04초였지만, 얼굴 좌표 검출 시간은 측정하지 않았습니다.</p><p>사전학습 SHuBERT는 얼굴 입력까지 받도록 배웠기 때문에 임의 생략이 성능을 유지한다고 가정할 수 없습니다. 이번에는 얼굴 제거를 실행하지 않았습니다. 노트북의 Arc140V GPU도 아직 이 경로에서 시험하지 않았습니다.</p>')
content+=section('6. 추가 학습은 타당한 다음 단계', '<p>공식 연구는 모델 전체와 속성 판정부를 미세조정해 미국수어의 손 모양·위치·움직임 속성을 평가했습니다. 우리처럼 원시 특징 거리만 비교한 조건과 다릅니다. <a href="https://aclanthology.org/2025.acl-long.1397.pdf">SHuBERT 공식 논문 부록B·표9</a></p><p>우리의 다음 순서는 <b>확인 가능한 한국수어 학습·평가 묶음 → 작은 시간별 판정부 학습 → 필요하면 일부 층 미세조정 → 기존 관절 기준과 비교</b>입니다. 본체를 고정한 판정부 학습과 본체 가중치를 바꾸는 미세조정을 구분합니다. 단어명만 배우는 것과 어느 시점에 무엇이 달랐는지 배우는 것은 다릅니다.</p><p>정답 자료 준비를 위해 모든 프레임을 사람이 읽자는 뜻은 아닙니다. 사전 설명과 기존 주석을 활용하되 실제 보이는 값·시간 문맥 추정·확인 불가를 분리하고, 최종 성능을 판정할 작은 기준은 별도로 확인해야 합니다. AI의 의견 일치를 독립 사람 정답으로 삼지는 않습니다.</p>')
content+=section('전체 설명과 검증 근거', '<p><a href="RESULT_KO.txt">쉬운 설명 전체</a> · <a href="EVALUATION.json">주평가</a> · <a href="REFERENCE_SCOPE_AUDIT.json">참고 범위 보충 감사</a> · <a href="INDEPENDENT_VERIFY.json">별도 산술 검산</a> · <a href="CONSENSUS.json">잠정 관측과 불확실 항목</a> · <a href="FROZEN.json">사전 고정 조건</a></p><p>추가 학습0 · 이번 새 신경망 추론0 · 사용자/멘토 촬영 접근0 · 운영 변경0. 출력 오류 최초 기록은 보존했습니다. 이번 완료는 작은 비교 시험이며 전체 피드백 프로젝트의 완성이 아닙니다.</p>')

payload={'pairs':C['stage_pairs'],'meta':{r['sample_id']:r for r in C['sample_metadata']},'clips':{r['alias']:r for r in F['rows']},'values':S['values'],'names':names}
script='''
const D=__PAYLOAD__;
const pair=document.getElementById('pair'), time=document.getElementById('time'), ctx=document.getElementById('context');
const unique=[...new Set(D.pairs.map(p=>p.query+'|'+p.reference))];
for(const key of unique){const [q,r]=key.split('|'),o=document.createElement('option');o.value=key;o.textContent=D.clips[q].label+' · 비교 ID03 / 참고 ID'+D.clips[r].recorded_person_id;pair.appendChild(o);}
pair.value=unique.find(x=>x==='V06|V05')||unique[0];
const vlabel={up:'위로',down:'아래로',still:'뚜렷한 이동 없음',closer:'가까워짐',farther:'멀어짐'};
function card(title,sid){const m=D.meta[sid],v=D.values[sid];return '<article><h3>'+title+'</h3><img alt="'+sid+'" src="display/'+sid+(ctx.checked?'_context':'')+'.jpg"><p>'+sid+' · '+(m.timestamp_us/1e6).toFixed(2)+'초 · 원본 프레임 '+m.frame_id+'</p><p>오른손: '+(vlabel[v[7]]||'확인 불가')+' / 왼손: '+(vlabel[v[8]]||'확인 불가')+'</p><p>현재 손 사진 근거: 오른손 '+(m.R_current?'있음':'없음')+' · 왼손 '+(m.L_current?'있음':'없음')+'</p></article>';}
function draw(){const [q,r]=pair.value.split('|'),ps=D.pairs.filter(p=>p.query===q&&p.reference===r),i=+time.value,qs=ps[0].query_sample_ids[i];let s=card('비교 대상 · 공개 ID03',qs);for(const p of ps)s+=card(D.names[p.method],p.reference_sample_ids[p.mapping[i]]);document.getElementById('cards').innerHTML=s;document.getElementById('cards').classList.toggle('context',ctx.checked);document.getElementById('position').textContent=(i+1)+'/9시점 · 원자료 활동구간 '+(D.meta[qs].annotation_active?'안':'밖');document.getElementById('detail').textContent='위 그림은 비교 대상 기준의 대표 장면입니다. 대표에서 빠지는 참고가 없는지는 별도 참고 범위 감사에서 확인했습니다.';}
pair.onchange=draw;time.oninput=draw;ctx.onchange=draw;document.getElementById('prev').onclick=()=>{time.value=Math.max(0,+time.value-1);draw();};document.getElementById('next').onclick=()=>{time.value=Math.min(8,+time.value+1);draw();};draw();
'''.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/'))
css='''body{font-family:Segoe UI,Malgun Gothic,sans-serif;background:#f3f5f8;color:#17283a;margin:0;line-height:1.75}main{max-width:1180px;margin:auto;padding:36px 22px 70px}header{padding:22px 4px}h1{font-size:32px;line-height:1.35}h2{font-size:23px;line-height:1.4}h3{font-size:17px}section{background:white;border:1px solid #dce3eb;border-radius:14px;padding:24px;margin:20px 0}.lead{font-size:26px;font-weight:700;line-height:1.5}.callout{background:#fff4d5;padding:16px;border-left:5px solid #ba8118}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:10px;border-bottom:1px solid #dce3eb;text-align:left}thead{background:#edf2f8}a{color:#155bb2}.barrow{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center;margin:14px 0}.bar{display:flex;height:32px;background:#d2d8e0;border-radius:6px;overflow:hidden;font-size:12px;text-align:center}.bar span{display:block}.agree{background:#2677ba;color:white}.different{background:#b64949;color:white}.unknown{background:#cdd3db;color:#243244}.controls{display:flex;align-items:center;flex-wrap:wrap;gap:12px}button,select{font:inherit;padding:7px;border:1px solid #b5c1ce;border-radius:6px;background:#f4f7fa}input[type=range]{width:200px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.cards.context{grid-template-columns:1fr 1fr}.cards article{border:1px solid #dde3ea;border-radius:8px;padding:10px;min-width:0}.cards img{width:100%;height:auto}.cards p{font-size:12px;margin:6px 0}summary{cursor:pointer;font-weight:600}.muted{color:#586a7d}@media(max-width:850px){.cards{grid-template-columns:1fr 1fr}.cards.context{grid-template-columns:1fr}.lead{font-size:22px}.barrow{grid-template-columns:1fr;gap:4px}main{padding:15px}section{padding:16px}}'''
page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SHuBERT 한국수어 비교와 추가 학습 방향</title><style>'+css+'</style><main><header><div class="muted">WordsInSilence · 공개 자료 연구 비교 · '+stamp+'</div><h1>SHuBERT는 지금 무엇까지 확인됐나</h1><p>추가 학습 전 비교 결과와 한국수어 맞춤 학습의 다음 조건</p></header>'+content+'</main><script>'+script+'</script></html>'
(U/'RESULT_KO.html').write_text(page,encoding='utf-8')
(U/'REPORT_MANIFEST.json').write_text(json.dumps({'created':stamp,'report':'RESULT_KO.html','visual_samples':108,'comparison_pairs':8,'training_runs':0,'new_model_forward_calls':0,'face_ablation':False,'quality_improvement_demonstrated':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'report':str(U/'RESULT_KO.html'),'samples':108,'status':'REPORT_WRITTEN'},ensure_ascii=False))
