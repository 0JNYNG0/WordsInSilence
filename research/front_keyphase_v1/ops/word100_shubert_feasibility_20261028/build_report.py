from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import html,json,statistics
U=Path(__file__).resolve().parent
def read(name):return json.loads((U/name).read_text(encoding='utf-8'))
v=read('FEATURE_VERIFICATION.json');lat=read('LATENCY_ISOLATED.json');result=read('PUBLIC_FEATURE_RESULT.json')
rows=lat['rows'];times=[r['timing']['cached_crops_to_features_seconds'] for r in rows]
assert len(rows)==3 and v['clips']==12 and v['frames']==1438
ind=read('ADAPTER_INDEPENDENT.json');assert ind['status']=='PASS'
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
decision='SHuBERT를 먼저 시험할 연구 후보로 유지한다. 현재 CPU 경로를 그대로 게임에 채택할 단계는 아니다.'
text=f'''SHuBERT 공개 한국수어 실행 시험 — {now}

결론
{decision}
그동안 검토한 후보 중 수어의 손·얼굴·몸과 시간 흐름을 함께 배우고, 수어 속성을 평가한 직접 연구 근거가 있어 우선했다. 이번에는 공식 가중치를 실제 노트북에서 실행했다. 한국수어 교정 정확도가 개선됐거나 최적 모델임을 증명한 것은 아니다.

1. 이번에 실제로 끝낸 일
- 공식 SHuBERT와 손/얼굴 DINO 가중치 3개 약1.283GB 확보, 출처·파일 해시 고정.
- 기존 Python 환경의 CPU에서 실행하도록 최소 연결부 구현. 새 라이브러리 설치 없음.
- 공개 한국수어 4표현(멀다·믿다·감사·별), 기록상 3수행자, 12영상의 원본 1,438프레임 전체 입력과 특징 추출 완료.
- 원본 시점/관측 배열과 현재 관측·이전 이미지 복사 출처를 보존했다. 실패 영상을 제외하지 않았다.
- 별도 Codex 검산 담당이 공식 소스 계산식과 실제 가중치로 11시험(공개 V01 포함)을 비교해 최대차이0을 확인했다. 자체검사와 별도 구현 검산이며 독립 수어 전문가의 정답 확인은 아니다.
- 별도 저장값 검산에서 12개 출력, 파일 해시, 시점, 관측 출처, 역순/중간제거 계산을 확인했다.
- 기존 C·T3-F·운영 웹·DTW 변경0, 사용자/멘토 촬영 접근·학습·튜닝·선택0, 새 학습0.

2. 실제 속도
다른 에이전트의 모델 계산과 겹치지 않게 고정 3영상을 다시 측정했다. CPU FP32,4스레드,배치8이다.
'''
for r in rows:
    t=r['timing']
    text+=f"- {r['alias']} {r['duration_seconds']:.2f}초 영상: 손·얼굴 이미지 특징 {t['three_dino_streams_seconds']:.3f}초 + SHuBERT {t['shubert_seconds']:.3f}초, 합계 {t['cached_crops_to_features_seconds']:.3f}초.\n"
text+=f'''중앙값 {statistics.median(times):.3f}초. 첫 SHuBERT 호출 약1.09초, 이후 약0.08초다. 모델 파일 로딩 약0.865초는 별도이며 파일 캐시가 따뜻한 상태일 수 있다.
이 숫자는 이미 준비된 손/얼굴 이미지와 관절부터의 처리시간이다. 원영상 읽기·MediaPipe 검출·이미지 잘라내기·DTW·피드백 문장·화면 표시까지의 전체 사용자 대기시간은 아직 측정하지 않았다.
영상 재생과 분석을 함께 진행하는 설계는 타당하지만, 약4초 영상 재생만으로 약16초 계산이 모두 가려지는 것은 아니다. 참고 특징은 미리 계산할 수 있어도 새 사용자 영상의 이미지 처리 비용은 남는다. GPU/NPU 가속과 재생 중 실제 추가 대기시간은 미시험이다.

3. 알아낸 핵심 문제
가. 병목은 SHuBERT의 시간 흐름 처리보다 세 RGB 스트림의 DINO 이미지 특징 추출이다. 그래서 새 모델을 바로 찾기보다 이 비용과 필요한 표본 수를 확인해야 한다.
나. 전체 2,876손-프레임 중 2,764는 현재 관측,112(왼손51·오른손61)는 직전 손 이미지를 복사한 입력이다. 공식 처리의 호환을 위해 복사했지만 source_kind=2로 보존했다. 모델에는 이 표지가 별도 입력되지 않는다. 이런 특징을 현재 손모양을 확인한 증거로 쓰거나, 가려진 손을 복원했다고 표현하면 안 된다. 주변 프레임에도 문맥 영향이 갈 수 있다.
다. 이번 모델 입력은 전체 영상 native 관측이다. 상반신 제한 masked 자료는 보존·대조했으나 SHuBERT 입력에 아직 적용하지 않았다. 실제 웹캠처럼 화면 밖 손이 없는 조건과 동일한 성능 비교가 아니다.
라. 원본30fps 전체를 보존한 실행이다. 논문 사전학습은 한 프레임씩 건너뛰어 평균14.89fps였다. 정확도와 속도를 판정할 다음 비교에는 이 시간 간격 차이도 고정 조건으로 다뤄야 한다. 짧은 동작을 놓치지 않는 확인 없이 임의로 프레임을 줄이지 않는다.

4. 확인한 것과 아직 확인하지 못한 것
- 정상 손·손 교차: 기존 입력 배열과 원시점은 유지됐다. 새 검출기를 돌리거나 좌우 배정을 바꾼 시험이 아니므로 손 인식 정확도 향상을 뜻하지 않는다.
- 순서/짧은 구간: 역순 및 중간10% 제거에서 특징 값이 변했다. 순서 변화에 계산이 반응한다는 뜻이며, 올바른 수어/틀린 수어를 구별했다는 뜻은 아니다.
- 속도: 0.75/1.25는 같은 프레임에 시간표만 바꾼 기술 점검이다. 실제 빠른 동작·흔들림·다른 수행자의 속도 차이 정확도 시험이 아니다.
- 재현: 같은 프로세스의 V01 반복 차이0. 별도 지연 측정의 RGB/body 특징 차이0, 최종 특징 최대차이0.0000419617. 별도 실행까지 완전히 같은 비트라고 보고하지 않는다.
- 손모양/단계/교육 피드백: 이를 판단하는 한국수어 판정부를 아직 학습하지 않았다. 기존 AI 관측 초안이나 사전의 대표 설명을 독립 정답으로 쓰지 않는다. 손모양 정확도·순서 오류 발견율·허용 차이 판별률은 아직 없다.

5. 다음 한 작업과 통과 기준
다음 목표는 '실제 동작 구분에 도움이 되는지, 기존 관절 기준보다 잘못된 지적을 줄이는지'를 작은 공개 한국수어 비교에서 확인하는 것이다.
(1) 현재 저장 특징을 재사용하며, 보이는 손의 모양·손의 위치 관계·변화 순서와 확인 불가 구간을 구분하는 소표본 평가 기준을 독립적으로 확인한다. 사전의 기대 동작을 촬영된 실제 동작 정답으로 복사하지 않는다. 언어적 허용 경계는 단순 영상 관측과 구분한다.
(2) 웹 상반신 사용범위·화면 밖 손 표지를 연결하고, 원30fps/사전학습에 가까운15fps 조건을 미리 고정한다. 전체 원프레임과 짧은 변화의 근거는 보존한다. 고속화만으로 채택하지 않는다.
(3) 검증된 관측 정답이 확보되면 가중치를 고정한 특징 위에 작은 속성·단계 판정부를 학습하고, 기존 관절 기준과 같은 사람 분리로 비교한다. 그동안 정상 오차/부족한 관측을 교육 오류로 바꾸지 않는다.
(4) 품질 이득이 있는 조건에 한해 Intel Arc 활용 또는 이미지 특징 계산 최적화, 촬영/재생 중 분산처리를 시험한다. 최종 평가는 촬영 종료→피드백 표시 시간이다. 3초를 기계적인 탈락 기준으로 삼지 않는다.
통과 조건은 정상 동작의 잘못된 교정 감소, 중요한 차이 구별, 관측 부족의 보류, 짧은 변화 보존이다. 단어 검색 점수나 기술 출력 변화만으로 통과시키지 않는다.

현재 판단의 범위
연구 방향과 첫 후보 선택은 타당하다. 실행 가능성은 실제 확인했다. 한국수어 피드백 품질과 현재 노트북의 최적 실행 방식은 아직 검증해야 한다. SHuBERT는 가이드를 읽고 바로 교정문을 생성하는 완성 AI가 아니라 시간별 관찰 특징을 만드는 기반이다. 한국수어 설명 연결·단계 판단·여러 참고 비교는 별도 구성한다. 얼굴 표정 채점은 도입하지 않았다.

검토 담당
Codex 주 담당: 확보·CPU DINO·실제 실행·저장 검산·통합 보고.
shubert_fit_audit: CPU 연결부·자체 검사·한계 반대검토.
shubert_adapter_verifier: 공식 소스와 독립 구현 동등성 검산11건.
shubert_public_probe_design: 공개 입력·출처 보존 및 별도3영상 지연.

공식 근거
https://aclanthology.org/2025.acl-long.1397.pdf
https://github.com/ShesterG/SHuBERT
사전학습 외국 수어의 규칙을 한국수어 정답으로 복사하지 않았다. 공식 README의 primarily MIT만으로 모든 가중치의 상업적 사용 조건을 확정하지 않는다.

실행 근거
AUTHORIZATION.json / FROZEN_FEASIBILITY.json / PUBLIC_INPUT_PROTOCOL.json
inputs/MANIFEST.json / inputs/VERIFY.json / ADAPTER_INDEPENDENT.json
PUBLIC_FEATURE_RESULT.json / FEATURE_VERIFICATION.json / LATENCY_ISOLATED.json
FEASIBILITY_REVIEW.txt
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
table=''.join(f"<tr><td>{r['alias']}</td><td>{r['duration_seconds']:.2f}초</td><td>{r['timing']['three_dino_streams_seconds']:.2f}초</td><td>{r['timing']['shubert_seconds']:.2f}초</td><td><b>{r['timing']['cached_crops_to_features_seconds']:.2f}초</b></td></tr>" for r in rows)
links=''.join(f'<a href="{n}">{html.escape(n)}</a> ' for n in ['RESULT_KO.txt','FEATURE_VERIFICATION.json','LATENCY_ISOLATED.json','ADAPTER_INDEPENDENT.json','FEASIBILITY_REVIEW.txt'])
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SHuBERT 실제 노트북 시험</title>
<style>body{{background:#f2f5f7;color:#152336;font:17px/1.7 system-ui,'Malgun Gothic';margin:0}}main{{max-width:1050px;margin:auto;padding:42px 22px}}h1{{font-size:34px;line-height:1.3}}h2{{font-size:24px}}section{{background:white;border-radius:16px;padding:24px;margin:20px 0;border:1px solid #dbe3eb}}.tag{{color:#146445;font-weight:700}}.alert{{background:#fff1d9;border-left:5px solid #d59428;padding:18px}}.cards{{display:flex;gap:18px;flex-wrap:wrap}}.card{{background:#eaf2fa;padding:16px 25px;flex:1;min-width:180px}}strong.big{{font-size:32px;display:block}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dde3e9}}small{{color:#4c5b6a}}img{{width:100%;max-width:760px}}pre{{white-space:pre-wrap;font:15px/1.8 system-ui}}a{{color:#1265a1;margin-right:12px;overflow-wrap:anywhere}}.flow{{font-size:20px;background:#edf6f1;padding:18px}}@media(max-width:600px){{td,th{{padding:6px;font-size:13px}}}}</style>
<main><div class="tag">실제 공개 영상 실행 완료 · 학습과 운영 적용은 별도</div><h1>SHuBERT가 노트북에서 돌아가는가?<br>실제로 확인한 결과</h1><p>{decision}</p>
<div class="cards"><div class="card"><strong class="big">12개 / 1,438프레임</strong>공개 한국수어 전체 프레임 처리</div><div class="card"><strong class="big">16.45초</strong>독립 3영상의 특징 추출 중앙값</div><div class="card"><strong class="big">아직 미측정</strong>한국수어 교정 정확도</div></div>
<section><h2>왜 이 후보부터 시작했나</h2><p>손 사진 한 장에 머무르지 않고, 양손·얼굴·몸의 앞뒤 흐름을 함께 배우는 수어 연구 모델이기 때문이다. 이번에는 공식 파일을 받아 실제 노트북에서 실행했다. 다만 한국수어 설명을 읽고 곧바로 교정해 주는 완성 모델은 아니다.</p><div class="flow">영상·관절 → 시간별 관찰 특징 <b>[이번 실행 완료]</b><br>→ 손모양·동작 단계 판단 → 여러 참고와 비교 → 피드백 <b>[아직 남음]</b></div></section>
<section><h2>시간은 어디에 쓰이는가</h2><table><tr><th>영상</th><th>영상 길이</th><th>손·얼굴 이미지 특징</th><th>시간 흐름 처리</th><th>합계</th></tr>{table}</table><p class="alert">이미 준비한 손·얼굴 이미지부터 측정한 값이다. MediaPipe 검출, 영상 준비, DTW와 피드백 표시까지 포함한 전체 대기시간은 아니다. 약4초 영상을 재생한다고 약16초 계산이 모두 숨겨지지는 않는다.</p><p>병목은 이미지 특징을 뽑는 DINO 부분이다. SHuBERT 본체는 첫 호출 이후 약0.08초였다. 현재 CPU 경로를 그대로 게임에 적용하는 것은 보류한다.</p></section>
<section><h2>보이는 손과 이전 이미지를 복사한 구간을 구분했다</h2><p>2,876손-프레임 중 <b>2,764는 현재 관측</b>, <b>112는 직전 이미지 복사</b>다. 복사 구간은 현재 손모양을 확인한 근거로 세지 않았다. 다만 SHuBERT 자체는 이 표지를 별도 입력받지 않으므로 이후 피드백에도 별도의 보류 처리가 필요하다.</p><p>정상 손과 교차의 원본 좌표·시점은 유지했다. 이는 새로 손 검출 정확도를 높였다는 뜻은 아니다.</p><details><summary>실제 모델 입력 예시 — 양손 / 눈·입 영역</summary><img src="inputs/QA_CROPS.jpg" alt="공개 네 영상의 중간 프레임에서 추출한 왼손 오른손 얼굴 입력"><p>얼굴 입력은 공식 방식의 일부 영역이다. 얼굴 표정 채점은 하지 않는다.</p></details></section>
<section><h2>남은 핵심 검증</h2><ol><li>실제 보이는 손모양·부위·단계와 확인 불가 구간을 구분하는 평가 근거를 확보한다.</li><li>웹 상반신 범위와 화면 밖 손 조건을 맞춘다. 이번은 원영상 입력 실행 시험이다.</li><li>30fps와 사전학습에 가까운15fps를 비교하되 짧은 변화를 놓치지 않는지 확인한다.</li><li>검증된 정답으로 작은 판정부를 학습해 기존 관절 기준보다 개선되는지 비교한다. 이득이 있으면 Intel 가속과 재생 중 분석을 시험한다.</li></ol><p>기존 C·T3-F·웹·DTW 유지. 사용자·멘토 촬영 사용 없음. 한국수어 교정 모델 학습 없음.</p></section>
<section><h2>상세 보고와 검산 근거</h2>{links}<details><summary>전체 설명 펼치기</summary><pre>{html.escape(text)}</pre></details><p><a href="https://aclanthology.org/2025.acl-long.1397.pdf">공식 논문</a><a href="https://github.com/ShesterG/SHuBERT">공식 코드</a></p></section><small>{now} · 연구 실행 가능성 보고 · 최종 제품 성능 보고 아님</small></main></html>'''
(U/'RESULT_KO.html').write_text(page,encoding='utf-8')
print('Wrote RESULT_KO.txt/html')
