"""Report saved measurements only; does not execute models or training."""
from pathlib import Path
import datetime as dt
import hashlib
import html
import json

U = Path(__file__).resolve().parent
def read(name):
    return json.loads((U / name).read_text(encoding='utf-8-sig'))

bench = read('BENCH_RESULT.json')
verify = read('BENCH_INDEPENDENT_VERIFY.json')
assert verify['status'] == 'PASS'
assert hashlib.sha256((U/'BENCH_RESULT.json').read_bytes()).hexdigest() == verify['result_sha256']
observations = read('OBSERVATION_TEST_RESULT_POST_REVIEW.json')
review = (U/'OBSERVATION_INDEPENDENT_REVIEW.txt').read_text(encoding='utf-8-sig')
assert observations['status'].startswith('PASS')
assert observations['corrected_execution']['exit_code'] == 0
assert observations['corrected_execution']['tests_run'] == 25
assert 'PASS_AFTER_FIX' in review
for filename, expected in observations['tested_code_sha256'].items():
    assert hashlib.sha256((U/filename).read_bytes()).hexdigest() == expected
manifest = json.loads((U.parent/'word100_shubert_feasibility_20261028/inputs/MANIFEST.json').read_text(encoding='utf-8-sig'))
names = {r['alias']: r['label'] for r in manifest['rows']}
now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()
rows = verify['per_clip']
speed = 1 - bench['summary']['int8_linear']['median'] / bench['summary']['fp32']['median']
table = '\n'.join(f"{names[r['alias']]} ({r['alias']}) | {r['fp32_median']:.2f}초 | {r['int8_median']:.2f}초 | {100*r['median_cost_reduction_fraction']:.1f}% 감소" for r in rows)
drift = [r for r in bench['differences'] if r['repeat'] == 0]
drift_table = '\n'.join(f"{names[r['alias']]}: 최종 특징 상대L2 중앙 {100*r['fields']['last_ffn']['relative_l2_median']:.1f}%, cosine 최소 {r['fields']['last_ffn']['cosine_min']:.3f}" for r in drift)

sections = [
('현재 판단',
 '현재의 SHuBERT 처리 경로를 사용자용 최종 모델로 확정하지 않는다. CPU에서 특징 계산만 14~18초 걸리는 경로는 부담이 크고, 아직 한국수어의 손 모양·동작 단계 판단에서 기존 방식보다 좋아졌다는 증거도 충분하지 않다. 다만 지금 결과는 한국수어 미세조정 전의 결과이므로, 학습해도 좋아질 수 없다는 뜻은 아니다.\n'
 '이번에는 새 모델을 또 받기 전에 시간이 어디에 쓰이는지 분리하고, 동일 모델을 가볍게 실행하는 한 조건만 비교했다. 속도는 좋아졌으나 모델이 내보내는 특징도 크게 달라졌다. 따라서 INT8 후보를 웹이나 실제 판정에 적용하지 않는다.'),
('시간을 실제로 비교한 결과',
 '같은 공개 영상 3개, 각 조건 2회, 순서를 바꿔 총 12회 측정했다. 모든 원래 프레임(30fps), 양손·얼굴 이미지와 몸 좌표를 유지했다. 기존 DINO의 Linear 계산만 FP32에서 동적 INT8로 바꾸고 SHuBERT 본체는 그대로 두었다. 이미지 모델 2개에서 각각 48/48개 Linear의 변환을 확인했다.\n'
 '영상 | 원래 계산 | 가볍게 한 계산 | 시간 변화\n'+table+'\n'
 f"6회 전체 중앙값 {bench['summary']['fp32']['median']:.2f}초 → {bench['summary']['int8_linear']['median']:.2f}초, 시간 {100*speed:.1f}% 감소. 각 측정 범위는 원래 14.51~17.74초, INT8 6.72~9.43초다. 영상별 표는 각 2회의 중앙값이다.\n"
 '이 시간은 이미 준비한 이미지에서 특징을 계산하는 구간이다. 원본 영상 읽기·MediaPipe 검출·손 이미지 준비·DTW·문장 생성·화면 표시를 모두 합한 시간이 아니다. 실제 촬영 종료부터 피드백까지의 대기 시간은 아직 측정하지 않았다. 모델 로딩/변환/준비 실행 시간은 별도 저장했다. FP32 첫 전체 영상(V01)의 SHuBERT 계산은 1.17초로 나머지 0.07~0.10초보다 컸으며 포함해서 보고했다. 표본 3개×2회로 다양한 PC의 보장 시간이나 p95를 주장하지 않는다.'),
('속도만 보고 적용하지 않는 이유',
 '기존 병목은 이미지에서 특징을 읽는 DINO 3개 스트림으로, 앞선 분리 측정에서 전체 특징 계산의 94~99.5%였다. SHuBERT 뒤에 작은 판정부를 붙이거나 미세조정하는 것만으로 이 비용이 사라지지는 않는다.\n'
 'INT8는 계산을 단순화해 빨라졌지만, 최종 특징 벡터 변화가 작지 않았다.\n'+drift_table+'\n'
 '여기서 31.6% 같은 값은 인식률 하락률이 아니다. 원래 특징과 수치적으로 얼마나 달라졌는지를 나타낸다. 손 모양·단계 정답률은 이번 속도 시험에서 측정하지 않았다. 같은 조건의 두 반복 출력은 완전히 같았고 원본 체크포인트·입력·기존 결과 해시도 유지됐다. 따라서 처리 시간 감소는 확인했지만, 품질 유지나 정상 손·교차의 판정 유지까지 확인한 것은 아니다.'),
('다음 학습을 위해 실제로 보완한 부분',
 '모델이 현재의 관측과 복사된 과거 정보를 구분할 수 있도록 입력 부품을 구현했다. 각 손의 현재 좌표·점별 유효 여부·손가락별 기하 정보 여부·현재/이전복사/없음·경과시간을 나눠 전달한다. 얼굴 및 몸 좌표의 현재/복사 상태도 구분한다. 이는 얼굴 표정 학습을 추가하는 작업이 아니다.\n'
 '손이 안 보이는 프레임도 영상 시간축에서는 유지한다. 실제 영상 길이를 맞추기 위해 붙인 빈칸과 손 미검출을 구별한다. 확정된 정답이 없는 항목은 그 항목만 학습에서 제외하며, 다른 손과 확인 가능한 항목은 남긴다. 반대로 RGB에서 정답을 신뢰할 수 있게 확인했다면 관절 미검출 자체만으로 그 학습 항목을 버리지 않는 구조다.\n'
 '공개 12영상, 1,438프레임의 저장 입력을 검사했다. 손 RGB 2,876개 시점 중 현재 관측 2,764개, 이전 복사 112개, 빈 이미지 0개다. 저장 특징과 시간·출처·해시를 대조해 입력 연결도 검사했다. 실제 정답을 새로 생성하거나 실제 모델을 학습시킨 작업은 아니다.\n'
 '초기 자체 검사 22개 통과 후, 독립 검토에서 한 손짜리 배열이 양손에 조용히 복제될 수 있는 입력 검사 부족과 몸/얼굴의 복사 상태가 수치 입력에 빠진 문제가 발견되었다. 반례를 기록하고 보완한 최종 검사는 25개 통과했다. 관측 입력은 손별 108차원이며 기존 SHuBERT 특징 768차원을 연결하면 876차원이다. 이 숫자는 모델 성능이 아니라 입력 구조를 설명한다. 최초 검사와 후속 수정 기록을 모두 남겼다.\n'
 '중요한 한계: 상태 정보를 옆에 붙인다고 기존 SHuBERT 특징 속에 이미 섞인 과거 이미지의 영향이 저절로 없어지지는 않는다. 입력 부품 검증 완료와 학습 준비 전체 완료는 다르다.'),
('모델을 선택하는 다음 기준과 작업',
 '1. 먼저 기존 공개 한국수어 자료에서 손 모양·이동·신체 부위·단계 정답의 출처와 사람별 분리를 갖춘 작은 묶음을 만든다. AI끼리 같은 답을 냈다는 이유만으로 교육적 정답으로 확정하지 않는다. 가려진 모양은 임의로 정답을 채우지 않는다.\n'
 '2. 같은 정답 묶음에서 관절·관측 상태를 읽는 작은 시간별 판정부를 기준으로, 저장된 SHuBERT 특징을 더했을 때 중요한 동작 구분과 잘못된 교정이 실제로 개선되는지 비교한다. 저장 특징으로 학습 시간을 아낄 수 있지만, 새 사용자 영상의 처리 비용이 줄었다고 해석하지 않는다.\n'
 '3. SHuBERT 추가 이득이 있을 때에만 실제 사용자 경로의 속도 최적화를 더 진행한다. Intel 가속 실행 또는 프레임 처리량 조정은 각각 별도 검증 조건이며, 현재 시험하지 않았다. 짧은 손 모양 변화·접촉·교차를 놓치지 않는지와 실제 총 대기 시간을 함께 확인한다. 이득이 부족하면 무거운 경로는 제품에서 제외하고 가벼운 판정부를 개선한다.\n'
 '4. 무거운 모델의 이득은 있지만 실행 시간만 맞지 않는 경우에 한해, 그 지식을 작은 모델로 옮기는 학습을 검토한다. SHuBERT가 맞는 답을 준다는 검증 전에는 정답 교사로 삼지 않는다.\n'
 '특징 차원이 같다는 이유만으로 DINO를 MobileNet 등에 바로 바꿔 끼울 수는 없다. SHuBERT가 배운 입력의 의미가 달라지므로 별도 학습·검증이 필요하다. 얼굴 분석 목표가 없다는 이유로 얼굴 입력을 0으로 만드는 것도 무해하다고 확인되지 않았다. 지금은 새로운 모델을 더 받거나 광범위한 미세조정을 시작하지 않는다.'),
('실행 환경과 적용 상태',
 'Samsung 960XHA, Intel Core Ultra 7 258V, 메모리 32GB, Intel Arc 140V 내장 GPU/NPU가 있는 노트북이다. 이번 실험은 Torch 2.12 CPU, 4스레드, batch 8, onednn으로 실행했다. GPU가 없는 노트북이 아니라, 현재 실험 코드가 CPU 실행 중인 상태다. Intel GPU 가속의 실제 속도는 미측정이다.\n'
 'C·T3-F 체크포인트와 운영 웹·DTW는 변경하지 않았다. 새 외부 모델/라이브러리 다운로드 0, 사용자·멘토 영상 접근 0, 한국수어 추가 학습 0회다. 공개 자료의 속도 시험과 학습 입력 부품을 검증했으며, 프로젝트의 최종 피드백 기능이 완성된 것은 아니다.\n'
 '구현: Codex root/observation_input_builder. 독립 검토: latency_quality_reviewer(저장 벤치 186검사), latency_path_audit(환경·입력 반례검토). 별도 에이전트 검산은 독립 수어 전문가의 정답 검증과 다르다.')
]
txt = 'SHuBERT 속도와 다음 학습 입력 검증\n'+now+'\n\n'+'\n\n'.join(k+'\n'+v for k,v in sections)
(U/'RESULT_KO.txt').write_text(txt,encoding='utf-8')
bars = ''.join(f'<div class="barrow"><b>{html.escape(names[r["alias"]])}</b><div><span style="width:{r["fp32_median"]/18*100:.1f}%">원래 {r["fp32_median"]:.2f}초</span><span class="q" style="width:{r["int8_median"]/18*100:.1f}%">INT8 {r["int8_median"]:.2f}초</span></div></div>' for r in rows)
links=['FROZEN.json','BENCH_RESULT.json','BENCH_INDEPENDENT_VERIFY.json','LATENCY_AUDIT.txt','OBSERVATION_TEST_RESULT.json','OBSERVATION_TEST_RESULT_POST_REVIEW.json','OBSERVATION_POST_REVIEW_TEST_LOG.txt','OBSERVATION_INDEPENDENT_REVIEW.txt','observation_input.py','test_observation_input.py']
body=''.join('<section><h2>'+html.escape(k)+'</h2>'+''.join('<p>'+html.escape(p).replace('\n','<br>')+'</p>' for p in v.split('\n'))+'</section>' for k,v in sections)
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>처리 속도와 입력 검증</title><style>body{font:17px/1.8 system-ui,sans-serif;max-width:1020px;margin:40px auto;padding:0 24px;color:#1f2937;background:#f4f6fa}h1{font-size:30px}h2{font-size:23px}section,.card{background:white;border:1px solid #d9dfe9;padding:24px;margin:20px 0;border-radius:14px}p{margin:12px 0}.badge{color:#934500;background:#fff0d5;padding:8px 16px;border-radius:8px}.barrow{display:grid;grid-template-columns:90px 1fr;align-items:center;margin:15px 0}.barrow span{display:block;min-width:150px;background:#54677e;color:white;padding:2px 8px;margin:5px 0;box-sizing:border-box}.barrow .q{background:#147a78}a{color:#176aab}small{color:#59687b}</style><h1>빨라졌지만, 바로 적용하기에는 아직 부족해요</h1><p class="badge">처리 시간 감소 확인 · 정확도 유지 미검증 · 운영 적용 없음</p><div class="card"><h2>같은 영상, 같은 프레임으로 비교</h2>'''+bars+'''<small>영상별 2회 중앙값. 준비된 RGB → 특징 계산 시간이며 최종 피드백까지의 시간이 아닙니다.</small></div>'''+body+'<section><h2>검증 근거</h2><ul>'+''.join(f'<li><a href="{x}">{x}</a></li>' for x in links)+'</ul><p><a href="https://aclanthology.org/2025.acl-long.1397/">SHuBERT 공식 논문</a> · <a href="https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html">PyTorch 동적 양자화 문서</a></p></section></html>'
(U/'RESULT_KO.html').write_text(page,encoding='utf-8')
print(json.dumps({'at':now,'text_chars':len(txt),'html_chars':len(page),'speed_reduction_fraction':speed},ensure_ascii=False))
