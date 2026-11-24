from common import *
import numpy as np, statistics, html
v=read(ROOT/'VERIFICATION.json');e=read(ROOT/'END_TO_END.json');s=read(ROOT/'STREAMING.json')
checks=[]
for title,data,folder in [('serial',e,'e2e'),('overlapped',s,'streaming')]:
    assert len(data['rows'])==6
    assert statistics.median(r['seconds'] for r in data['rows'])==data['median_seconds']
    for r in data['rows']:
        a=np.load(ROOT/'features'/f"ov_gpu_fp16_b1_{r['alias']}_r0.npy",allow_pickle=False)
        b=np.load(ROOT/folder/f"{r['alias']}_r{r['repeat']}.npy",allow_pickle=False)
        assert np.array_equal(a,b)
        checks.append(dict(path=title,alias=r['alias'],repeat=r['repeat'],identical_gpu_outputs=True,file_sha256=sha(ROOT/folder/f"{r['alias']}_r{r['repeat']}.npy")))
    assert data['protocol_sha256']==sha(ROOT/('END_TO_END_PROTOCOL.json' if title=='serial' else 'STREAMING_PROTOCOL.json'))
rows=v['conditions'];cpu=rows[0];gpu=next(r for r in rows if r['condition']=='ov_gpu_fp16_b1')
latency_pass=s['median_seconds']<=5
status='SIGNREP_GPU_RUNTIME_VERIFIED_KSL_TRAINING_NOT_STARTED'
state=dict(status=status,public_videos=3,frames=368,windows=164,repeated_timings=36,benchmark_arithmetic_checks=v['checks'],raw_path_checks=checks,
  latency_prerequisite=latency_pass,ksl_accuracy_measured=False,ksl_training_runs=0,production_changes=0,private_data_access=0,
  max_feature_relative_error=gpu['max_relative_l2'],feature_error_is_not_accuracy_loss=True,
  operator='Codex local implementation and separate arithmetic/script checks; not independent expert labels',
  pending='Time-localized public KSL handshape/movement/body-relation labels, ROI input consistency and person-separated evaluation before adaptation',
  files={p.name:sha(p) for p in [ROOT/'FROZEN.json',ROOT/'ACQUISITION.json',ROOT/'BENCHMARK.json',ROOT/'VERIFICATION.json',ROOT/'END_TO_END.json',ROOT/'STREAMING.json',ROOT/'TRAINING_READINESS.json']})
save('FINAL_AUDIT.json',state)
def sec(n):return f'{n:.2f}'
def span(d):return f"{sec(d['range_seconds'][0])}~{sec(d['range_seconds'][1])}초"
text=f'''SignRep 실행 가능성 검증 결과 — 2026-10-29

결론
이 노트북의 Intel Arc140V GPU로 SignRep를 실제 실행했다. 공개 영상 전체를 읽고 특징값을 준비하는 시간을 중앙 {sec(s['median_seconds'])}초({span(s)})로 줄였다. 영상을 읽는 작업과 GPU 계산을 겹쳐 진행했고 프레임이나 동작 구간은 줄이지 않았다. 이번에 확인한 것은 실행 가능성과 계산의 일관성이며, 한국수어 손모양 정확도나 피드백 성능 향상을 확인한 결과는 아니다.

무엇을 확보했나
- SignRep 공식 코드 commit06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2와 공식 checkpoint222,676,746bytes(약212.4MiB)를 확보하고 공개 SHA256을 검증했다.
- 학습된55,644,329개 파라미터를 누락 없이 로드했다. 이 사전학습은 저자들이 미리 수행한 학습이며, 우리 한국수어 자료로 추가 학습한 것은 아직0회다.
- 기존CPU PyTorch를 유지하고 별도 가상환경에 Intel OpenVINO를 설치했다. 실제 실행장치 GPU.0을 확인했다. NVIDIA CUDA가 없어도 이 Intel GPU로 추론할 수 있다. 드라이버나 전역Python 환경은 바꾸지 않았다.

속도: 같은 공개3영상·각2회
1. CPU 원래 정밀도 특징 계산: 중앙 {sec(cpu['inference_median_seconds'])}초.
2. GPU 원래 정밀도 특징 계산: 중앙 {sec(rows[1]['inference_median_seconds'])}초.
3. GPU 가벼운 정밀도(FP16) 특징 계산: 중앙 {sec(gpu['inference_median_seconds'])}초, 범위 {sec(gpu['inference_range_seconds'][0])}~{sec(gpu['inference_range_seconds'][1])}초.
4. GPU에4구간씩 묶어 전달: 중앙 {sec(rows[3]['inference_median_seconds'])}초. 이 노트북에서는1구간씩 전달하는 조건보다 느렸으므로 우선경로에서 제외했다.
5. 영상읽기→변환→GPU 계산을 차례대로 실행: 중앙 {sec(e['median_seconds'])}초({span(e)}).
6. 영상읽기와GPU 계산을 겹쳐 실행: 중앙 {sec(s['median_seconds'])}초({span(s)}).

1~4는 미리 준비한RGB에서 특징 계산까지,5~6은 원본파일 읽기부터 특징 준비까지 잰 실제 시간이다. 모델은 이미 로드하고 예열한 상태이며 첫 시작의 모델로드/컴파일은 별도다. GPU컴파일은 조건별 약2.6~3.5초였다. 최종 피드백판정부·MediaPipe·DTW·웹표시 시간은 포함하지 않았다. 노트북은 배터리로 동작 중이었다. 다른 사용자의 PC에서도 같은 시간을 보장하지 않는다.

속도를 줄이면서 무엇을 확인했나
- 원본368프레임을 모두 사용했다. 16프레임씩2프레임 간격으로164구간을 비교했고, 마지막프레임이 빠지는2영상은 마지막 온전한구간을 추가했다.
- 원본공식출력과 불필요한출력계산만 뺀 경로가9구간에서 완전히 같았다. Intel변환기의 빈토큰 처리 호환오류도 공식 저장소의 동등한 계산으로 고쳤고9구간 동일출력을 확인했다. 원본공식파일은 수정하지 않았다.
- GPU FP16의 특징값 차이는 CPU FP32 대비 최대 상대L2 {100*gpu['max_relative_l2']:.3f}%였고, 전체창에서 사전수치기준을 통과했다. 이 숫자는 인식률 하락률이 아니다. 앞으로 학습할 판정 결과도 같다는 뜻은 아니다.
- 영상읽기를 겹쳐 처리한 결과는 순차GPU결과와 모든창에서 비트단위로 같았다. 이 가속은 입력을 생략한 효과가 아니다.
- 별도 계산기로168개 해시·프레임·출력·수치조건을 재검산하고, 실제원본경로12회 결과도 비교했다. 이 검산은 같은Codex의 별도검사이며 사람 수어전문가의 정답확인은 아니다.

추가 학습은 어떤 자료로 하나
기존에 확보한 공개 한국수어 영상과 출처가 있는 동작설명을 사용한다. SignRep 본체는RGB영상의 시간흐름을 읽는다. MediaPipe 좌표만 넣는 모델이 아니며, 설명문을 스스로 읽고 한국수어 규칙을 모두 이해하는 언어모델도 아니다. 손모양·방향·신체부위관계·움직임·순서를 판별할 작은시간판정부를 연결할 계획이다.

아직 학습하지 않은 이유와 다음 작업
이번 승인에 따른 첫 작업인 실행시간/장치검증을 끝냈다. 속도는 기존CPU중심 경로보다 실용적인 방향을 확인했지만, 학습 전에 입력·정답 준비가 남았다. 현재36공개영상의 시간별주석에는 AI관찰초안/불확실표시가 섞여 있고, 단어설명은 모든프레임의 실제손모양 정답이 아니다. 이를 검증 없이 정답으로 복사해 학습시키지 않았다.
다음은 기존 공개한국수어의 손별·시간별로 확인 가능한 물리적특징만 학습대상에 연결하는 일이다. 가려져 모르는 항목은 학습손실에서 빼고, 확인 가능한 손위치·다른손정보는 유지한다. 상반신사용범위와RGB입력조건을 맞추고, 사람별 학습/평가분리를 고정한다. 이후SignRep본체를 우선고정하고 작은시간판정부를 학습하여 같은정답을 쓰는 관절기반방법과 비교한다. 필요성이 확인된 뒤 본체일부미세조정으로 간다. 현재OpenVINO는추론가속이며학습엔진이 아니므로GPU미세조정경로는별도확인대상이다. 승인된 이 준비작업에 모델선택을 다시 묻거나 다른모델을 자동다운로드하지 않는다.

남은 한계
- 이번3영상은 이미 살펴본 개발영상이다. 새사람에 대한 인식성능평가가 아니다.
- GPU가속 성공은 손모양/동작단계 정확도 개선을 의미하지 않는다.
- 이번은공식전체RGB입력의 실행시험이다. 실제웹의상반신관측범위와미관측표시까지통합한최종제품경로는아직아니다.
- 정상수어의 허용범위와 교육적으로 잘못된동작의 정답은 자동검출·문구만으로 확정하지 않는다.
- 최종판정과웹표시까지붙이면대기시간이더해질수있다. 촬영중·복습영상재생중계산은아직웹에구현하지않았다.

보존
기존C·T3-F/운영웹·DTW는변경없다. 사용자·멘토촬영접근·학습·튜닝·모델선택0. 새외부수어데이터다운로드0,영상외부전송0,카메라0,Git0. SignRep는확인한비상업연구조건하의별도후보다.

근거
공식논문 https://arxiv.org/abs/2503.08529
공식코드 https://github.com/ryanwongsa/SignRep
로컬: ACQUISITION.json, WRAPPER_PARITY.json, EXPORT_COMPAT_PARITY.json, EXPORT.json, BENCHMARK.json, VERIFICATION.json, END_TO_END.json, STREAMING.json, TRAINING_READINESS.json, FINAL_AUDIT.json
'''
(ROOT/'RESULT_KO.txt').write_text(text,encoding='utf-8')
table=''.join(f'<tr><td>{html.escape(name)}</td><td>{sec(value)}초</td></tr>' for name,value in [('CPU 특징 계산',cpu['inference_median_seconds']),('GPU FP32 특징 계산',rows[1]['inference_median_seconds']),('GPU FP16 특징 계산',gpu['inference_median_seconds']),('원본읽기 포함·순차',e['median_seconds']),('원본읽기 포함·동시',s['median_seconds'])])
document='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SignRep 노트북 GPU 실행 검증</title><style>body{font-family:system-ui,sans-serif;background:#f4f6f8;color:#142330;max-width:1000px;margin:36px auto;padding:0 22px;line-height:1.7}h1{font-size:30px}.summary,section{background:white;padding:24px;border-radius:14px;margin:18px 0}.badge{color:#13664a;font-weight:700}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;font:inherit}a{color:#1765aa}.caution{border-left:4px solid #c88414;padding:12px;background:#fff5db}</style><h1>SignRep가 노트북 GPU에서 작동합니다</h1><div class="summary"><div class="badge">실행환경·시간 검증 완료 / 한국수어 추가학습 아직0회</div><p>원본 영상 읽기와 GPU 계산을 겹쳐 처리해, 특징 준비까지 중앙 <b>'''+sec(s['median_seconds'])+'''초</b>가 걸렸습니다. 원본 프레임과 계산 결과를 유지했습니다.</p><p class="caution">수어 정확도 개선 결과나 최종 피드백 대기시간이 아닙니다. 판정부·DTW·웹 표시를 연결한 검증은 남아 있습니다.</p></div><section><h2>같은 공개3영상·각2회 중앙값</h2><table><tr><th>처리 조건</th><th>시간</th></tr>'''+table+'''</table><p>배터리 상태의 실제Intel Arc140V 측정. 모델로드/예열 제외. 전체368프레임/164창 보존.</p></section><section><h2>무엇을 했고 무엇이 남았나</h2><pre>'''+html.escape(text)+'''</pre></section><p><a href="RESULT_KO.txt">텍스트 보고서</a> · <a href="FINAL_AUDIT.json">검산 기록</a> · <a href="STREAMING.json">원본 영상 처리시간</a></p></html>'''
(ROOT/'RESULT_KO.html').write_text(document,encoding='utf-8')
summary=f'공식SignRep55,644,329파라미터/212.4MiB strict로드·SHA검증완료. 공개개발3영상368프레임164창,4조건각2회24측정. CPU특징중앙{sec(cpu["inference_median_seconds"])}초/GPU FP16 {sec(gpu["inference_median_seconds"])}초. 원본읽기포함순차중앙{sec(e["median_seconds"])}초→동시처리{sec(s["median_seconds"])}초({span(s)}). GPU수치차최대상대L2 {100*gpu["max_relative_l2"]:.3f}%;인식률아님. 동시/순차GPU출력전창동일. 독립산술168검사+원본경로12회검산. KSL추가학습0/인식향상미측정/운영변경0/개인자료접근0. IntelGPU추론경로확보,학습라벨·ROI연결준비다음. 실행종료.'
(ROOT/'STATUS_KO.txt').write_text(summary,encoding='utf-8')
next_text='SignRep 공식가중치와 검증된Intel GPU FP16 실행경로를 재사용한다. 기존공개한국수어의 손별·시간별 확인가능한손모양/관계/이동정답과미확인항목을구분하고실제상반신입력조건/사람분리를고정한뒤,본체고정+작은시간판정부학습과관절기준을비교한다. 기존AI초안=독립사람정답으로승격금지. 모델이만든768특징은의미단계정답이아님. 이미승인된준비·조건부학습에반복승인요청없음. 추가모델순회/새데이터/개인자료/운영교체금지. 이번시간검증완료결과재추론불필요.'
(ROOT/'NEXT_KO.txt').write_text(next_text,encoding='utf-8')
(ROOT/'CURRENT_EXECUTION.json').write_text(json.dumps(dict(stage='COMPLETE',session_id=None,running=False,report='RESULT_KO.html',next='NEXT_KO.txt'),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=status,gpu_features=gpu['inference_median_seconds'],serial=e['median_seconds'],streaming=s['median_seconds'],range=s['range_seconds'],latency_pass=latency_pass),ensure_ascii=False))
