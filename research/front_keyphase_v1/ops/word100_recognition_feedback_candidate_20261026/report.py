from pathlib import Path
import json,html,hashlib,datetime
U=Path(__file__).resolve().parent;D=U.parent/'word100_dev_image_feedback_20261026'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=read(U/'MANIFEST.json');v=read(U/'VERIFIED.json');assert v['responses']==900 and v['manifest_sha256']==sha(U/'MANIFEST.json')
original=read(D/'VIEW_DATA.json');data=json.loads(json.dumps(original));rows={r['query_id']:r for r in f['rows']}
for q in data:
    q['predictions']['E3']=rows[q['query']]['predictions']['E3']
for old,new in zip(original,data):
    assert old['references']==new['references'] and old['predictions']=={k:z for k,z in new['predictions'].items() if k!='E3'}
text='''공개 자료를 재생하는 인식·피드백 연구 후보

만든 것: 보존한 C·T3-F·E3의 인식 결과를 선택하고, 같은 영상의 참고별 DTW 차이와 관측 근거를 함께 읽는 연구용 연결부입니다. 공개100단어·300영상만 허용합니다. 실제 웹캠용 운영 서버나 학습 모델을 교체하지 않았습니다.

E3는 기존 T3-F3개 모델의 확률을 같은 비중으로 합친 고정 후보입니다. 이미 저장되고 검증된 개발자료 결과는 C294/300, T3-F295/300, E3 300/300입니다. 이 결과는 이번 연결 작업으로 새로 오른 정확도가 아니며, 새 사용자나 모든 수어에서100%라는 뜻도 아닙니다. 세 모델의 계산을 쓰므로 단일 모델과 같은 처리비용이라고 볼 수 없습니다. 이번에는 새 추론이나 속도 측정을 하지 않았습니다.

피드백 개선: 비교 가능한 각도 중56.09%에는 두 영상의 전체장면·손확대·원래위치 재확인 근거를 연결했습니다. 연속 각도비교는41.06%입니다. 기존 좌표와 팔 정보,시간대응,차이값은 모두 보존했습니다. 이 비율은 정확도나 올바른 수행 비율이 아닙니다. 같은 MediaPipe의 반복 관측은 독립적인 정답이 아닙니다.

YOLO는 앞선 공개자료 실험에서 MediaPipe가 놓친 손을 찾는 선택적 보조로 사용했습니다. 기존 제안은 보존하지만, 손가락 좌표가 뭉치는 등의 문제가 있어 이번 정밀각도 근거에 자동채택하지 않았습니다. 현재300영상의 새 이미지재확인은 MediaPipe입니다. MANO/HaMeR도 추정모양 자체를 정답으로 쓰지 않습니다.

인식과 교정은 다릅니다. 단어가1위여도 모든 손모양이 올바르다는 뜻은 아닙니다. 비교는 사용자가 연습하는 목표 단어의 참고를 유지합니다. 인식모델을 바꾸어도 비교 목표나 DTW수치가 바뀌지 않습니다. 촬영측 가림,참고측 부족,재확인 불일치를 구분하고, 확인할 수 없는 부분을0차이·성공·사용자 오류로 처리하지 않습니다.

검증:300입력·600고정비교쌍의 원본보존 및 출처별시간 검산을 재사용하고, 새 연결부900응답을 원래 저장 인식·피드백과 대조했습니다. 모델선택이 참고나 차이를 바꾸지 않았으며, 허용되지 않은 입력3종은 거절했습니다. 같은Codex의 별도 산술·계약 검산이며, 사람이 관절정답을 달아 검증한 결과는 아닙니다.

남은 일: 정상적인 사람 간 차이의 허용범위, 숨겨진 손가락의 실제모양, 구체적교정방향과 실시간 처리비용은 미확정입니다. 다음에는 기존 공개TRAIN 여러 수행자 자료로 관측이 충분한 경우의 차이범위를 정리하고, 한 수행자를 제외해도 적용되는지와 짧은동작 누락·관측부족을 정상으로 넘기지 않는지 검증합니다. 운영 적용은 별개입니다. 사용자·멘토·추가199/200자료는 접근하지 않았습니다.

실행 예: python adapter.py NIA_SL_WORD1519_REAL11_F --model E3
이는 저장된 공개자료 재생입니다. 새 촬영,학습,재분류를 수행하지 않습니다.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
old=(D/'RESULT_KO.html').read_text('utf8');start=old.index('<script>');script=old[start:];a=script.index('const DATA=');b=script.index(';\nconst labels=',a);script=script[:a]+'const DATA='+json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</',r'<\/')+script[b:]
oldline="$('query').textContent='비교 영상: '+q.query+' | 기존 C: '+q.predictions.C.predicted_label+' | 기존 T3-F: '+q.predictions.T3F.predicted_label+' (저장 결과, 새 인식 아님)';"
newline="const model=$('model').value;$('query').textContent='연습 목표: '+q.label+' | '+model+'의 1위: '+q.predictions[model].predicted_label+' | 상위5: '+q.predictions[model].top5_labels.join(', ')+' (이전 저장 결과)';"
assert oldline in script;script=script.replace(oldline,newline).replace("['word','side','channel'].forEach","['word','side','channel','model'].forEach")
css=old[old.index('<style>'):old.index('</style>')+8]
body='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공개자료 인식·피드백 연구 후보</title>'''+css+'''<main><header><div class="small">공개 자료 재생 · 운영 모델 유지</div><h1>단어 인식과 동작 비교를 함께 확인하기</h1><p>인식 후보를 바꿔도 같은 촬영·같은 참고로 비교합니다. 확인하기 어려운 부분은 따로 표시합니다.</p></header><section><h2>같은 촬영으로 세 후보 살펴보기</h2><div class="controls"><select id="word"></select><select id="model"><option value="C">기존 C</option><option value="T3F">기존 T3-F</option><option value="E3">고정 E3 후보</option></select><select id="side"><option value="0">촬영자의 왼손</option><option value="1">촬영자의 오른손</option></select><select id="channel"></select></div><p id="query"></p><p class="small">파란선: 좌표 차이 · 초록표시: 양쪽 영상의 추가 이미지 근거 · 회색: 비교 불가. 낮은 차이가 곧 올바른 수어 수행을 뜻하지는 않습니다.</p><div id="cards"></div></section><section><h2>지금 사용할 수 있는 결과와 남은 문제</h2><pre>'''+html.escape(text)+'''</pre></section></main>'''
(U/'RESULT_KO.html').write_text(body+script,encoding='utf8')
with (U/'VIEW_DATA.json').open('x',encoding='utf8') as fp:json.dump(data,fp,ensure_ascii=False,allow_nan=False)
with (U/'REPORT_CHECK.json').open('x',encoding='utf8') as fp:json.dump(dict(at=datetime.datetime.now().astimezone().isoformat(),queries=300,existing_graphs_unchanged=True,models=['C','T3F','E3'],html_sha256=sha(U/'RESULT_KO.html'),view_sha256=sha(U/'VIEW_DATA.json'),browser_rendered=False),fp,ensure_ascii=False)
print('RESEARCH_CANDIDATE_REPORT_READY_NO_NEW_SCORES')
