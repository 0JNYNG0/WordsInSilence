from pathlib import Path
import json,html
U=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    v=read(U/'VERIFICATION.json');r=read(U/'RESULT.json');f=read(U/'FROZEN.json')
    sums=v['summary'];a,b=[s['end_to_render_ack_seconds'] for s in sums]
    text=f'''현재 확인하는 범위와 웹 연결 시험 결과

현재 기능은 수어 전체의 올바름을 판단하는 완성된 교사가 아닙니다.
검증한 출력은 10개 물리 특징입니다: 오른 검지만 펴기, 오른/왼 네 손가락 펴기, 오른손과 얼굴의 화면상 겹침, 양손 화면상 겹침, 오른손의 상/하/좌/우 이동, 왼손 아래 이동.
어깨·팔꿈치·손목과 손가락 좌표를 함께 읽지만 전신 자세 교정은 검증하지 않았습니다. 얼굴의 특정 부위 접촉, 손바닥의 정확한 3D 방향, 필수 동작/반복의 생략, 수어로 허용되는 차이도 아직 검증 대상이 남아 있습니다. 얼굴 표정은 제외합니다.
예: '오른손이 위로 이동했다'라는 관측과 '이 표현은 꼭 두 번 올려야 하므로 잘못했다'라는 교육 판단은 다릅니다. 현재는 앞쪽을 시험했습니다.

이번 작업: 공개 수어/멀다/별 3영상, 각 방식2회, 총12실행 {v['total_processed_frames']}프레임. 고정 관절 기반 시간별 판정부와 각 표현의 다른 수행자 참고3개를 DTW로 비교하고 브라우저에 실제 결과표를 표시했습니다.
종료 후 처리: 표시 확인 중앙 {a['median']:.3f}초, 범위 {a['min']:.3f}~{a['max']:.3f}초.
촬영 중 처리: 표시 확인 중앙 {b['median']:.3f}초, 범위 {b['min']:.3f}~{b['max']:.3f}초.
6쌍 모두 전체 관절/좌우배정 배열, 시간별 모델 출력, DTW 경로·결과 JSON 일치. 이전 동일 공개파일 관절 산출물과도 일치. 속도를 위해 프레임을 버리거나 비교 범위를 줄이지 않았습니다. 전체 미관측 통제에서는 참고 분모를 유지하고 비교 근거 부족/거리 없음으로 나왔습니다.

시간 해석: 원본 마지막 프레임이 공급된 서버 시각부터, 결과표/안내를 브라우저의 두 화면 갱신 콜백 후 확인한 응답이 서버로 돌아온 시각까지입니다. 물리 모니터 표시/영상 재생준비를 정밀 측정한 값이 아닙니다. 앱·모델·참고 준비는 촬영 전에 완료했습니다. 파일 PTS 재생이며 실제 웹캠이 아닙니다. 3개 짧은 영상/고정 실행순서의 작은 속도 시험입니다. 기존 1.134초와 이번0.411초는 실행범위·부하가 달라 별도 최적화 효과로 빼서 비교하지 않습니다.
이 경로에는 SignRep 영상 분석과 C/T3-F 단어 분류가 포함되지 않습니다. CPU 관절+고정 작은 판정부+DTW+HTTP/화면 시험입니다. 새 특징 모델을 채택하거나 새 인식률을 얻은 것이 아닙니다.

남은 문제: 손 검출의 불확실성과 정상 동작에도 나오는 특징 차이 후보가 남습니다. 기존 generic 관측기준을 출력동등성 시험에 고정했으며, 최신 각 손가락별 근거 기준과 통합하여 품질을 다시 검증해야 합니다. 자동 의미단계/필수 반복/언어적 허용경계 및 교육적 오경고율은 미완료입니다. 현재 차이 후보 수는 틀린 수어 횟수가 아닙니다.
시행 중 문제: 별도검토에서 시험 Chrome 고정포트 충돌 위험을 발견해 실행 전에 자기 프로필의 임시포트만 쓰도록 수정했습니다. 영상 전환 중 이전 전송취소 WinError10053 로그는 별도 보존했습니다. 결과 실패는 없으며 별도 전송 보완은 그 취소만 기록하도록 제한했습니다.

다음 목표: 계산과 화면 연결은 이 작은 범위에서 확인했습니다. 이제 정상 공개 동작에 잘못된 교정이 나오지 않도록, 단계별 손 모양·방향·신체부위 관계의 근거와 차이 후보를 검증하는 것이 우선입니다. 실제 웹캠 전체 대기는 이후 별도 시험이 필요합니다. 속도가 빨라졌다는 이유로 미검증 교정문을 운영에 적용하지 않습니다.
기존 C/T3-F/운영웹/DTW 보존. 사용자·멘토 자료 접근, 학습, 튜닝, 새 다운로드, 카메라, 외부 전송 없음. 별도 Codex 검산은 사람/수어교육자 정답 검증이 아닙니다.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf8')
    files={x['alias']:Path(x['source']).as_uri() for x in f['rows']+f['references']}
    data={}
    for alias in ['P005','P043','P079']:
        key=f'{alias}_2_during_recording';rr=next(x for x in r['rows'] if x['key']==key)
        data[alias]=dict(key=key,alias=alias,payload=read(U/'runs'/key/'feedback.json'),metrics=rr)
    template=(U/'index.html').read_text(encoding='utf8')
    style=template[template.index('<style>'):template.index('</style>')+8]
    panel=template[template.index('<section id="result"'):template.index('<script>')]
    panel=panel.replace('비교한 특징 / 필요했던 특징','비교한 특징×시점 / 필요했던 특징×시점').replace('관측 부족</th>','대응점 부족 / 원본 전체 미관측</th>')
    script=template[template.index('const $='):template.index('async function runOne')]
    script=script.replace("'/video/'+job.alias","SOURCE_URLS[job.alias]").replace("'/video/'+$('#ref').value","SOURCE_URLS[$('#ref').value]")
    script=script.replace("'개 (정렬 전 누락 '","'개 (원본 전체 미관측 특징 '")
    script=script.replace("$('#scope').textContent=p.scope", "$('#scope').textContent=p.scope+' 현재 상태: '+(r.status==='PARTIAL_EVIDENCE'?'부분 근거 — 원본에 미관측 부분이 남아 있습니다':r.status==='NO_COMPARISON_EVIDENCE'?'비교 근거 부족':'관측한 부분의 연구 비교')")
    options=''.join(f'<option value="{k}">{html.escape(x["payload"]["label"])}</option>' for k,x in data.items())
    table=''.join('<tr><td>'+name+'</td><td>'+f'{s["median"]:.3f}초'+'</td><td>'+f'{s["min"]:.3f}~{s["max"]:.3f}초'+'</td></tr>' for name,s in [('종료 후 계산',a),('촬영 중 계산',b)])
    paragraphs=''.join('<p>'+html.escape(p).replace('\n','<br>')+'</p>' for p in text.split('\n\n'))
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>무엇을 확인했나 · 웹 연결 결과</title>'+style
    page+='<h1>일부 동작 특징의 비교를 웹 화면까지 연결했습니다</h1><section><p><b>몸 전체의 정답·오답 피드백을 완성한 시험은 아닙니다.</b> 손 모양 일부, 손 이동 방향, 겹침을 확인하는 작은 비교 경로입니다.</p><table><tr><th>계산 시점</th><th>표시 확인 중앙값</th><th>6회 범위</th></tr>'+table+'</table><p>공개 파일 3개 × 각 2회. CPU 관절 + 고정 판정부 + DTW + 비교표 표시 확인. 실제 웹캠·SignRep·TCN 단어 분류 제외.</p></section>'
    page+='<section><h2>저장 결과를 직접 확인</h2><select id="saved" onchange="choose()">'+options+'</select><p>원본 공개 영상과 참고를 보고, 막대로 DTW가 대응시킨 시점을 확인할 수 있습니다. ‘차이 후보’는 모델의 관측 차이이며 교정 지시가 아닙니다. 대응점 부족이 0이어도 원본 전체의 미관측이 별도로 있으면 충분한 관측을 뜻하지 않습니다.</p></section>'+panel
    page+='<section><h2>진행 내용·해석·다음 목표</h2>'+paragraphs+'</section>'
    page+='<script>const SOURCE_URLS='+json.dumps(files,ensure_ascii=False)+';const DATA='+json.dumps(data,ensure_ascii=False,allow_nan=False)+';'+script+"function choose(){let x=DATA[$('#saved').value];render(x);$('#metrics').textContent+=' · 표시 확인까지 '+x.metrics.end_to_render_ack_seconds.toFixed(3)+'초';}choose();</script></html>"
    (U/'RESULT_KO.html').write_text(page,encoding='utf8')
    print('report written')
if __name__=='__main__':main()
