"""Report only completed, verified W comparisons. Never selects or changes models."""
from pathlib import Path
import json,sys,hashlib,datetime,html
sys.stdout.reconfigure(encoding='utf8')
J=Path(__file__).resolve().parent
SEEDS=(20261013,20261014,20261015)
VIEWS={'sensor':'카메라 밖 손을 볼 수 없는 조건','masked':'전체 영상 검출 후 상반신 정보만 사용하는 조건'}
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(head,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def main():
    assert not (J/'RUNNING.json').exists()
    assert not (J/'RESULT_KO.html').exists()
    c=read(J/'CONTEXT.json');v=read(J/'VERIFIED.json');cross=read(J/'CROSS_VERIFIED.json');imp={k:read(J/(k+'_INPUT_IMPACT.json')) for k in ('TRAIN','DEV')}
    assert v['status']=='PASS_SAVED_SCORES_AND_ACTUAL_BATCH_TRACE' and cross['status']=='PASS_SAVED_CROSSOVER_SCORES'
    for p,h in {**v['pins'],**cross['pins']}.items():assert sha(p)==h,p
    for value in imp.values():
        assert value['status']=='VERIFIED_INPUT_IMPACT_AGGREGATE'
        for p,h in value['pins'].items():assert sha(p)==h,p
    at=datetime.datetime.now().astimezone().isoformat();sections=[];text=[]
    def para(title,body):
        text.extend([title,body,'']);sections.append('<section><h2>'+html.escape(title)+'</h2><p>'+html.escape(body)+'</p></section>')
    passed=v['gate']['passed'];verdict='동결한 개발 비교 기준 통과. 연구 통합 후보로 검토할 수 있지만 운영 교체나 일반 사용자 성능 보장은 아니다.' if passed else '동결한 개발 비교 기준 미통과. 이번 W 후보를 개선된 운영 모델로 채택하지 않는다. 좋아진 사례와 나빠진 사례를 모두 남긴다.'
    para('결론',verdict)
    para('무엇을 바꿨나','기존 C와 같은 TCN 구조·학습 자료·초기값·학습 순서를 유지했다. MediaPipe가 놓친 손을 현재 영상에서 재검출하고, 그래도 놓친 경우에만 손 전용 YOLO를 보조로 사용했다. 추가한 것은 손의 몸 기준 위치와 관측 표시뿐이다. 기존 손가락 정보·몸 크기 기준·동작 시작과 끝·시간 간격은 그대로 두었다. W는 별도 후보이며 기존 C·T3-F·E3와 운영 웹·DTW는 보존했다.')
    para('자료와 평가 의미','공개 100단어의 TRAIN899영상(기록ID1~9)과 DEV300영상(기록ID11~13)을 사용했다. 두 관측 조건에 대해 같은 세 초기값으로 비교했다. 각 수치는300영상 중 단어를1위로 맞힌 수이며 손동작의 정확한 수행 점수가 아니다. DEV는 이미 개발과 학습 시점 선택에 사용한 자료다. 사용자·멘토 촬영 및 이미 노출된 추가199·200평가는 사용하지 않았다.')
    for view,name in VIEWS.items():
        rows=[]
        for seed in SEEDS:
            key=str(seed);a=v['baseline'][key][view];b=v['results'][key][view];p=v['paired'][key][view]
            assert a['total']==b['total']==300
            rows.append([seed,f"{a['top1_correct']}/300 ({a['top1_percent']:.2f}%)",f"{b['top1_correct']}/300 ({b['top1_percent']:.2f}%)",f"{p['percentage_point_difference']:+.2f}%p",len(p['improved']),len(p['regressed'])])
        a=v['summary']['C'][view]['top1_percent'];b=v['summary']['W'][view]['top1_percent']
        summary=f"세 번 평균: C {a['mean']:.2f}% → W {b['mean']:.2f}% ({b['mean']-a['mean']:+.2f}%p). 범위: C {a['minimum']:.2f}~{a['maximum']:.2f}%, W {b['minimum']:.2f}~{b['maximum']:.2f}%."
        text.extend([name,summary,*[' | '.join(map(str,x)) for x in rows],''])
        sections.append('<section><h2>'+html.escape(name)+'</h2><p>'+html.escape(summary)+'</p>'+table(['초기값','기존 C','후보 W','차이','새로 맞힘','새로 틀림'],rows)+'</section>')
    crossrows=[]
    for view,name in VIEWS.items():
        for seed in SEEDS:
            a=v['baseline'][str(seed)][view]['top1_correct'];b=cross['evidence'][f'C_on_W_input_{seed}_{view}']['metrics']['top1_correct'];d=cross['evidence'][f'W_on_C_input_{seed}_{view}']['metrics']['top1_correct'];e=v['results'][str(seed)][view]['top1_correct']
            crossrows.append([name,seed,a,b,d,e])
    sections.append('<section><h2>입력 효과와 재학습 효과 구분</h2><p>같은300개를 네 조합으로 비교했다. 교차 결과는 원인 진단용이며 모델 선택에 사용하지 않았다. 입력이 완전히 같은 경우 저장 점수를 재사용했다.</p>'+table(['조건','초기값','C+기존입력','C+보완입력','W+기존입력','W+보완입력'],crossrows)+'</section>')
    text.extend(['입력 교차 진단(각300개 정답 수)',*[' | '.join(map(str,x)) for x in crossrows],''])
    inputrows=[]
    for role,value in imp.items():
        for view,a in value['by_view'].items():
            t=value['fully_uncached_trigger_frame_extra_time'].get(view)
            duration=f"{t['median_ms']:.1f} / {t['p95_ms']:.1f}ms" if t else '캐시 재사용 — 신규 시간 측정 없음'
            inputrows.append([role,VIEWS[view],a['views'],a['added_wrist_frames'],a['mp_wrist_frames'],a['yolo_wrist_frames'],duration])
    sections.append('<section><h2>추가된 위치 정보와 계산 비용</h2>'+table(['자료','조건','영상','추가 손·프레임','MP 재검출','YOLO 보조','추가 처리 중앙값 /95백분위'],inputrows)+'<p>손·프레임은 한 손의 한 시점이다. 실제 정답 복원 수가 아니다. 시간은 캐시 없이 추가 검출을 수행한 장면의 호출 시간 합이며, 기존 검출·영상 처리·TCN·DTW·웹 대기 시간은 빠져 있다. 전체 서비스 FPS로 해석할 수 없다.</p></section>')
    text.extend(['입력/추가 처리시간',*[' | '.join(map(str,x)) for x in inputrows],'손·프레임 수는 복원 정답률이 아니다. 기록 시간은 추가 검출 호출만이며 전체 FPS가 아니다.',''])
    para('검증과 남은 문제','원래 관절·손가락·크기·시간·동작 범위 보존을 전체 입력에서 검산했다. 모든 YOLO 추가 장면과 미리 정한 위험도/기록ID 표본은 영상으로 확인했다. 손목점이 손바닥이나 손등 쪽으로 치우치는 정밀도 문제가 남아 있다. 이 결과는 손 위치를 이용한 단어 분류 시험이며 정확한 손가락 복원이나 검증된 교정 피드백을 의미하지 않는다. 관측 부족을0차이 또는 올바른 동작으로 간주하지 않는다.')
    para('검증 범위','저장된 점수·학습 이력·선택된 체크포인트를 별도 계산으로 확인하고, 실제 학습 배치에 들어간 영상ID·관측 조건·입력 텐서 해시를 재구성해 대조했다. 구현과 검산은 같은 Codex의 로컬 작업이다. 독립된 사람의 관절 정답 검증이나 새로운 사용자의 평가와는 구분한다.')
    para('다음 판단','이번 결과의 개선·회귀와 추가 처리 비용을 함께 보고 연구 후보의 채택 여부를 정한다. 정밀 교정에는 위치 정확도와 손 모양 근거 검증이 별도로 필요하다. DTW에서는 정상 속도 차이에도 민감한 연속 변화 경고를 곧바로 교정 지시로 쓰지 않고, 실제 관측에서 찾은 시간 대응과 짧은 동작 누락 보존을 함께 검증한다. 작은 비교 완료를 프로젝트 전체 완료로 간주하지 않는다.')
    txt='손목 위치 보완 입력의 C/W 개발 비교\n'+at+'\n\n'+'\n'.join(text)
    document='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손목 보완 C/W 비교</title><style>body{margin:0;background:#f1f4f8;color:#172435;font:16px/1.7 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:32px}header,section{background:white;border-radius:16px;padding:26px;margin-bottom:20px}h1{font-size:30px}h2{font-size:22px;margin-top:0}p{white-space:pre-line}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dbe3eb;text-align:left}th{background:#eef3f8}@media(max-width:800px){main{padding:12px}section{overflow:auto}}@media print{body{background:white}section{break-inside:avoid;border:1px solid #ddd}}</style><main><header><h1>손목 위치 보완 입력의 C/W 개발 비교</h1><p>'+html.escape(at)+'</p><p>인식 효과 · 회귀 사례 · 추가 계산 비용 · 정밀도 한계</p></header>'+''.join(sections)+'</main></html>'
    for name,content in [('RESULT_KO.txt',txt),('RESULT_KO.html',document)]:
        with (J/name).open('x',encoding='utf8') as out:out.write(content)
    pins={str(J/name):sha(J/name) for name in ['CONTEXT.json','VERIFIED.json','CROSS_VERIFIED.json','TRAIN_INPUT_IMPACT.json','DEV_INPUT_IMPACT.json','RESULT_KO.txt','RESULT_KO.html']}
    with (J/'REPORT_CHECK.json').open('x',encoding='utf8') as out:json.dump(dict(at=at,status='REPORT_FROM_VERIFIED_RESULTS',gate_passed=passed,pins=pins,goal_complete=False),out,ensure_ascii=False,indent=2)
    print('REPORT_FROM_VERIFIED_RESULTS',passed)
if __name__=='__main__':main()
