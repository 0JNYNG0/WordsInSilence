from pathlib import Path
import json,hashlib,html,datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
U=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text('utf-8-sig'))
r=read(U/'RESULT.json');v=read(U/'VERIFIED.json')
assert v['result_sha256']==hashlib.sha256((U/'RESULT.json').read_bytes()).hexdigest() and v['status']=='PASS_SEPARATE_TIME_INVERSE_DESCRIPTOR_AND_RECOUNT'
def cohort(name):
    aa=[x for k,x in r['aggregate'].items() if k.startswith(name+'__')]
    out={k:sum(x[k] for x in aa) for k in ('expected','common','actual_comparable','oracle_comparable','lost_oracle','gained_oracle')}
    for k in ('actual_new','oracle_new','removed_by_phase','remaining','new_after_phase'):out[k]=sum(x[k+'_'+str(j)] for x in aa for j in (3,4))
    return out
k=cohort('known');c=cohort('cross')
text=f'''정상 속도에서 생긴 움직임 경고의 원인 분리

결론
정상 속도 대조에서 커진 차이 표시에는, 같은 장면을 조금 다른 시점에 연결한 영향이 크게 나타났다. 정답 시간 대응을 알고 있는 시험에서 그 연결을 보정하면 새 차이 표시가 줄었다. 그러나 이는 원인 진단이며, 실제 사용자 영상의 정렬을 고친 결과가 아니다. 정답 시점을 실제 서비스에 넣을 수는 없고, 경고 문턱도 바꾸지 않았다.

쉽게 설명하면
같은 춤을 빠르게 재생해도 같은 자세끼리 비교하면 모습은 비슷해야 한다. 그런데 손을 올리는 순간과 손을 다 올린 순간을 짝지으면 움직임이 다른 것처럼 보인다. 이번에는 원본을 얼마나 빠르게 바꿨는지 알고 있으므로, 원래 짝을 정확히 되짚어 이 영향만 확인했다. 여러 사람이 서로 다른 속도로 수행한 자료에서도 각 사람의 원본 연결을 유지한 채 속도 변화의 영향만 비교했다. 원래 사람 간 연결 자체가 의미상 정답이라는 뜻은 아니다.

확인한 자료와 방법
공개 TRAIN 3개 단어·기록ID9개·27영상에서 이미 만든 정상 속도 대조972개를 재사용했다. 같은 원본108개, 다른 사람 참고864개이다. 빠르게·천천히·부분 속도 변화·잠깐 멈춤을 포함한다. 새 MediaPipe·YOLO·TCN·DTW 탐색은 하지 않았다.
A: 저장되어 있던 실제 시간 연결 결과.
B: 원래 대응 시점을 알려준 결과. 속도가 바뀐 입력의 값은 그대로이고 연결만 원래 시점으로 되짚었다.
C: 변형 전 원본의 비교 결과. 새로 생긴 경고인지 판단할 기준이다.
세 조건 모두 비교할 수 있던 동일한 시간·부위에서만 A와 B의 새 차이를 셌다. 손을 못 봐서 비교할 수 없어진 칸은 별도로 기록했다.

새로 생긴 변화량 차이 표시
같은 원본 참고108대조: {k['actual_new']:,} → {k['oracle_new']:,}항목. 공통 비교 근거 {k['common']:,}칸.
다른 사람 참고864대조: {c['actual_new']:,} → {c['oracle_new']:,}항목. 공통 비교 근거 {c['common']:,}칸.
여기서 항목은 각도·위치의 앞→중간 또는 중간→뒤 변화량 한 개이다. 같은 장면이 여러 참고와 비교되며, 한 시간·관절에서 두 변화량을 센다. 영상 수·정답률·실사용 오류율이 아니다.

좋아진 부분만 세지 않았다
다른 사람 참고에서는 기존 새 차이 중 {c['removed_by_phase']:,}항목이 사라졌지만 {c['remaining']:,}항목은 남았다. 알려준 시점으로 바꾼 뒤 새로 나타난 항목도 {c['new_after_phase']:,}개라서 최종 {c['oracle_new']:,}개이다. 정상 속도 변형에서도 입력을 다시 샘플링하면 짧은 봉우리나 회전 구간의 값이 조금 달라질 수 있고, 현재 여러 사람의 허용 범위가 이런 변화를 모두 다루는 것은 아니다.
또 다른 사람 비교에서 기존 비교 근거 {c['lost_oracle']:,}칸을 잃고 {c['gained_oracle']:,}칸을 얻었다. 위 차이 감소는 이 손실을 이득으로 센 것이 아니라, 세 조건 공통 근거에서만 계산한 결과다. 전체 비교 필요 범위 {c['expected']:,}칸은 그대로다. 새 시점이 미관측이나 확인된 손내림을 가로지르면 그 연속 비교는 보류한다.

보존한 것과 하지 않은 것
기존 순간별 비교, 원촬영의 미관측 시간표, 손내림·필요 손의 비교 범위, 관절 좌표와 좌우 배정은 그대로다. 관측 부족을 정상이나0차이로 바꾸지 않았다. 교차 장면의 좌표를 새로 수정하지도 않았다. C·T3-F·E3와 운영 웹·DTW는 유지했다. 사용자·멘토 촬영과 추가199/200평가를 열거나 사용하지 않았다.
독립적으로 작성한 시간 역대응·구간·차이 계산으로972대조, {v['component_cells']:,}개 항목 상태를 검산했다. 같은 Codex의 별도 계산 검증이며 수어 전문가의 동작 단계 정답 검토는 아니다.

다음 작업
현재 진행 중인 공개899 학습 입력 준비와 손목-only 후보 비교를 먼저 끝낸다. 피드백에서는 현재의 순간별 차이와 근거 부족 표시는 유지하고, 정상 속도에도 민감한 연속 변화 경고를 곧바로 교정 지시로 사용하지 않는다. 다음 시간 정렬 후보는 실제 입력만 사용해야 하며, 알려진 시간 대응은 평가 때만 사용한다. 속도 변형에서 대응 시점이 안정되는지와 짧은 동작을 지운 대조에서도 변화를 놓치지 않는지를 함께 통과해야 한다. 손·팔 위치와 방향을 단순히 더 강조한 이전 후보는 효과가 확인되지 않았으므로 그대로 반복하지 않는다. 이번 진단만으로 통과 판정이나 전체 목표 완료를 선언하지 않는다.
'''
(U/'RESULT_KO.txt').write_text(text,'utf8')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
fig,axes=plt.subplots(1,2,figsize=(10,4.7))
for ax,d,title in zip(axes,[k,c],['같은 원본 참고 · 108대조','다른 사람 참고 · 864대조']):
    vals=[d['actual_new'],d['oracle_new']];ax.bar([0,1],vals,color=['#d17b54','#39867e'],width=.55);ax.set_xticks([0,1],['기존 시간 연결','정답 시점을 알려준 진단']);ax.set_ylim(0,max(vals)*1.25);ax.set_title(title);ax.set_ylabel('새 변화량 차이 표시 (항목 수)')
    for i,y in enumerate(vals):ax.text(i,y+max(vals)*.025,f'{y:,}',ha='center')
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('같은 동작의 속도 차이: 시간 연결이 경고에 미친 영향',fontsize=15)
fig.text(.5,.02,'세 조건의 공통 비교 근거에서 집계 · 정답 시점을 쓰는 원인 진단이며 실제 기능 개선 결과가 아님',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(U/'normal_speed_diagnosis.png',dpi=160);plt.close(fig)
paragraphs=text.strip().split('\n\n');body=''.join('<section><p>'+html.escape(p).replace('\n','<br>')+'</p></section>' for p in paragraphs[1:])
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>정상 속도 경고 원인 분리</title><style>body{margin:0;background:#eef2f5;color:#182c3c;font:17px/1.8 system-ui,"Malgun Gothic",sans-serif}main{max-width:1040px;margin:40px auto;padding:32px}h1{font-size:32px;line-height:1.4}section{background:white;padding:24px 30px;margin:18px 0;border-radius:16px}img{width:100%;border-radius:16px}.tag{color:#86551d;font-weight:700}</style><main><div class="tag">연구 진단 · 운영 적용 없음 · 전체 개선 목표 진행 중</div><h1>정상 속도에서도 차이가 커 보였던 이유</h1><img src="normal_speed_diagnosis.png" alt="기존 시간 연결과 정답 대응을 제공한 진단의 비교">'''+body+'</main></html>'
(U/'RESULT_KO.html').write_text(page,'utf8')
out=dict(at=datetime.datetime.now().astimezone().isoformat(),status='VERIFIED_DIAGNOSIS_NOT_RUNTIME_IMPROVEMENT',self_source=k,cross_person=c,training=0,production_changes=0,user_access=0,results=[str(U/p) for p in ('RESULT_KO.txt','RESULT_KO.html','normal_speed_diagnosis.png')],goal_complete=False)
with (U/'DONE.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('REPORT WRITTEN; no actual alignment candidate adopted')
