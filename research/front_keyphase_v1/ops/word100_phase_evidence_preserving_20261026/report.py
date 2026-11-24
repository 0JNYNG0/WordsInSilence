from pathlib import Path
import json,html,hashlib,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf8');plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def table(head,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table>'
def main():
    v=read(U/'VERIFIED.json');a=read(U/'REGRESSION_AUDIT.json');assert not v['gate_passed'];assert not (U/'RESULT_KO.html').exists()
    for p,h in {**read(U/'RESULT.json')['pins'],**a['pins']}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    t=v['known_effective_timing'];c=v['normal_cross'];om=v['omissions'];ct=v['counts'];pct=100*(c['old_channel']-c['new_channel'])/c['old_channel']
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    values=[[t['old_frame_mae'],t['new_frame_mae']],[c['old_channel'],c['new_channel']],[om['skip3']['old'],om['skip3']['new']]]
    titles=['알려진 시점 오차\n낮을수록 좋음','정상 속도에서 추가 차이 경고\n비교 근거는 동일','3프레임 삭제 감지\n81변형 중 5/8참고 이상']
    for ax,vals,title in zip(axes,values,titles):
        ax.bar(['기존','후보'],vals,color=['#657c98','#c67b36']);ax.set_title(title,fontsize=12);ax.set_ylim(0,max(vals)*1.25)
        for i,x in enumerate(vals):ax.text(i,x,f'{x:.3f}' if max(vals)<1 else f'{int(x):,}',ha='center',va='bottom',fontsize=12)
    axes[0].set_ylabel('입력 프레임 차이');axes[1].set_ylabel('채널·시점 수');axes[2].set_ylabel('변형 수');fig.suptitle('이득과 남은 회귀 — 짧은 삭제 2개 새 감지, 1개 감지 감소',fontsize=14);fig.savefig(U/'summary.png',dpi=140);plt.close(fig)
    facts=[['공개 비교',2808,'3단어·9기록ID·27영상, 반복 사용한 개발 자료'],['원래 연속 비교 근거',f"{ct['sequence_coverage_preserved']:,}",'전부 보존'],['새 정렬 근거 부족',f"{ct['fallback_cells']:,}",'원래 비교로 유지, 새 관측으로 승격하지 않음'],['새 정상 속도 경고',f"{c['old_channel']:,} → {c['new_channel']:,}",f'{pct:.2f}% 감소; 영상 오류율 아님'],['짧은 삭제 감지',f"{om['skip3']['old']} → {om['skip3']['new']} /81",'새로 감지2·감소1'],['원래 순간 비교도 함께 볼 때','75 → 75 /81','새로 놓친 변형0; 보조 집계, 실패 기준 대체 아님']]
    conclusion='일부 시간 대응과 정상 속도 경고는 개선됐지만, 연속 비교만 보면 기존에 감지하던 짧은 삭제 1개가 기준을 넘지 못해 사전 조건은 미통과다. 운영에는 적용하지 않았다.'
    body=[('무엇을 바꿨나','앞뒤 동작을 비교하는 새 시간 대응에 필요한 관측이 없으면, 원래 확인 가능하던 비교 5개 항목(앞·중간·뒤·두 변화량)을 통째로 유지했다. 작은 차이가 나오는 쪽을 선택한 것이 아니라, 관측이 충분한지 여부만으로 선택했다. 손가락별 출처를 남겨 서로 다른 시점을 하나의 확정 손 모양으로 포장하지 않았다.'),('남은 1개를 확인한 결과','공개 영상에서 중간 3프레임을 인위적으로 뺀 한 변형이 8개 참고 중 5개 반응에서 4개 반응으로 바뀌었다. 한 참고에서는 비교하는 원본 자체의 경고가 새로 생겨 추가 변화가 가려졌고, 다른 참고에서는 실제 변형의 각도 차이가 줄었다. 후자의 순간별 차이는 원래 기록에 남아 있다. 관측 부족 때문에 정상으로 바꾼 것은 아니지만, 연속 비교가 구별 동작을 더 잘 보존한다고 확정할 수도 없다.'),('수치의 의미','시점 오차는 정답 시간 변형을 알고 만든 시험에서만 계산했다. 서로 다른 사람 영상의 정확한 동작 시점 정답은 아직 없다. 경고 개수는 손·각도·시점의 계산 항목 수이며 수어 수행 정답률이 아니다. 3프레임 삭제는 인위적 시험이고 언어학적으로 필수 단계라는 정답 주석은 아니다.'),('앞으로의 작업','이번 후보를 운영에 넣지 않고, 이미 준비된 공개 100단어·300영상으로 시간 대응 시험 범위를 넓힌다. 새 검출이나 재학습 없이 저장된 영상 관측과 시간만 바꾼 대조를 사용한다. 같은 소수 단어에서만 개선되는지, 정상 손 모양·짧은 동작·관측 부족이 넓은 범위에서도 유지되는지 확인한다. 미관측을 정상으로 간주하거나 단어별 규칙을 넣지 않는다.'),('보존한 것','기존 C·T3-F·E3, 운영 웹·DTW, 원본 자료를 유지했다. 사용자·멘토 영상은 접근·학습·튜닝·모델 선택에 사용하지 않았다. 구현과 독립 수식 검산은 같은 Codex의 로컬 작업이며 사람의 독립 정답 검증은 아니다.')]
    txt='비교 근거를 보존하는 시간 대응 후보\n\n'+conclusion+'\n\n'+'\n'.join(' | '.join(map(str,row)) for row in facts)+'\n\n'+'\n\n'.join(k+'\n'+b for k,b in body)
    doc='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>비교 근거 보존과 시간 대응</title><style>body{margin:0;background:#f0f4f7;color:#203144;font:17px/1.8 "Malgun Gothic",sans-serif}main{max-width:1150px;margin:auto;padding:30px}header,section{padding:26px;background:white;border-radius:15px;margin-bottom:20px}h1{font-size:30px}h2{font-size:23px}img{width:100%}table{width:100%;border-collapse:collapse;font-size:15px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dbe3eb}th{background:#edf3f8}.notice{border-left:6px solid #b77b3a}p{white-space:pre-line}</style><main><header><h1>근거는 보존하고, 시간 대응은 개선하기</h1><p>기존 인식 모델 유지 · 공개 자료 검증 · 개선과 회귀 함께 보고</p></header><section class="notice"><h2>현재 판단</h2><p>'+html.escape(conclusion)+'</p></section><section><img src="summary.png" alt="시점·경고·짧은 누락 비교 그래프">'+table(['항목','결과','해석'],facts)+'</section>'+''.join('<section><h2>'+html.escape(k)+'</h2><p>'+html.escape(b)+'</p></section>' for k,b in body)+'</main></html>'
    for name,s in [('RESULT_KO.txt',txt),('RESULT_KO.html',doc)]:
        with (U/name).open('x',encoding='utf8') as f:f.write(s)
    with (U/'REPORT_CHECK.json').open('x',encoding='utf8') as f:json.dump(dict(status='PASS_FROM_VERIFIED_AND_REGRESSION_AUDIT',gate_passed=False,pins={str(U/n):hashlib.sha256((U/n).read_bytes()).hexdigest() for n in ['VERIFIED.json','REGRESSION_AUDIT.json','RESULT_KO.txt','RESULT_KO.html','summary.png']}),f,ensure_ascii=False,indent=2)
    print('REPORT_COMPLETE_NOT_ADOPTED',pct,flush=True)
if __name__=='__main__':main()
