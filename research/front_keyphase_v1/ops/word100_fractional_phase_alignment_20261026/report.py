from pathlib import Path
import json,html,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf8')
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
U=Path(__file__).resolve().parent;P=U.parent/'word100_point_sequence_split_20261022'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def table(head,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table>'
def main():
    v=read(U/'VERIFIED.json');assert v['status']=='PASS_NUMERIC_AND_PRESERVATION_NOT_SEMANTIC_TRUTH'
    r=read(U/'RESULT.json');assert hashlib.sha256((U/'RESULT.json').read_bytes()).hexdigest()==v['result_sha256'];assert not (U/'RESULT_KO.html').exists()
    normals=[x for x in v['rows'] if x['folder']=='known' and 'old_phase_mae' in x];order=sorted(normals,key=lambda x:(x['new_phase_mae']-x['old_phase_mae'],x['name']));chosen=[order[0],order[-1]]
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for row,(ax,bx),label in zip(chosen,axs,['가장 많이 줄어든 사례','가장 많이 늘어난 사례']):
        z=load(P/'known'/(row['name']+'.npz'));n=load(U/'cases'/('known__'+row['name']+'.npz'));j=np.arange(len(n['index']));m=np.array([np.median(z['path'][z['path'][:,1]==i,0]) for i in j]);ix=z['source_index'];old=np.interp(m,np.arange(len(ix)),ix);new=np.interp(n['index'],np.arange(len(ix)),ix)
        ax.plot(j,j,'--',color='black',label='시험의 정답 대응');ax.plot(j,old,label='기존 대응',color='#677d98');ax.plot(j,new,label='연속화 후보',color='#d77724');ax.set_title(label+'\n'+row['name'],fontsize=10);ax.set_xlabel('참고 영상 시점(입력 프레임)');ax.set_ylabel('원본에 환산한 수행 시점');ax.legend(fontsize=8)
        bx.plot(j,abs(old-j),label='기존 시점 오차',color='#677d98');bx.plot(j,abs(new-j),label='후보 시점 오차',color='#d77724');bx.set_title('정답 시점과의 거리 — 낮을수록 좋음');bx.set_xlabel('참고 시점');bx.set_ylabel('입력 프레임 차이');bx.legend(fontsize=8)
    fig.suptitle('정답 시점을 아는 속도 변형 시험: 이득과 회귀를 함께 확인',fontsize=16);fig.savefig(U/'phase_examples.png',dpi=140);plt.close(fig)
    p=v['normal_phase'];warn=v['normal_warnings'];old=sum(x['old'] for k,x in warn.items() if k.startswith('cross_'));new=sum(x['new'] for k,x in warn.items() if k.startswith('cross_'))
    conclusion='사전 조건 통과. 더 넓은 공개 검증이 필요한 연구 후보이며 운영 적용·동작 정답 판정은 아직 하지 않는다.' if v['gate_passed'] else '사전 조건 미통과. 이 시간 대응 연속화 후보를 운영에 적용하지 않는다. 좋아진 수치만으로 정상 동작과 짧은 동작 보존이 해결됐다고 결론 내리지 않는다.'
    t=[['같은 원본 속도 변형의 정답 대응 오차',f"{p['known_old']['mean']:.4f}",f"{p['known_new']['mean']:.4f}",'평균 입력 프레임'],['다른 사람 참고에서 속도 변형 전후 대응 변화',f"{p['cross_old']['mean']:.4f}",f"{p['cross_new']['mean']:.4f}",'사람 간 정답 시점 아님'],['다른 사람 참고의 새 차이 항목',old,new,'동일 비교 가능 범위에서만 계산']]
    om=[[k,x['variants'],x['old_at_least5'],x['new_at_least5'],x['gained'],x['lost']] for k,x in v['omissions'].items()]
    loss=v['normal_coverage_loss'];txt='DTW 시간 대응 연속화 시험\n\n'+conclusion+'\n\n'+'\n'.join(' | '.join(map(str,x)) for x in t)+'\n\n누락 시험\n'+'\n'.join(' | '.join(map(str,x)) for x in om)+'\n\n정상 속도에서 비교 근거 감소: '+str(loss)+'\n\n관절은 바꾸지 않고 기존 DTW 대응 시점만 최대1입력프레임 이내 조정했다. 원래 순간별 차이·고정 비교범위·미관측은 모두 보존했다. 새 근거는 만들어내지 않았다. 새 대응 시점의 보간값은 관측 자체가 아닌 계산된 추정이다.\n공개TRAIN3단어9기록ID27영상의 이미 사용한 개발 묶음이다. 실제 사용자의 정답률이나 모든 수어의 정상 범위는 아니다. 짧은 삭제 시험은 인위적 데이터 변형이며 언어학적으로 필수 동작인지 확정하지 않는다. 구현과 별도 산술검산 모두 Codex 로컬 작업이며 사람의 독립 정답 검증은 아니다. 기존C/T3F/E3/운영보존,사용자자료미사용.\n'
    doc='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DTW 시간 대응 연속화 시험</title><style>body{font:17px/1.8 "Malgun Gothic",sans-serif;background:#eef3f8;color:#183046;margin:0}main{max-width:1150px;margin:auto;padding:30px}header,section{background:white;padding:25px;border-radius:15px;margin-bottom:20px}h1{font-size:30px}h2{font-size:23px}img{width:100%}table{width:100%;border-collapse:collapse;font-size:15px}th,td{padding:10px;text-align:left;border-bottom:1px solid #d9e1e8}th{background:#edf2f6}.note{border-left:5px solid #c27332}p{white-space:pre-line}</style><main><header><h1>속도가 달라도 같은 시점끼리 비교하기</h1><p>고정한 시간 대응 후보 · 정상 동작 · 짧은 누락 · 비교 근거 보존</p></header><section class="note"><h2>결론</h2><p>'+html.escape(conclusion)+'</p></section><section><h2>바꾼 부분</h2><p>관절 좌표를 바꾸지 않고, 기존 DTW가 고른 대응 시점의 작은 계단을 최대 1개 입력 프레임 범위에서 연속적으로 정리했다. 원래 순간별 비교 결과와 손이 안 보인 시간은 그대로 보존했다. 정답 시간 정보는 시험 채점에만 사용했다.</p></section><section><h2>정상 속도 비교</h2>'+table(['항목','기존','후보','해석'],t)+'<p>같은 원본 108개·다른 사람 참고 864개의 정상 속도 변형 시험이다. 차이 항목은 손·각도·시점별 계산 수이며 영상 오류 수나 정답률이 아니다. 비교 근거가 사라져 경고가 줄어든 경우는 이득으로 세지 않았다.</p></section><section><h2>짧은 동작을 놓치지 않는가</h2>'+table(['삭제 조건','변형 수','기존:8참고 중5개 이상 감지','후보','새로 감지','놓치게 됨'],om)+'<p>삭제 위치는 미리 고정했다. 3프레임 또는 핵심 구간 20%를 제거한 인위적 시험이며 필수 수어 단계의 사람 정답 주석은 아니다.</p></section><section><h2>비교 근거 보존</h2><p>'+html.escape(str(loss))+'\n정상 속도에서 연속 비교가 불가능해진 항목 수다. 손내림과 관측 부족을 가리지 않고, 새 대응에 필요한 모든 관측이 있는지 보수적으로 확인했다. 원래 순간별 결과와 미관측 기록은 모두 그대로다.</p></section><section><h2>이득과 회귀 사례</h2><img src="phase_examples.png" alt="가장 많이 개선·악화된 두 대응 사례"><p>이미 만들어 둔 시점 정답을 이용한 시험 그래프다. 실제 사용자 영상에서 정답 시점을 알고 정렬한 것은 아니다.</p></section><section><h2>현재 한계</h2><p>공개 TRAIN 3단어·9기록ID·27영상의 개발 시험이다. 독립 새 사람 평가나 모든 수어 검증이 아니다. 보간된 손 모양은 실제 새 관측이 아니다. 분류 모델 C·T3-F·E3와 운영 웹·DTW는 그대로이며 사용자·멘토 촬영은 사용하지 않았다. 작은 실험의 완료가 프로젝트 목표 완료는 아니다.</p></section></main></html>'
    for name,s in [('RESULT_KO.txt',txt),('RESULT_KO.html',doc)]:
        with (U/name).open('x',encoding='utf8') as f:f.write(s)
    out=dict(status='REPORT_FROM_VERIFIED_RESULTS',gate_passed=v['gate_passed'],selected_graph_cases=[x['name'] for x in chosen],pins={str(U/x):hashlib.sha256((U/x).read_bytes()).hexdigest() for x in ['VERIFIED.json','RESULT_KO.txt','RESULT_KO.html','phase_examples.png']})
    with (U/'REPORT_CHECK.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print('REPORT',v['gate_passed'])
if __name__=='__main__':main()
