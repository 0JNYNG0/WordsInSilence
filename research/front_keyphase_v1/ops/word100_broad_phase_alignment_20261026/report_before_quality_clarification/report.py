import study as S
import numpy as np
import html
def main():
 v=S.read(S.U/'VERIFIED.json');r=S.read(S.U/'RESULT.json');qc=S.read(S.U/'QUALITY_CORRECTION.json');assert qc['counts']['conditions']==1200;S.check(qc['pins']);n=v['totals']['normal'];reg=v['normal_regressions'];gains=sum(x['delta_frames']<-1e-9 for x in reg);loss=sum(x['delta_frames']>1e-9 for x in reg);ties=len(reg)-gains-loss
 names={'angle':'손가락 각도','palm':'손바닥 위치','elbow':'팔꿈치 위치','pose_wrist':'팔에서 찾은 손목','hand_wrist':'손에서 찾은 손목'}
 group_rows=''
 for key,g in n['groups'].items():
  a,b=g['old_mean'],g['new_mean'];unit='도' if key=='angle' else '어깨너비 단위';change=(b/a-1)*100 if a else 0
  group_rows+=f'<tr><td>{names[key]}</td><td>{a:.6f}</td><td>{b:.6f}</td><td>{change:+.2f}%</td><td>{unit}</td></tr>'
 gate=v['gate'];normal_pct=(n['time_new_mean_frames']/n['time_old_mean_frames']-1)*100
 missing=v['totals']['missing'];omission=v['totals']['omission'];zero=v['deleted_window_positive_to_zero_cells']
 report=f'''공개 100단어 · 300영상 시간정렬 확대 검증

결론: 연구 비교 계산·별도 산술검산 완료. 사전 진단 기준 {'통과' if gate['passed'] else '미통과'}. 운영 적용·TCN 변경 없음.
YOLO는 선택적 손 검출 보조다. 주 관측은 MediaPipe이며, YOLO가 찾았다는 이유만으로 정확한 손가락이나 올바른 수행을 확정하지 않는다.

1. 왜 이 작업을 했나
손목 관측을 더한 W를 3회 학습했지만 기존 C보다 인식률이 일관되게 좋아지지 않았다. 공개 DEV300 기준 sensor 평균98.67→98.11%, masked98.44→98.22%로 미채택했다. 검출 증가와 인식·피드백 개선을 구분해야 한다.
그 뒤 DTW 시간 맞추기를 보완했다. 3단어 시험에서 정상속도 차이에 따른 비교수치가 줄었지만 짧은 삭제1건의 연속비교 경고가 사라지는 회귀가 있었다. 기존 순간비교 정보는 남아 있었다. 그래서 100단어로 범위를 확대했다.

2. 이번 범위
공개 AIHub 100단어, 촬영ID11/12/13의 기존 개발검증300영상. 원래300 + 정상속도변형1200 + 짧은/긴 구간삭제1800 + 관측누락1200 =4500비교. 새로 촬영한4500영상이 아니다. 관절좌표의 시간·누락을 인위적으로 바꾼 시험이며 실제 RGB 가림이나 수어의 의미 오류를 직접 검증한 것은 아니다.
학습0회·새 검출0회·TCN 추론0회. 사용자·멘토·별도199/200 평가자료 사용0.

3. 바꾼 점
원래 DTW가 고른 대응 시점을 최대1입력프레임 이내에서 부드럽게 연결한다. 관절값은 원래 관측을 사용한다. 새 연결에서 비교 근거가 없어지면 그 부위의 원래 연속비교 전체를 남긴다. 원래 순간비교도 따로 보존한다. 낮은 점수만 골라서 좋은 결과로 바꾸지 않는다.
시간변형은 좌표·관측표시에서 시작해 새 시간에 맞춰 속도정보도 다시 계산했다. 이 시험의 기본 DTW는 속도그룹 가중치0이므로 이 수정만으로 정렬이 좋아졌다고 해석하면 안 된다.

4. 정상 속도변형1200비교
알려진 원래 시점과의 평균차이: {n['time_old_mean_frames']:.6f}→{n['time_new_mean_frames']:.6f}프레임 ({normal_pct:+.2f}%).
영상·변형별 시점 오차 감소{gains}, 증가{loss}, 동일{ties}. 정지·반복장면의 동일한 자세는 영상만으로 정확한 시간의 정답이 하나로 정해지지 않을 수 있다. 이 수치는 알려진 합성 시간표에 대한 진단이다.
기존 연속비교 근거 {n['sequence_old']:,}개, 후보 {n['sequence_new']:,}개, 잃은 근거 {n['lost_support']:,}개. 원래 경로로 보존한 채널 {n['fallback']:,}개.

5. 놓친 부분을 숨기지 않았나
관측누락1200조건의 고정 비교범위를 유지했고, 손 전체를 숨긴600조건에서 해당 손가락·손 위치를 일치로 처리하지 않았다. 참조의 원래 미관측도 남아 있다. 손의 관측 부족은 수어를 틀리게 수행했다는 판정과 다르다.
삭제구간 앞·중간·뒤의 기존 양수 차이가 새 연속비교에서 수치상0으로 줄어든 채널 {zero:,}개. 이는 수어 오류 정답률이 아니라 잃을 수 있는 비교 신호를 찾는 진단이다. 원래 순간비교와 원래 연속비교 값도 결과에 함께 보존한다.
한 손 동작에서 쉬는 손까지 반드시 써야 한다는 판정은 하지 않았다. 이 시험은 고정된 양손·부위 조사표이며, 각 손의 필수 수행구간에 대한 의미 주석은 아니다.
검산 중 손만 숨기는 시험이 팔꿈치·몸 관측 손목의 품질 표시도 낮추는 문제를 발견했다. 동결된 첫 결과를 보존하고 별도 정정표로 기존 {qc['counts']['old_arm_quality_restored']:,}개·후보 {qc['counts']['new_arm_quality_restored']:,}개의 팔 근거 표시를 복구했다. 관절·시간정렬·차이값·채택기준은 바뀌지 않았다. QUALITY_CORRECTION.json과 quality_correction/이 품질 표시의 최신 정정 근거다.

6. 현재 판단과 다음 작업
이 결과만으로 올바른 수행의 허용 범위나 교정 문구를 확정할 수 없다. 좋은 평균만 보지 않고, 오차가 증가한 공개 장면과 삭제 신호가 줄어든 부분을 확인한다. 이미 보존한 순간·연속 차이와 관측 비율을 함께 보여주는 연구 비교 경로로 통합할 수 있는지 검토하되, 검증되지 않은 부분은 비교 근거 부족 또는 시간 대응 불확실로 남긴다. 기존 C·T3-F·E3 및 운영 웹·DTW는 그대로다.
검증: 같은 Codex의 별도 path 최적비용·descriptor 계산12조건 및4500개 저장배열/분모/집계 검산. 독립적인 사람의 손관절 정답 검증은 아니다. 전체 개선 목표는 진행 중이다.
'''
 (S.U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
 worst=''.join(f"<tr><td>{html.escape(x['clip_id'])}</td><td>{x['kind']}</td><td>{x['old']:.4f}</td><td>{x['new']:.4f}</td><td>{x['delta_frames']:+.4f}</td></tr>" for x in reg[:12])
 doc='<!doctype html><html lang="ko"><meta charset="utf-8"><title>공개100단어 시간정렬 확대검증</title><style>body{margin:0;background:#edf2f6;color:#172b43;font:17px/1.8 system-ui}main{max-width:1120px;margin:30px auto;background:white;border-radius:18px;padding:34px}h1{font-size:30px}h2{font-size:23px;margin-top:34px}table{width:100%;border-collapse:collapse;font-size:15px}td,th{padding:10px;border-bottom:1px solid #dae3ea;text-align:left}pre{white-space:pre-wrap;font:inherit}.note{background:#fff0d6;padding:18px;border-radius:9px}.good{background:#e9f5ee;padding:18px}img{max-width:100%}small{color:#52677c}a{color:#1768a6}</style><main>'
 doc+='<h1>공개100단어로 넓혀 확인한 시간정렬</h1><p class="note">'+('사전 진단 기준 통과. 운영 채택은 별도 검증 필요.' if gate['passed'] else '평균 개선과 남은 회귀를 함께 확인한 연구 결과. 운영 미적용.')+'</p><img src="summary.png" alt="시간 오차와 비교 근거"><h2>정상 속도에서 같은 근거만 비교한 차이</h2><table><tr><th>항목</th><th>기존</th><th>후보</th><th>변화</th><th>단위</th></tr>'+group_rows+'</table><small>앞·중간·뒤의 위치·각도3개와 변화량2개, 총5구성요소 평균. 사람에게 주는 점수나 정답률이 아닙니다.</small><h2>전체 설명</h2><pre>'+html.escape(report)+'</pre><details><summary>시점 오차 증가 상위 장면</summary><table><tr><th>공개영상</th><th>시간변형</th><th>기존</th><th>후보</th><th>증가량</th></tr>'+worst+'</table></details><p><a href="VERIFIED.json">전체 검산 근거</a> · <a href="FROZEN.json">사전 고정 조건</a></p></main></html>'
 (S.U/'RESULT_KO.html').write_text(doc,encoding='utf-8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
 fig,axs=plt.subplots(1,3,figsize=(14,4.5));fig.patch.set_facecolor('#f4f7fb')
 for ax in axs:ax.set_facecolor('#f4f7fb');ax.spines[['top','right']].set_visible(False)
 vals=[n['time_old_mean_frames'],n['time_new_mean_frames']];axs[0].bar(['기존','후보'],vals,color=['#73879d','#24867d']);axs[0].set_title('정상 속도 · 평균 시점 차이');axs[0].set_ylabel('원래 영상 프레임 단위')
 for i,vv in enumerate(vals):axs[0].text(i,vv,f'{vv:.4f}',ha='center',va='bottom')
 axs[1].bar(['감소','증가','동일'],[gains,loss,ties],color=['#24867d','#d98b35','#8a9bad']);axs[1].set_title('1200개 속도변형 조건별 결과');axs[1].set_ylabel('비교 건수')
 for i,vv in enumerate([gains,loss,ties]):axs[1].text(i,vv,str(vv),ha='center',va='bottom')
 allold=sum(t['sequence_old'] for t in v['totals'].values());allnew=sum(t['sequence_new'] for t in v['totals'].values());axs[2].bar(['기존','후보'],[allold/1e6,allnew/1e6],color=['#73879d','#24867d']);axs[2].set_title('전체4500조건 · 연속비교 근거');axs[2].set_ylabel('백만 채널·시점')
 for i,vv in enumerate([allold,allnew]):axs[2].text(i,vv/1e6,f'{vv:,}',ha='center',va='bottom',fontsize=9)
 fig.suptitle('공개100단어·300영상의 합성시간 진단 — 인식률/수어 정답률 아님',fontsize=15);fig.tight_layout();fig.savefig(S.U/'summary.png',dpi=150);plt.close(fig)
 S.write(S.U/'REPORT_CHECK.json',dict(at=S.now(),gate=gate,normal_gains=gains,normal_losses=loss,normal_ties=ties,files={str(S.U/n):S.sha(S.U/n) for n in ['RESULT_KO.txt','RESULT_KO.html','summary.png']},production_changed=False))
 print('REPORT_SAVED',gate,flush=True)
if __name__=='__main__':main()
