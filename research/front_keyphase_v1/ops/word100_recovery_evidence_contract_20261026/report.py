from common import *
from html import escape

def main():
    f=read(U/'FROZEN.json');r=read(U/'RESULT.json');v=read(U/'VERIFIED.json');assert v['status']=='PASS'
    check(f['pins']);check(f['protected']);check(r['output_pins'])
    labels=read(O/'word100_body_boundary_20261017/training_CONTEXT.json')['labels']
    labelmap={row['word_code']:labels[row['label_index']] for row in f['rows']}
    obs=[]
    for row in r['rows']:
        q=read(row['receipt']);q['label']=labelmap[q['word_code']]
        for model,p in q['recognition']['saved_original_predictions'].items():p['top5_words']=[labels[i] for i in p['top5']]
        obs.append(q)
    comparisons=[]
    for row in r['comparison_rows']:
        q=read(row['receipt']);code=q['query'].split('_')[2];q['label']=labelmap.get(code,code);comparisons.append(q)
    dataset=json.dumps(dict(observations=obs,comparisons=comparisons),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    text=f'''인식과 피드백 근거를 분리한 연구 연결 결과

무엇을 만들었나
공개 DEV300 영상의 원래 단어 인식 입력·저장 결과와, MediaPipe 재검출·선별 YOLO로 보완한 관측을 따로 전달하는 연구 연결부를 구현했다. 손목 위치만 찾은 장면에서 손가락 모양까지 확인했다고 말하지 않는다. 어떤 부위·시간대를 비교했는지, 어디에 정보가 부족한지 읽을 수 있는 분석 기록을 만들었다.

이번 확인 범위
DEV300·37,295프레임: 원래 분류 입력6,113,800값이 바이트까지 동일하다. 손의 원래 관측580,494개 관절·프레임 값과 화면상 손목 순서 교차60프레임을 보존했다. 몸통·팔 좌표도 원본 그대로다. 손목만 보충한 YOLO29손·프레임에서 손바닥/손가락 근거가 생성되지 않았다.
TRAIN46 자료 중 기존126개 사람 간 비교(21영상·3단어)의 저장 DTW 결과를 소비하는 연결부를 검증했다. 비교 범위는 참고에 고정된다. 손이 안 보이면 그 범위를 없애거나 차이0으로 만들지 않고 누락 시간과 null 차이로 남긴다. 같은 데이터의 재검출·분류 추론·시간 정렬을 반복하지 않았다.

수치의 의미
아래는 관측한 채널·프레임의 수이며 정확도가 아니다. 손목/손바닥의 x,y는 각각 다른 채널로 센다. 각도는 한 손15개이고 2D 영상에 투영된 굽힘이다. 3D 해부학 각도와 같지 않다.
DEV300 손목 위치 비교 가능 근거: 기존65,192 → MP재검출69,454 → MP+YOLO69,550.
DEV300 각도 재확인 근거: 기존16,588 → MP재검출43,319 → MP+YOLO43,451. 이는 두 예측의 기하학적 불일치가 없고 좌우 충돌 표시가 없는 범위다. 실제 손가락 정답 인증이 아니다.
기존 TRAIN126 비교에서 손목 채널 비교15,922 → 16,492 → 16,514 / 고정18,444; 각도75,648 → 78,489 → 78,489 / 고정138,330. 기존에 비교 가능하던 채널 손실0. 이는 지난 계산의 전달 검증이며 새 정확도 개선 실험이 아니다.

남은 중요한 문제
1. 재검출은 누락 순간에 집중했다. 최종 계산 가능한 각도434,418개 중386,101개는 재확인 정보가 없다. 원래 검출이 잘못됐다는 뜻이 아니고, 정밀 교정의 근거까지 검증됐다고 할 수 없다는 뜻이다. 모든 좌표와 계산 가능한 각도는 보존했다.
2. 현재 영상 근거에서 좌우/중복 가능성이 엇갈리는261손·프레임은 추가 확인 대상으로 표시했다. 실제 오류261개라는 뜻이 아니다. 좌우를 강제 교체하지 않았고, 해당 손의 피드백만 보류한다. 별도 팔 위치 근거는 유지한다.
3. 두 크기의 영상에서 비슷하게 추정했어도 가려진 손가락을 함께 틀리게 추정할 수 있다. 보인 점 증가만으로 정확한 교정이 완성되지 않는다. 독립적인 사람 손가락 정답 주석은 아직 없다.
4. 정상 속도 차이와 중요한 짧은 동작을 정확히 시간 정렬했는지, 어느 정도 차이가 실제로 틀린 수어인지도 별도 문제다. 이번 기록은 올바른 수행 판정이나 수정 방향을 자동으로 내리지 않는다.

인식 성능은 어떻게 됐나
이번에는 인식 모델·입력·결과를 바꾸지 않았다. 직전 공개 DEV300 비교에서 C13은294→295개, T3-F세모델은295/299/297→296/298/298개였다. 평균99.00→99.11%지만 한 모델은 떨어졌다. YOLO 추가만으로 정답이 늘지는 않았다. 현재 웹 실사용 정확도나 전체 수어의 성능을 뜻하지 않는다.

다음 작업
1. 연구용 연결부에서 관측한 차이와 추정/재확인 부족을 구분하는 기능을 유지한다. 아직 보이지 않은 손가락을 정답처럼 복원하지 않는다.
2. 공개 TRAIN899에서 같은 누락 순간 보완을 준비하되, 다음 인식 후보는 손가락 모양을 억지로 채우지 않고 새 손목 위치만 사용하는 조건을 먼저 고정한다. 기존 좌표·손 크기·시작/종료 구간은 보존해 영향이 섞이지 않게 한다. 학습 전 모든 입력을 검증한다.
3. 동일 학습 조건과3개seed로 기준 입력과 손목 보완 입력을 비교하는 조건을 동결한다. 이미 사용한 DEV를 새로운 최종 시험으로 부르지 않고, 사용자·멘토·추가199/200 자료는 쓰지 않는다. 단어별 예외나 결과를 보고 조정하는 임계값은 없다.
4. 피드백은 별도 검증한다. 일반 검출과 손목-only를 손가락 정답으로 승격하지 않고, 정상 수행·가림·짧은 변화의 관측 품질과 경고를 구분한다. 기존 C·T3-F·웹·DTW 운영은 그대로 유지한다.

구현·검증: Codex 구현, 별도 복소수 유사변환/각도·채널 계산·구간 재구성 검산. 같은 Codex의 별도 검산이며 독립적인 사람 정답 검증이 아니다. {v['counts']['independent_geometry_pairs']:,}쌍 기하학·{v['counts']['angle_states']:,}각도상태·{v['counts']['comparison_cells']:,}비교 셀 검산 PASS. 사용자자료 접근0·학습0·새 검출/모델 추론0·운영교체0. 전체 개선 목표는 진행 중이다.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf8')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>인식 결과와 피드백 근거 분리</title>
<style>*{box-sizing:border-box}body{margin:0;background:#edf2f8;color:#172b43;font:16px/1.65 system-ui}main{max-width:1150px;margin:30px auto;padding:28px;background:white;border-radius:18px}h1{font-size:30px}h2{font-size:23px;margin:32px 0 12px}h3{margin:16px 0 8px}p{margin:9px 0}.note{padding:16px;background:#fff2d8;border-left:4px solid #db9c23}.good{background:#e8f4ee;padding:15px;border-radius:9px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{padding:18px;background:#f3f6fa;border-radius:12px}label{display:block;margin:10px 0}select{max-width:100%;padding:10px;border:1px solid #b3c1d2;border-radius:7px;font:inherit}table{width:100%;border-collapse:collapse}th,td{padding:9px;text-align:left;border-bottom:1px solid #dbe3ed}td{font-variant-numeric:tabular-nums}small{color:#4d6278}details{margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;font:14px/1.7 system-ui}.bar{height:9px;background:#d6e0ed;border-radius:5px;margin-top:4px;overflow:hidden}.bar>i{display:block;height:100%;background:#247e76}.warn{color:#945118}a{color:#1768a6}@media(max-width:780px){.grid{grid-template-columns:1fr}main{margin:0;padding:16px}td,th{padding:5px;font-size:13px}}</style>
<main><h1>무슨 단어인지와, 얼마나 확인했는지를 따로 보기</h1><p>MediaPipe → 빠진 순간 재검출 → 남은 누락에 YOLO 보조 → 같은 상반신 범위 → 위치·손가락 근거 분리</p>
<p class="note">연구용 공개 영상 기록입니다. 기존 웹 모델을 교체하지 않았습니다. <b>차이가 작아도, 확인하지 못한 동작까지 잘했다고 판단하지 않습니다.</b></p>
<div class="grid"><div class="card"><h3>300개 공개 영상</h3><p>원래 단어 인식 결과와 보완 관측을 분리해 전달</p></div><div class="card"><h3>126개 기존 사람 간 비교</h3><p>기준 범위를 유지하며 비교한 부위와 놓친 시간대를 표시</p></div></div>
<h2>① 공개 영상의 인식 결과와 관측 정보</h2><p>이 부분은 관측 현황입니다. 목표 동작의 필수 범위를 계산하지 않았으므로 손이 안 보인 시간을 바로 오류로 판정하지 않습니다.</p>
<label>영상 <select id="clip"></select></label><label>관측 조건 <select id="obsCondition"><option value="0">기존 MediaPipe</option><option value="1">MediaPipe 재검출 추가</option><option value="2" selected>재검출 + 선별 YOLO</option></select></label><div id="recognition"></div><div id="observation" class="grid"></div>
<h2>② 실제 DTW 비교 기록 읽기</h2><p>아래는 기존 TRAIN 자료의 같은 단어·서로 다른 기록ID 비교입니다. 새 DEV300 성능 결과와 합쳐 계산하지 않습니다. 세 조건은 동일 참고 범위를 사용합니다.</p>
<label>비교 영상 <select id="pair"></select></label><label>비교할 관측 <select id="pairCondition"><option value="0">기존 MediaPipe</option><option value="1">MediaPipe 재검출 추가</option><option value="2" selected>재검출 + 선별 YOLO</option></select></label><div id="comparison" class="grid"></div>
<p class="note">관측 비율은 정답률이 아닙니다. 각도 차이는 2D 투영 각도(°), 위치 차이는 어깨 폭을1로 정규화한 좌표 차이입니다. 참고와 비교 가능한 시간대에 한해서 계산했으며, 정상 수어 허용 기준이나 정확한 교정 방향은 아직 검증하지 않았습니다.</p>
<h2>검증 결과와 다음 작업</h2><details open><summary>쉬운 설명 전체</summary><pre>__TEXT__</pre></details><p><a href="VERIFIED.json">별도 검산 기록</a> · <a href="RESULT.json">전체 출처와 집계</a> · <a href="../word100_recovery_dev300_20261026/RESULT_KO.html">직전 인식 비교 및 실제 영상 장면</a></p>
</main><script type="application/json" id="dataset">__DATA__</script><script>
const data=JSON.parse(document.getElementById('dataset').textContent),$=id=>document.getElementById(id), esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={angle:'손가락 각도',palm:'손바닥 중심',elbow:'팔꿈치',pose_wrist:'팔에서 찾은 손목',hand_wrist:'손에서 찾은 손목'},conditions=['기존 관측','MP 재검출','MP + YOLO'];
function times(rs){return rs.length?rs.map(r=>r.start_seconds.toFixed(2)+'–'+r.end_seconds.toFixed(2)+'초 ('+r.samples+'표본)').join(', '):'해당 표본 없음'}
function populate(id,arr,label){$(id).innerHTML=arr.map((r,i)=>'<option value="'+i+'">'+esc(label(r))+'</option>').join('')}
populate('clip',data.observations,r=>r.label+' / ID'+r.signer_id+' / '+r.word_code);populate('pair',data.comparisons,r=>r.label+' / '+r.query.split('_')[3]+' → '+r.reference.split('_')[3]);
function obs(){let q=data.observations[+$('clip').value],k=+$('obsCondition').value;
 $('recognition').innerHTML='<div class="good"><b>인식은 원래 입력·저장 결과 그대로</b><p>'+Object.entries(q.recognition.saved_original_predictions).map(([n,p])=>esc(n)+': '+p.top5_words.map(esc).join(' → ')).join('<br>')+'</p><small>관측 조건을 바꿔도 이 인식 결과는 바뀌지 않습니다. 새 인식 추론을 실행한 화면이 아닙니다.</small></div>';
 $('observation').innerHTML=q.feedback_observation.conditions[k].hands.map(h=>'<div class="card"><h3>'+['왼손','오른손'][h.side]+'</h3><table><tr><th>부위</th><th>계산 가능</th><th>근거 조건 충족</th></tr>'+Object.entries(h.groups).map(([g,a])=>'<tr><td>'+names[g]+'</td><td>'+a.computable_cells.toLocaleString()+'</td><td>'+a.quality_eligible_cells.toLocaleString()+'</td></tr>').join('')+'</table><small>채널·프레임 수입니다. 정확한 해부학 정답을 뜻하지 않습니다.</small><p>각도 재확인 없음: '+h.angle_state_counts[2].toLocaleString()+'개 / 두 예측 불일치: '+h.angle_state_counts[3].toLocaleString()+'개</p><details><summary>손목을 관측하지 못한 시간</summary>'+esc(times(h.wrist_not_observed))+'</details><details><summary>좌우 근거가 엇갈린 시간</summary>'+esc(times(h.identity_disagreement))+'</details><details><summary>일부 손가락 재확인 근거가 부족한 시간</summary>'+esc(times(h.finger_quality_incomplete))+'</details></div>').join('');}
function compare(){let q=data.comparisons[+$('pair').value],k=+$('pairCondition').value;
 $('comparison').innerHTML=q.conditions[k].hands.map(h=>'<div class="card"><h3>'+['왼손','오른손'][h.side]+'</h3><table><tr><th>부위</th><th>비교한 범위</th><th>확인한 부분의 차이</th></tr>'+Object.entries(h.groups).map(([g,a])=>'<tr><td>'+names[g]+'</td><td>'+a.compared_cells+'/'+a.expected_cells+(a.coverage===null?'':'<div class="bar"><i style="width:'+a.coverage*100+'%"></i></div>')+'</td><td>'+(a.mean_difference===null?'근거 없음':a.mean_difference.toFixed(3)+(g==='angle'?'°':''))+'</td></tr>').join('')+'</table><p class="warn">누락 근거: '+h.groups.angle.missing_cells+'개 각도 표본. 낮은 차이가 전체 동작의 일치를 뜻하지 않습니다.</p><details><summary>비교하지 못한 시간대</summary>'+Object.entries(h.groups).map(([g,a])=>'<p><b>'+names[g]+'</b>: '+esc(times(a.missing_intervals))+'</p>').join('')+'</details><details><summary>15개 각도를 따로 확인</summary>'+h.angles.map(a=>'<p>각도 '+(a.angle+1)+': '+a.compared+'/'+a.expected+'표본, '+(a.mean_degrees===null?'차이 계산 불가':a.mean_degrees.toFixed(2)+'°')+'<br><small>부족 시간: '+esc(times(a.missing_intervals))+'</small></p>').join('')+'</details></div>').join('');}
['clip','obsCondition'].forEach(id=>$(id).addEventListener('change',obs));['pair','pairCondition'].forEach(id=>$(id).addEventListener('change',compare));obs();compare();
</script></html>'''.replace('__TEXT__',escape(text)).replace('__DATA__',dataset)
    (U/'RESULT_KO.html').write_text(html,encoding='utf8')
    write(U/'DONE.json',dict(at=now(),status='RESEARCH_CONSUMER_VERIFIED_NOT_OPERATIONAL',goal_complete=False,
        evidence_pins={str(p):sha(p) for p in [U/'FROZEN.json',U/'RESULT.json',U/'VERIFIED.json',U/'RESULT_KO.html',U/'RESULT_KO.txt',U/'report.py']},
        new_accuracy_gain_measured=False,fully_verified_correction=False,anatomical_ground_truth=False))
    print('REPORT',len(obs),len(comparisons),flush=True)
if __name__=='__main__':main()
