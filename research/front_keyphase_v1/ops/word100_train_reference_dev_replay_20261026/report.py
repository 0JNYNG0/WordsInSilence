"""Research response and accessible report from saved, verified results only."""
import study as S
import numpy as np,json,html,sys,traceback
from collections import defaultdict,Counter
from pathlib import Path
STATUS=['대상 구간 밖','관절 관측 부족','이미지 재확인 부족','참고 인원 부족','참고 차이 범위 안','참고 차이 범위 밖']

def intervals(mask,t):
    starts=np.flatnonzero(mask&~np.r_[False,mask[:-1]]);ends=np.flatnonzero(mask&~np.r_[mask[1:],False]);return [[round(float(t[a])/1e6,3),round(float(t[b])/1e6,3)] for a,b in zip(starts,ends)]
def mean(a):
    good=np.isfinite(a);return float(a[good].mean()) if good.any() else None
def groups(z,ref):
    result=[]
    for h,side in enumerate(('left','right')):
        for mode in ('point','sequence'):
            for name,a,b in S.GROUPS:
                state=z[mode+'_status'][:,h,a:b];raw=z[mode+'_ok'][:,h,a:b];delta=z[mode+'_delta'][:,h,a:b];domain=z['domain'][:,h,a:b]
                counts={str(k):int((state==k).sum()) for k in range(1,6)};n=int(domain.sum());supported=counts['4']+counts['5'];rawmean=mean(delta if mode=='point' else delta[...,:3]);change=mean(delta[...,3:]) if mode=='sequence' else None
                result.append(dict(side=side,mode=mode,group=name,unit='degrees_2d' if name=='angle' else 'shoulder_width',domain_cells=n,raw_compared_cells=int(raw.sum()),state_counts=counts,supported_cells=supported,
                    raw_level_difference=rawmean,raw_change_difference=change,
                    support_fraction=supported/n if n else None,
                    message='비교 근거 부족' if not supported else '일부 구간만 비교 가능' if supported<n else '관측·참고 근거 확보',
                    out_of_band_is_error=False,within_band_is_correct=False,
                    missing_intervals=intervals((state==1).any(1),ref['timestamp_us']),image_uncertain_intervals=intervals((state==2).any(1),ref['timestamp_us']),reference_shortage_intervals=intervals((state==3).any(1),ref['timestamp_us']),difference_intervals=intervals((state==5).any(1),ref['timestamp_us']),
                    timeline_counts=[[int((row==k).sum()) for k in range(1,6)] for row in state]))
    return result

def build():
    f=S.read(S.U/'FROZEN.json');done=S.read(S.U/'DONE.json');v=S.read(S.U/'VERIFIED.json');controls=S.read(S.U/'CONTROLS_DONE.json')
    assert v['tests']['verified_pairs']==2697;S.check(done['pins']);S.check(controls['pins']);S.check(f['protected'])
    by=defaultdict(list)
    for p in done['rows']:by[p['query']].append(p)
    meta={r['clip_id']:r for r in f['train']+f['dev']};prior={r['query_id']:r for r in f['recognition']};samples={};rows=[];view=[];tot=Counter();changes=[]
    for qid,pp in sorted(by.items()):
        refs=[];q=meta[qid]
        for p in pp:
            rid=p['reference']
            if rid not in samples:samples[rid]=S.load(meta[rid]['output'])
            r=samples[rid];z=S.load(p['output']);gg=groups(z,r);refs.append(dict(reference_id=rid,recorded_id=meta[rid]['signer_id'],groups=gg,reference_seconds=(r['timestamp_us']/1e6).tolist(),pair_path=p['output'],pair_sha256=done['pins'][p['output']]))
            for g in gg:
                prefix=g['mode']+'_'+g['group'];tot[prefix+'_domain']+=g['domain_cells'];tot[prefix+'_supported']+=g['supported_cells']
                for k,val in g['state_counts'].items():tot[prefix+'_state'+k]+=val
        receipt=dict(query_id=qid,recorded_id=q['signer_id'],practice_target=dict(word_code=q['word_code'],label=q['label']),predictions=prior[qid]['predictions'],references=refs,whole_gesture_correct=None,validated_correction=None,skill_score=None,
            evidence_policy='Full reference core xleft/right xchannels is an audit grid, not a declaration that bothhands are linguistically required. Missing never shrinks it or becomes0difference. Position raw observation and same-MediaPipe image-corroborated finger angles are separate.',
            difference_policy='Empirical TRAIN residual range, not correctness probability or permissible sign error. Compare references separately; no automatic correction direction. Alignment uncertainty is not solved by within-band output.')
        op=S.U/'queries'/(qid+'.json');S.write(op,receipt);rows.append(dict(query_id=qid,receipt=str(op),sha256=S.sha(op)))
        compact=[]
        for ref in refs:
            compact.append(dict(id=ref['reference_id'],person=ref['recorded_id'],groups=[{k:g[k] for k in ('side','mode','group','domain_cells','raw_compared_cells','state_counts','supported_cells','raw_level_difference','raw_change_difference','missing_intervals','image_uncertain_intervals','reference_shortage_intervals','difference_intervals')} for g in ref['groups']]))
        view.append(dict(id=qid,label=q['label'],word=q['word_code'],person=q['signer_id'],predictions=prior[qid]['predictions'],refs=compact))
    control_counts=defaultdict(Counter);missed=defaultdict(list)
    for r in controls['rows']:
        control_counts[r['kind']].update(r['counts'])
        if r['no_target_new_signal']:missed[r['kind']].append(r['query'])
    summary=dict(counts=dict(tot),controls={k:dict(v) for k,v in control_counts.items()},no_new_target_signal=dict(missed),fixed_mask_controls=controls['fixed_path_verified'],pair_count=2697,query_count=300,train_references=899,
        conclusion='Public TRAIN-to-DEV replay and evidence separation verified. Automatic correction release remains withheld: within band is not sign correctness, natural timing and short omissions unresolved.',
        new_training=0,new_classifier=0,new_detector=0,operating_change=False)
    S.write(S.U/'SUMMARY.json',summary)
    manifest=dict(schema='train_reference_public_replay_v1',rows=rows,allow_personal_input=False,models=['C','T3F','E3'],default_model='C',pins={str(p):S.sha(p) for p in [S.U/'FROZEN.json',S.U/'VERIFIED.json',S.U/'CONTROLS_DONE.json',S.U/'SUMMARY.json',S.U/'adapter.py']})
    S.write(S.U/'MANIFEST.json',manifest);S.write(S.U/'VIEW_DATA.json',view)
    print('RESEARCH_RECEIPTS300_REFERENCES2697',flush=True)

def pct(a,b):return f'{100*a/b:.2f}%' if b else '—'
def report():
    summary=S.read(S.U/'SUMMARY.json');view=S.read(S.U/'VIEW_DATA.json');tot=summary['counts'];done=S.read(S.U/'DONE.json');times=np.array([r['seconds'] for r in done['rows']]);table=[]
    labels={'point_angle':'한 순간 손가락 각도','sequence_angle':'앞·중간·뒤 손가락 변화','point_position':'한 순간 손·팔 위치','sequence_position':'앞·중간·뒤 손·팔 변화'}
    for key,label in labels.items():
        n=tot[key+'_domain'];table.append([label,pct(tot[key+'_supported'],n),pct(tot[key+'_state1'],n),pct(tot[key+'_state2'],n),pct(tot[key+'_state3'],n),pct(tot[key+'_state5'],n)])
    text='공개 검증300개 × 학습 참고899개의 DTW 근거 분리 검증\n\n이번 목표 단계 완료: 공개 검증 영상300개를 단어별8~9개의 학습 참고와 각각 비교해2697개 결과와900개 모델별 재생 응답을 준비했다. 기준 범위에는 검증 촬영자11~13을 넣지 않았다. C·T3-F·E3, 운영 웹, 사용자 촬영은 변경하지 않았다.\n\n'
    text+='비교 근거가 충분한 비율(정답률 아님):\n'+'\n'.join(f'{r[0]}: {r[1]} / 원래 관측 부족 {r[2]} / 이미지 재확인 부족 {r[3]} / 참고5명 미달 {r[4]} / 전체 비교칸 중 범위 밖 {r[5]}' for r in table)
    text+='\n\n비율의 분모는 참고 핵심시간×손×항목이며 같은 영상의 여러 참고 비교가 중복된다. 이미지 근거는 같은 MediaPipe의 전체/확대 재확인 일치로, 사람의 정답 관절이 아니다. 범위 안=올바른 수어, 범위 밖=틀린 수어가 아니다. 정상 한손/손내림의 미관측도 수정 지시로 바꾸지 않았다.\n'
    text+='\n검산: 899개 TRAIN전용 참고 범위,2697개 저장 정렬·차이·근거·상태를 별도 산술로 재구성. 단일각도 누락100건과 손전체 누락200건에서 이웃각도·반대손·팔·비교범위 보존. 별도10단어×7종류70개에서는 실제 시간 정렬을 다시 계산해 손누락·짧은삭제·속도변형의 남은 문제를 확인했다. 같은 Codex의 별도 산술검산이며 독립 전문가 정답 검증은 아니다.\n'
    for kind,c in summary['controls'].items():
        text+=f"\n{kind}: 정상 원본 범위 안→새 범위 밖 연속위치 {c['sequence_position_new_excess']}/{c['sequence_position_domain']}칸, 원래 비교가능→근거부족 {c['sequence_position_supported_to_unknown']}칸, 표적 시간대 새 신호 없는 사례 {len(summary['no_new_target_signal'].get(kind,[]))}/10."
    text+='\n\n추가 확인: missing_L·gap3_L에서 새 신호가 없던2영상(WORD2713/WORD2867,ID11)은 원래 해당 왼손 각도·위치 관측이0이었다. 제거할 새 정보가 없던 시험이므로 실패나 새 누락 검출 성공으로 세지 않는다. 0.75배 시험에서 새 신호가 없는 것은 기대한 불변 결과다.\n'
    text+='\n한계: 위 속도는 저장된 관측을 재표본화한 시험이고 실제 빠른RGB 검출 시험은 아니다. 3시점 삭제는 의미상 필수 동작 정답이 아니다. 다른 사람 간 정렬의 실제 정답 시점은 없다. 따라서 자연속도·짧은동작 누락 검출, 가려진 손가락 정답과 자동 교정 방향의 신뢰성은 아직 미검증이다.\n'
    text+=f'\n처리시간: 저장관절의 한 참고 DTW 경로 계산 중앙값 {np.median(times)*1000:.2f}ms,95백분위 {np.percentile(times,95)*1000:.2f}ms. 검출/웹/이미지 재확인 제외이므로 전체 서비스 속도가 아니다.\n'
    text+='\n적용 상태: 별도 공개자료 연구 재생만 연결. 실제 웹캠·운영 DTW에는 미적용. 이번 결과를 TCN 인식률 향상이나 검증된 자동 교정 완성으로 보고하지 않는다.\n\n다음: 다른 수행자 정렬에서 생긴 정상 속도 경고와 짧은구간 누락을 원본 시점대응·표본손실로 분리한다. 우선 손·팔 위치에서 근거가 확실한 비교를 만들고, 손가락 보류 기준은 완화하지 않는다. 새 학습보다 이 비교 경로 검증이 우선이다.\n'
    (S.U/'RESULT_KO.txt').write_text(text,encoding='utf8')
    head=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in table)
    controls=''.join(f'<tr><td>{html.escape(k)}</td><td>{pct(c["sequence_position_new_excess"],c["sequence_position_domain"])}</td><td>{c["sequence_position_supported_to_unknown"]:,}</td><td>{len(summary["no_new_target_signal"].get(k,[]))}/10</td></tr>' for k,c in summary['controls'].items())
    data=json.dumps(view,ensure_ascii=False).replace('<','\\u003c');notes=html.escape(text)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DTW 공개 검증 · 참고와 근거 구분</title><style>
*{box-sizing:border-box}body{font:16px/1.65 system-ui,sans-serif;background:#f4f6fa;color:#182438;margin:0}main{max-width:1200px;margin:auto;padding:36px 24px}h1{font-size:32px;line-height:1.3}h2{font-size:23px;margin-top:36px}.intro,.box{background:white;border:1px solid #dce3ed;border-radius:16px;padding:24px;margin:18px 0}.tag{color:#205c9e;font-weight:700}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.stats strong{font-size:32px;display:block}.warn{border-left:5px solid #b36a17;padding:16px;background:#fff5df}.bar{display:flex;height:15px;border-radius:4px;overflow:hidden;min-width:150px}.s1{background:#9aa4b0}.s2{background:#cfb3e9}.s3{background:#e5c87c}.s4{background:#75b2df}.s5{background:#e69b79}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:12px 8px;border-bottom:1px solid #e4e8ed;vertical-align:top}th{background:#f6f8fb}select,input{font:inherit;padding:9px;border:1px solid #b9c7d9;border-radius:8px;max-width:100%}.scroll{overflow:auto}details{margin:12px 0}summary{cursor:pointer;font-weight:650}small{color:#526378}pre{white-space:pre-wrap;font:14px/1.7 system-ui}.key{display:flex;gap:18px;flex-wrap:wrap}.key span:before{content:'■';margin-right:6px}.k1{color:#7d8997}.k2{color:#a17ac5}.k3{color:#b69843}.k4{color:#4c92c5}.k5{color:#ca805c}@media(max-width:650px){.stats{grid-template-columns:1fr}main{padding:20px 12px}}
</style><main><p class="tag">공개 자료 연구 후보 · 실제 웹캠 미적용</p><h1>차이가 작다는 것과<br>잘 확인했다는 것을 구분했습니다.</h1><div class="intro">검증 영상은 기준을 만드는 데 넣지 않았습니다. 단어별 여러 참고를 각각 유지하고, 손이 보이지 않거나 참고가 부족한 구간을 ‘잘 맞음’으로 바꾸지 않습니다.</div><div class="stats"><div class="box"><strong>300개</strong>공개 개발 검증 영상</div><div class="box"><strong>899개</strong>학습 참고 영상</div><div class="box"><strong>2,697쌍</strong>개별 참고 비교</div></div><p class="warn">이 결과는 수어 정답률이나 교정 정확도가 아닙니다. C·T3-F·E3와 실제 테스트 웹은 그대로입니다.</p><h2>어디까지 비교 근거가 충분했나</h2><div class="box scroll"><table><thead><tr><th>항목</th><th>근거 충분</th><th>관측 부족</th><th>이미지 재확인 부족</th><th>참고 인원 부족</th><th>전체 중 범위 밖</th></tr></thead><tbody>__TABLE__</tbody></table><small>근거 충분 = 참고 차이 범위 안 + 범위 밖. 범위 밖을 틀린 동작으로 판정하지 않습니다. 분모는 참고 시간·손·항목의 비교칸입니다.</small></div><h2>공개 영상별 결과 보기</h2><div class="box"><label>표현·촬영 ID <select id="query"></select></label><p id="prediction"></p><div class="key"><span class="k1">관측 부족</span><span class="k2">이미지 재확인 부족</span><span class="k3">참고 부족</span><span class="k4">참고 범위 안</span><span class="k5">참고 범위 밖</span></div><p><small>각 막대는 서로 다른 참고 사람과의 비교입니다. 가장 좋은 참고만 고르지 않습니다. 파란색도 ‘올바른 수행’ 인증은 아닙니다.</small></p><div id="references"></div></div><h2>정상 속도와 누락 통제에서 남은 문제</h2><div class="box scroll"><table><thead><tr><th>입력 변형</th><th>연속 위치 새 범위 밖</th><th>비교가능→근거부족 칸</th><th>표적 새 신호 없음</th></tr></thead><tbody>__CONTROLS__</tbody></table><small>10개 공개 표현의 저장 관측에 동일한 변형을 적용했습니다. 속도 변화로 생긴 차이와 근거 감소를 개선으로 세지 않습니다. 짧은 삭제는 의미상 필수 동작의 정답 주석이 아닙니다.</small></div><details class="box"><summary>검증 방법과 남은 한계 전문</summary><pre>__NOTES__</pre></details></main><script id="data" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.querySelector('#data').textContent),select=document.querySelector('#query');const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.label+' · ID'+r.person+' · '+r.word;select.append(o)});function render(){const r=rows[Number(select.value)||0];document.querySelector('#prediction').textContent='기존 저장 분류: '+Object.entries(r.predictions).map(([k,p])=>k+' → '+p.predicted_label).join(' / ')+' (새 추론 없음)';document.querySelector('#references').innerHTML=r.refs.map(ref=>'<details><summary>참고 촬영 ID '+ref.person+'</summary><div class="scroll"><table><tr><th>손·항목</th><th>근거 상태 분포</th><th>근거 충분</th><th>원래 관측 차이</th></tr>'+ref.groups.map(g=>{const n=g.domain_cells;return '<tr><td>'+(g.side==='left'?'왼손':'오른손')+' · '+(g.mode==='point'?'순간':'앞·중간·뒤')+' · '+(g.group==='angle'?'각도':'손·팔 위치')+'</td><td><div class="bar">'+[1,2,3,4,5].map(k=>'<span class="s'+k+'" title="'+g.state_counts[k]+'칸" style="width:'+(n?100*g.state_counts[k]/n:0)+'%"></span>').join('')+'</div></td><td>'+(n?(100*g.supported_cells/n).toFixed(1)+'%':'—')+'</td><td>'+(g.raw_level_difference==null?'비교 불가':g.raw_level_difference.toFixed(3)+(g.group==='angle'?'°':' 어깨폭'))+'<br><small>관측된 부분의 차이 · 정확도 아님</small></td></tr>'}).join('')+'</table></div></details>').join('')}select.addEventListener('change',render);render();
</script><script>
function showTimes(){const r=rows[Number(select.value)||0];[...document.querySelectorAll('#references>details')].forEach((node,i)=>{const detail=document.createElement('details');const title=document.createElement('summary');title.textContent='근거가 부족하거나 차이가 관측된 시간대';detail.append(title);r.refs[i].groups.forEach(g=>{const box=document.createElement('p');box.style.fontSize='13px';const times=a=>a.length?a.map(x=>x[0].toFixed(2)+'–'+x[1].toFixed(2)+'초').join(', '):'없음';box.textContent=(g.side==='left'?'왼손':'오른손')+' / '+(g.mode==='point'?'순간':'연속')+' / '+(g.group==='angle'?'각도':'위치')+' — 관측 부족: '+times(g.missing_intervals)+'; 이미지 재확인 부족: '+times(g.image_uncertain_intervals)+'; 참고 부족: '+times(g.reference_shortage_intervals)+'; 관측 차이 범위 밖: '+times(g.difference_intervals);detail.append(box)});const note=document.createElement('small');note.textContent='시간은 해당 참고 영상 기준입니다. 일부 각도 또는 위치 항목의 문제이며 손 전체가 틀렸다는 뜻은 아닙니다.';detail.prepend(note);node.append(detail)})}select.addEventListener('change',showTimes);showTimes();
</script></html>'''
    page=page.replace('__TABLE__',head).replace('__CONTROLS__',controls).replace('__NOTES__',notes).replace('__DATA__',data)
    page=page.replace('10개 공개 표현의 저장 관측에', '왼손 누락 시험에서 새 신호가 없던 2개는 원래 그 손의 관측이 없던 영상입니다. 제거할 새 정보가 없었으므로 실패나 검출 성공으로 세지 않습니다. 느리게 한 시험의 새 신호 없음은 기대한 결과입니다. 10개 공개 표현의 저장 관측에')
    (S.U/'RESULT_KO.html').write_text(page,encoding='utf8')
    manifest_name='REPORT_MANIFEST_FINAL.json' if (S.U/'REPORT_MANIFEST.json').exists() else 'REPORT_MANIFEST.json'
    S.write(S.U/manifest_name,dict(at=S.now(),files={str(S.U/x):S.sha(S.U/x) for x in ['RESULT_KO.html','RESULT_KO.txt','VIEW_DATA.json','SUMMARY.json']},phase='BUILT_PENDING_RENDER_CHECK'))
    print(text,flush=True)

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        p=S.U/('FIRST_ERROR_report_'+sys.argv[1]+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
