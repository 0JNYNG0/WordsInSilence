from pathlib import Path
from collections import Counter
import json,gzip,html
import numpy as np
U=Path(__file__).resolve().parent;P=U.parent/'word100_repeat_alignment_20261020'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
write=lambda n,x:(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
esc=lambda x:html.escape(str(x))
s=read(U/'SUMMARY.json');rows=read(U/'ROWS.json');stress=read(U/'STRESS.json');lower=read(U/'LOWERING.json')
old8=read(P/'DIAGNOSTICS.json')['previous8escaped']
keys={(r['clip_id'],r['side'],r['key'],r['condition']) for r in old8}
eight=[r for r in stress if (r['clip_id'],r['side'],r['key'],r['condition']) in keys]
assert len(eight)==8
changed_lower=[r for r in lower if r['key']=='location' and (r['old_strong_s']>1e-8 or r['new_strong_s']>1e-8)]
deletion=[r for r in rows if r['scenario'].startswith('delete')]
counts=Counter(m['status'] for r in deletion for m in r['new_metrics'] if m['side']==r['meta']['side'])
original=[r for r in rows if r['view']=='sensor' and r['scenario']=='original']
diag=dict(previous8=eight,previous8_strong=sum(r['new_strong_s']>1e-8 for r in eight),
    previous8_possible=sum(r['new_possible_s']>1e-8 for r in eight),lowering_cases=changed_lower,
    deletion_rows=len(deletion),deletion_statuses=dict(counts),
    deletion_count_abstained=sum(r['decision']['count'] is None for r in deletion),
    original_count_changes=[dict(clip_id=r['clip_id'],label=r['label'],old_count=r['old_count'],decision=r['decision']) for r in original if r['old_count']!=r['decision']['count']])
diag['reason_counts']=dict(Counter(r['decision']['reason'] for r in rows))
diag['incorrect_count_controls']=[]
for r in rows:
    expected=1 if r['scenario']=='original' else int(r['scenario'][-1]) if r['scenario'].startswith('repeat') else None
    if expected and r['decision']['count'] is not None and r['decision']['count']!=expected:
        with gzip.open(P/r['source_path'],'rt',encoding='utf-8') as h:source=json.load(h)
        diag['incorrect_count_controls'].append(dict(clip_id=r['clip_id'],label=r['label'],view=r['view'],scenario=r['scenario'],decision=r['decision'],
            reference_costs=[[dict(cycles=e['cycles'],score=e['score'],available=e['available']) for e in bank] for bank in source['result']['bank']]))
diag['boundary_gate_incremental_rejections']=diag['reason_counts'].get('return_boundary_not_jointly_supported',0)
write('DIAGNOSTICS.json',diag)
loc=s['lowering']['location'];sensor=[s['groups']['sensor_'+name] for name in ('original','repeat2','repeat3')]
labels=['원래 영상','같은 영상 2회 연결','같은 영상 3회 연결']
txt=['여러 참고의 반복 경계 합의 — 공개 자료 검증',
     '학습·웹 적용 없이 별도 후보를 구현했다. 비용이 낮다는 이유만으로 반복 횟수를 확정하지 않고 참고 사람들의 동의와 실제 팔 위치의 되돌아옴을 함께 검사한다.',
     '여기서 횟수는 참고 영상 전체를 몇 번 순회해 맞추는지이다. 수어에서 반드시 해야 할 반복 횟수나 단어 정답 여부가 아니다.',
     '공개100단어, TRAIN899(01~09) 참고 / DEV300(11~13) 평가. 같은DEV의 후속 진단이며 새로운 최종시험이 아니다.',
     '',
     '촬영 범위에 맞춰 손을 가린 sensor 경로, 각300개:']
for label,g in zip(labels,sensor):
    txt.append(f"{label}: 기존 횟수 일치 {g['old_matching']}/300 → 새 확정 일치 {g['confirmed_matching']}, 확정 불일치 {g['confirmed_different']}, 보류 {g['abstained']}.")
txt+=['보류를 성공으로 세지 않는다. 합성 반복은 원래 영상을 그대로 연결한 대조이며 자연스러운 반복의 정답 검증은 아니다.',
      f"정상 손내림 추정 구간과 강한 관측 요구의 겹침(위치): {loc['old_strong_s']:.3f} → {loc['new_strong_s']:.3f}초. 불확실 범위까지 포함한 겹침: {loc['old_possible_s']:.3f} → {loc['new_possible_s']:.3f}초.",
      f"이전 짧은 동작 누락 8항목손조건: 강한 근거 {diag['previous8_strong']}/8, 불확실 범위 포함 {diag['previous8_possible']}/8. 서로 다른8영상이 아니다.",
      f"전체7092가림조건에서 가능 비교 범위 보존. 짧은peak10 실관측 제거 {s['stress']['peak10']['effective']}조건 중 불확실 포함 {s['stress']['peak10']['new_possible']}조건에서 추가 근거 부족 확인.",
      '같은5사람이 모든 경계를 지지해야 하며 다른 횟수를 지지하는 참고가 하나라도 있으면 보류한다. 손 누락·경로의 단순 감김·동일 비용은 되돌아옴의 증거가 아니다.',
      '보류 시 강한 범위는 가능한 정렬들의 교집합, 불확실 범위는 합집합이다. 확정된 경우에도 불확실 범위는 모든 가능한 정렬의 합집합으로 보존한다.',
      '따라서 애매한 범위가 늘어날 수 있다. 이를 사용자 동작 오류나 필수 동작으로 표시하면 안 된다. 관측한 부분이 비슷해도 전체 동작을 빠짐없이 수행했다는 뜻은 아니다.',
      '남은 확정 불일치2조건은 같은 실명 영상의2회/3회 연결이다. 9명 참고 모두1회 정렬 비용이 더 낮아 합의 규칙을 통과했다. 여러 참고가 같은 방향으로 틀릴 수 있음을 확인했다. 특정 단어 예외로 고치지 않았다.',
      '반복으로 합의된869대조는 실제 팔 귀환의 공동 근거도 통과했다. 이번 자료에서는 경계 확인 단계가 합의 단계에 더해 추가로 보류한 사례가0개였다. 따라서 잘못된 확정 감소는 이번 결과상 의견 불일치 보류의 효과이며, 경계 검사만의 추가 개선은 입증되지 않았다.',
      '시각·의미 한계: 원래 영상의 공식 단어시간 주석과 기존 전체화면 손상태 추정만 대조했다. 전문가가 확정한 반복 경계/필수 단계 정답은 없다.',
      '속도 검사는 시간표시 변경 불변성이다. 실제 빠른 촬영·새 웹캠의 검출성능을 검증한 것이 아니다. 한손 동작은 기존에 확인된3단어만 반대손 무영향을 검증했다.',
      '기존 C·T3-F6가중치와32보호파일 보존. 사용자·멘토촬영/보류14~18 미접근, 새학습·분류추론·운영 DTW/웹 변경 없음.',
      '다음: 이번 결과를 검토한 뒤, 1회에 모두 동의해도 다른 반복 가설의 경계 증거가 있는지 확인하고, 짧은 단계가 한 구간에 과도하게 압축되는지 대조하는 계획. 추가학습은 사용자 결과 확인 후.']
(U/'RESULT_KO.txt').write_text('\n'.join(txt)+'\n','utf-8')
def number(v):return f'{v*100:.2f}%'
table=''.join(f"<tr><td>{label}</td><td>{g['old_matching']}</td><td class='good'>{g['confirmed_matching']}</td><td class='bad'>{g['confirmed_different']}</td><td class='warn'>{g['abstained']}</td></tr>" for label,g in zip(labels,sensor))
bars=''.join(f"""<div class="barrow"><b>{label}</b><div class="bar"><i style="width:{g['confirmed_matching']/3}%"></i><i class="wrong" style="width:{g['confirmed_different']/3}%"></i><i class="unsure" style="width:{g['abstained']/3}%"></i></div></div>""" for label,g in zip(labels,sensor))
reasons={'unopposed_single_traversal':'한 번으로 의견 일치','reference_count_disagreement':'참고마다 횟수 판단이 다름','same_people_observed_return_boundaries':'횟수와 경계 근거 일치'}
resttable=''.join(f"<tr><td>{esc(r['label'])} / {'오른손' if r['side']=='R' else '왼손'}</td><td>{r['decision']['count'] if r['decision']['count'] is not None else '보류'}</td><td>{r['old_strong_s']:.3f}</td><td>{r['new_strong_s']:.3f}</td><td>{esc(reasons.get(r['decision']['reason'],'근거 부족'))}</td></tr>" for r in changed_lower)
shorttable=''.join(f"<tr><td>{mode}</td><td>{v['effective']}</td><td>{v['old_strong']} → {v['new_strong']}</td><td>{v['old_possible']} → {v['new_possible']}</td></tr>" for mode,v in s['stress'].items())
localization=''.join(f"<tr><td>{label}</td><td>{number(g['localization']['old']['core']['recall'])} → {number(g['localization']['new']['core']['recall'])}</td><td>{number(g['localization']['old']['core']['precision'])} → {number(g['localization']['new']['core']['precision'])}</td><td>{number(g['localization']['new']['possible_core']['recall'])}</td><td>{number(g['localization']['new']['possible_core']['precision'])}</td></tr>" for label,g in zip(labels,sensor))
legacy=' / '.join(f"{label}: {s['groups']['legacy_'+name]['confirmed_matching']} 확정일치, {s['groups']['legacy_'+name]['confirmed_different']} 불일치, {s['groups']['legacy_'+name]['abstained']} 보류" for label,name in zip(labels,('original','repeat2','repeat3')))
timelines=''
for r in original:
    if r['old_count']==1:continue
    with gzip.open(U/r['evidence_path'],'rt',encoding='utf-8') as h:z=json.load(h)
    with gzip.open(P/r['source_path'],'rt',encoding='utf-8') as h:old=json.load(h)
    p=z['projection'];oldp=old['result']['multi'];n=len(p['core'])
    svg=['<svg viewBox="0 0 900 220" role="img" aria-label="기존과 새 후보의 비교 범위">']
    for j,(label,mask,color) in enumerate([('기존 강한 범위',oldp['expected']['R']['location'],'#e17e55'),('새 강한 범위',p['expected']['R']['location'],'#238987'),('새 불확실 포함 범위',p['possible_expected']['R']['location'],'#b5c3cf')]):
        y=15+j*55;svg.append(f'<text x="0" y="{y+19}">{label}</text><rect x="210" y="{y}" width="680" height="25" fill="#eef1f5"/>')
        for k,v in enumerate(mask):
            if v:svg.append(f'<rect x="{210+680*k/n:.3f}" y="{y}" width="{680/n+.1:.3f}" height="25" fill="{color}"/>')
    svg.append('<text x="210" y="205">영상 시작</text><text x="800" y="205">영상 끝</text></svg>')
    timelines+=f"<h3>{esc(r['label'])} — 기존 {r['old_count']}회 → {r['decision']['count'] or '보류'}</h3><p>오른손 위치의 비교 구간. 회색 범위는 동작 오류가 아니라 판단을 남겨 둔 범위입니다.</p>{''.join(svg)}"
v=read(U/'VERIFICATION.json') if (U/'VERIFICATION.json').exists() else None
verif=f"독립 구현 검산 PASS: {v['rows']:,}대조 · {v['reference_evidence']:,}참고 판단 · {v['hash_pins']:,}해시." if v else "독립 구현 검산 진행 중. 완료 전 결과 확정 금지."
htmltext=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>반복 경계 합의 검증</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#edf1f5;color:#172a3d;font-family:'Malgun Gothic',sans-serif;line-height:1.65}}main{{max-width:1160px;margin:auto;padding:42px 24px}}section{{background:white;border:1px solid #d9e1e9;border-radius:18px;padding:28px;margin:22px 0}}h1{{font-size:34px;line-height:1.35;margin:12px 0}}h2{{font-size:24px}}h3{{font-size:18px}}p{{margin:10px 0}}.eyebrow{{color:#267674;font-weight:bold}}.tag{{display:inline-block;background:#dcecea;padding:5px 14px;border-radius:20px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{padding:20px;background:#f3f6f9;border-radius:12px}}.big{{font-size:30px;font-weight:bold}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #dae1e8}}th{{background:#f0f4f7}}.scroll{{overflow-x:auto}}.good{{color:#087f79;font-weight:bold}}.bad{{color:#a63a30;font-weight:bold}}.warn{{color:#956311;font-weight:bold}}.note{{background:#fff4de;padding:16px;border-left:4px solid #c98d29}}.barrow{{margin:18px 0}}.bar{{display:flex;height:28px;background:#e5ebef;border-radius:5px;overflow:hidden}}.bar i{{background:#238987}}.bar .wrong{{background:#cf635c}}.bar .unsure{{background:#e0bd78}}svg{{width:100%;font-size:16px}}li{{margin:10px 0}}a{{color:#075d80}}footer{{font-size:14px;color:#526476}}@media(max-width:650px){{main{{padding:18px 12px}}section{{padding:20px 15px}}h1{{font-size:26px}}.cards{{grid-template-columns:1fr}}table{{min-width:600px}}}}</style>
<main><span class="eyebrow">WordsInSilence · 공개 자료 검증 · 2026.10.20</span><h1>반복 횟수를 억지로 정하지 않기</h1><p>여러 참고 사람이 같은 경계와 실제 팔의 되돌아옴을 함께 지지하는지 확인했습니다.</p><span class="tag">별도 연구 후보 · 학습 및 웹 적용 없음</span>
<section><h2>어떻게 바뀌었나</h2><div class="cards"><div class="card"><b>이전</b><p>1·2·3회로 맞춰 보고 계산 비용이 가장 낮은 횟수를 선택</p></div><div class="card"><b>추가 확인</b><p>참고 사람들의 횟수 판단이 일치하고, 같은 5명 이상이 모든 반복 경계에서 관측된 귀환 자세를 지지</p></div><div class="card"><b>엇갈리면</b><p>‘정렬 불확실’로 보류. 짧은 동작이 사라지지 않도록 가능한 비교 범위를 남김</p></div></div>
<p class="note">횟수는 ‘참고 영상 전체를 몇 번 맞추는가’입니다. 수어에서 반복을 몇 번 해야 정답인지 판정하는 기능은 아닙니다.</p></section>
<section><h2>공개 영상 비교 결과</h2><p>공개100단어 · 참고 TRAIN899개 / 평가 DEV300개. 아래는 손이 화면 밖으로 나가는 조건을 반영한 sensor 경로, 각300개입니다.</p><div class="scroll"><table><tr><th>대조 조건</th><th>기존 횟수 일치</th><th>새 확정 일치</th><th>새 확정 불일치</th><th>판단 보류</th></tr>{table}</table></div>{bars}<p>초록: 확정 일치 / 빨강: 확정 불일치 / 노랑: 보류. 보류를 성공으로 세지 않았습니다.</p><p>2·3회는 원래 영상 전체를 그대로 이어 붙인 대조입니다. 자연스러운 반복에 대한 전문가 정답은 없습니다.</p><details><summary>원래 전체 관측 경로도 보기</summary><p>{legacy}</p></details></section>
<section><h2>정상적인 손내림은 보존됐나</h2><div class="cards"><div class="card">강한 비교 요구와 손내림 겹침<div class="big">{loc['old_strong_s']:.3f} → {loc['new_strong_s']:.3f}초</div></div><div class="card">불확실 범위 포함 겹침<div class="big">{loc['old_possible_s']:.2f} → {loc['new_possible_s']:.2f}초</div></div><div class="card">점검한 손내림·미관측 시간<div class="big">{loc['rest_missing_s']:.2f}초</div></div></div><p>전체화면 관측으로 추정한 내려놓은 손 상태와 비교했습니다. 이 값은 수어 수행의 오류율이 아닙니다. 불확실 범위의 겹침을 ‘손을 잘못 내렸다’는 경고로 바꾸면 안 됩니다.</p><div class="scroll"><table><tr><th>영상 / 손</th><th>새 횟수</th><th>기존 강한 겹침(초)</th><th>새 겹침(초)</th><th>결정 근거 코드</th></tr>{resttable}</table></div>{timelines}</section>
<section><h2>짧은 동작을 놓치지 않도록</h2><p>같은 관절 경로를 유지한 채 손의 관측 근거만 가린 7,092조건에서 가능한 비교 범위가 줄지 않는지 점검했습니다.</p><div class="scroll"><table><tr><th>가림 조건</th><th>실제 근거가 제거된 조건</th><th>강한 근거 발견 기존→새</th><th>불확실 포함 발견 기존→새</th></tr>{shorttable}</table></div><p>whole: 손 전체 시간 / center10: 가운데 짧은 구간 / peak10·20: 기존에 정한 움직임 주변 구간. 숫자는 영상 수가 아닌 손×항목×조건 수입니다.</p><p><b>이전에 놓쳤던8조건: 강한 근거 {diag['previous8_strong']}개, 불확실 범위 포함 {diag['previous8_possible']}개.</b> 영상 프레임 자체를 지운591대조에서는 여전히 관측한 부분만 충분해 보일 수 있습니다. 따라서 ‘근거가 남아 있음’을 ‘동작을 빠짐없이 수행함’으로 판정하지 않습니다.</p></section>
<section><h2>보류에 따른 비용도 함께 확인</h2><div class="scroll"><table><tr><th>조건</th><th>강한 범위 포착률 기존→새</th><th>강한 범위 정밀도 기존→새</th><th>불확실 포함 포착률</th><th>불확실 포함 정밀도</th></tr>{localization}</table></div><p>각300영상의 시간 주석 중첩 중앙값입니다. 포착률은 주석 동작 시간을 얼마나 포함했는지, 정밀도는 포함한 범위가 얼마나 주석 안에 있었는지입니다. TCN 단어 인식률이 아닙니다. 빈 강한 범위는 포착률0, 정밀도는 계산 대상에서 제외했습니다.</p><p class="note">강한 판단을 줄이면 잘못 확정하는 문제는 줄지만, 보류와 넓은 불확실 범위가 늘어납니다. 이 결과만으로 실사용 피드백이 완성됐다고 볼 수 없습니다.</p></section>
<section><h2>아직 남아 있는 문제</h2><p><b>같은 방향으로 함께 틀릴 수 있습니다.</b> 남은 확정 불일치2조건은 같은 ‘실명’ 영상의2회·3회 연결입니다. 참고9명 모두1회 정렬 비용이 더 낮아 합의를 통과했습니다. 1회 선호도 다른 반복 가설의 경계 증거와 대조할 필요가 있습니다.</p><p><b>경계 검사 자체의 추가 개선은 아직 입증되지 않았습니다.</b> 반복 합의869대조는 공동 귀환 근거도 모두 통과했습니다. 이번 공개 대조에서 잘못된 확정 감소는 의견 불일치 보류에서 나왔고, 경계 검사가 추가로 보류한 사례는0개였습니다.</p><p>특정 단어용 예외를 추가하거나 평가 결과에 맞춰 규칙을 다시 조정하지 않았습니다.</p></section>
<section><h2>확인한 범위와 다음 단계</h2><ul><li>같은5사람의 경계 합의, 근거 충돌 시 보류, 관측 누락을 귀환으로 오해하지 않는 자체검사12개 통과.</li><li>한손3단어18경로의 반대손 누락 무영향 유지. 나머지 미확정 손 역할을 임의로 확정하지 않았습니다.</li><li>속도 검사는 시간표시를 바꾼 대조입니다. 실제 빠른 동작·웹캠 센서 변화는 별도입니다.</li><li>{verif} 같은 Codex의 별도 구현 검산이며 별도 전문가 검토는 아닙니다.</li><li>참고와 평가 사람은 분리됐지만, 같은 개발 평가 자료를 반복 분석했습니다. 새로운 최종 검증 결과는 아닙니다.</li><li>다음은1회에 모두 동의해도 다른 반복 가설의 경계 근거가 있는지, 짧은 단계가 과도하게 압축되는지 대조하는 계획입니다. 추가 학습은 결과 확인 뒤 진행합니다.</li></ul></section>
<footer>기존 C·T3-F6가중치 및32보호파일 유지 · 사용자/멘토 촬영 미접근 · 추가학습/운영 DTW/웹 변경 없음<br><a href="RESULT_KO.txt">텍스트 보고</a> · <a href="SUMMARY.json">집계 근거</a> · <a href="VERIFICATION.json">독립 검산</a> · <a href="FROZEN.json">고정 조건</a></footer></main></html>"""
(U/'RESULT_KO.html').write_text(htmltext.replace('결정 근거 코드','판단 근거'),'utf-8')
write('DECISION.json',dict(status='OFFLINE_VERIFIED_NOT_DEPLOYED' if v else 'PENDING_INDEPENDENT_VERIFICATION',
    training=0,private_data_read=0,production_applied=False,
    conclusion='Reference consensus and count abstention implemented. Report abstention coverage, boundary evidence and normal-lowering/short-stage limits; no semantic repetition guarantee.'))
print('REPORT',len(htmltext))
