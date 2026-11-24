"""Summarize saved development scores and measured latency; no model inference."""
from pathlib import Path
import datetime,html,json,statistics,hashlib
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def avg(xs):return float(statistics.mean(xs))
def pc(x):return f'{x*100:.2f}'
def f3(x):return f'{x:.3f}'
def interval(xs):return [float(min(xs)),float(max(xs))]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
labels={'v3':'MobileNetV3-Small','repvit':'RepViT-M1.0','v4hybrid':'MobileNetV4 Hybrid-Medium'}
def main():
    assert read(U/'TRAINING_COMPLETE.json')['runs']==9
    p=read(U/'FROZEN_PROTOCOL.json');names=p['models'];seeds=p['seeds']
    manifest=read(U/'TRAINING_MANIFEST.json');fit_classes={r['target'] for r in manifest['rows'] if r['role']=='fit'};dev_classes={r['target'] for r in manifest['rows'] if r['role']=='validation'}
    assert len(fit_classes)==45 and len(dev_classes)==30
    audit=read(U/'verification/SAVED9_INDEPENDENT.json')
    assert audit['status']=='PASS' and audit['fixed_run_count']==9 and audit['protocol_sha256']==sha(U/'FROZEN_PROTOCOL.json')
    lat=read(U/'LATENCY_RESULTS.json')
    assert lat['status']=='COMPLETE' and lat['protocol_sha256']==sha(U/'CPU_LATENCY_PROTOCOL.json')
    assert len(lat['models'])==3 and {r['name'] for r in lat['models']}==set(names)
    lm={r['name']:r for r in lat['models']}
    for name in names:
        assert lm[name]['checkpoint_sha256']==sha(U/'models'/f'{name}_28101.pt')
    models=[]
    base={s:np.load(U/'models'/f'v3_{s}_dev.npz',allow_pickle=False) for s in seeds}
    for name in names:
        rs=[read(U/'models'/f'{name}_{s}.json') for s in seeds]
        score=[r['balanced_selection_macro_f1'] for r in rs];acc=[r['dev']['accuracy'] for r in rs]
        delta=[]
        for seed in seeds:
            z=np.load(U/'models'/f'{name}_{seed}_dev.npz',allow_pickle=False);b=base[seed]
            assert np.array_equal(z['ids'],b['ids']) and np.array_equal(z['targets'],b['targets'])
            y=z['targets'];bp=b['logits'].argmax(1);npred=z['logits'].argmax(1)
            delta.append({'seed':seed,'improved':int(((bp!=y)&(npred==y)).sum()),'regressed':int(((bp==y)&(npred!=y)).sum()),'both_wrong':int(((bp!=y)&(npred!=y)).sum())})
        people={s:{'n':rs[0]['per_signer'][s]['n'],'accuracy_mean':avg([r['per_signer'][s]['accuracy'] for r in rs]),'macro_f1_mean':avg([r['per_signer'][s]['macro_f1_supported'] for r in rs])} for s in rs[0]['per_signer']}
        models.append({'name':name,'label':labels[name],'accuracy_mean':avg(acc),'accuracy_range':interval(acc),'balanced_macro_f1_mean':avg(score),'balanced_macro_f1_range':interval(score),'per_signer':people,'paired_vs_v3':delta,'training_seconds':sum(r['seconds'] for r in rs),'total_parameters':p['specs'][name]['total_parameters'],'trainable_parameters':p['specs'][name]['trainable_parameters'],'latency':lm[name]})
    best=max(r['balanced_macro_f1_mean'] for r in models)
    pool=[r for r in models if best-r['balanced_macro_f1_mean']<=.01+1e-12]
    chosen=min(pool,key=lambda r:r['latency']['single_full_model']['p95_ms'])
    selection={'selected_model':chosen['name'],'selected_label':chosen['label'],'eligible_within_one_point':[r['name'] for r in pool],'rule':p['selection']+' '+p['latency_tiebreak'],'reason':'Development-only selection; not KSL feedback deployment approval','production_changed':False}
    sp=U/'SELECTION.json'
    if sp.exists():assert read(sp)==selection,'Do not overwrite a changed selection'
    else:write(sp,selection)
    summary={'at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'status':'DEVELOPMENT_COMPARISON_COMPLETE_NOT_DEPLOYED','models':models,'selection':selection,'fit_images':2489,'development_images':488,'classes':45,'runs':9,'seeds':seeds,'previous_test382_accessed':False,'user_mentor_data_used':False,'operating_changed':False,'end_to_end_web_latency_measured':False,'scope':'Public German sign-language handshape image classification; not KSL word accuracy or educational correction accuracy','environment':lat.get('environment',{})}
    fused_path=U/'REPVIT_FUSED_RESULTS.json'
    fused_html=''
    if fused_path.exists():
        fused=read(fused_path);summary['repvit_fused_auxiliary']=fused
        assert fused['native_selection_rule_unchanged'] and fused['protocol_sha256']==sha(U/'REPVIT_FUSED_LATENCY_PROTOCOL.json')
        if fused['status']=='COMPLETE':
            assert fused['parity']['passed'] and fused['parity']['argmax_equal']==300
            fc={r['hand_crops']:r for r in fused['clips']}
            fs=f'RepViT의 계산식을 합쳐 실행하는 보조 시험도 진행했습니다. 학습용 사진 300장의 출력이 고정된 허용오차 안에서 일치했고, 1위 결과는 300장 모두 같았습니다. 최대 출력 차이는 {fused["parity"]["max_abs_logit_difference"]:.8f}입니다. CPU 한 장 p95는 {fused["single_full_model"]["p95_ms"]:.2f}ms, 사진 읽기를 포함한 120손 입력은 {fc[120]["rgb_decode_preprocess_full_model"]["p95_seconds"]:.3f}초, 300손 입력은 {fc[300]["rgb_decode_preprocess_full_model"]["p95_seconds"]:.3f}초였습니다. 원래 체크포인트와 모델 선택 규칙은 유지했습니다.'
        else:fs='RepViT 계산 최적화는 고정된 출력 일치 검사를 통과하지 못하여 속도를 측정하지 않았다. 허용오차는 완화하지 않았다.'
        fused_html='<section><h2>RepViT의 계산만 줄였을 때</h2><p>'+html.escape(fs)+'</p><p>위 표는 세 모델의 기본 실행 비교이며, 이 값은 별도의 배포 최적화 참고 결과입니다. 새 학습이나 정답률 재평가가 아닙니다.</p></section>'
    summary['development_supported_classes']=30;summary['classes_missing_in_development']=[manifest['classes'][i] for i in sorted(fit_classes-dev_classes)]
    summary['engineering_recommendation']={'quality_candidate':chosen['name'],'full_clip_default_adoption':False,'reason':'Native RGB component latency plus incomplete handshape accuracy does not establish reliable sub3s end-to-end feedback. Validate selective RGB support without dropping important or uncertain motion; preserve current operating pipeline.','selective_candidate_verified':False,'end_to_end_latency_verified':False}
    write(U/'SUMMARY.json',summary)
    v3=models[0];delta=(chosen['accuracy_mean']-v3['accuracy_mean'])*100
    text=[f'CPU 손모양 모델 비교 완료 — 개발후보 {chosen["label"]}',
          f'공개 수동 손모양45종 학습2489장/다른 사람 개발488장. 개발사진에는30종이등장하며나머지15종의일반화는이번시험에서확인하지못함. 3모델×3회=9회. 이전최종382장·개인촬영 미사용.',
          '단어 인식 TCN을 바꾼 실험이 아니라 RGB사진으로 손모양을 읽는 보조 모델 비교다.',
          '모델 | 개발 정답률3회평균(범위) | 사람·종류 균형점수 | CPU한장p95 | 120손(사진읽기포함)p95']
    for r in models:
        c={x['hand_crops']:x for x in r['latency']['clips']}
        text.append(f'{r["label"]} | {pc(r["accuracy_mean"])}% ({pc(r["accuracy_range"][0])}~{pc(r["accuracy_range"][1])}%) | {pc(r["balanced_macro_f1_mean"])}점 | {r["latency"]["single_full_model"]["p95_ms"]:.2f}ms | {c[120]["rgb_decode_preprocess_full_model"]["p95_seconds"]:.3f}초')
    text += [f'정해둔 선택규칙에 따른 후보: {chosen["label"]}. V3 대비 전체 개발 정답률 {delta:+.2f}%p.',
             '정답률은488사진 중45수형을 맞힌 비율이다. 균형점수는ID03/07 각각의 수형별F1을 평균하여 한사람·많은종류에 치우침을 줄인 별도지표다. ID06은11장으로 선택이 아닌 진단에만 포함했다.',
             'CPU시간은 이 노트북·PyTorch FP32·4스레드 기준이다. 120손은60시점×양손 정도의 예시이며 실제촬영후전체응답시간이 아니다. MediaPipe/TCN/DTW/영상디코딩/화면표시 및 첫 모델로딩을 포함하지 않는다. PNG사진읽기와letterbox포함시간은 별도로 표시했다.',
             '개발사진을 이용해 모델을 골랐으므로 처음보는 최종시험으로 주장하지 않는다. 2024후보들을 포함해 공개가중치·CPU실행·표현력·이용조건을 검토했으며 세계최적 보장은 아니다. 원모델전처리 전체동일이라고하지않는다: 손 전체 보존letterbox공통, V3/Rep224 Hybrid256으로 모델패키지를 비교했다.',
             '이전 보고서와는 사람 분리 및 학습 경로가 달라 절대점수를 직접 빼서 개선률이라고 부르지 않는다. 모델 간 차이는 이번 동일 조건의 표 안에서 비교한다.',
             '다음: 고정한후보로 독립적인 새공개자료의 정상수형·회전·손교차·짧은변화를 확인하고, 선택적RGB보조 및 DTW전체지연을 시험한다. 검출이안되는손·관측부족을정상으로판단하는지별도확인이필요하다. Holistic검출기비교는별도이며이번RGB모델비교로대신하지않는다.',
             'C·T3-F·운영웹/DTW 유지. 새 후보를 사용자촬영으로 학습/튜닝/선택하지 않는다.']
    cc={r['hand_crops']:r for r in chosen['latency']['clips']}
    practical=f'{chosen["label"]}를 손모양 구별의 다음 개발후보로 선정했습니다. 모든 프레임에 쓰는 CPU 기본모델로 바로 채택하지는 않습니다. 300손 사진의 RGB 분석만 {cc[300]["rgb_decode_preprocess_full_model"]["p95_seconds"]:.2f}초여서, 다른 분석까지 더한 전체 응답이 3초 또는 5초 미만이라고 보장할 수 없습니다. 정답률 평균 {pc(chosen["accuracy_mean"])}%도 단독으로 교정을 판정하기에는 부족합니다. 다음에는 필요한 시점에서 RGB로 보완하면서 짧고 중요한 동작과 관측 부족을 놓치지 않는지, 전체 피드백 시간을 줄일 수 있는지 검증합니다. 이 선택적 보완 경로는 아직 검증 전입니다.'
    text.append(practical)
    if fused_path.exists():text.append(fs)
    (U/'RESULT_KO.txt').write_text('\n\n'.join(text),encoding='utf-8')
    rows=''
    for r in models:
        c={x['hand_crops']:x for x in r['latency']['clips']};selected=r['name']==chosen['name']
        rows+=f'<tr class="{ "chosen" if selected else ""}"><th>{html.escape(r["label"])}{ "<small>이번 개발후보</small>" if selected else ""}</th><td><b>{pc(r["accuracy_mean"])}%</b><small>{pc(r["accuracy_range"][0])}–{pc(r["accuracy_range"][1])}%</small></td><td>{pc(r["balanced_macro_f1_mean"])}점</td><td>{r["latency"]["single_full_model"]["p95_ms"]:.2f}ms</td><td>{f3(c[120]["rgb_decode_preprocess_full_model"]["p95_seconds"])}초</td><td>{f3(c[300]["rgb_decode_preprocess_full_model"]["p95_seconds"])}초</td></tr>'
    bars=''.join(f'<div class="barrow"><span>{r["label"]}</span><div class="bar"><i style="width:{100*r["accuracy_mean"]:.2f}%"></i></div><b>{pc(r["accuracy_mean"])}%</b></div>' for r in models)
    perrows=''.join(f'<tr><th>{r["label"]}</th>'+''.join(f'<td>{pc(r["per_signer"][s]["accuracy_mean"])}%</td>' for s in ['signer03','signer06','signer07'])+'</tr>' for r in models)
    runrows=''
    for n in names:
        for seed in seeds:
            r=read(U/'models'/f'{n}_{seed}.json')
            runrows+=f'<tr><td>{labels[n]}</td><td>{seed}</td><td>{r["dev"]["correct"]}/488</td><td>{pc(r["dev"]["accuracy"])}%</td><td>{pc(r["balanced_selection_macro_f1"])}점</td></tr>'
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>노트북 CPU 손모양 모델 비교</title>
<style>body{{margin:0;background:#f2f5f9;color:#172637;font:17px/1.7 'Malgun Gothic',sans-serif}}main{{max-width:1160px;margin:auto;padding:42px 24px 80px}}header{{background:#152d46;color:#fff;padding:36px;border-radius:20px}}header p{{color:#d7e4ef}}h1{{font-size:34px;line-height:1.35;margin:10px 0 16px}}h2{{font-size:24px;line-height:1.4;margin:0 0 18px}}section{{margin-top:24px;padding:28px;background:#fff;border:1px solid #dde5ee;border-radius:16px}}.tag{{font-size:14px;color:#91dec8}}.lead{{font-size:20px}}.notice{{padding:16px 20px;background:#fff4db;border-left:4px solid #cf961b}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;padding:13px 10px;border-bottom:1px solid #dfe6ed;white-space:nowrap}}thead{{color:#516277;background:#f5f8fb}}small{{display:block;color:#64768a;font-size:12px}}.chosen{{background:#e9f6f1}}.barrow{{display:grid;grid-template-columns:270px 1fr 90px;align-items:center;gap:14px;margin:16px 0;font-size:15px}}.bar{{height:18px;background:#e8eef5;border-radius:6px;overflow:hidden}}.bar i{{height:100%;background:#218975;display:block}}a{{color:#186dba}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}.grid p{{margin-top:0}}details{{margin-top:18px}}summary{{cursor:pointer;font-weight:bold}}footer{{font-size:14px;color:#52657b;padding-top:26px}}@media(max-width:700px){{main{{padding:20px 12px}}header,section{{padding:22px 18px}}h1{{font-size:27px}}.grid{{grid-template-columns:1fr}}.barrow{{grid-template-columns:1fr 75px;gap:8px}}.barrow span{{grid-column:1/-1}}}}</style>
<main><header><div class="tag">2026.10.28 · 실제 CPU 학습 9회 · 운영 변경 없음</div><h1>MobileNet을 당연한 정답으로 두지 않고,<br>세 모델을 직접 비교했습니다.</h1><p class="lead">이번 개발후보: <strong>{chosen['label']}</strong></p><p>사진 속 손모양을 더 정확하게 읽는 모델을 고르는 시험입니다.<br>기존 TCN의 단어 인식률이나 완성된 수어 피드백 정확도를 뜻하지 않습니다.</p></header>
<section><h2>정확도와 실제 CPU 비용</h2><p>45가지 손모양 · 학습 2,489장 / 다른 사람의 개발 비교 488장 · 모델마다 3회</p><div class="scroll"><table><thead><tr><th>모델</th><th>손모양 정답률 평균 / 범위</th><th>사람·종류 균형점수</th><th>한 손 사진 p95</th><th>120손 입력 p95</th><th>300손 입력 p95</th></tr></thead><tbody>{rows}</tbody></table></div><p>묶음 시간은 PNG 사진 읽기·전처리·전체 RGB 모델 계산을 포함합니다. p95는 측정값 중 느린 쪽을 보는 값이며, 묶음은 5회로 표본이 작습니다.</p><div class="notice">이 시간은 촬영 종료부터 피드백 표시까지의 전체 시간이 아닙니다. MediaPipe·TCN·DTW·영상 처리·화면 표시와 첫 모델 로딩은 별도입니다.</div></section>
<section><h2>한눈에 보는 손모양 구별 결과</h2>{bars}<p>이번 후보의 정답률 변화는 V3 대비 <b>{delta:+.2f}%p</b>입니다. 모델 선택은 단순 정답률 대신 사람·수형 종류를 고르게 반영한 사전 지정 점수로 했습니다. 그 점수가 최고와 1점 이내인 경우 CPU 한 장 p95가 더 빠른 모델을 선택했습니다.</p></section>
<section><h2>어떤 비교를 했나</h2><div class="grid"><div><p><b>같은 출발 조건</b><br>세 모델 모두 공식 배포된 ImageNet 사전학습 가중치에서 다시 시작했습니다. 마지막 학습 단계와 분류 부분을 동일한 20회 반복 조건으로 미세조정했습니다.</p><p><b>원래 모델 경로 보존</b><br>일부 영상 특징만 꺼내 비교했던 이전 한계를 보완했습니다. V3·RepViT는 224, V4 Hybrid는 256 크기의 입력을 사용합니다.</p></div><div><p><b>사람을 나누어 확인</b><br>ID01·02·04·05로 학습하고 ID03·06·07로 비교했습니다. 이전 최종시험 382장과 사용자·멘토 촬영은 사용하지 않았습니다.</p><p><b>이름보다 실제 결과</b><br>최근 후보의 출처·가중치·CPU 실행·이용조건을 검토했습니다. 큰 모델이나 최신 모델이 자동으로 더 좋다고 가정하지 않았습니다.</p></div></div></section>
<section><h2>사람이 달라도 잘 읽었나</h2><div class="scroll"><table><thead><tr><th>모델</th><th>ID03 · 204장</th><th>ID06 · 11장</th><th>ID07 · 273장</th></tr></thead><tbody>{perrows}</tbody></table></div><p>ID06은 사진이 11장뿐이라 후보 선택에는 쓰지 않고 결과만 공개합니다. ID03와 ID07의 손모양별 F1 평균을 선택점수에 같은 비중으로 반영했습니다.</p></section>
<section><h2>이번에 확인한 것과 아직 남은 것</h2><p><b>확인:</b> 공개 손 사진에서 세 모델의 구별 성능과 노트북 CPU 계산 비용. 기존 C·T3-F·웹·DTW는 보존했습니다.</p><p><b>남음:</b> 새 공개 한국수어에서 정상 손모양·회전·손 교차·짧은 변화가 보존되는지, 관측이 부족할 때 틀린 교정을 하지 않는지, 전체 피드백 응답이 충분히 빠른지 확인해야 합니다.</p><p>독일수어의 공개 손모양 주석으로 실시한 개발 비교입니다. 교육적 정오와 허용 범위의 정답은 별도입니다. 이 자료로 모델을 골랐으므로 ‘처음 보는 최종시험’이나 ‘모든 수어에 최적’으로 주장하지 않습니다.</p><p>다음에는 고정한 후보를 선택적 RGB 보조로 검증합니다. Holistic은 관절 검출기의 비교 과제이며, 이번 영상 모델 선택과 구분합니다.</p><details><summary>9회 전체 결과</summary><div class="scroll"><table><thead><tr><th>모델</th><th>반복 seed</th><th>정답 수</th><th>정답률</th><th>균형점수</th></tr></thead><tbody>{runrows}</tbody></table></div></details></section>
<footer><p>실행·구현: Codex / 문헌 및 평가 계약: 별도 Codex 에이전트 검토. 독립적인 사람의 한국수어 정오 판정과 구분합니다.</p><p><a href="research/CPU_MODEL_REVIEW.txt">후보 선택 근거</a> · <a href="FROZEN_PROTOCOL.json">사전 고정 조건</a> · <a href="verification/SAVED9_INDEPENDENT.json">저장 점수 독립 검산</a> · <a href="SUMMARY.json">전체 수치</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></p><p>원출처: <a href="https://github.com/THU-MIG/RepViT">RepViT</a> · <a href="https://huggingface.co/timm/mobilenetv4_hybrid_medium.ix_e550_r256_in1k">MobileNetV4 Hybrid 가중치</a>. V4는 timm 제작자의 학습 가중치입니다. 공개 DeepHand 연구자료의 이용조건은 기존 자료 계약을 유지합니다.</p></footer></main></html>'''
    page=page.replace('45가지 손모양 · 학습 2,489장 / 다른 사람의 개발 비교 488장 · 모델마다 3회','45종 학습 2,489장 / 다른 사람의 개발 비교 488장에는 30종 포함 · 모델마다 3회',1)
    page=page.replace('<div class="scroll"><table>',f'<p>측정 환경: {html.escape(str(lat.get("environment",{}).get("cpu_registry_name","현재 노트북 CPU")))} · CPU 4스레드 · FP32</p><div class="scroll"><table>',1)
    page=page.replace('ID06은 사진이 11장뿐이라','개발사진에는 45종 중 30종이 등장합니다. 나머지 15종의 일반화는 이번 시험에서 확인하지 못했습니다. ID06은 사진이 11장뿐이라',1)
    page=page.replace('이 시간은 촬영 종료부터','앞선 보고서와는 사람 분리·학습 경로가 달라, 모델의 우열은 이번 표 안에서 비교해야 합니다.<br>이 시간은 촬영 종료부터',1)
    page=page.replace('MobileNet을 당연한 정답으로 두지 않고,<br>세 모델을 직접 비교했습니다.','노트북 CPU에서 직접 비교한<br>손모양 모델 3종',1)
    page=page.replace('<section><h2>어떤 비교를 했나','<section><h2>실제 프로젝트에는 어떻게 쓰는가</h2><p>'+html.escape(practical)+'</p></section><section><h2>어떤 비교를 했나',1)
    page=page.replace('<footer>',fused_html+'<footer>',1)
    (U/'RESULT_KO.html').write_text(page,encoding='utf-8')
    print(json.dumps({'selected':chosen['label'],'accuracy':chosen['accuracy_mean'],'delta_pp':delta},ensure_ascii=False))
if __name__=='__main__':main()
