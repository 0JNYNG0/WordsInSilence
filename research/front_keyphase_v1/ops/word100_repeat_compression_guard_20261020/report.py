from pathlib import Path
from collections import Counter
import json,gzip,html
import numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent;B=OPS/'word100_repeat_boundary_consensus_20261020';P=OPS/'word100_repeat_alignment_20261020'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
write=lambda name,obj:(U/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n','utf-8')
esc=lambda x:html.escape(str(x))
s=read(U/'SUMMARY.json');rows=read(U/'ROWS.json');stress=read(U/'STRESS.json');lower=read(U/'LOWERING.json')
index={(r['clip_id'],r['view']):r for r in read(P/'INDEX.json')}
def result(row):
    with gzip.open(U/row['evidence_path'],'rt',encoding='utf-8') as h:return json.load(h)
recovered=[]
for row in rows:
    target=int(row['scenario'][-1]) if row['scenario'].startswith('repeat') else 1
    if row['scenario'].startswith('delete'):continue
    if row['old_count'] is not None and row['old_count']!=target and row['new_count'] is None:
        z=result(row);n0=index[row['clip_id'],row['view']]['frames']
        seams=[n0*j for j in range(1,target)]
        alt=z['alternative_evidence'][str(target)]
        recovered.append(dict(clip_id=row['clip_id'],label=row['label'],view=row['view'],scenario=row['scenario'],
            old_count=row['old_count'],new_count=row['new_count'],guard_reasons=row['guard_reasons'],alternative=z['alternative_evidence'],
            known_concat_seams=seams,target_boundary_exact_seam_hits=[seam in band or seam-1 in band for seam,band in zip(seams,alt['boundaries'])]))
deletion=[r for r in rows if r['scenario'].startswith('delete')]
deleted_signals=[]
for r in deletion:
    if not r['compression_frames']:continue
    z=result(r);locations=np.flatnonzero(z['compression_mask'])
    keep=np.array(r['meta']['keep_rows']);left=np.flatnonzero(np.diff(keep)>1);join=np.unique(np.r_[left,left+1])
    distance=int(min(abs(int(q)-int(j)) for q in locations for j in join)) if len(join) else None
    deleted_signals.append(dict(clip_id=r['clip_id'],label=r['label'],scenario=r['scenario'],locations=locations.tolist(),
        join_adjacent_frames=join.tolist(),distance_to_join_frames=distance,old_count=r['old_count'],new_count=r['new_count']))
old8=read(B/'DIAGNOSTICS.json')['previous8'];keys={(r['clip_id'],r['side'],r['key'],r['condition']) for r in old8}
eight=[r for r in stress if (r['clip_id'],r['side'],r['key'],r['condition']) in keys];assert len(eight)==8
stress_summary={}
for condition in ('whole','center10','peak10','peak20'):
    rr=[r for r in stress if r['condition']==condition and r['removed_frames']>0]
    stress_summary[condition]=dict(effective=len(rr),old_strong=sum(r['old_strong_s']>1e-8 for r in rr),new_strong=sum(r['new_strong_s']>1e-8 for r in rr),
        old_possible=sum(r['old_possible_s']>1e-8 for r in rr),new_possible=sum(r['new_possible_s']>1e-8 for r in rr))
native_losses=[dict(clip_id=r['clip_id'],label=r['label'],reasons=r['guard_reasons']) for r in rows if r['view']=='sensor' and r['scenario']=='original' and r['old_count']==1 and r['new_count'] is None]
normal_losses={sc:s['groups']['sensor_'+sc]['lost_correct'] for sc in ('speed075','speed150','pause')}
diag=dict(recovered_unanimous_cases=recovered,deleted_compression_signals=deleted_signals,
    deleted_join_exact=sum(r['distance_to_join_frames']==0 for r in deleted_signals),
    deleted_signal_count=len(deleted_signals),deleted_total=len(deletion),previous8=eight,
    previous8_strong=sum(r['new_strong_s']>1e-8 for r in eight),previous8_possible=sum(r['new_possible_s']>1e-8 for r in eight),
    stress_summary=stress_summary,native_sensor_lost_confirmations=native_losses,normal_control_lost_confirmations=normal_losses,
    compression_scope='Ten TRAIN observed arm-travel portions, not annotated essential/semantic stages. Only exact one-query-sample collapse is gated; positive duration ratios are descriptive.',
    spatial_boundary_limit='Shared return witnesses may be multiple disconnected times. A witness is not an exact or linguistically mandatory repetition boundary.',
    deletion_limit='Synthetic removal of a peak window; its semantic necessity is unknown. Compression elsewhere after deletion does not establish localization of the omitted movement.')
write('DIAGNOSTICS.json',diag)
v=read(U/'VERIFICATION.json') if (U/'VERIFICATION.json').exists() else None
verif=f"PASS · {v['rows']:,}대조 / {v['reference_evidence']:,}참고 근거 / 신규 정렬 {v['new_control_paths']:,}경로 / 해시 {v['hash_pins']:,}개" if v else '검산 진행 중'
labels={'original':'원래 영상','repeat2':'2회 연결','repeat3':'3회 연결','speed075':'빠른 재생 · 길이75%','speed150':'느린 재생 · 길이150%','pause':'중간15프레임 정지'}
low=s['lowering']['location']
txt=['1회 합의의 대안 반복 근거와 짧은 움직임 압축 — 공개 검증',
     '별도 후보 구현·공개 비교 완료. 기존 모델/웹 적용 및 추가학습 없음.',
     '결론: 남은 잘못된1회 확정2대조를 다른 반복 근거로 보류했다. 그러나 정상 속도 변형에서 추가 보류가 늘어 자동 판정용으로 채택하지 않는다.',
     '1) 비용이 가장 낮은 정렬뿐 아니라 가능한2·3회 정렬의 실제 팔 귀환 근거를 모두 확인한다. 같은5명 이상이 경계를 함께 지지하면1회 확정을 보류한다.',
     '2) TRAIN 참고의 관측된 팔 이동량을10구간으로 나눠, 움직이는 여러 참고 시점이 촬영본1시점에 몰리는지 확인한다. 같은5명 이상이 같은 촬영 시점에서 지지해야 한다.',
     '이10구간은 수어의 필수 단계가 아니다. 압축은 동작 오류가 아니라 정렬·관측 근거의 불확실성이다.',
     '공개100단어 TRAIN899참고/DEV300평가. 기존2391경로묶음 재사용+새 속도/정지900대조. 같은DEV후속분석이며 최종 새사람 성능시험이 아니다.',
     '']
for scenario in labels:
    g=s['groups']['sensor_'+scenario]
    txt.append(f"{labels[scenario]}(300개): 기존 확정일치{g['old_correct']}/불일치{g['old_wrong']}/보류{g['old_abstain']} → 새 확정일치{g['new_correct']}/불일치{g['new_wrong']}/보류{g['new_abstain']}.")
txt.extend([
    f"잘못된1회확정 차단{len(recovered)}대조: 같은실명영상2회/3회연결. 실제2회/3회를정답확정한것은아니며 보류로전환.",
    f"짧은구간삭제591대조 중 압축신호{len(deleted_signals)}, 삭제연결지점과정확히붙은신호{diag['deleted_join_exact']}. 필수동작생략탐지율이아님. 대부분의삭제는이기준으로확인하지못함.",
    f"빠른정상재생에서도압축신호16개. 따라서한시점압축신호만으로동작생략이라고판정할수없음.",
    f"정상손내림위치겹침강한범위 {low['old_strong_s']:.3f}→{low['new_strong_s']:.3f}초, 불확실포함 {low['old_possible_s']:.2f}→{low['new_possible_s']:.2f}초. 악화하지않았으나기존겹침미해결.",
    f"7092손근거가림대조의가능범위그대로보존. 이전8항목조건 강한{diag['previous8_strong']}/불확실포함{diag['previous8_possible']}. 서로다른8영상이아님.",
    '정상 반복을 잘못 수행했다고 출력한 사례는 없다(그 판정 자체를 만들지 않음). 대신 정상 반복/속도에서도 판단 보류가 생기므로 무해한 개선으로 단정하지 않는다.',
    '원래sensor영상3개(초대권·가평군수어통역센터·남양주시수어통역센터)도추가보류. 내부귀환동작이전체반복근거처럼보일수있어귀환만으로충분하지않음.',
    '다음계획: 전체횟수불확실성과일부구간관측부족을분리하고,대안반복에서앞·중간·끝동작이각각순서대로충분히보이는지검증. 정상속도/정지대조의정보손실과실제단계누락을분리해비교.',
    '기존C·T3-F6가중치/32보호파일유지. 사용자·멘토촬영/보류14~18미접근. 추가학습/분류추론/운영DTW·웹변경없음.',
    '검산: '+verif+'; 같은Codex별도구현,이미검산된기존투표/경로검사부품재사용. 별도수어전문가아님.'])
(U/'RESULT_KO.txt').write_text('\n'.join(txt)+'\n','utf-8')
def tr(sc):
    g=s['groups']['sensor_'+sc]
    return f"<tr><td>{labels[sc]}</td><td>{g['old_correct']} → {g['new_correct']}</td><td>{g['old_wrong']} → {g['new_wrong']}</td><td>{g['old_abstain']} → {g['new_abstain']}</td></tr>"
tables=''.join(tr(sc) for sc in ('original','repeat2','repeat3'))
normal=''.join(tr(sc) for sc in ('speed075','speed150','pause'))
signal_table=''.join(f"<tr><td>{esc(r['label'])}</td><td>{'왼손 주변' if r['scenario']=='delete_L' else '오른손 주변'}</td><td>{r['distance_to_join_frames']}</td></tr>" for r in deleted_signals)
timelines=''
for r in recovered:
    n0=index[r['clip_id'],r['view']]['frames'];k=int(r['scenario'][-1]);n=n0*k;witness=r['alternative'][str(k)]
    svg=['<svg viewBox="0 0 900 175" role="img" aria-label="알려진 연결 경계와 참고가 지지한 경계 후보">']
    for cycle in range(k):
        x=180+cycle*700/k;svg.append(f'<rect x="{x}" y="15" width="{700/k-2}" height="35" fill="#dae9f1"/><text x="{x+12}" y="39">{cycle+1}번째 영상</text>')
    for seam in r['known_concat_seams']:
        x=180+700*seam/n;svg.append(f'<line x1="{x}" x2="{x}" y1="8" y2="145" stroke="#e09837" stroke-width="3"/>')
    svg.append('<text x="0" y="40">연결한 원본</text><text x="0" y="105">참고5명 공동근거</text>')
    for band in witness['boundaries']:
        for j in band:svg.append(f'<rect x="{180+700*j/n}" y="83" width="{max(1,700/n)}" height="30" fill="#208c8a"/>')
    svg.append('<text x="180" y="165">주황선: 실제 연결 위치 / 초록: 공동 지지 경계 후보</text></svg>')
    timelines+=f"<h3>{esc(r['label'])} · {k}회 연결 → 1회 확정 보류</h3><p>참고 {', '.join(map(str,witness['people']))}번의 근거가 함께 지지했습니다. 후보에 실제 연결 경계가 포함됐지만, 다른 시점도 포함돼 정확한 경계 위치까지 확정한 것은 아닙니다.</p>{''.join(svg)}"
htmltext=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>반복·짧은 동작 압축 검증</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#edf2f6;color:#183047;font-family:'Malgun Gothic',sans-serif;line-height:1.7}}main{{max-width:1140px;margin:auto;padding:40px 24px}}section{{background:#fff;border:1px solid #dce4eb;border-radius:18px;padding:28px;margin:22px 0}}h1{{font-size:34px;line-height:1.35}}h2{{font-size:23px}}h3{{font-size:19px}}.eyebrow{{color:#267c78;font-weight:bold}}.lead{{font-size:20px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{padding:18px;background:#f2f6f9;border-radius:12px}}.big{{font-size:29px;font-weight:bold}}.note{{padding:16px;background:#fff2dd;border-left:4px solid #cb8c29}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:15px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #dae3e9}}th{{background:#f0f5f8}}li{{margin:9px 0}}svg{{width:100%;font-size:15px}}footer{{font-size:14px;color:#526779}}a{{color:#075b7b}}@media(max-width:650px){{main{{padding:20px 12px}}section{{padding:20px 16px}}h1{{font-size:27px}}.cards{{grid-template-columns:1fr}}table{{min-width:570px}}}}</style>
<main><div class="eyebrow">WordsInSilence · 공개 자료 검증 · 2026.10.20</div><h1>모두1회라고 해도,<br>반복 근거와 짧은 동작을 다시 확인</h1>
<p class="lead">남은 오확정2건은 차단했지만, 정상 속도 변화에서도 보류가 늘었습니다.</p><p>별도 후보 구현 및 검증 · 기존 C·T3-F / 웹 유지 · 추가 학습 없음</p>
<section><h2>이번에 보완한 내용</h2><div class="cards"><div class="card"><b>다른 반복 근거도 확인</b><p>1회 정렬 점수가 가장 좋아도2·3회 정렬에 실제 팔의 되돌아옴이 있는지 확인합니다. 같은5명 이상이 경계를 함께 지지하면1회 확정을 보류합니다.</p></div><div class="card"><b>한 시점에 몰린 움직임 확인</b><p>참고의 관측된 팔 이동량을10구간으로 나눕니다. 움직이는 여러 시점이 촬영본 한 시점에 몰려 대응되면 근거 부족 신호를 남깁니다.</p></div><div class="card"><b>동작 오류와 구별</b><p>판단을 보류하거나 해당 구간을 불확실하게 남깁니다. 정상 손내림·반복·속도 차이를 ‘틀린 동작’으로 판정하지 않습니다.</p></div></div><p class="note">10구간은 자동으로 나눈 관측 움직임입니다. 수어의 필수 단계10개라는 뜻이 아닙니다. 횟수도 참고 영상 전체를 몇 번 맞추는지이며 언어적 반복 정답이 아닙니다.</p></section>
<section><h2>반복 대조 결과</h2><p>공개100단어 · TRAIN899 참고 / DEV300 평가. 아래는 화면 밖 손을 반영한 sensor 관측 경로, 조건마다300개입니다.</p><div class="scroll"><table><tr><th>조건</th><th>확정 일치 기존→새</th><th>확정 불일치 기존→새</th><th>보류 기존→새</th></tr>{tables}</table></div><p>2·3회는 같은 영상을 그대로 연결한 합성 대조입니다. 새 후보가 정확한 횟수를 더 많이 맞춘 것이 아니라, 틀린 확정을 보류했습니다. 보류를 성공으로 세지 않습니다.</p>{timelines}</section>
<section><h2>정상 속도와 잠깐 멈춤에서도 점검</h2><div class="scroll"><table><tr><th>정상 변형 · 각각300개</th><th>확정 일치 기존→새</th><th>확정 불일치 기존→새</th><th>보류 기존→새</th></tr>{normal}</table></div><p>빠른 재생은 예를 들어4초를3초로, 느린 재생은4초를6초로 바꾼 대조입니다. 원래 관측이 있는 두 시점 사이에서만 좌표를 보간하고, 없는 관측을 새로 만들지 않았습니다. 정지는 중간 자세를15프레임 더 유지했습니다.</p><p class="note"><b>자동 판정에는 아직 과도하게 보수적입니다.</b> 빠른 재생16개, 느린 재생28개, 정지3개가 추가로 보류됐습니다. 한 시점에 몰린 대응이나 되돌아옴의 근거가 있다는 사실만으로 실제 동작 생략·반복 오류를 뜻하지는 않습니다.</p></section>
<section><h2>짧은 동작과 정상 손내림</h2><div class="cards"><div class="card">짧은 구간을 지운 대조<div class="big">{len(deletion)}개 중 {len(deleted_signals)}개</div><p>여러 참고가 지지하는 한 시점 압축 신호</p></div><div class="card">삭제 연결 지점 바로 옆의 신호<div class="big">{diag['deleted_join_exact']}개</div><p>나머지는 다른 위치에도 나타나므로 삭제 구간을 정확히 찾았다고 볼 수 없음</p></div><div class="card">이전 짧은 관측 누락8조건<div class="big">{diag['previous8_possible']} / 8 보존</div><p>강한 근거{diag['previous8_strong']}개, 나머지는 불확실 범위</p></div></div><p>여기서 삭제 구간이 수어의 필수 단계인지는 확정돼 있지 않습니다. 14/591을 필수 동작 누락 탐지율로 해석하면 안 됩니다. 정상 빠른 재생에서도 압축 신호가 나왔습니다.</p>
<div class="scroll"><table><tr><th>정상 손내림과 비교 요구의 겹침</th><th>기존</th><th>새 후보</th></tr><tr><td>강한 비교 범위</td><td>{low['old_strong_s']:.3f}초</td><td>{low['new_strong_s']:.3f}초</td></tr><tr><td>불확실 범위 포함</td><td>{low['old_possible_s']:.2f}초</td><td>{low['new_possible_s']:.2f}초</td></tr></table></div><p>손내림 추정945.88초에서 추가 악화는 없었지만 기존 겹침은 남았습니다. 불확실 범위를 잘못된 손내림으로 표시하지 않습니다. 전체7,092개 손 관측 가림 조건의 가능한 비교 범위는 그대로 보존했습니다.</p><details><summary>삭제 대조의 압축 신호 위치 보기</summary><div class="scroll"><table><tr><th>영상 단어</th><th>삭제 구간 선정</th><th>삭제 연결점과 거리(프레임)</th></tr>{signal_table}</table></div></details></section>
<section><h2>결론과 다음 보완 방향</h2><ol><li><b>전체 횟수와 일부 구간의 관측 부족을 분리해야 합니다.</b> 빠른 정상 수행에서 일부 시점이 압축됐다는 이유만으로 전체 횟수를 보류하는 처리는 과도할 수 있습니다.</li><li><b>되돌아옴만으로 반복을 확정하면 안 됩니다.</b> 대안 정렬에서도 앞·중간·끝의 동작이 각각 순서대로 관측되는지 확인하는 방향이 필요합니다. 원래 정상 영상3개도 이번 규칙에서 추가 보류됐습니다.</li><li><b>현재 후보는 진단용으로 보존합니다.</b> 정상 변형에서 생기는 정보 부족과 실제 생략을 구별할 근거가 충분해진 뒤 적용 여부를 검토합니다. 추가 학습은 결과를 함께 확인한 다음 진행합니다.</li></ol></section>
<section><h2>검증 범위</h2><p>{verif}</p><p>기존2,391개 저장 경로 묶음과 새 속도·정지900개 대조를 검사했습니다. 같은 Codex가 별도로 작성한 보완 규칙 검산이며, 이미 교차검산된 기존 투표·경로 검사 부품은 재사용했습니다. 별도 수어 전문가 검토는 아닙니다.</p><p>같은 개발 평가 자료를 반복 점검한 결과로, 새 사람의 최종 성능 시험이 아닙니다. 움직임은 팔꿈치·몸 관절의 손목 좌표를 기준으로 정렬하므로 손가락만의 짧은 구별 동작을 모두 검증한 것도 아닙니다.</p></section>
<footer>기존 C·T3-F6가중치 및32보호파일 보존 · 사용자/멘토 촬영 미접근 · 추가학습/분류추론/운영 DTW/웹 변경 없음<br><a href="RESULT_KO.txt">텍스트 보고서</a> · <a href="SUMMARY.json">집계</a> · <a href="DIAGNOSTICS.json">남은 문제 근거</a> · <a href="VERIFICATION.json">검산</a> · <a href="FROZEN.json">고정 조건</a></footer></main></html>"""
htmltext=htmltext.replace(
    '나머지는 다른 위치에도 나타나므로 삭제 구간을 정확히 찾았다고 볼 수 없음',
    '삭제 연결점의 압축을 찾았어도, 그 구간이 수어의 필수 동작인지까지 증명한 것은 아님')
htmltext=htmltext.replace(
    '정상 손내림·반복·속도 차이를 ‘틀린 동작’으로 판정하지 않습니다.',
    '‘틀린 동작’이라는 판정은 만들지 않습니다. 다만 정상 동작에서도 보류가 늘어나는지 별도로 확인했습니다.')
for before,after in [
    ('모두1회','모두 1회'),('오확정2건','잘못된 확정 2건'),
    ('좋아도2·3회','좋아도 2·3회'),('같은5명','같은 5명'),
    ('보류합니다.','보류합니다.'),('지지하면1회','지지하면 1회'),
    ('이동량을10구간','이동량을 10구간'),('단계10개','단계 10개'),
    ('공개100단어 · TRAIN899 참고 / DEV300 평가','공개 100단어 · 학습 자료 899개를 참고 / 개발 검증 자료 300개로 평가'),
    ('조건마다300개','조건마다 300개'),('들어4초를3초','들어 4초를 3초'),
    ('재생은4초를6초','재생은 4초를 6초'),('자세를15프레임','자세를 15프레임'),
    ('빠른 재생16개','빠른 재생 16개'),('느린 재생28개','느린 재생 28개'),('정지3개','정지 3개'),
    ('누락8조건','누락 8조건'),('추정945.88초','추정 945.88초'),
    ('전체7,092개','전체 7,092개'),('영상3개','영상 3개'),
    ('기존2,391개','기존 2,391개'),('정지900개','정지 900개'),
    ('C·T3-F6가중치 및32보호파일','C·T3-F 가중치 6개 및 보호 파일 32개')]:
    htmltext=htmltext.replace(before,after)
legacy=[]
for sc in ('original','repeat2','repeat3'):
    g=s['groups']['legacy_'+sc]
    legacy.append(f"<tr><td>{labels[sc]}</td><td>{g['old_correct']} → {g['new_correct']}</td><td>{g['old_wrong']} → {g['new_wrong']}</td><td>{g['old_abstain']} → {g['new_abstain']}</td></tr>")
htmltext=htmltext.replace(
    '<section><h2>검증 범위</h2>',
    '<section><h2>기존 전체 화면 관측 경로도 확인</h2><p>아래는 같은 영상의 legacy 관측 경로입니다. 조건별 300개이며 새로 촬영한 영상이 아닙니다. 이 경로에서도 보류가 늘어 모든 관측 조건에 무해하다고 볼 수 없습니다.</p><div class="scroll"><table><tr><th>조건</th><th>확정 일치 기존→새</th><th>확정 불일치 기존→새</th><th>보류 기존→새</th></tr>'+''.join(legacy)+'</table></div></section><section><h2>검증 범위</h2>')
(U/'RESULT_KO.html').write_text(htmltext,'utf-8')
write('DECISION.json',dict(status='OFFLINE_VERIFIED_NOT_ADOPTED' if v else 'PENDING_VERIFICATION',
    training=0,production_applied=False,private_data_read=0,
    conclusion='Unanimous miscounts abstained with alternative evidence; extra abstentions on normal speed/pause and poor deletion localization prevent automatic adoption. Diagnostic candidate retained.'))
print('REPORT',len(htmltext), 'deleted-local',diag['deleted_join_exact'])
