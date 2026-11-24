"""Human-readable record of frozen coordinate alignment diagnostics."""
import html,json
from pathlib import Path
import study as S
U=S.U
def esc(v):return html.escape(str(v))
def fmt(v):return '비교 불가' if v is None else f'{v:.2f}'
def run():
    f=S.read(U/'FROZEN.json');d=S.read(U/'DONE.json');v=S.read(U/'VERIFIED.json');vis=S.read(U/'VISUALS.json')
    assert v['status']=='PASS_SAVED_ARRAY_AND_ARITHMETIC_VERIFICATION'
    comments={
      '1290':'감사: 초반과 마지막뿐 아니라 58·66·75프레임에서도 B가 전체 위치의 밀림을 줄였다. C는 일부 손목을 위쪽으로 옮겨 보이는 손과 어긋난다. 가려진 손가락의 정확한 3D 모양까지 확인한 것은 아니다.',
      '2029':'무산: B로 전체 손 위치는 가까워지지만 60·67·74프레임의 손가락 배치는 여전히 실제 영상과 다르거나 가려져 있다. 손목을 옮기는 것만으로 손 모양을 고칠 수 없다는 사례다.',
      '0003':'수어: A/B의 차이는 작고 C가 일부 손을 위쪽으로 밀어 정상에 가까웠던 위치를 악화시킨다. 59프레임의 개별 손가락 불일치는 공통 이동으로 해결되지 않는다.',
      '0185':'후원: B는 작은 위치 변화에 그친다. 60·72프레임의 겹침에서는 C가 각 손을 따로 이동시켜 두 손의 관계를 바꾼다. 겹친 손의 정확한 좌우·손가락 정답을 이 화면만으로 확정하지 않는다.',
      '0920':'추출: 53·64·76프레임에서 C가 손가락이나 손바닥을 실제보다 위쪽으로 옮기는 모습이 보인다. A에 있던 교차를 바꾸지 않는 방식과 구분해야 한다.',
      '1545':'수영: A의 동작 중 손 위치가 상대적으로 잘 맞는다. B의 일정 이동도 일부 손끝을 아래로 밀고, C는 일부 점을 팔 위나 손 아래로 옮긴다. 모든 자료에 보정을 일괄 적용하면 안 된다.'}
    review=[]
    for r in vis['clips']:
        key=r['clip_id'].split('_WORD')[1][:4]
        review.append(dict(clip_id=r['clip_id'],frames=[x['frame'] for x in r['frames']],observation=comments[key],reviewer='Codex visual inspection of all saved panels',independent_human_groundtruth=False))
    S.write('VISUAL_REVIEW.json',dict(status='30_SOURCE_FRAMES_VISUALLY_REVIEWED_NOT_GROUNDTRUTH',clips=review,frames=30,annotation='No certified wrist/finger point coordinates were generated. Original RGB is shown beside all candidates.'))
    plan=[
      '우선 실제 RGB에서 보이는 손목·손끝만 따로 확인하는 작은 위치 평가표를 만든다. MediaPipe 결과를 답으로 복사하지 않고, 가림·흐림은 판단 보류로 남긴다. 같은 Codex의 시각 표시와 외부 독립 사람 정답을 구분한다.',
      '영상 전체의 일정한 좌표 밀림과 개별 손 검출 오류를 분리한다. 어깨·팔꿈치로 구한 보정값이 다른 시점과 보이는 손에서도 유지되는지 검증한다. 특정 단어별 예외 없이 적용·보류 기준을 먼저 고정한다.',
      '통과한 자료에만 원본 좌표·보정량·검증 근거를 함께 보관한 참고 답안을 만든다. 손목 확인을 가린 손가락 정답으로 확대하지 않고, 시간 누락은 위치 보정으로 채우지 않는다.',
      '참고 답안의 품질과 정상 사례 보존이 확인된 다음 필요한 다른 TRAIN ID 자료의 추가 확보와 학습 효과 비교로 이어간다. 현재 단계에서는 추가 다운로드와 학습을 시작하지 않는다.'
    ]
    S.write('NEXT_PLAN.json',dict(status='VERIFIED_GEOMETRIC_DIAGNOSTIC_NOT_CERTIFIED_TARGETS',steps=plan,first_next_action='Build source-RGB visible-point evaluation table and test stable global offsets against those points; preserve unknown points.',not_adopted=['global translation for all clips','per-frame wrist forced alignment','teacher targets inferred from MP agreement alone'],operating_change=False,training=False))
    rows={r['clip_id']:r for r in d['rows']}
    summary=v['summary']['post_calibration'];group=summary['previous_large10'];other=summary['other90']
    table=[]
    for x in vis['clips']:
        r=rows[x['clip_id']];m=r['metrics']['annotated_action']
        table.append('<tr><td>'+esc(r['label'])+'</td>'+''.join('<td>'+fmt(m[k]['paired_wrist']['median'])+'</td>' for k in 'ABC')+'</tr>')
    auditrows=[]
    for r in d['rows']:
        m=r['metrics']['annotated_action'];a=m['A']['paired_wrist']['median'];b=m['B']['paired_wrist']['median']
        auditrows.append('<tr><td>'+esc(r['label'])+'</td><td>'+esc(r['clip_id'].split('_')[2])+'</td>'+''.join('<td>'+fmt(m[k]['paired_wrist']['median'])+'</td>' for k in 'ABC')+'<td>'+('원래 자료의 동작 구간 비교 불가' if a is None else ('MP와 가까워짐' if b<a else 'MP와 멀어짐'))+'</td></tr>')
    sections=[]
    for x in vis['clips']:
        r=rows[x['clip_id']];key=r['clip_id'].split('_WORD')[1][:4]
        links=' · '.join(f'<a href="{esc(a["full"])}" target="_blank">{a["frame"]}프레임 원본 크기</a>' for a in x['frames'])
        sections.append(f'<article id="WORD{key}"><h3>{esc(x["label"])}</h3><p>{esc(comments[key])}</p><a href="{esc(x["sheet"])}" target="_blank"><img src="{esc(x["sheet"])}" alt="{esc(x["label"])} 원본·제공 좌표·일정 이동·손목 맞춤 5시점 비교"></a><p class="links">{links}</p></article>')
    intro='일정한 위치 밀림은 보정할 여지가 있다. 하지만 손마다 매 순간 옮기거나 모든 자료를 일괄 보정하는 방식은 정상 좌표를 해칠 수 있다.'
    text=f'''AI Hub 제공 좌표 — 위치 보정 비교
{intro}

범위: 현재 100단어의 공개 TRAIN REAL02 정면 100영상. 여러 사람에 대한 검증은 아니다.
A 제공 좌표 그대로 / B 초반15프레임 어깨·팔꿈치 차이의 중앙값을 한 번 구해 이후에도 같은 이동 / C 매 순간 팔의 손목 위치에 각 손을 따로 이동.
여기서 실험 C는 기존 TCN C 모델과 전혀 다르다. C는 팔꿈치를 관측 유효성에만 사용하며 팔꿈치–손목 방향에 따른 회전·크기 보정은 시험하지 않았다.
초반15프레임 이후 손목차이 기준: 이전에 차이가 컸던10개는 B10/10 가까워짐; 나머지90개는 B27/90 가까워짐,63/90 멀어짐. C는 나머지90개 중81개 멀어짐.
동작 구간 손목 차이(px): 감사 68.31→8.06(B), 무산87.67→23.17(B), 수영6.60→9.76(B). 모두 저장 MediaPipe와의 차이이며 실제 정답오차나 인식률이 아니다.
무산 손끝 차이는104.83→93.20(B)로 크게 남았다. 손목 위치 개선은 손가락 모양 복원과 다르다.
B는 양손 사이 관계·손 모양·시간별 이동량을 보존했다. C는 손 내부 모양은 유지하지만 양손 사이 위치와 궤적을 바꿨다. 양손이 비교 가능한9534프레임 중52프레임에서 화면 가로 순서의 부호가 달라짐. 이는52건의 좌우 배정 오류가 확인됐다는 뜻은 아니다.
동작 구간 비교 가능80/100; 나머지20은 기존 제공 좌표·관측 공백 때문에 계산 불가로 표시. 제외 후 성공률로 포장하지 않음.
6영상30시점 RGB 시각 검토 완료. 같은 Codex의 검토이며 독립 사람의 손가락 정답 주석은 아님.
100개 배열·출처·계산 별도 검산 PASS, 기존32개 보호파일 해시 유지. 기존 C/T3F/E3·운영웹·DTW 변경0. 새 학습0, 새검출0, 추가다운로드0, 사용자/멘토촬영접근0.
일괄 보정 및 보정값의 학습 정답 채택 보류. 이번 보정은 같은 원본영상의 좌표 정합 진단이다. 사용자 손을 참고 손 위치로 강제로 옮기면 피드백에 필요한 실제 위치 오류를 지워버릴 수 있으므로 그렇게 적용하지 않는다.
다음 작업:
'''+ '\n'.join(f'{i+1}. {s}' for i,s in enumerate(plan))
    with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as w:w.write(text)
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Hub 좌표 위치 보정 검증</title><style>
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f2f5fa;color:#17253b;font:16px/1.75 "Malgun Gothic",sans-serif}main{max-width:1120px;margin:auto;padding:44px 28px}header{background:#142a49;color:#fff;border-radius:22px;padding:34px}h1{font-size:32px;line-height:1.4;margin:8px 0 18px}h2{font-size:23px;margin:0 0 16px}h3{margin-top:0;font-size:21px}p{margin:12px 0}.badge{font-size:13px;color:#b9dcff;letter-spacing:.08em}.lead{font-size:20px;line-height:1.7}section,article{background:white;border:1px solid #dce4ef;border-radius:18px;padding:26px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.box{border-radius:12px;background:#eff4fb;padding:18px}.box b{display:block;font-size:20px}.note{border-left:5px solid #d49327;background:#fff7e8;padding:16px 20px}.good{border-left:5px solid #239376;background:#edf9f5;padding:16px 20px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:12px;border-bottom:1px solid #dce4ef;text-align:left}th{background:#edf3fa;font-size:14px}.scroll{overflow-x:auto}img{display:block;width:100%;height:auto;border:1px solid #dce4ef;border-radius:12px}a{color:#2455ae;overflow-wrap:anywhere}.links{font-size:13px}.sub{color:#506079;font-size:14px}li{margin:10px 0}summary{cursor:pointer;font-weight:bold}nav{display:flex;flex-wrap:wrap;gap:16px;margin-top:18px}header a{color:#c5e3ff}code{font-size:13px;overflow-wrap:anywhere}.step{display:block;font-weight:bold;color:#225a99;margin-bottom:8px}@media(max-width:640px){main{padding:18px 12px}header,section,article{padding:20px}.grid{grid-template-columns:1fr}h1{font-size:25px}.lead{font-size:18px}th,td{padding:9px;font-size:13px}}
    </style><main><header><div class="badge">PUBLIC DATA · COORDINATE QUALITY · 2026.10.26</div><h1>좌표가 조금 밀렸다면,<br>옮겨 맞추면 해결될까?</h1><p class="lead">'''+esc(intro)+'''</p><nav><a href="#methods">비교 방법</a><a href="#results">수치 결과</a><a href="#visual">직접 보기</a><a href="#next">다음 작업</a></nav></header>
    <section><h2>이번에 실제로 한 작업</h2><div class="grid"><div class="box"><b>100개 영상</b>공개 TRAIN REAL02의 정면 영상<br>100단어 · 촬영 ID 1개</div><div class="box"><b>3가지 방식</b>원본 좌표 / 일정 이동 / 매 순간 손목 맞춤</div><div class="box"><b>30장면 확인</b>6영상의 시작·중간·끝 RGB와 좌표를 직접 겹쳐 검토</div></div><p>기존에 저장한 공개 좌표와 전체 화면 MediaPipe 관측을 재사용했다. 새 검출이나 학습은 하지 않았다. 시각화의 확대는 화면 설명용이며 검출 범위를 새로 자른 것이 아니다.</p></section>
    <section id="methods"><h2>‘위치 보정’과 ‘없는 정보를 채우기’는 다르다</h2><p>이번에는 있는 좌표를 일정 거리 옮기는 <b>평행 이동</b>을 시험했다. 사라진 손가락을 만들어 넣거나, 없는 시간 구간을 보간하지 않았다.</p><div class="grid"><div class="box"><b>A · 제공 좌표 그대로</b>AI Hub의 기존 손 좌표를 수정하지 않는다.</div><div class="box"><b>B · 일정하게 옮기기</b>초반 15프레임의 어깨·팔꿈치에서 이동값을 정하고, 두 손의 모든 점에 영상 끝까지 같은 값을 적용한다. 손가락은 이동값 계산에 쓰지 않는다.</div><div class="box"><b>C · 매 순간 손목 맞춤</b>팔의 손목 관측을 기준으로 각 손을 매 프레임 따로 옮긴다. 팔꿈치·손목이 확인될 때만 계산한다.</div></div><p class="note">이번 표의 <b>C는 비교 방법 이름</b>이며 기존 TCN C 모델과 별개다. 팔꿈치–손목의 방향으로 회전하거나 크기를 바꾸는 보정은 이번에 시험하지 않았다.</p><p>B의 이동값을 계산한 첫 15프레임은 이후 결과 집계에서 분리했다. 비교 기준인 MediaPipe 자체에도 오류가 있을 수 있으므로, 수치와 실제 영상 관찰을 함께 확인했다.</p></section>
    <section id="results"><h2>어떤 자료에서는 도움이 됐고, 어떤 자료에서는 해가 됐다</h2><div class="scroll"><table><thead><tr><th>이전 검사로 나눈 자료</th><th>B로 가까워짐</th><th>B로 멀어짐</th><th>C로 멀어짐</th></tr></thead><tbody><tr><td>기존 차이가 컸던 10개</td><td>10 / 10</td><td>0 / 10</td><td>0 / 10</td></tr><tr><td>나머지 90개</td><td>27 / 90</td><td>63 / 90</td><td>81 / 90</td></tr></tbody></table></div><p class="sub">초반 15프레임 이후의 손목 거리 중앙값을 영상별로 비교. ‘나머지 90개’가 모두 정확한 정답이라는 뜻은 아니다. 가까워짐/멀어짐은 MediaPipe와의 차이이며 실제 관절 정확도·TCN 인식률이 아니다.</p><h3>동작 구간에서의 예시</h3><div class="scroll"><table><thead><tr><th>표현</th><th>A 원본</th><th>B 일정 이동</th><th>C 손목 맞춤</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div><p class="sub">단위: 원본 영상의 픽셀. 같은 시점의 제공 손목과 저장 MediaPipe 손목 사이 거리 중앙값. 작은 값은 두 관측이 가깝다는 뜻이며 ‘올바른 행동 점수’가 아니다.</p><p class="good"><b>감사:</b> 초반 팔 관측에서 구한 ‘왼쪽 약65px, 위쪽 약23px’ 이동을 이후 동작에도 적용하자 손목 차이가 <b>68.31 → 8.06px</b>로 줄었다. 실제 RGB에서도 전체 위치가 더 가까워졌다.</p><p class="note"><b>무산:</b> 손목 차이는 <b>87.67 → 23.17px</b>로 줄었지만, 손끝 차이는 <b>104.83 → 93.20px</b>로 여전히 컸다. <b>손의 위치를 맞추는 것과 손가락 모양을 바로잡는 것은 다른 문제다.</b></p><p>주석의 동작 구간은 80/100영상에서 비교 가능했다. 나머지 20개는 기존 제공 좌표·공통 관측이 해당 구간에 없어 ‘비교 불가’로 남겼다. 초반 이후 집계에는 일부 영상의 준비 구간만 포함될 수 있다.</p></section>
    <section><h2>정상 손과 손 교차는 유지됐나?</h2><p><b>B는 손 모양, 양손의 상대적 위치, 시간별 이동량을 그대로 보존했다.</b> 동일한 이동값을 두 손 전체에 적용했기 때문이다. 다만 실제 영상과의 위치 일치까지 좋아진다는 뜻은 아니다.</p><p><b>C는 손 내부 모양만 유지하고, 양손 사이의 위치와 이동 궤적을 바꿀 수 있었다.</b> 양손을 계산할 수 있었던 9,534프레임 중 52프레임에서는 화면상 두 손목의 가로 순서가 원본과 달라졌다. 이것을 ‘좌우 배정 오류 52건’으로 해석하면 안 된다. 원본도 틀릴 수 있고, 좌우 슬롯을 재배정한 시험은 아니기 때문이다.</p><p>팔 관측 유효성 조건으로 기존 제공 손 좌표가 추가 탈락한 경우는 이 표본에서 0건이었다(19,068 손×프레임 유지). 원래 없던 관절이나 시간 구간은 만들지 않았다.</p><p class="note">매 순간 다른 손목 추정값에 손 전체를 붙이면, 이미 잘 맞던 손가락까지 함께 밀릴 수 있다. ‘팔과 연결되어 있다’는 사실은 유용한 근거지만, 그 팔 관측값도 검증 없이 정답으로 삼을 수는 없다.</p></section>
    <section id="visual"><h2>실제 영상에서 확인하기</h2><p>각 장면의 왼쪽부터 <b>원본 RGB → A 제공 좌표 → B 일정 이동 → C 손목 맞춤</b>이다. 분홍색은 자료의 왼손 슬롯, 노란색은 오른손 슬롯이다. 피부 위에 점이 놓이는지, 손목이 맞아도 손끝이 벗어나는지, 두 손의 겹침 관계가 바뀌는지 확인하면 된다.</p><p>아래 화면의 색은 정답 판정이 아니다. 이미지와 프레임 링크를 누르면 크게 볼 수 있다.</p></section>
    '''+''.join(sections)+'''
    <section id="next"><h2>그래서 앞으로 어떻게 쓸까?</h2><p class="lead">‘확인된 위치 밀림만 보정한 참고 답안’으로 활용하는 방향은 유지한다. 현재 결과를 모든 자료에 적용하거나 학습 정답으로 채택하지는 않는다.</p><ol>'''+''.join('<li>'+esc(x)+'</li>' for x in plan)+'''</ol><p class="note"><b>사용자 동작을 정답 위치로 끌어다 맞추는 보정은 하지 않는다.</b> 이번에는 같은 원본 영상에 붙은 좌표가 어긋났는지 검사했다. 사용자와 참고의 카메라 위치·체형 차이는 공통 몸 기준으로 정규화하되, 피드백에 필요한 손 위치 차이는 남겨야 한다.</p></section>
    <section><h2>검증 범위와 현재 상태</h2><ul><li>100개 저장 배열과 보정 계산을 별도 코드로 재구성·검산 완료. 정상 내부 모양·시간 변화·공백 보존 검사 통과.</li><li>6영상 30시점 시각 검토 완료. 같은 Codex의 검토이며 독립 사람의 정답 좌표 검증은 아니다.</li><li>기존 C·T3-F·E3와 웹·DTW 등 32개 보호 파일 해시 유지. 사용자·멘토 촬영 접근 0.</li><li>새 학습·새 관절 검출·추가 다운로드·운영 적용 0. 이번 성과는 좌표 보정의 유효 범위 확인이며 수어 인식률 개선 결과는 아니다.</li></ul><p class="links"><a href="VERIFIED.json">별도 검산</a> · <a href="FROZEN.json">고정 비교 조건</a> · <a href="VISUAL_REVIEW.json">시각 검토</a> · <a href="NEXT_PLAN.json">다음 작업 기록</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></p><details><summary>전체 100개 영상의 동작 구간 수치</summary><div class="scroll"><table><thead><tr><th>표현</th><th>ID</th><th>A</th><th>B</th><th>C</th><th>B 변화</th></tr></thead><tbody>'''+''.join(auditrows)+'''</tbody></table></div></details></section></main></html>'''
    with (U/'RESULT_KO.html').open('x',encoding='utf-8') as w:w.write(doc)
    names=['RESULT_KO.html','RESULT_KO.txt','FROZEN.json','DONE.json','VERIFIED.json','VISUALS.json','VISUAL_REVIEW.json','NEXT_PLAN.json','study.py','verify.py','visuals.py','report.py']
    S.write('REPORT_MANIFEST.json',dict(files={str(U/n):S.sha(U/n) for n in names},images={str(U/x['sheet']):S.sha(U/x['sheet']) for x in vis['clips']}))
    print('REPORT_COMPLETE',len(vis['clips']),vis['total_frames'],flush=True)
if __name__=='__main__':run()
