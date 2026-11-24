from pathlib import Path
import json,html
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_multiref_observation_compare_20261019'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
E=lambda x:html.escape(str(x))
def main():
    s=read(U/'SUMMARY.json');v=read(U/'VERIFICATION.json');prep=read(U/'PREPARE.json');visual=read(U/'VISUAL.json');details=read(U/'DEV_QUERY_DETAILS.json')
    assert v['status']=='PASS';names={'legacy':'기존 전체화면 관측','masked':'전체 검출 후 상반신 범위 제한','sensor':'웹캠 경계 모사 관측'}
    rows='';txt=[]
    for key,x in s['views'].items():
        d=x['DEV'];rows+=f'<tr><th>{names[key]}</th><td>{d["old_whole"]}</td><td>{d["new_whole"]}</td><td>{d["partial"]}</td><td>{d["unavailable"]}</td><td>{d["whole_lost"]}</td></tr>'
        txt.append(f'{key}: 기존 전체굽힘 {d["old_whole"]}/591 → 보완15각도 모두측정 {d["new_whole"]}/591, 일부각도 {d["partial"]}, 전체보류 {d["unavailable"]}, 기존측정가능→불가 {d["whole_lost"]}. 각도별근거 {d["ready_angles"]}/8865.')
    cases='';case_text=[]
    for item in visual:
        cid=item['clip_id'];query=next(q for q in details if q['clip_id']==cid and q['view']=='sensor');cs=''
        for side,d in query['sides'].items():
            cs+=f'<tr><th>{"왼손" if side=="L" else "오른손"}</th><td>{d["old_refs"]}명</td><td>{d["angles_ready"]}/15각도</td><td>{min(d["new_refs"])}~{max(d["new_refs"])}명</td></tr>'
            case_text.append(f'{item["label"]} {cid} {side}: 기존굽힘참고{d["old_refs"]}, 보완측정각도{d["angles_ready"]}/15,각도별참고{min(d["new_refs"])}~{max(d["new_refs"])}.')
        frames=''
        for frame in item['frames']:
            h=frame['hands'];frames+=f'<tr><td>{frame["time"]:.2f}초</td>'+''.join(f'<td>{h[side]["angles"]}/15 · 위치 {"있음" if h[side]["position"] else "없음"}</td>' for side in ('L','R'))+'</tr>'
        src='../'+OLD.name+'/visual/'+Path(item['previous_public_rgb_sheet']).name
        cases+=f'<details><summary>{E(item["label"])} · 촬영자 ID{cid.split("REAL")[1][:2]}</summary><table><thead><tr><th>손</th><th>기존 전체굽힘 유효참고</th><th>보완 후 충분한 근거</th><th>각도마다 유효참고</th></tr></thead><tbody>{cs}</tbody></table><p>전체 영상 비교는 각도마다 최소5명·유효대응50%·서로 다른 프레임8개 조건을 유지합니다. 아래의 단일 프레임 각도 개수와 구분해야 합니다.</p><table><thead><tr><th>확인 장면</th><th>왼손: 관측 각도 / 위치</th><th>오른손: 관측 각도 / 위치</th></tr></thead><tbody>{frames}</tbody></table><a href="visual/{Path(item["timeline"]).name}"><img src="visual/{Path(item["timeline"]).name}" alt="{E(item["label"])} 관측 타임라인"></a><p><a href="{src}">이전 작업에서 확인한 공개 원본·관절 장면</a></p></details>'
    t=s['views']['sensor']['trajectory_common_measurable'];source=s['views']['sensor']['source_diagnostics']
    lowered=sum(source['development_only_'+side+'_lowered_observed_missing_in_view'] for side in ('L','R'))
    inside=sum(source['development_only_'+side+'_annotated_source_inside_view_missing'] for side in ('L','R'))
    text=f'''손 위치·각도별 근거 분리 / 구간 진단 / 궤적 시작 기준 점검

공개100단어, TRAIN899영상·DEV300영상, 세 저장 관측에 보완 후보를 적용했다. 기존 C·T3-F·웹·DTW는 그대로다. 인식 보조·학습·개인/멘토 촬영 사용 없음.

1. 실제 보완
손가락15개 굽힘각을 각자 필요한 관절3개의 관측 여부로 계산한다. 손목이나 다른 손가락이 없다는 이유로 독립적으로 계산할 수 있는 각도까지 버리지 않는다. 다만 관측되지 않은 각도는 NaN/보류이며 보간하거나 0점으로 넣지 않는다. 위치는 기존 어깨 기준 좌표를 그대로 사용한다. 어깨가 없더라도 손가락 자체의 세 점이 있으면 그 국소각은 기록 가능하다. 단어명/촬영자별 예외는 없다.
프레임별 기존각도 {prep['counts']['old_angle_values_preserved']:,}값은 보존됐고, 추가로 {prep['counts']['angle_frames_added']:,}개의 각도 관측 셀을 기록했다. 반복 프레임·세 관측이 포함된 숫자로 새로운 영상 수나 정답 횟수가 아니다.

2. 공개 DEV 비교
'''+'\n'.join(txt)+f'''
591은 DEV300영상의 손 수에서 기존 오른손 전용3단어의 반대손9개를 뺀 것이다. 전체 손 점수는15각도 모두 충분한 참고가 있을 때만 기술적으로 계산하며, 수어 합격 판정이 아니다. 새 종합값은 각도별 참고 중앙값의 평균이므로 기존의 프레임평균→참고중앙값과 집계 순서가 다르다. 새 점수가 낮다는 이유로 개선이라 보고하지 않았다. 현재는 관측·근거의 보존/분리가 개선된 것이며 인식률 개선을 측정하지 않았다.

3. 내려놓은 손과 미관측 분리
각 영상에 현재 관측의 손점/위치/각도·팔 위치 힌트·내부 관측 공백을 따로 기록했다. 화면 밖 손목 추정을 손가락 관측으로 바꾸지 않는다. 전체 원본을 볼 수 있는 오프라인 진단에서는 손21점 전체가 하단 밖에 있고, 낮은 위치에서 짧은 시간 안정적으로 관측된 경우에만 '관측된 낮은 정지 구간' 표식을 붙였다. 원본 손이 관측되지 않으면 불명이다. 위치 표식은 점수 제외나 필수손 판정에 쓰지 않았다. 정지한 받침손을 무조건 불필요손으로 판단하지 않았다.
DEV sensor의 손-프레임 단위로 전체 원본에서 낮은 정지가 확인되지만 잘린 관측에서 손점이 없는 경우 {lowered:,}개, 공개 주석 동작 구간 안에서 원본은 손 전체를 범위 안에 관측했지만 잘린 관측의 손점이 없는 경우 {inside:,}개를 분리했다. 이는 검출된 근거의 조건별 건수이지 실제 수어오류/가림 정답 수가 아니다. 모든 누락 원인에 라벨을 확정하지 않았다.
실제 웹캠은 화면 밖 원본을 볼 수 없다. 그 경우 원인을 모르면 불명으로 남겨야 하며, 공개 전체화면 진단의 분류 능력을 실사용 능력으로 보고하지 않는다. 전체 원본 관측도 MediaPipe 추정이므로 진실값이 아니다.

4. 궤적
기존 '각 손의 첫 관측점'과 '같은 DTW 경로에서 양쪽이 함께 관측된 첫 대응점'을 비교했다. 경로/대응쌍은 고정했고 내부 공백을 채우지 않았다. sensor DEV의 공통 측정가능 {t['absolute_change']['n']}개 손-궤적에서 점수 변화 절댓값 중앙값 {t['absolute_change']['median']:.3f},90백분위 {t['absolute_change']['p90']:.3f}였다. 단위는 어깨너비를100으로 놓은 거리 단위이며 정답률 %가 아니다. 출발점 차이만으로 수치가 달라질 수 있음을 확인했지만, 첫 대응점으로 바꾸면 실제로 다른 시작 위치도 지워질 수 있고 같은 의미의 동작 단계라는 보장도 없다. 이 방식을 운영에 채택하지 않았다.

5. 공개 장면 재확인
'''+'\n'.join(case_text)+'''
이전 공개6영상18장면과 새 근거 타임라인을 연결했다. 눈으로 확인한 기존 관측 장면은 재사용했으며 새로운 언어적 정답 라벨은 만들지 않았다. 전체 누락의 시각 전수검토가 아니다.

6. 검증과 한계
기본13검사, 원입력/삼각점 코사인 별도검산, 기존 위치보존, 각도별 집계·자기촬영자제외, 고정선정300쌍의 각도·궤적 산술검산과 보호파일 해시검사를 수행했다. 동일 Codex의 별도 검산이며 다른 전문가/에이전트 검토는 아니다.
남은 문제: 관절이 통째로 미관측이면 이번 분리만으로 복구되지 않는다. 현재 각도는2D 투영 각도이며 실제3D 굽힘이나 손의 방향을 완전히 설명하지 못한다. 각도마다 충분한 관측이 있어도 짧은 구별 동작이 빠질 수 있다. 내부 관측 공백은 실제 손 겹침/배정 보류/검출 실패 중 어떤 원인인지 추가 근거 없이는 확정하지 않는다. 동작이 틀렸다는 교육적 판정은 아직 하지 않는다.
다음 제안: 확인된 항목별 근거를 연구용 DTW 출력에 연결하는 범위와 구간별 정보 누락 표시를 검토한다. 먼저 정상 자료의 짧은 핵심 구간을 빠뜨린 채 낮은 차이를 내는 경우가 없는지 점검하고, 이후 인식 보조/추가학습 여부를 결정한다. 기존 C·T3-F와 사용자 촬영은 그대로 유지한다.
'''
    (U/'RESULT_KO.txt').write_text(text,'utf-8')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손 위치와 각도별 관측 근거</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f9;color:#163047;font:17px/1.75 'Malgun Gothic',sans-serif;word-break:keep-all;overflow-wrap:break-word}}main{{max-width:1120px;margin:auto;padding:45px 25px 75px}}h1{{font-size:38px;line-height:1.35}}h2{{font-size:25px;line-height:1.5;margin-top:0}}section{{background:white;padding:30px;border:1px solid #dae5ed;border-radius:16px;margin:22px 0}}.lead{{font-size:21px}}.tag{{color:#26689a;font-size:14px;font-weight:700}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.card{{padding:20px;background:#eef5fa;border-radius:12px}}.card strong{{font-size:30px;display:block}}.note{{padding:17px 22px;border-left:5px solid #ce8b28;background:#fff4de;margin:20px 0}}table{{width:100%;border-collapse:collapse;font-size:15px}}td,th{{padding:13px;border-bottom:1px solid #dce5ee;text-align:left}}thead{{background:#edf4f9}}.scroll{{overflow:auto}}details{{overflow-x:auto;min-width:0;border:1px solid #dce5ee;border-radius:10px;padding:18px;margin:15px 0}}summary{{font-weight:700;cursor:pointer}}img{{width:100%;height:auto}}a{{color:#1768a0}}li{{margin:12px 0}}small{{display:block;color:#4e6576}}@media(max-width:700px){{main{{padding:22px 13px}}h1{{font-size:28px}}section{{padding:20px}}.grid{{grid-template-columns:1fr}}}}</style><main><span class="tag">WORDS IN SILENCE · 공개 자료 검증 / 2026.10.20</span><h1>손 위치를 아는 것과<br>손가락 모양을 아는 것을 분리했습니다.</h1><p class="lead">계산할 근거가 있는 각도는 살리고, 보이지 않는 각도는 보류합니다.<br>손을 내려놓은 구간과 화면 안에서 관측이 끊긴 구간도 별도로 기록합니다.</p><div class="grid"><div class="card"><strong>{prep['counts']['old_angle_values_preserved']:,}</strong>기존에 유효했던 프레임별 각도값 보존</div><div class="card"><strong>+{prep['counts']['angle_frames_added']:,}</strong>추가로 기록한 프레임별 각도 근거</div></div><p>각도값 개수에는 여러 프레임과 세 관측 경로가 포함됩니다. 새로운 촬영 수나 정답 횟수가 아닙니다.</p>
    <section><h2>01. 무엇을 바꿨나</h2><p>이전에는 손 전체의 관측 조건이 부족하면 굽힘을 한꺼번에 보류했습니다. 이제 <b>15개의 굽힘각마다 필요한 관절3개</b>를 확인합니다. 검지 끝이 없더라도 중지의 세 관절이 보이면 그 중지 각도는 기록합니다.</p><p>손 위치는 기존 좌표를 그대로 보존했습니다. 손목을 추정했다고 손가락을 본 것으로 처리하지 않습니다. 각도가 일부만 보이면 전체 손의 점수를 내지 않습니다.</p><div class="note"><b>이번 개선은 관측 정보의 보존과 구분입니다.</b> 인식률 상승이나 수어를 올바르게 수행했는지를 검증한 결과는 아닙니다. 기존 C·T3-F·웹·DTW는 바꾸지 않았습니다.</div></section>
    <section><h2>02. 같은 공개 영상에서 확인한 결과</h2><p>100단어의 TRAIN899영상을 참고로 두고, 다른 촬영자3명의 DEV300영상으로 비교했습니다. 표의 분모는 기존 손 역할을 유지한 <b>591개 손-영상</b>입니다.</p><div class="scroll"><table><thead><tr><th>관측 경로</th><th>기존 전체굽힘 가능</th><th>보완15각도 모두 가능</th><th>일부각도만 가능</th><th>전체 보류</th><th>기존 가능→불가</th></tr></thead><tbody>{rows}</tbody></table></div><div class="note"><b>웹캠 경계 모사: 전체측정447건은 그대로 유지됐고, 전체를 보류하던14건에서 일부 각도 근거를 확보했습니다. 전체측정 가능 건수 자체는 늘지 않았습니다.</b></div><p>각도별로 <b>최소5명의 참고·50%의 유효 대응·서로 다른 프레임8개</b>라는 근거 조건을 유지했습니다. 비교 경로도 이전 DTW 결과를 재사용했습니다. 특정 단어의 기준을 낮춰 통과시키지 않았습니다.</p><p>새 종합값은 각도별 참고 중앙값을 평균한 값이어서 이전 집계 순서와 다릅니다. 수치가 낮아졌다고 성능 향상으로 해석하지 않았습니다. 15각도 모두 계산돼도 짧은 핵심 동작을 전부 관측했다는 보장은 없습니다.</p></section>
    <section><h2>03. 내려놓은 손과 미관측은 다르게 기록합니다</h2><div class="grid"><div class="card"><h3>관측된 낮은 정지 구간</h3><p>전체 원본에서 손21점이 모두 하단선 아래에 있고, 낮은 위치에서 안정적으로 보이는 경우에만 표시합니다. 화면 밖이라고 추정한 것만으로는 붙이지 않습니다.</p></div><div class="card"><h3>영역 안 관측 공백</h3><p>전체 원본에는 범위 안 손 근거가 있지만, 잘린 관측에는 손점이 없는 경우를 분리합니다. 손 겹침인지 배정 보류인지까지 자동 확정하지 않습니다.</p></div></div><p>공개 DEV의 웹캠 경계 모사에서 전자는 <b>{lowered:,}손-프레임</b>, 주석 동작 구간 안의 후자는 <b>{inside:,}손-프레임</b>을 구분했습니다. 실제 오류나 가림의 정답 개수가 아니라, 저장된 관측 근거의 조건별 건수입니다.</p><div class="note">실제 웹캠에서는 화면 밖 원본을 볼 수 없습니다. <b>그 원인을 모르면 ‘관측 불명’으로 남깁니다.</b> 공개 원본을 이용한 진단 능력을 실사용 능력으로 보고하지 않았습니다. 이 표식으로 받침손이나 정지 구간을 자동 제외하지도 않았습니다.</div></section>
    <section><h2>04. 궤적의 출발점도 수치에 영향을 줍니다</h2><p>‘각 손이 처음 보인 위치’와 ‘DTW상 서로 대응되며 함께 관측된 첫 위치’를 각각 출발점으로 두고 같은 대응쌍을 비교했습니다. 웹캠 경계 모사의 공통 측정 가능한 <b>{t['absolute_change']['n']}개 손-궤적</b>에서 차이 변화의 절댓값 중앙값은 <b>{t['absolute_change']['median']:.2f}</b>, 90백분위는 <b>{t['absolute_change']['p90']:.2f}</b>였습니다.</p><p>단위는 어깨너비를100으로 놓은 거리 단위입니다. 정답률 %가 아닙니다. 시작점을 바꾸면 실제 시작 위치 차이도 지워질 수 있으므로, <b>새 시작점 방식을 운영에 적용하지 않았습니다.</b> 내부 빈 구간은 채우지 않았고, 대응점이 같은 의미의 동작 단계인지도 이번 수치만으로 확정하지 않습니다.</p></section>
    <section><h2>05. 실제 공개 장면과 근거 타임라인</h2><p>이전에 확인한6개 공개 영상의18장면에 새 근거를 연결했습니다. 그래프 파란선은 관측된 각도 수, 회색선은 기존 손 전체 조건을 통과한 각도 수입니다. 빨간 표시는 전체 원본의 범위 안 손 근거와 잘린 관측의 불일치입니다. 전체 원본 관측도 MediaPipe 추정이며 진실값은 아닙니다.</p>{cases}</section>
    <section><h2>06. 남은 문제와 다음 단계</h2><ul><li><b>손이 통째로 미관측이면 이번 보완으로 복구되지는 않습니다.</b> 손 위치·각도·팔 힌트·관측 공백을 구분해 근거 부족을 정확히 표시합니다.</li><li>측정값은 화면에 투영된2D 각도입니다. 실제3D 굽힘·손 방향·수어의 언어적 정확성과 같습니다라는 뜻이 아닙니다.</li><li>다음에는 항목별 관측 근거를 연구용 DTW 출력에 연결할 범위와, 짧은 구별 구간을 놓친 상태에서 낮은 차이를 내지 않도록 하는 검증 조건을 확인하는 것이 적절합니다.</li></ul><p><b>인식 보조·추가 학습은 결과 확인 뒤로 남겼습니다.</b> 사용자·멘토 촬영은 읽거나 학습·튜닝·선택에 쓰지 않았고 기존 모델은 보존했습니다. 새로운 사람에 대한 정확도 시험은 하지 않았습니다.</p><p>검증: 기본13검사, 원입력{v['inputs']:,}개, 각도값{v['source_angle_values']:,}개 별도 코사인 검산, 위치보존·저장집계·고정300쌍 각도/궤적 검산. 같은 Codex의 별도 산술 검산이며 다른 전문가나 에이전트 검토는 아닙니다.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="SUMMARY.json">결과 집계</a> · <a href="VERIFICATION.json">별도 검산</a> · <a href="FROZEN.json">고정 조건</a></p></section></main></html>'''
    page=page.replace('언어적 정확성과 같습니다라는 뜻이 아닙니다.','언어적 정확성을 그대로 뜻하지 않습니다.')
    (U/'RESULT_KO.html').write_text(page,'utf-8');print('\n'.join(txt));print('source DEV',lowered,inside);print('trajectory',t)
if __name__=='__main__':main()
