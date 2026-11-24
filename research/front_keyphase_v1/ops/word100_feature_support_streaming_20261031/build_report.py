from quality import *
import html
LABELS={'R_index_only':'오른손 검지만 폄','R_open_four':'오른손 네 손가락 폄','L_open_four':'왼손 네 손가락 폄','R_face_overlap':'오른손·얼굴 겹침','hands_overlap':'두 손 겹침','R_motion_up':'오른손 위로 이동','R_motion_down':'오른손 아래로 이동','R_motion_left':'오른손 화면 왼쪽 이동','R_motion_right':'오른손 화면 오른쪽 이동','L_motion_down':'왼손 아래로 이동'}
CONDS={'frozen_previous_pose':'기존 관절 판정부','frozen_previous_rgb':'기존 관절+SignRep 판정부','shared_shape_interval_motion':'공유 손모양+구간 이동 후보'}
def main():
 scores=read(U/'FRESH_SCORES.json');summ=scores['summary'];idx={(r['condition'],r['role']):r for r in summ};speed=read(U/'review/INPUT_SPEED_INDEPENDENT.json')['speed'];audit=read(U/'fresh/LABEL_AUDIT.json');di=read(U/'SAVED_OUTPUT_DIAGNOSTICS.json');control=read(U/'CONTROLS.json');train=read(U/'TRAINING.json');rows=get_rows(True)
 def pct(x):return f'{x*100:.2f}%'
 table=[]
 for c,name in CONDS.items():
  cells=[]
  for group in ['fresh_known','fresh_expression']:
   r=idx[c,group];cells.append(f"{pct(r['mean'])} ({r['range'][0]*100:.2f}–{r['range'][1]*100:.2f}%)")
  table.append([name,*cells])
 gain=(idx['shared_shape_interval_motion','fresh_known']['mean']-idx['frozen_previous_pose','fresh_known']['mean'])*100;novelgain=(idx['shared_shape_interval_motion','fresh_expression']['mean']-idx['frozen_previous_pose','fresh_expression']['mean'])*100
 targetrows=[]
 for j,f in enumerate(TARGETS):
  vals=[idx[c,'fresh_known']['target_means'][j] for c in CONDS];targetrows.append([LABELS[f],*[pct(v) if v is not None else '근거 없음' for v in vals],f'{(vals[2]-vals[0])*100:+.2f}%p' if vals[2] is not None and vals[0] is not None else '—'])
 delta=100*(1-speed['single_decode_paced']['median']/speed['legacy_double_decode_batch']['median'])
 lines=['두 우선순위 작업 결과 — 공개 자료의 특징 판별과 촬영 중 처리', '',
 '1. 무엇을 바꿨나',
 '손 모양은 오른손과 왼손에 같은 판정부를 사용했다. 기존의 관절 특징과 이미 확인한 공개 손모양 주석을 함께 사용해, 왼손 자료가 적은 문제를 줄이고자 했다. 단어별 예외나 사용자 촬영은 넣지 않았다.',
 '이동 방향은 구간 가운데만 보는 대신 시작부터 끝까지 모든 원본 프레임을 작은 GRU에 순서대로 넣었다. 가려진 좌표는 채우지 않고 관측 여부를 별도로 전달했다. 손 모양 판정부는 앞 2개·현재 1개·뒤 2개, 합계 5프레임을 본다. 자동으로 수어의 필수 단계를 알아낸 모델은 아니다.',
 '기존 공개36영상으로 새 판정부를 seed42/43/44 각160epoch 학습했다. 새로 다운로드한 모델은 없고, C/T3-F/SignRep 본체는 재학습하지 않았다. 추가 판정부9990개 파라미터, 3회 학습 합계 %.2f초. 관계2개출력을 위해 기존관절판정부47914개도 쓰므로 합성후보전체는57904개 파라미터다.'%sum(r['seconds'] for r in train['runs']), '',
 '2. 다른 촬영자32영상에서 비교한 결과',
 '앞서 결과를 확인한 held32는 새 후보의 품질 튜닝에 다시 쓰지 않았다. 이번32개는 기록ID7/8, 기존12표현24영상과 판정부가 학습하지 않은4표현8영상이다. 모든 영상은 과거TCN 학습목록에 포함된 공개자료이므로, TCN에게도 완전히 새로운 시험이라고 부르지는 않는다.',
 '평가지표는 단어 인식률이 아니라 10가지 물리적 특징의 균형정확도다. 아래는3회 평균(최소–최대). 지원이 한쪽 클래스뿐인 특징은 그쪽 재현율이며 상세 분모를 저장했다.',
 *[' | '.join(r) for r in table],
 f'기존 관절 판정부 대비 배운표현 {gain:+.2f}%p, 새표현 {novelgain:+.2f}%p. 새 평가점수를 보고 학습·임계값·seed를 다시 고르지 않았다.',
 '두 손의 겹침과 얼굴 겹침 출력은 기존 관절 판정부를 그대로 썼다. 변경한 손모양/이동항목과 전체 결과를 함께 보고한다.',
 '새 후보는 모든 비교에서 우세하지 않았다. 배운표현에서 기존RGB보조82.01%보다 낮은80.78%이며, 오른손 화면왼쪽이동은75.33→57.59%, 왼손펼침은96.38→92.56%로 기존관절보다 나빠졌다. 기존 모델을 전면 교체할 근거는 부족해 미채택한다.',
 '손겹침이 실제로 표시되고 손모양정답도 있는28셀에서는 기존관절24/25/25개 정답에서 새후보23/23/23개로 줄었다. 관절좌표와좌우배정의보존은확인했지만 손겹침상황의손모양분류까지유지됐다고할수없다. 물리 손모양 양성 주석 66셀은61/63/64→64/64/64개로, 개선과회귀가함께있었다.',
 f'RGB 주석은 Codex가 직접 화면을 보고 작성한 임시 물리 주석이다. 8영상의 두 번째 검토에서 양쪽 모두 답을 붙인 {audit["joint_known"]}셀 중 {audit["known_disagreements"]}셀은 달랐고, 이견·불확실 항목은 보류했다. 최종 사용가능 표시는 17후보 전체 {audit["known"]}셀이고, 이번 10평가특징에는 636셀(배운 표현 462·새 표현 174)을 사용했다. 수어 교육자의 독립 정답은 아니다.', '',
 '3. 촬영이 끝난 뒤 기다리는 시간을 줄인 결과',
 '공개 원본3개를 각2회, 총3조건18회 처리했다. 같은 MediaPipe 모델, 원본 모든 프레임, 같은 좌우 배정으로 비교했다.',
 f"기존 두 번 영상읽기: 중앙 {speed['legacy_double_decode_batch']['median']:.3f}초 / p95 {speed['legacy_double_decode_batch']['p95']:.3f}초.",
 f"한 번 영상읽기: 중앙 {speed['single_decode_batch']['median']:.3f}초 / p95 {speed['single_decode_batch']['p95']:.3f}초.",
 f"촬영 속도에 맞춰 들어오는 프레임을 곧바로 처리: 종료 후 중앙 {speed['single_decode_paced']['median']:.3f}초 / p95 {speed['single_decode_paced']['p95']:.3f}초. 기존 대비 중앙대기 {delta:.1f}% 감소.",
 '18회1884프레임 유지. 새3조건의12비교쌍과 과거CPU관측18건 모두 관절·누락표시·좌우배정이 정확히 같았다. 정상 손이나 교차의 관측을 바꿔 얻은 속도 향상이 아니다.',
 '단, 이 시간은 관절 검출·좌우 배정·파일저장까지다. SignRep/판정부/DTW/웹 화면은 포함하지 않았다. 실제 웹캠 대신 원본PTS에 맞춘 파일재생이며, 모델준비는 촬영 전에 따로 했다. 원본3개·6회 표본, 고정 실행순서, 큐지연이 있어 모든환경의 지연을 보장하지 않는다.',
 '현재 Windows MediaPipe는 손/몸 모두 GPU delegate 생성 시 GPU processing is disabled in build flags로 실패했다. 실패기록을 보존했다. 기존 SignRep의 Intel GPU 사용은 정상적으로 유지했다. 새 작은 판정부는 CPU에서 학습했다.', '',
 '4. 남은 문제와 다음 판단',
 '손이 없으면 분류기 자체는 숫자를 낼 수 있다. 그래서 숫자가 생성되었다는 표시와 실제 관측 근거를 분리하는 부품을 만들었다. 이 보류 부품은 격리 시험용이며 운영 웹의 판정에는 아직 연결하지 않았다. 관절 근거가 없으면 비교 근거 부족이며, 비교 범위의 분모도 그대로 유지한다. 관측 부족을 수어 오류나 잘한 동작으로 단정하지 않는다.',
 '개발자료의 시간만 .75/1.25배로 바꿔 속도값을 조정했을 때, 방향판정172셀 중 seed/조건별2~9셀이 바뀌었다. 실제 빠른 촬영·흐림 검증은 아니며, 정상 속도에 완전히 불변하다는 주장은 할 수 없다.',
 '고정된 원본 단어 구간을4등분해 비교했으므로, 중요한 단계의 시작·끝을 스스로 찾는 능력이나 필수동작 누락 판정까지 검증된 것은 아니다. 표현 설명과 실제 영상형의 일치 및 정상 변이와 오류의 정답 근거도 계속 필요하다.',
 '이번에는 새 후보와 처리 부품을 격리된 연구 단위에 구현했다. 운영 웹/TCN/DTW를 교체하지 않았으며, 사용자·멘토 촬영을 접근·학습·튜닝·선택에 쓰지 않았다.',
 '다음은 촬영중 처리 부품을 별도 웹시험 경로에 연결해 실제 종료→피드백 화면 시간을 확인하는 것, 그리고 이번 후보의 회귀 특징·속도 민감도를 새 평가 점수로 튜닝하지 않고 학습/개발자료의 근거와 설계로 해결하는 것이다. 특징 점수 향상을 교육피드백 완성으로 취급하지 않는다.'
 ]
 (U/'RESULT_KO.txt').write_text('\n'.join(lines),encoding='utf8')
 def tab(headers,rs):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in rs)+'</tbody></table>'
 cards='';movies=[]
 for alias in ['F001','F017','F025','F029']:
  r=next(r for r in rows if r['alias']==alias);movies.append(dict(alias=alias,label=r['label'],url=Path(r['source']).as_uri()))
  cards+=f'<a href="fresh/visuals/{alias}/states.jpg"><img src="fresh/visuals/{alias}/states.jpg" alt="{alias} 원본손모양 세시점"></a>'
 body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>두 우선순위 비교 결과</title><style>body{{font:16px/1.8 'Malgun Gothic',sans-serif;background:#f3f6fa;color:#152739;margin:0}}main{{max-width:1120px;margin:auto;padding:34px 24px 80px}}h1{{font-size:30px}}h2{{margin-top:42px;font-size:23px}}p{{max-width:1000px}}.lead{{background:#e4f1f1;border-left:6px solid #007c82;padding:18px 23px}}.warn{{background:#fff0d5;padding:16px 22px;border-radius:10px}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{border:1px solid #cfdae4;padding:9px;text-align:left}}th{{background:#e3eaf2}}.scroll{{overflow-x:auto}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}img{{width:100%;border:1px solid #bbcbd7}}video{{max-width:100%;max-height:500px;background:#111}}select,button{{padding:8px;font:inherit}}small{{color:#526374}}.bar{{height:26px;background:#007c82;color:white;padding-left:10px;box-sizing:border-box;min-width:90px}}@media(max-width:650px){{main{{padding:20px 14px}}.grid{{grid-template-columns:1fr}}h1{{font-size:24px}}}}</style><main><h1>특징 판별을 다시 배우고,<br>촬영 중 미리 처리했습니다</h1><p>2026-10-31 · 공개 한국수어 자료 · 기존 C·T3-F 및 운영 웹 유지</p><div class="lead"><b>관절 결과가 같아도 기다리는 시간은 줄일 수 있었습니다.</b><br>촬영 종료 후 관절 결과까지 대기 중앙값 6.08초 → 1.13초. 아래 정확도는 <b>단어 인식률이 아닌 손모양·이동 등 특징 판별</b>입니다.</div><h2>1. 무엇을 고쳤나</h2><p>왼손 자료가 적다고 왼손만 따로 배우게 두지 않고, 양손에 공통인 손가락 모양을 같은 판정부가 배웁니다. 이동은 구간 가운데 몇 프레임 대신 <b>해당 구간의 모든 원본 프레임</b>을 순서대로 봅니다. 가려진 좌표를 정답인 것처럼 채우지는 않습니다.</p><p>학습 36영상, 개발 12영상, 새 평가 32영상. 추가 판정부 9,990개 파라미터를 세 번 각각 160회 학습했습니다. 기존 관계 판정부까지 합친 구성은 57,904개입니다. C·T3-F와 SignRep 본체는 고정했습니다.</p><h2>2. 다른 촬영자에서 같은 조건으로 비교</h2><p>지표: <b>10특징 평균 균형정확도</b>. 임시 RGB 정답 17후보의981셀 중 이번10특징의636셀(배운표현462·새표현174)을 비교했습니다.</p><div class="scroll">{tab(['조건','배운 12표현 · 새 촬영 ID · 24영상','학습하지 않은 4표현 · 새 촬영 ID · 8영상'],table)}</div><p>세 번 학습의 평균(최소–최대). 기존 관절 판정부 대비 각각 <b>{gain:+.2f}%p / {novelgain:+.2f}%p</b>. 같은 새 영상과 같은 정답 표시로 비교했습니다.</p><div class="warn">이 자료는 과거 TCN 학습에 포함된 공개 영상입니다. 새 특징 판정부에만 새 평가입니다. 정답 표시도 Codex의 영상 검토로 만든 물리 주석이며, 수어 전문가가 보증한 교육적 정오가 아닙니다.</div><h2>3. 어떤 특징이 좋아졌고 나빠졌나</h2><p>아래 특징별 표는 <b>배운 12표현을 새 촬영 ID가 수행한 24영상</b>의 결과입니다.</p><div class="scroll">{tab(['물리 특징','기존 관절','기존 관절+영상','새 후보','기존 관절 대비'],targetrows)}</div><p>두 손 겹침·얼굴 겹침 출력은 기존 관절 판정부를 유지했습니다. 표의 변동은 바뀐 손모양·이동 판정부에서 나옵니다. 새 결과를 보고 좋은 seed나 특징만 골라 교체하지 않았습니다.</p><div class="warn"><b>새 판정부 전체는 미채택입니다.</b> 배운 표현에서는 기존 영상 보조82.01%보다 낮은80.78%였습니다. 손겹침이 표시된28개 손모양 비교도 기존24/25/25개 정답에서23/23/23개로 줄었습니다. 관절을 그대로 유지했다는 것과 손모양 분류까지 좋아졌다는 것은 다릅니다.</div><h2>4. 속도는 프레임을 버리지 않고 개선</h2>'''
 for key,name in [('legacy_double_decode_batch','기존: 종료 후 전체 처리'),('single_decode_batch','중복 영상읽기 제거'),('single_decode_paced','촬영 중 처리: 종료 후 남은 작업')]:
  value=speed[key]['median'];body+=f'<p>{name}</p><div class="bar" style="width:{value/7.3*100:.1f}%">{value:.3f}초</div>'
 body+='''<p><b>같은 원본, 같은 모든 프레임, 같은 관절·좌우 배정</b>을 확인했습니다. 실제18회1884프레임, 새조건12비교쌍과 기존관측18건 모두 일치했습니다.</p><div class="warn">위 시간은 관절 추출·좌우 배정·저장까지입니다. 최종 피드백까지1.13초라는 뜻은 아닙니다. SignRep/판정부/DTW/화면은 별도이며, 실제 카메라 대신 공개파일을 촬영속도로 공급했습니다. GPU MediaPipe는 현재 Windows빌드가 지원하지 않았고, SignRep는 기존 Intel GPU를 유지했습니다.</div><h2>5. 원본 손과 교차를 직접 확인</h2><p>아래 이미지는 새 평가 자료의 원본 세 시점입니다. 클릭하면 크게 볼 수 있습니다. 접촉 여부나 수어의 옳고 그름을 표시한 정답 그림은 아닙니다.</p>'''+f'<div class="grid">{cards}</div><p><label for="clip">원본 영상 </label><select id="clip"></select></p><video id="video" controls preload="metadata"></video><p id="caption"></p>'
 body+='''<h2>6. 남은 문제와 다음 단계</h2><p>숫자를 낸다고 근거가 있는 것은 아닙니다. 손이 안 보인 구간의 비교 범위를 줄이지 않고 ‘비교 근거 부족’으로 둡니다. 정상적인 손내림을 곧바로 잘못된 수어라고 지적하지 않습니다.</p><p>속도만 바꾼 개발 시험에서도 일부 방향판정이 달라졌습니다. 이번 구간은 원본 단어 구간을 균등하게 나눈 것이므로, 필수 동작 단계나 누락을 스스로 정확히 찾았다는 뜻도 아닙니다. 위치·모양·움직임의 차이를 교육적 교정으로 바꾸려면 그 차이가 실제 오류인지 확인할 근거가 더 필요합니다.</p><p>다음은 촬영 중 처리 부품을 별도 웹시험 경로에 연결해 <b>실제 종료부터 피드백 화면까지</b> 측정하고, 판정부의 회귀 항목과 속도 민감도를 학습·개발 근거로 보완하는 것입니다. 현재 운영 모델은 보존했습니다.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="FRESH_SCORES.json">정확도와 특징별 분모</a> · <a href="streaming/RESULT.json">속도 원자료</a> · <a href="fresh/LABEL_AUDIT.json">RGB 주석 이견·보류</a> · <a href="review/INPUT_SPEED_INDEPENDENT.json">별도 입력·속도 검산</a></p><small>구현·영상 주석·별도 검산: Codex 및 분담 에이전트. 독립된 사람 또는 한국수어 교육자의 검증과 구분합니다. 사용자·멘토 자료, 새 다운로드, 카메라, 외부 전송, Git 사용 없음.</small></main><script>const movies=DATA;const s=document.querySelector('#clip'),v=document.querySelector('#video');for(const r of movies){let o=document.createElement('option');o.value=r.alias;o.textContent=r.alias+' · '+r.label;s.append(o)}function change(){let r=movies.find(r=>r.alias===s.value);v.src=r.url;document.querySelector('#caption').textContent=r.alias+' · 공개 원본 전체 영상. 촬영자의 속도와 손내림도 그대로 보존합니다.'}s.onchange=change;change();</script></html>'''.replace('const movies=DATA;', 'const movies='+json.dumps(movies,ensure_ascii=False)+';')
 (U/'RESULT_KO.html').write_text(body,encoding='utf8')
 report_data=dict(table=table,target_table=targetrows,known_delta_pp=gain,expression_delta_pp=novelgain,speed_reduction_pct=delta,conditions=CONDS,limitations=['provisional_Codex_labels','TCN_TRAIN_history','uniform_not_semantic_intervals','speed_sensitive','capture_not_end_to_end'],crossing=di['rows'])
  # Report prose/layout may be revised; numerical results are never rescored.

 (U/'REPORT_DATA.json').write_text(json.dumps(report_data,ensure_ascii=False,indent=2),encoding='utf8')
 print('report written',gain,novelgain)
if __name__=='__main__':main()
