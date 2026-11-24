"""Separate arithmetic review, report, and status sync for the planning task only."""
from review import U,O,ROOT,BASE,read,sha,save,now
import math,html
from collections import defaultdict

def main():
 a=read(U/'ANALYSIS.json');p=read(U/'PLAN.json');rows=read(U/'TRAIN_TIMING_ROWS.json')
 for path,h in a['pins'].items():assert sha(path)==h,path
 assert len(rows)==3596 and len({r['clip_id'] for r in rows})==899
 groups=defaultdict(list)
 for r in rows:groups[r['view']+'/'+r['temporal']].append(r)
 for k,rr in groups.items():
  lengths=sorted(r['n'] for r in rr);s=a['summary'][k]
  assert len(rr)==899 and lengths[449]==s['length_frames']['median'] and min(lengths)==s['length_frames']['min'] and max(lengths)==s['length_frames']['max']
  assert sum(r['n']>61 for r in rr)==s['longer_than_nominal_conv_RF61']
  assert all(all(m>0 for m in r['effective_phase_frames']) for r in rr)
  assert all(abs(sum(r['effective_phase_frames'])-r['n'])<1e-9 for r in rr)
 # Independent scalar reference for triangular basis and synthetic example.
 def one(u):return ([1-2*u,2*u,0] if u<=.5 else [0,2-2*u,2*u-1])
 h=[[1,0,0]]*3+[[0,1,0]]*3+[[0,0,1]]*3
 w=[one(i/8) for i in range(9)];g=[sum(r[j] for r in h)/9 for j in range(3)]
 d=[[sum(w[i][k]*h[i][j] for i in range(9))/sum(r[k] for r in w)-g[j] for j in range(3)] for k in range(3)]
 assert all(abs(d[k][j]-a['synthetic_example']['forward_order_contrasts'][k][j])<1e-12 for k in range(3) for j in range(3))
 original=(280*32+32)+8*(32*32*3+32)+8*2*32+(32*100+100)
 assert original==37636 and original+3*32*100==p['architecture']['parameters']['T3']
 assert p['training_authorized_now'] is False
 save('VERIFICATION.json',{'at':now(),'status':'PASS_PLAN_ARITHMETIC_AND_PROTECTED_PINS','operator':'same Codex separate scalar arithmetic, no independent agent/expert','rows':3596,'clips':899,'scalar_reference_agrees':True,'parameter_arithmetic':{'C':original,'T3':original+9600},'synthetic_checks':9,'original_model_hashes_preserved':15,'new_model_inference':0,'training':0,'user_mentor_access':False,'pins':{str(U/n):sha(U/n) for n in ['ANALYSIS.json','TRAIN_TIMING_ROWS.json','PLAN.json','review.py']}})
 facts=[
 '추천: 기존 C의 전체 요약을 그대로 두고, 시간상 앞·중간·뒤의 특징을 부드럽게 요약해 추가하는 T3 후보 한 가지를 먼저 검증합니다. 지금은 검토와 계획만 완료했으며 추가 학습·모델 추론·운영 교체는 하지 않았습니다.',
 '현재 TCN도 시간 순서를 학습합니다. 코드에서 마지막 시간별32차원 특징을 전체 평균 하나로 합치는 구조를 확인했습니다. 앞에 나온 모양과 뒤에 나온 모양을 구분하는 정보가 추가 요약으로 도움이 될 수 있다는 가설이며, 오답 원인으로 확정하지 않습니다.',
 '공개 자료의 기존 검토에서 백육십이–백칠십팔은 뒤쪽 손 모양 변화, 흡연실–감히는 후반 경로와 손 모양, 상처–믿다는 교대·반복 흐름이 구분 단서였습니다. 기존 표본 검토를 재사용한 것이며, 수어 전문가의 정오 판정이나 순서 때문이라는 인과 증거는 아닙니다.',
 '공개 TRAIN899개의 기존 두 관측조건×전체/핵심 구간3596시계열만 새로 점검했습니다. 길이는31~138프레임, 세 요약 영역이 비는 경우0건입니다. 이는 각 영역에 정보가 일부 있다는 뜻이며 모든 프레임의 손 검출이 정확하다는 뜻은 아닙니다.',
 '앞·중간·뒤는 수어의 의미 단계를 자동 판정한 이름이 아닙니다. 현재 남긴 동작 구간의 상대적인 시간 위치입니다. 중간에 잠깐 손이 안 보여도 실제 시간과 관측 마스크를 그대로 유지하며, 구간을 삭제하거나 손을 만들어 채우지 않습니다.',
 '딱3등분해서 자르는 대신 앞·중간·뒤를 중심으로 부드럽게 겹치는 가중치를 사용합니다. 작은 시간 차이로 요약 구간이 갑자기 바뀌는 문제를 줄이기 위한 설계입니다. 손이 올라오는 시점이나 단계별 유지 시간이 크게 달라지는 문제까지 해결하지는 않습니다.',
 '전체 요약32개에 세 구간의 차이96개를 추가합니다. 모델 크기는37,636→47,236파라미터(+9,600,+25.5%)입니다. 추가 분류 가중치는0으로 시작해 처음에는 같은 초기값 C와 같은 출력을 내도록 검증합니다. 기존 C를 덮어쓰거나 이어 학습하지 않습니다.',
 '승인 후 순서: ① 별도 후보 구현·초기출력/패딩/미검출/기울기 검증 ② 입력·조건 고정 ③ 같은899학습/300개발 자료로3초기값 비교 ④ 저장점수·실제자료노출·선택모델 검산 및 보고. 기존 C의 저장3회 결과를 재사용합니다.',
 '균등 배치는 보류하므로 학습 묶음은 원래 C의32×28+3으로 돌아갑니다. 입력·정규화·관측조건·학습률·max80/patience12·선택규칙은 유지합니다. 단계 수나 가중치 위치를 DEV오답에 맞춰 바꾸지 않습니다.',
 '평가: 전체300개 정확도 평균/범위·상위5위·촬영자별·한손/재등장 그룹별 회귀·후반61~80회 흔들림을 확인합니다. known오답 몇 개만 좋아졌다고 채택하지 않습니다. 주정확도평균과3회중2회가 개선되고 보호그룹 회귀가 없어야 다음검토로 넘기며 자동운영교체는 하지 않습니다.',
 '한계: 미세 관절 오검출이나 손 가림을 복구하는 방법은 아닙니다. 추가 파라미터의 효과도 섞이므로 좋아져도 순서보존만의 효과라고 단정하지 않습니다. 실제 새 사용자 성능과 독립 최종 검증은 별도입니다.',
 '현재 범위에서는 학습0·모델추론0·새영상추출0·사용자/멘토자료접근0. 기존 C·웹·DTW 보존, 구현 전 계획 확인 대기입니다.'
 ]
 text='동작 단계·순서 보완 검토 및 다음 비교 계획\n\n'+'\n\n'.join(facts)+'\n\n참고1: https://arxiv.org/abs/1503.01224\n참고2: https://arxiv.org/abs/1803.01271\n'
 (U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
 def paras(ix):return ''.join('<p>'+html.escape(facts[i])+'</p>' for i in ix)
 doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>동작 단계·순서 보완 계획</title><style>body{margin:0;background:#eff3f6;color:#172c3a;font:17px/1.75 "Malgun Gothic",sans-serif}main{max-width:1050px;margin:auto;padding:38px 24px}h1{font-size:34px;line-height:1.4}h2{font-size:24px}section{background:white;margin:24px 0;padding:28px;border-radius:16px}.tag{color:#18786d;font-weight:700}.flow{display:flex;gap:15px;align-items:stretch;flex-wrap:wrap}.box{flex:1;min-width:210px;background:#eaf4f2;border-radius:12px;padding:18px}.tiny{font-size:14px}a{color:#146e65}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:12px;border-bottom:1px solid #dde6eb;text-align:left}</style><main><div class="tag">WORDS IN SILENCE · 계획 검토 완료 · 추가 학습 없음</div><h1>전체 동작은 유지하고,<br>앞·중간·뒤의 차이를 더 잘 남기기</h1>'''
 doc+='<section><h2>추천하는 한 가지 후보</h2>'+paras([0,1])+'<div class="flow"><div class="box"><b>기존 C</b><br>관절 시계열 → TCN<br>→ 전체 요약 → 단어</div><div class="box"><b>T3 후보</b><br>같은 관절 시계열 → 같은 TCN<br>→ 전체 요약 + 앞·중간·뒤 요약 → 단어</div></div><p>예를 들어 A→B와 B→A는 나타나는 모양은 같아도 순서가 다릅니다. 인공 특징으로는 전체 평균이 같고 세 구간 요약은 달라짐을 확인했습니다. 실제 C가 역순 동작을 구분하지 못한다는 실험은 아닙니다.</p></section>'
 doc+='<section><h2>공개 자료와 코드에서 확인한 근거</h2>'+paras([2,3])+'<p>시간별 합성곱의 명목상 관찰 범위는61프레임입니다. 실제 전체 입력 중 일부는 더 길지만, 전체 평균은 모든 프레임을 보므로 이것을 모델의61프레임 제한이라고 해석하지 않습니다.</p></section>'
 doc+='<section><h2>잘라 버리지 않고 부드럽게 요약</h2>'+paras([4,5,6])+'<p>수학 점검9항목: 순서 대비, 패딩 제외, 시간 원점 이동, 균일 시간 배율에서 가중치 유지, 일정 특징, 짧은 입력, 전체 패딩, 미검출 프레임 시간 보존. 모델 전체의 속도 불변성이나 인식률 향상을 증명한 검사는 아닙니다.</p></section>'
 doc+='<section><h2>승인 후 구체적인 진행 순서</h2>'+paras([7,8,9])+'<table><tr><th>항목</th><th>고정 조건</th></tr><tr><td>자료</td><td>공개 TRAIN899 / DEV300 · 사용자·멘토·보류촬영자 제외</td></tr><tr><td>새 학습</td><td>T3 3회 · 초기값20261013/14/15 · 기존C 저장3회와 비교</td></tr><tr><td>기존 입력</td><td>280채널, 전체/핵심 및 두 관측조건 일정 유지</td></tr><tr><td>배치·학습</td><td>원래32×28+3,29단계 · lr0.001 · 최대80회 · patience12</td></tr><tr><td>모델 선택</td><td>기존과 같은 양쪽DEV 평균 손실 최소 · 최고정확도 회차로 재선택 금지</td></tr></table></section>'
 doc+='<section><h2>기대할 수 있는 것과 남은 한계</h2>'+paras([10,11])+'</section>'
 doc+='<section><h2>검토 근거</h2><p>시간 구간별 요약은 <a href="https://arxiv.org/abs/1503.01224">Temporal Pyramid Pooling 원 논문</a>에 관련 선례가 있습니다. 논문의 입력과 과제는 우리와 다르므로 우리 성능의 보장은 아닙니다. <a href="https://arxiv.org/abs/1803.01271">TCN 원 연구</a>도 합성곱의 시계열 모델링 능력을 다룹니다. 현재 코드의 구조는 로컬에서 직접 확인했습니다.</p><p><a href="PLAN.json">구체 알고리즘·비교 계획</a> · <a href="ANALYSIS.json">공개 학습 입력 점검</a> · <a href="VERIFICATION.json">별도 산술 검산</a> · <a href="RESULT_KO.txt">텍스트 보고서</a></p><p class="tiny">작성·점검: Codex. 독립 에이전트나 수어 전문가 검토를 받았다는 뜻은 아닙니다.</p></section></main></html>'
 (U/'RESULT_KO.html').write_text(doc,encoding='utf-8')
 at=now();key=U.name
 status='COMPLETE_PLAN_ONLY_AWAITING_USER_REVIEW'
 summary='ops/'+key+'. 균등배치보류·C유지. 현TCN시간순서학습+마지막masked globalmean 구조확인. 공개TRAIN899×2관측×2구간=3596시계열(1798파일) 시간범위검산, 길이31~138·빈phase0. T3제안: 기존global32 + 시간정규화 삼각가중 앞/중간/뒤 대비96, zero-init bias없는추가head, 37636→47236(+9600,+25.5%). 수학9검사/별도scalar검산PASS이며 성능증거아님. 상대시간phase≠의미단계, 손미검출복원없음. 기존공개혼동검토재사용. 승인후동일C조건으로새3seed vs저장C3회 계획; 원래32×28+3배치, 입력/정규화/학습률/CE선택유지. 추가파라미터효과와순서효과분리확정불가. 학습0/모델추론0/새추출0/사용자·멘토접근0/운영변경0. PLAN·ANALYSIS·VERIFICATION·RESULT_KO.html/txt.'
 nxt='T3시간단계요약 후보의 구체계획을 사용자와 확인한다. 승인 전 구현·추가학습·운영교체 없음. 승인 시 별도모듈 초기C출력동등성·패딩/미검출/기울기 검증→동일899/300입력·조건동결→3seed학습과저장C비교. 균등배치보류, C/웹/DTW유지, 사용자/멘토자료제외.'
 path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=path.read_text(encoding='utf-8-sig')
 for heading,title,body in [('## 8. 현재 상태','2026-10-19 단계·순서 보완 계획 검토 완료 — 학습 없음',summary),('## 9. 다음 단일 작업','2026-10-19 T3 단계요약 계획 확인',nxt)]:
  pos=t.index('\n',t.index(heading));t=t[:pos+1]+'\n### '+title+'\n'+body+'\n\n'+t[pos+1:]
 path.write_text(t,encoding='utf-8')
 state=read(O/'STATE.json');state.update(current_stage='WORD100_TEMPORAL_PHASE_PLAN_COMPLETE',activity_state='AWAITING_USER_PLAN_REVIEW',updated_date_kst=at,next_single_action=nxt)
 state[key]={'status':status,'at':at,'unit':key,'public_train_clips':899,'timing_sequences_checked':3596,'new_training':0,'model_inference':0,'production_changed':False,'user_mentor_data_used':False,'training_allowed':False,'plan':str(U/'PLAN.json'),'report':str(U/'RESULT_KO.html'),'candidate_parameters':47236,'next_single_action':nxt}
 (O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 handoff=f'''# WordsInSilence 최신 인수인계

갱신: {at}

현재 완료: {key}. 사용자 요청은 균등 배치 보류 후 동작 단계·순서 보완 검토/구체계획이며 학습승인은 아직 없음.

- 기존 C·웹·DTW 유지. 균등배치 평균98.67→98.56%, 불일관안정화 결과는 이전완료단위에 보존. 새학습0/모델추론0/MediaPipe0/새추출0/사용자·멘토접근0/다운로드0/외부자료전송0/Git0.
- C코드는 시계열합성곱 뒤 시간별32차원을globalmean으로요약. 순서를전혀모르는모델아님. nominal conv RF61이전체모델시간제한아님.
- 공개TRAIN899의1798개파일/3596시계열만새점검: 길이31~138, 빈softphase0, dt로시간위치복원가중치오차max2.17e-9. phase내손일부관측있음이전프레임정확검출뜻아님.
- 제안T3: 같은C입력·trunk·globalhead유지 + 상대시간0/.5/1 삼각가중요약3개에서global을뺀96차원추가. zero-init bias없는head9600개 추가, 전체47236(+25.5%). FreshC와초기출력/RNG동등성은실제후보구현후검증대상. 지금은수학prototype만,모델구현/추론없음.
- 수학9체크 및 별도scalar/파라미터산술PASS. 같은Codex검산,독립사람/agent검토아님. 공개반복혼동RGB표본검토는이전 VISUAL_REVIEW 재사용. 최신별도영상검토아님.
- 승인후계획: 격리후보구현·초기출력동등성/패딩/미검출/기울기검증→동결→3seed신규 vs기존C저장3회. 입력899/DEV300·원래32×28+3배치·lr0.001·80/patience12·CE선택동일. 성능/촬영자/관측그룹회귀·후반흔들림보고. 단어별분기/DEV로phase선택/사용자자료/원본변경금지.
- 의미단계자동분할아님. 머무는시간·반복·시작/끝차이에민감할수있음. 손누락복원아님. 추가head크기효과와순서효과를이비교만으로분리확정불가. 논문은설계선례일뿐KSL성능보장아님.
- PLAN.json, ANALYSIS.json, TRAIN_TIMING_ROWS.json, VERIFICATION.json, RESULT_KO.html/txt, FINAL_AUDIT.json. 실행중학습/잠금없음.

다음 한 작업: {nxt}
'''
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
 save('FINAL_AUDIT.json',{'at':now(),'status':status,'pins':{str(U/n):sha(U/n) for n in ['SCOPE.json','ANALYSIS.json','PLAN.json','VERIFICATION.json','RESULT_KO.html','RESULT_KO.txt','review.py','finish.py']},'root_document_pins':{str(q):sha(q) for q in [ROOT/'NEW_CHAT_HANDOFF.md',ROOT/'WordsInSilence_PROJECT_CONTEXT.md',O/'STATE.json']},'protected_models_unchanged':15,'new_training':0,'new_inference':0,'user_mentor_access':False,'production_changed':False,'next_single_action':nxt})
 assert read(O/'STATE.json')[key]['training_allowed'] is False
 print(json.dumps({'status':status,'report':str(U/'RESULT_KO.html'),'new_training':0,'new_inference':0},ensure_ascii=False))
if __name__=='__main__':main()
