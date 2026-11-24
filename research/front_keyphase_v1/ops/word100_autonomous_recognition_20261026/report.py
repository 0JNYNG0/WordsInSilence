"""Report fixed comparisons without changing model selection or inputs."""
from common import *
import math,html,re

def paired_uncertainty(pair):
 g=len(pair['gains']);l=len(pair['losses']);n=g+l
 p=min(1.,2*sum(math.comb(n,k) for k in range(min(g,l)+1))/2**n) if n else 1.
 return dict(gains=g,losses=l,discordant=n,nominal_exact_two_sided_p=p,interpretation='Descriptive paired-binomial only; repeated signers/words are correlated and DEV was reused for development. Not confirmatory independent-person significance.')

def main():
 f=U/'fresh199';fresh=read(f/'COMPARISON.json');verify=read(f/'SCORES_VERIFIED.json');assert verify['status']=='PASS'
 for p,h in verify['pins'].items():assert sha(p)==h,p
 frozen=read(U/'FROZEN.json')
 for p,h in frozen['protected'].items():assert sha(p)==h,p
 dev=read(U/'SAVED_COMPARISON.json');timing=read(U/'TIMING.json');ctx=read(f/'EVAL_CONTEXT.json');byid={r['clip_id']:r for r in ctx['rows']}
 evidence=read(f/'INPUTS_VERIFIED.json');groups={};failures={};known=dev['scores']['known200']
 individual=read(f/'T3F_20261013_SCORES.json')['predictions'];c=read(f/'C_20261013_SCORES.json')['predictions'];ensemble=read(f/'ENSEMBLE_ROWS.json')
 for view in VIEWS:
  groups[view]={};bb={r['clip_id']:r for r in individual[view]};ee={r['clip_id']:r for r in ensemble[view]}
  for group in sorted(set(evidence['groups'][view].values())):
   ids=[cid for cid,g in evidence['groups'][view].items() if g==group]
   groups[view][group]=dict(n=len(ids),T3F13_correct=sum(bb[i]['correct'] for i in ids),E3_correct=sum(ee[i]['correct'] for i in ids))
  failures[view]=[dict(clip_id=i,label=byid[i]['canonical_label'],model_prediction_index=ee[i]['prediction']) for i in sorted(ee) if not ee[i]['correct']]
 checks=fresh['checks'];success=fresh['scores']['sensor']['E3']['correct']>fresh['scores']['sensor']['T3F13']['correct'] and checks['masked']['top1_nondecreasing'] and all(checks[v]['top5_nondecreasing'] for v in VIEWS)
 delta={v:paired_uncertainty(fresh['scores'][v]['E3_vs_T3F13']) for v in VIEWS}
 status='DEV_GAIN_AND_FRESH_GAIN_VERIFIED_NOT_DEPLOYED' if success else 'DEV_GAIN_VERIFIED_FRESH_ADDITIONAL_GAIN_NOT_DEMONSTRATED_NOT_DEPLOYED'
 result=dict(at=now(),status=status,fresh_frozen_success=success,development=dev['scores']['dev'],known_exposed200=known,fresh199=fresh['scores'],fresh_subgroups=groups,remaining_fresh_errors=failures,paired_uncertainty=delta,limits=['199 videos represent2recordedIDs, not verified independent people.','Only existing100word vocabulary, not allKSL words.','DEV reused for development; fresh candidate fixed before predictions; no postscore tuning.','Synthetic sensor crop and mask views are not real webcam recordings.','SameCodex separate numericalverification, not independent expert groundtruth.','Ensemble does not recover fingers, solve10HaMeRconflicts, or validate DTW feedback correctness.'],protected_files=len(frozen['protected']),training=0,user_data_access=0,production_changed=False,next='KeepE3 as isolated research candidate. Report fresh gain/regression honestly; no weight selection on199. Before product adoption compare frozen E3/T3F on new prospectively fixed webcam-style public evaluation or explicitly authorized tester-only paired review. Hand-recovery needs reliable currentRGB anatomical evidence first.')
 write(U/'RESULT.json',result)
 def score(m):return f"{m['correct']}/{m['n']} ({m['accuracy']:.2f}%)"
 rows=[]
 for label,data in [('개발 검증 300영상',result['development']),('앞서 이미 확인한 200영상',known),('이번 추가 공개 검증 199영상',fresh['scores'])]:
  for view in VIEWS:
   d=data[view];rows.append([label,'화면 밖 미검출 재현' if view=='sensor' else '전체 검출 후 상반신 제한',score(d['C13']),score(d['T3F13']),score(d['E3']),f"{d['E3_vs_T3F13']['delta_pp']:+.2f}%p",f"{len(d['E3_vs_T3F13']['gains'])} / {len(d['E3_vs_T3F13']['losses'])}"])
 title='개발 자료에서 인식 개선 확인, 추가 공개 자료 비교 완료'
 verdict='이번 추가 공개199영상에서도 미리 정한 개선 조건을 통과했다.' if success else '개발 자료에서는 개선됐지만, 이번 추가 공개199영상에서 추가 인식률 개선 조건은 충족하지 못했다. 기존 모델이 이미 모두 맞힌 경우에는 더 좋아졌다고 표현하지 않는다.'
 lines=[title,'',verdict,'','무엇을 바꿨나','기존 T3-F 모델3개의 단어별 확률을 같은 비중으로 평균한 E3 후보1개를 만들었다. 새 학습·손좌표 변경·기존 모델 교체는 하지 않았다. 단어마다 유리한 모델을 고르거나 새 평가 결과에 맞춰 비중을 바꾸지 않았다.','', '결과(같은 촬영·같은 입력으로 비교, 기준 T3F13)']
 lines += [' | '.join(r) for r in rows]
 lines += ['', '평가에서 발견한 중요한 문제','새 평가 대상으로 남겨 둔 REAL16 자료 중4영상이 과거 DEV REAL11 영상과 SHA256까지 같았다. 새 점수 확인 전에 해당ID전체100을 격리했고, 원래 영상연결이 확정되지 않았던1개를 제외하여 REAL14/15의199개로 고정했다. 16번ID가 같은 실제 사람이라는 뜻은 아니며, 제공ID만으로 실제 독립된 사람을 보장하지 않는다. 얼굴을 비교하여 동일인 판정하지 않았다.','', '보존과 남은 문제','관절 입력은 그대로이므로 정상 손·교차 좌표를 새로 지우거나 교체하지 않았다. 이는 모든 인식 결과가 좋아진다는 보장은 아니다. 새 평가의 개선·퇴보·손 재등장 그룹을 별도 집계했다.','손목 평가기준은 보이는 연결부5장면, 흐린8장면, 부분가림3장면으로 근거를 구분했다. 기존16장면의 대략적범위를 다시 분류한 것으로 새 정밀정답을 확보한 것은 아니다. 후원10프레임의 충돌과 가려진 손가락 복원은 미해결이다. E3 인식 개선을 손복원이나DTW교정정확도 향상으로 해석하면 안 된다.','', '속도','이 노트북 CPU4스레드에서 순수 모델 계산 중앙값 T3F13약2.30ms→E3약6.78ms였다. 입력영상 검출·DTW·웹처리시간은 빠져 있어 실제 서비스FPS가 아니다.','', '검증의 한계','개발300자료는 여러 번 개발에 사용됐다. 신규199는 로컬 노출감사를 거쳐 조건을 먼저 고정했으나, 같은 데이터셋 촬영 환경과 기록ID2개에 한정된다. 모든 수어·새로운 웹캠 사용자 성능을 입증하지 않는다. 독립 구현 산술검산은 같은 Codex가 작성했으며 독립 전문가의 손가락 정답 검토가 아니다.','', '다음 작업','이번 E3는 연구 후보로 보존하고 C·T3-F와 운영 웹을 유지한다. 새로운 웹캠 조건의 공개 평가 또는 사용자가 명시적으로 요청한 동일촬영 비교에서 E3를 고정한 채 검증한 후 적용 여부를 결정한다. 이번199점수를 보고 재학습·가중치선택하지 않는다. 손 보완은 정확히 보이는 위치 근거를 확보한 뒤 재개한다.']
 for v in VIEWS:
  lines+=['',v+' 기록ID별 결과: '+json.dumps(fresh['scores'][v]['E3']['by_signer'],ensure_ascii=False),v+' 손 관측별 결과: '+json.dumps(groups[v],ensure_ascii=False),v+' 남은 오답: '+json.dumps(failures[v],ensure_ascii=False),v+' 짝비교 불확실성: '+json.dumps(delta[v],ensure_ascii=False)]
 (U/'RESULT_KO.txt').write_text('\n'.join(lines),encoding='utf8')
 esc=html.escape
 table='<table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in ['자료','관측 조건','C','기존T3-F','E3','차이','개선/퇴보'])+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
 sections=''
 for head,body in [('쉽게 말하면','한 모델이 답을 고르던 방식에, 같은 방식으로 배운 세 모델의 의견을 동일하게 합쳤다. 손을 새로 만들어 넣는 기능이 아니라 단어 판단을 합치는 기능이다.'),('평가 자료의 중복을 먼저 걸러냈다',lines[lines.index('평가에서 발견한 중요한 문제')+1]),('손 검출 문제는 별도로 남아 있다',lines[lines.index('보존과 남은 문제')+2]),('검증 범위와 다음 단계',lines[lines.index('검증의 한계')+1]+' '+lines[lines.index('다음 작업')+1])]:sections+=f'<section><h2>{esc(head)}</h2><p>{esc(body)}</p></section>'
 source='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E3 공개 인식 검증</title><style>body{margin:0;background:#f2f5fa;color:#17273f;font-family:Segoe UI,Malgun Gothic,sans-serif;line-height:1.7}main{max-width:1250px;margin:40px auto;padding:32px;background:white;border-radius:20px}h1{font-size:32px;line-height:1.35}h2{font-size:21px;margin-top:0}.tag{color:#3267a1;font-weight:700}.lead{padding:20px;background:#eaf3ff;border-left:5px solid #2871ce}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:14px;text-align:left;border-bottom:1px solid #dce3ed}th{background:#eaf0f7}section{margin-top:24px;padding:24px;background:#f7f9fc;border-radius:12px}a{color:#2869ba}details{margin-top:18px}pre{white-space:pre-wrap;font-size:13px}footer{margin-top:24px;color:#516175}</style><main><p class="tag">WordsInSilence · 2026.10.26 · 연구 후보 / 운영 미적용</p><h1>'+esc(title)+'</h1><p class="lead">'+esc(verdict)+'</p>'+sections[:sections.find('<section><h2>평가')]+ '<h2>같은 입력에서 비교한 단어 인식률</h2><div class="scroll">'+table+'</div><p>정답 단어가 1위인 비율이다. 동작을 얼마나 올바르게 수행했는지의 점수가 아니다. %p는 두 정확도 비율의 차이다.</p>'+sections[sections.find('<section><h2>평가'):]+ '<section><h2>계산 비용</h2><p>순수 모델 중앙값 2.30ms → 6.78ms. 영상 검출·DTW·웹 시간은 제외.</p></section><details><summary>촬영자 ID·손 관측 그룹·오답 상세</summary><pre>'+esc(json.dumps({'by_id':{v:fresh['scores'][v]['E3']['by_signer'] for v in VIEWS},'groups':groups,'errors':failures,'paired_uncertainty':delta},ensure_ascii=False,indent=2))+'</pre></details><footer><a href="RESULT_KO.txt">전체 설명</a> · <a href="RESULT.json">결과 수치</a> · <a href="fresh199/SCORES_VERIFIED.json">별도 수치 검산</a> · <a href="DUPLICATE_AUDIT.json">자료 중복 감사</a></footer></main></html>'
 (U/'RESULT_KO.html').write_text(source,encoding='utf8')
 assert source.count('<section>')==source.count('</section>') and source.count('<tr>')==7
 for link in re.findall(r'href="([^"]+)"',source):assert (U/link).exists(),link
 write(U/'REPORT_STATIC_QA.json',dict(status='PASS_STATIC_ONLY',html_balanced_sections=True,result_table_rows=6,all_local_links_exist=True,browser_render_verified=False,reason='Earlier file browser policy not bypassed; static QA only.'))
 write(U/'COMPLETION.json',dict(at=now(),status=status,source_result=sha(U/'RESULT.json'),score_verifier=sha(f/'SCORES_VERIFIED.json'),report=sha(U/'RESULT_KO.html'),training=0,production_changed=False,protected_files_unchanged=len(frozen['protected']),pending='Human-readable final report and root context synchronization'))
 print(status,flush=True)
if __name__=='__main__':main()
