"""Human-readable closeout after reviewing saved comparison and verifiers. No inference."""
from common import *
import html,shutil,re

def main():
 f=U/'fresh199';r=read(U/'RESULT.json');c=read(f/'EVAL_CONTEXT.json');v=read(f/'INPUTS_VERIFIED.json');s=read(f/'SCORES_VERIFIED.json');frozen=read(U/'FROZEN.json');at=now()
 assert s['status']==v['status']=='PASS' and v['records']==199
 for pins in [s['pins'],frozen['protected'],read(U/'PIPELINE_FROZEN.json')['pins']]:
  for p,h in pins.items():assert sha(p)==h,p
 assert not (U/'AUTONOMOUS_DRIVER_LOCK.json').exists() and not (f/'PREP_RUNNING.json').exists() and not (f/'EVAL_RUNNING.json').exists()
 assert read(U/'DRIVER_STATUS.json')['stage']=='COMPLETE_PENDING_REPORT_REVIEW_AND_HANDOFF'
 rows={x['clip_id']:x for x in c['rows']};labels=c['labels']; details={}
 for view in VIEWS:
  pair=r['fresh199'][view]['E3_vs_T3F13'];err=r['remaining_fresh_errors'][view]
  details[view]={'gains':[rows[i]['canonical_label'] for i in pair['gains']],'new_errors':[rows[i]['canonical_label'] for i in pair['losses']],'remaining':[{'target':x['label'],'predicted':labels[x['model_prediction_index']]} for x in err]}
 paragraph='''추가 공개199개에서 기존 T3-F의 오답2개를 맞혔지만, 전에 맞혔던1개를 새로 틀렸다. 따라서 순이득은1개(+0.50%p)이며, 모든 단어가 좋아진 결과는 아니다. 화면 밖 미검출 재현 조건에서 새로 맞힌 것은 ‘진도군수어통역센터’, ‘육백만’이다. ‘성남시수어통역센터’를 ‘전주시수어통역센터’로 고른 것은 새 회귀이고, ‘팔만’을 ‘팔십오만’으로 고른 것은 남은 오류다.
한 손이 계속 관측되지 않은 그룹은64/66→65/66, 손이 다시 나타나는 그룹은21/22→22/22였지만, 나머지 그룹은111/111→110/111이었다. 관측 패턴에 따른 분류이며 수어 의미상의 한손·반복 정답 주석은 아니다.
개발 개선3건 또는5건은 모두 기록ID12에, 새 평가 순개선1건은 기록ID14에 몰렸다. 별도199개에서도 수치상 개선은 이어졌지만 범위가 작고 실제 독립된 사람의 수가 확인된 것은 아니다. 짝비교3건(개선2/퇴보1)의 명목상 양측 p=1.0이며, 이 자료는 사람·단어가 반복되어 단순 독립표본 검정으로 우월성을 입증할 수 없다.
개별 T3-F3모델도 모두 공개한다. 새 평가 sensor는196/194/197개, masked는196/195/196개였다. E3의 sensor197개는 가장 높은 개별 모델과 동률이다. 이번 새 평가를 본 뒤 유리한seed로 갈아타거나 가중치를 바꾸지 않았다. 세 모델은 같은 학습자료를 사용해 오류가 겹칠 수 있다.
결론: 연구 후보로 남길 근거는 생겼지만, 운영 기본 교체나 손가락 복원·DTW 피드백 개선을 주장할 단계는 아니다. 추가 단어에도 적용 가능한 합산 원리지만 지금 검증한 어휘는100개뿐이다.'''
 generated=U/'generated_report';generated.mkdir(exist_ok=False)
 for name in ['RESULT_KO.html','RESULT_KO.txt']:shutil.copy2(U/name,generated/name)
 h=(U/'RESULT_KO.html').read_text('utf8');h=h.replace('<h2>같은 입력에서 비교한 단어 인식률</h2>','<section><h2>소폭 개선과 새 오류를 함께 확인했다</h2>'+''.join('<p>'+html.escape(p)+'</p>' for p in paragraph.splitlines())+'</section><h2>같은 입력에서 비교한 단어 인식률</h2>',1)
 (U/'RESULT_KO.html').write_text(h,encoding='utf8');text=(U/'RESULT_KO.txt').read_text('utf8');(U/'RESULT_KO.txt').write_text(text+'\n\n최종 검토 — 소폭 개선과 새 회귀\n'+paragraph+'\n',encoding='utf8')
 assert h.count('<section>')==h.count('</section>') and h.count('<tr>')==7
 for link in re.findall(r'href="([^"]+)"',h):assert (U/link).exists()
 review=dict(at=at,status='VERIFIED_SMALL_GAIN_WITH_ONE_REGRESSION_RESEARCH_ONLY',details=details,inputs={'records':199,'matched_channel_values':v['matched_channel_values'],'hidden_points_perturbed':v['hidden_points_perturbed'],'failures':v['input_failures']},score_recount={'raw_rows':s['raw_prediction_rows'],'ensemble_rows':s['ensemble_rows']},protected_files_unchanged=len(frozen['protected']),prep_exit_code=0,driver_exit_code=0,source_result_pin=sha(U/'RESULT.json'),report_pins={str(U/n):sha(U/n) for n in ['RESULT_KO.html','RESULT_KO.txt']},static_html_qa=True,browser_render_verified=False,reviewer='Codex, separate arithmetic implementation; no independent expert groundtruth',same_new_public_cohort_reused_for_tuning=False,training=0,production_changed=False,user_data_access=0)
 write(U/'FINAL_REVIEW.json',review)
 nexttext='E3를 고정 연구 후보로 보존한다. 공개199는 이제 평가에 사용됐으므로 이후 미사용 검증이라고 부르거나 튜닝하지 않는다. 운영 기본 C·T3-F·웹·DTW를 유지한다. 다음 적용 검토는 별도 웹캠 조건 공개자료 또는 사용자가 명시 요청한 동결된 동일촬영 비교를 먼저 설계하고, 새 자료의 결과로 현재후보를튜닝하지않는다. 손복원10충돌은 실제RGB의독립위치근거없이는강제채택하지않는다. 현재고정비교완료,추가실행없음.'
 summary=f'''사용자 승인 자율 비교 완료. E3는 기존T3-F세모델의softmax확률을동일1/3평균; 새학습/입력좌표변경없음. DEV300 sensor T3F296→E3 299(+1.00%p),masked295→300(+1.67%p),개발개선모두기록ID12. 이미노출200은200→200유지.
추가평가199(기록ID14/15)에서sensor C193/T3F196/E3 197,masked C194/T3F196/E3 197. T3F대비두조건각+0.5025%p,개선2/퇴보1. sensor개선진도군수어통역센터·육백만,masked개선백육십이·육백만. 새회귀성남시수어통역센터→전주시수어통역센터,남은팔만→팔십오만. 모든모델top5는C13/T3F13/E3에서199/199. ID14는T3F98/99→99/99,ID15는98/100유지. 개별T3F3seed sensor196/194/197,masked196/195/196; E3sensor는최고단일seed와동률. 새점수로seed/가중치교체없음. 단순짝비교p1.0·상관자료/작은순이득으로일반우월성주장금지.
REAL16의4파일이이전DEV REAL11과SHA256까지같아신규점수전에ID16전체100격리;기존영상연결미확정1도대상밖. 원본보존/최초freeze오류보존. 실제사람신원확인아님,얼굴동일인판정없음. 신규199전수소스hash가이전1199/노출200및서로다름,로컬공개노출감사수행.
입력199완료exec1215종료0(1172.23초);전수검산8775480채널·939927숨긴점교란/398검출인스턴스PASS,입력실패0. exec65920순차driver종료0. 4고정모델×2관측새1592예측,저장E3 398행별도scalar검산PASS. 신규추론재실행없음. 원본/보호32파일불변. 입력3영상9프레임시각ROI확인,전수의미정답검증아님. 구현/별도검산동일Codex·전문가GT아님.
순수CPU4thread모델중앙값2.30→6.78ms;영상검출/DTW/웹시간제외. 기존손목16장면근거재분류5명확/8흐림/3부분가림,새정밀정답없음/기존10충돌미해결. E3를손가락복원·DTW교정개선으로해석금지. 운영C·T3-F·웹·DTW보존,사용자·멘토자료접근/학습/튜닝/선택0,카메라/외부전송/다운로드/Git0.
결과 RESULT_KO.html/txt,RESULT.json,FINAL_REVIEW.json,COMPLETION_FINAL.json. 정적HTML·링크검사완료/브라우저렌더미확인. 원래자동생성보고는generated_report/보존. 진행작업/잠금없음.'''
 context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';body=context.read_text('utf8')
 body=body.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n### 2026-10-26 고정 E3 공개 검증 완료 — 순개선1·회귀1·운영유지\n'+summary+'\n',1)
 body=body.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n### 2026-10-26 E3 검증 이후 — 연구후보 보존·새 조건 검증 설계\n'+nexttext+'\n',1);context.write_text(body,encoding='utf8')
 statepath=O/'STATE.json';state=read(statepath);state['updated_date_kst']=at;state['current_stage']=review['status'];state['next_single_action']=nexttext;unit=state[U.name];unit.update(status=review['status'],at=at,next=nexttext,running=False,prep_exit_code=0,driver_exit_code=0,final_review=str(U/'FINAL_REVIEW.json'),report=str(U/'RESULT_KO.html'),fresh_counts={view:{key:r['fresh199'][view][key]['correct'] for key in ['C13','T3F13','E3']} for view in VIEWS},fresh_improvements=2,fresh_regressions=1,additional_training=0,followup_status='delete_after_final_report');write(statepath,state,new=False)
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n갱신 '+at+'\n현재단위: '+str(U)+'\n\n'+summary+'\n\n다음: '+nexttext+'\n후속heartbeat automation은이번최종보고후삭제.\n',encoding='utf8')
 write(U/'CURRENT_EXECUTION.json',dict(at=at,status=review['status'],running=False,prep_session=1215,prep_exit_code=0,driver_session=65920,driver_exit_code=0,next=nexttext),new=False)
 items=read(U/'WORK_ITEMS.json');statuses=['criterion_recorded_exact_wrist_GT_unresolved','development_gain_verified','PASS','COMPLETE_small_gain_one_regression','COMPLETE']
 for item,status in zip(items['items'],statuses):item['status']=status
 write(U/'WORK_ITEMS.json',items,new=False)
 write(U/'COMPLETION_FINAL.json',dict(at=at,status=review['status'],final_review_sha256=sha(U/'FINAL_REVIEW.json'),report_sha256=sha(U/'RESULT_KO.html'),documents_synchronized=True,pending='Delete current followup automation and deliver final report',running=False,training=0,production_changed=False))
 print(json.dumps({'status':review['status'],'fresh199':unit['fresh_counts'],'remaining':details},ensure_ascii=False))
if __name__=='__main__':main()
