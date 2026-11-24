from common import *
from html import escape
import statistics
r=read(U/'COMPARISON_INDEPENDENT.json');a=read(U/'ANALYSIS.json');s=r['summary']
assert r['status'].startswith('PASS') and a['comparison_sha256']==sha(U/'COMPARISON_INDEPENDENT.json')
gate=r['primary_review_gate_vs_T3F']['pass'] and r['primary_review_gate_vs_FG16']['pass']
sensor_errors=[e for e in a['error_cases'] if e['view']=='sensor' and not e['new_correct']]
aux_errors=sum(e['aux_prediction']!=e['label'] for e in sensor_errors)
margin_worse=sum(e['new_target_margin']<e['old_target_margin'] for e in sensor_errors)
conclusion={'at':now(),'status':'COMPLETE_NOT_PROMOTED_PRIMARY_NO_GAIN' if not gate else 'COMPLETE_PASS_REVIEW_GATE_NO_AUTO_PROMOTION',
 'candidate':ARM,'runs':3,'gate_pass':gate,'sensor_mean':s[ARM]['sensor']['accuracy_mean'],'sensor_range':s[ARM]['sensor']['accuracy_range'],
 'masked_mean':s[ARM]['masked']['accuracy_mean'],'sensor_delta_vs_T3F_pp':r['contrasts']['T3F']['sensor']['mean_pp'],'masked_delta_vs_T3F_pp':r['contrasts']['T3F']['masked']['mean_pp'],
 'sensor_delta_vs_C_pp':r['contrasts']['C']['sensor']['mean_pp'],'masked_delta_vs_C_pp':r['contrasts']['C']['masked']['mean_pp'],
 'sensor_gain_events':sum(len(e['gains']) for e in r['paired']['T3F']['sensor']),'sensor_loss_events':sum(len(e['losses']) for e in r['paired']['T3F']['sensor']),
 'masked_gain_events':sum(len(e['gains']) for e in r['paired']['T3F']['masked']),'masked_loss_events':sum(len(e['losses']) for e in r['paired']['T3F']['masked']),
 'sensor_remaining_unique_error_videos':len({e['clip_id'] for e in sensor_errors}),'sensor_remaining_seed_error_events':len(sensor_errors),'aux_alone_wrong_in_remaining_sensor_events':aux_errors,'remaining_sensor_margin_worse_events':margin_worse,
 'epochs':r['completed_epochs'][ARM],'selected_epochs':r['selected_epochs'][ARM],'training_seconds_sum':a['training_seconds_sum'],
 'actual_batches_verified':sum(x['actual_batches'] for x in r['traces'].values()),'actual_source_exposures_verified':sum(x['actual_source_exposures'] for x in r['traces'].values()),
 'saved_final_predictions_verified':7200,'saved_component_predictions_verified':3600,'frozen_T3F_preserved':True,'new_inference_during_verification':0,
 'user_recordings_read':0,'user_tuning_or_selection':False,'existing_C_T3F_web_DTW_changed':False,'class_specific_exceptions':False,'tested_word_classes':100,'all_signs_validated':False,
 'next':'Keep C/T3F. No automatic retuning, extra training, holdout use or deployment. Review whether future public stage/completion evidence and broader vocabulary verification are warranted.'}
write(U/'CONCLUSION.json',conclusion,new=True)
nextplan={'status':'COMPARISON_COMPLETE_KEEP_C_T3F','training_complete':True,'automatic_further_training':False,'automatic_adoption':False,
 'evidence':'Auxiliary classification learns, but primary accuracy unchanged and only one secondary-view seed event improved.',
 'next_proposal':'먼저 이번 결과를 검토한다. 다음 실험을 택한다면 같은 DEV에서 계수·학습시간을 계속 바꾸기보다, 공개 자료의 구별 동작/순서/생략 여부를 직접 확인할 평가 기준과 어휘 확대 시 사람별 자료 분리 계획을 마련한다. 해당 자료는 사람이 검증해야 하며 임의 역순 영상을 모두 틀린 수어로 라벨링하지 않는다.',
 'reason':'Same word labels and frozen features did not provide complementary answers on the remaining seven sensor error events. This does not establish order/completion semantics or causally identify the detector as the sole issue.',
 'holdout':'Do not use ID14-16 as a new tuning set. Final fixed candidate and separate review required before a single heldout evaluation.',
 'generality':'One word-independent objective; empirical proof limited to current100class development set. New labels/data and head dimensions require a separate expansion experiment.',
 'private_data':'User remains an end-user tester only, never train/tuning/model-selection source.'}
write(U/'NEXT_PLAN.json',nextplan,new=True)
names={'C':'기존 C','T3F':'기존 T3-F','FG16':'직전 F-G16',ARM:'이번 F-G16-AUX01'}
modelrows=''.join(f'<tr><th>{names[m]}</th><td>{s[m]["sensor"]["accuracy_mean"]:.2f}%</td><td>{s[m]["masked"]["accuracy_mean"]:.2f}%</td><td>{s[m]["sensor"]["top5_mean"]:.2f}%</td></tr>' for m in names)
seedrows=''.join(f'<tr><td>{i+1}</td><td>{s["T3F"]["sensor"]["top1_counts"][i]} → {s[ARM]["sensor"]["top1_counts"][i]} / 300</td><td>{s["T3F"]["masked"]["top1_counts"][i]} → {s[ARM]["masked"]["top1_counts"][i]} / 300</td><td>{x["epochs"]}회 / {x["selected_epoch"]}회</td></tr>' for i,x in enumerate(a['rows']))
auxrows=''.join(f'<tr><td>{i+1}</td><td>{x["sensor_aux_correct_first_selected_last"][0]/3:.2f}% → {x["sensor_aux_correct_first_selected_last"][2]/3:.2f}%</td><td>{x["sensor_aux_correct_first_selected_last"][1]/3:.2f}%</td><td>{x["sensor_final_correct_first_selected_last"][0]} → {x["sensor_final_correct_first_selected_last"][2]} / 300</td></tr>' for i,x in enumerate(a['rows']))
errorrows=''.join(f'<tr><th>{escape(c["label"])}</th><td>{escape(" · ".join(sorted({p["predicted"] for p in c[ARM+"_sensor_errors"]})))}</td><td>{len(c[ARM+"_sensor_errors"])} / 9회</td></tr>' for c in r['all_classes'] if c[ARM+'_sensor_errors'])
gnames={'internal_reentry':'중간에 손이 다시 관측됨','one_hand_never_usable':'한 손이 계속 미관측','other':'그 외 관측 집합'}
grouprows=''.join(f'<tr><th>{gnames[g["group"]]}</th><td>{g["videos"]}개</td><td>{g[ARM]["mean_percent"]:.2f}%</td><td>{g[ARM]["mean_percent"]-g["T3F"]["mean_percent"]:+.2f}%p</td></tr>' for g in a['groups'] if g['view']=='sensor')
text=f'''보조 단어 학습 신호 3회 비교 완료 — 기존 C·T3-F 유지

결론: 보조 부분은 단어 구별을 학습했으나 주검증 평균 인식률은 T3-F와 같은99.22%로 추가개선0.00%p. 전체검출후범위적용 조건은99.00→99.11%,+0.11%p의작은개선. 사전채택기준미달이므로새후보는웹에적용하지않음.

검증범위: 공개100단어/TRAIN899(9명)/DEV300(다른3명), 동일3seed. 실제사용자시험아님. 같은300영상을3번평가한것이며독립900영상아님. 반복사용한DEV결과여서새사람일반화증명아님.
sensor=공개영상화면제한재검출: C98.67/T3F99.22/FG1699.22/AUX0199.22%. masked=전체검출후범위적용: C98.44/T3F99.00/FG1699.00/AUX0199.11%. sensor상위5는TF/FG/AUX모두99.89%.
새후보sensor정답296/299/298(각300),범위98.67~99.67%,TF대비새정답0/새오답0. masked295/299/298,3회차속다ID12영상1개만개선,새오답0. 이득1건은개발검증의한seed사례이지반복재현된개선아님. C대비sensor+0.56%p는기존TF이득을계승한것이고새학습신호의효과로보면안됨.

설계: TF47236변수와BN전부고정. 동일GRU16/head4000만CE(기존+보조)+0.1CE(보조)로학습. 같은초기해시/입력/ROI/정규화/스케줄유지. 코드에단어별/사람별예외없음. 사용자실사용테스터자료읽기·학습·튜닝·선택0. 기존모델·웹·DTW수정0.

배운것: 첫배치gradient는이전약1e-5에서이번약.044~.050으로바뀌었고이후memorygradient도확인. 기울기배율을인식률향상이나AdamW갱신배율로해석하지않음. 학습기록상보조단독sensor정답률은첫회→마지막21.00→45.67%,27.67→53.00%,31.33→54.33%. 선택시점에서는21.00/27.67/45.33%. 강해진학습신호와새부분의학습은확인되나최종답개선과다름.
남은sensor오답6개영상에서3seed합7건: 속다·동대문·상처·남해군지부·별·백육십이. 보조혼자도7건모두오답,그중5건은합친답에서정답과경쟁후보간점수차가더불리해짐. 보조분류학습이기존모델의약점을보완하는답을충분히만들지못한사실. 단일원인이나수어순서능력에대한인과증명은아님.

학습종료13/13/20회,사전규칙으로선택1/1/8회. 두번째run의마지막은보조정답수늘어도최종정답299→298. 세번째run마지막은299이나최종양viewCE가선택시점보다나빠서채택하지않음. 최고정답수가나온시점으로선택기준을사후변경하지않음. 예정61~80후반창미도달이므로그안정성개선주장불가.
검증: 46epoch,1334실제배치,41354실제학습영상노출. 최종예측7200건+새base/aux예측3600건산술검산,합산정확일치,base저장logit과TF완전동일,실제batchID/선택checkpoint/새4000optimizer/모든step동결확인. 최종재추론0. 같은Codex별도검산,독립agent/수어전문가검증아님. 실제학습합계{a['training_seconds_sum']:.2f}초(준비/검산/보고제외).

범용목표: 학습식은단어종류와무관한공통방법. 100개외수어에도적용할목표지만현재모델출력100개이며미학습단어를지금인식하는것은아님. 어휘확대는해당공개자료/출력종류/재학습/사람분리검증필요. 합성class수7/100/137 및단어번호순열불변검사는학습식의범용구현을확인할뿐그단어인식성능검증아님.

다음: 기존모델유지. 이번조건학습/비교완료,자동추가학습없음. 계수·epoch재탐색보다공개자료의구별단계/순서/생략을직접확인할평가기준및어휘확대자료분리계획을검토. 새조건은사용자검토후. ID14~16미사용299개보존,ID17/18원시접근도없음.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf8')
style=(R/'RESULT_KO.html').read_text(encoding='utf8').split('<style>')[1].split('</style>')[0]
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WordsInSilence · 보조 단어 학습 비교 결과</title><style>{style}</style><main>
<header><div class="tag">WORDS IN SILENCE · 2026.10.19 · 3회 학습·검산 완료</div><h1>보조 부분은 배웠지만,<br>주검증 인식률은 그대로였습니다</h1><p>새 후보의 채택은 보류하고 기존 C·T3-F를 유지합니다.</p><p>한 보조 검증 조건에서 작은 개선이 있었으나, 반복된 주성능 개선으로 확인되지는 않았습니다.</p></header>
<section><h2>같은 자료에서 비교한 결과</h2><div class="grid"><div class="card"><div class="big">99.22% · +0.00%p</div>화면 제한 후 재검출 · T3-F와 동일</div><div class="card green"><div class="big">99.11% · +0.11%p</div>전체 검출 후 범위 적용 · 작은 개선</div></div><p>공개 100단어의 학습899영상·9명 / 개발 검증300영상·다른3명을 사용했습니다. 같은 조건으로 세 번 학습했으며 관절 추출·손 배정·화면 범위·정규화는 바꾸지 않았습니다.</p><div class="tablewrap"><table><thead><tr><th>모델</th><th>화면 제한 관측</th><th>전체 검출 후 범위 적용</th><th>화면 제한 상위5</th></tr></thead><tbody>{modelrows}</tbody></table></div><p><b>이번 주검증에서 C보다 높은 +0.56%p는 기존 T3-F의 이득을 이어받은 것입니다.</b> 새 보조 학습 신호가 추가로 높인 주검증 정확도는 0.00%p입니다.</p><p class="small">화면 제한 관측(sensor)은 공개 영상의 화면을 제한해 다시 검출한 조건이고, 전체 검출 후 범위 적용(masked)은 전체 영상에서 검출한 좌표에 사용 범위를 적용한 조건입니다. 모두 공개 자료의 개발 검증이며 실제 웹캠 사용자 정확도가 아닙니다. 세 번의 평가는 독립된900영상이 아닙니다.</p></section>
<section><h2>개선된 부분과 남은 오류</h2><div class="tablewrap"><table><thead><tr><th>반복</th><th>화면 제한: T3-F → 새 후보</th><th>전체 검출: T3-F → 새 후보</th><th>학습 종료 / 선택</th></tr></thead><tbody>{seedrows}</tbody></table></div><p><b>전체 검출 조건의 세 번째 학습에서 ‘속다’ 영상1개가 새로 정답이 됐습니다.</b> 두 조건 모두 선택된 모델에서 T3-F가 맞히던 영상을 새로 틀린 경우는 없었습니다. 그러나 주검증에서 새 정답이 없고, 두 번 이상 반복되는 개선도 없어 채택 기준을 통과하지 못했습니다.</p><div class="tablewrap"><table><thead><tr><th>남은 단어</th><th>잘못 고른 단어</th><th>3영상 × 3회 중 오답</th></tr></thead><tbody>{errorrows}</tbody></table></div><p>주검증에서 고유6영상에 걸친 7번의 오답이 남았습니다. 같은 영상이 두 초기값에서 틀린 경우도 있어 ‘7명의 사용자’나 ‘7개 서로 다른 영상’이 아닙니다.</p></section>
<section><h2>보조 부분이 정말 학습되었나요?</h2><p><b>네. 합친 답만 채점하던 방식에 보조 답 자체의 채점을 추가했고, 보조 부분의 단어 분류 능력은 학습 중 높아졌습니다.</b> 하지만 그 능력이 기존 모델의 오답을 고치는 능력으로 충분히 이어지지는 않았습니다.</p><p class="formula">합친 답의 오차 + 0.1 × 보조 답의 오차</p><div class="tablewrap"><table><thead><tr><th>반복</th><th>보조 단독: 첫 회 → 마지막</th><th>선택 시점의 보조 단독</th><th>합친 정답 수: 첫 회 → 마지막</th></tr></thead><tbody>{auxrows}</tbody></table></div><p class="small">이 표는 학습 중 기록된 공개 개발 검증 성적입니다. 최종 모델의 선택 기준은 처음부터 정한 ‘합친 답의 검증 오차’입니다. 보조 성적이나 순간 최고 정답 수로 유리한 시점을 다시 고르지 않았습니다.</p><div class="warn"><b>보조 부분이 더 잘 맞힌다고 합친 답까지 반드시 좋아지는 것은 아닙니다.</b><br>두 번째 실험에서는 보조 성적이 올라가는 동안 마지막 최종 정답 수가299개에서298개로 줄었습니다. 세 번째 마지막은299개지만 전체 검증 오차가 선택 시점보다 나빠서 선택하지 않았습니다.</div><p>선택된 후보의 남은 주검증 오답7건은 보조 부분도 혼자 맞히지 못했습니다. 그중5건은 합친 점수에서 정답과 경쟁 후보의 차이가 더 불리해졌습니다. <b>‘학습 신호를 강하게 주기’는 확인했지만 ‘기존 모델에 부족한 구별 정보를 보충하기’는 충분하지 않았습니다.</b> 이는 결과에 근거한 해석이며 단일 원인이 확정됐다는 뜻은 아닙니다.</p><p>모든 실험이13·13·20회에 종료되어 미리 정한61~80회 후반 구간은 없었습니다. 후반 안정성이 향상됐다고 보고하지 않습니다.</p></section>
<section><h2>손 미관측 조건은 유지됐나요?</h2><div class="tablewrap"><table><thead><tr><th>주검증 관측 집합</th><th>영상 수</th><th>새 후보 평균</th><th>T3-F 대비</th></tr></thead><tbody>{grouprows}</tbody></table></div><p>사람별·미관측 집합별 주검증 정답 수는 T3-F와 같았습니다. 이 구분은 관절 관측 상태의 집합이며, 언어학적으로 한 손만 쓰는 수어인지 판정한 분류는 아닙니다.</p></section>
<section><h2>100개 단어를 넘어 적용하려는 공통 방법입니다</h2><p><b>특정 단어·사람을 위한 예외나 임계값은 넣지 않았습니다.</b> 단어 정답을 사용하는 같은 학습식을 모든 학습 자료에 적용했습니다. 단어 번호를 바꿔도 결과가 같은지, 다른 정답 종류 수에도 수식이 작동하는지를 검사했습니다.</p><p>다만 <b>현재 학습된 모델의 인식 범위는100개 단어</b>입니다. 나머지 모든 수어에 대한 성능을 검증한 것은 아닙니다. 범위를 넓히려면 추가 단어 자료와 출력 종류를 준비하고 사람을 분리해 다시 학습·검증해야 합니다.</p><p>단어 이름 정답만으로는 구별 동작의 순서나 생략을 직접 가르친 것이 아닙니다. 같은 개발 자료에서 계수나 학습 시간을 계속 바꾸기보다, 공개 자료의 구별 단계·순서·생략을 확인할 평가 근거와 어휘 확대 계획을 먼저 검토할 필요가 있습니다. 임의로 영상을 뒤집었다고 모두 잘못된 수어로 처리하지는 않습니다.</p><p><b>사용자는 실제 사용 환경을 확인하는 테스터입니다.</b> 사용자·멘토 촬영은 이번 작업에서 읽거나 학습·튜닝·계수 결정·모델 선택에 사용하지 않았습니다.</p><div class="warn">기존 C·T3-F·웹·DTW 보존 완료. 새 후보 자동 적용 없음. 승인된3회 비교를 마쳤으며 추가 학습이나 조건 재탐색은 진행하지 않습니다.</div><details><summary>검증 근거와 상세 기록</summary><p>기존 T3-F47,236변수와BN를 모든 학습 단계에서 고정하고 보조4,000변수만 갱신했습니다. 학습46회차·1,334실제 배치·41,354영상 노출의ID·입력 일정·선택 시점·가중치를 검산했습니다.</p><p>네 모델의 저장 최종 예측7,200건과 새 후보의 보조/기존 점수3,600건을 별도 산술 검산했습니다. 기존 점수는 T3-F 저장 점수와 정확히 같았고, 보조와 합친 점수도 일치했습니다. 검산 중 최종 재추론은 하지 않았습니다.</p><p>구현과 별도 검산은 같은 Codex가 수행했습니다. 독립 연구자나 수어 전문가 검증은 아닙니다. 학습 자체 합계{a['training_seconds_sum']:.1f}초이며 준비·검산·보고 시간은 제외한 값입니다.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="COMPARISON_INDEPENDENT.json">저장 점수·추적 검산</a> · <a href="ANALYSIS.json">원인과 학습 기록</a> · <a href="NEXT_PLAN.json">다음 검토 방향</a></p></details></section></main></html>'''
(U/'RESULT_KO.html').write_text(page,encoding='utf8')
print(json.dumps(conclusion,ensure_ascii=False))
