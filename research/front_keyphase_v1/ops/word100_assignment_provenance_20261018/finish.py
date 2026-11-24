"""Record reviewed outcome, readable report, and synchronized project state."""
from run import *
import html,re
def main():
 s=read(U/'SUMMARY.json');iv=read(U/'INDEPENDENT.json');assert iv['status']=='PASS'
 counts=s['counts'];ann=read(U/'ANNOTATED_COMPARISON.json');ch=read(U/'CHANGES.json');ev=read(U/'VISUAL_EVIDENCE.json');assert len(ch)==len(ev)==1
 assert ch[0]['clip_id']=='NIA_SL_WORD2193_REAL03_F' and ch[0]['index']==26
 assert counts['crossed_projection_preserved']==counts['crossed_projection_frames']==315
 normal=[r for r in ann if r['verdict']=='normal_identity'];errors=[r for r in ann if r['verdict']=='error_identity']
 assert len(normal)==441 and all(r['new_side']==r['old_side'] for r in normal)
 assert len(errors)==13 and sum(r['new_side']<0 for r in errors)==1
 write(U/'VISUAL_REVIEW.json',{'status':'ALL_HARD_CHANGES_REVIEWED','operator':'Codex visual inspection; independent arithmetic verification is separate, not a second human annotator.','hard_changes':1,'review':{'clip_id':ch[0]['clip_id'],'view':'sensor','frame':26,'verdict':'beneficial_withholding_of_wrong_anatomical_identity','basis':'At f25 actual Right is observed at the right forearm. At f26 the raw candidate is actual Left on the other forearm; old proposal says Right. Candidate withholds identity, retaining raw points. f27/f28 outputs match old results. No left-hand reconstruction claimed.','evidence':str(U/'evidence/change_001.jpg')},'reused_evidence':{'normal441':'Previous VISUAL_REVIEW anatomical labels + exact unchanged output checks','crossing315':'Full output equality; previous8-video crossing visual review reused, not all315 independently reannotated.'},'limits':'Single changed frame reviewed; unchanged outputs retain prior limitations, no certification of every finger or all old identities.'})
 remaining=[]
 for r in errors:
  if r['new_side']<0:continue
  d=r['trace'];eid=r['id']
  if eid in (89,91,294):reason='손목/손가락 가설이 왜곡되어 실제 중복이어도 전체21점 영역 겹침이 고정0.80 기준 미달. 기준을 결과에 맞춰 낮추지 않음.'
  elif eid==309:reason='영역 겹침은0.80 이상이지만 두 후보 모두 확정된 독립 좌우 추적이 없어 어느 후보를 버릴지 안전하게 결정할 근거 부족.'
  else:reason='사용 가능한 독립 Pose 손목 또는 확정 추적이 없어 좌우를 고칠 근거 부족. 라벨 가설을 임시 상태로 보존.'
  remaining.append({'audit_id':eid,'state':r['state'],'region_overlap':d['region_overlap_min_area'],'reason':reason})
 write(U/'REMAINING_ERRORS.json',remaining)
 result={'status':'VALIDATED_RESEARCH_CANDIDATE_AWAITING_REVIEW','public_train_clips':899,'detector_streams':2,'frames':226362,'raw_candidates':305602,'normal441_preserved':441,'crossing315_preserved':315,'hard_changes':1,'new_withheld':1,'wrong_identity_remaining_in_audited13':12,'wrong_identity_correctly_reassigned':0,'new_duplicate_guard_actual_hits':counts.get('overlapping_region_with_confirmed_opposite_anchor',0),'previous_duplicate_guard_preserved':True,'unchanged_raw_candidate_associations':305601,'other_non_label_associations_preserved':302738,'identity_states':{k:counts['state_'+k] for k in ('confirmed','provisional','unassigned','withheld_conflict')},'known_normal_provisional':436,'known_normal_confirmed':5,'all13_known_errors_not_confirmed':True,'training':0,'classifier_inference':0,'production_changed':False,'user_mentor_dev_heldout_reads':0,'independent':'PASS','limits':['Known TRAIN review informed candidate; no unseen recognition accuracy.','Withheld is not a recovered correct hand.','Provisional does not mean wrong;436/441 known normal cases remain provisional.','Confirmed means independent algorithmic evidence, not guaranteed anatomical/finger truth.','No new hard association regression in inspected changed output and fixed controls; pre-existing errors remain.']}
 write(U/'RESULT.json',result)
 next_plan={'status':'PROPOSAL_AWAITING_USER_REVIEW_NO_TRAINING','recommendation':'Keep C and separate candidate. Do not feed only confirmed hands or claim recognition improvement.','next_evidence_needed':['For startup errors, investigate additional visible forearm-to-hand evidence when Pose wrist is missing; do not use screen half or cropped-away pixels as truth.','For duplicate errors, distinguish warped/merged landmarks from two real overlapping hands with independent visible evidence. Do not lower hull-overlap threshold simply to catch these known examples.','Before any later training, freeze how provisional observations remain represented, then compare coverage/errors/regressions with C under fixed public splits. A metadata flag alone is not justification for changing the model.'],'unresolved_actual_errors':12,'ground_truth_uncertain':2,'normal_preservation_required':True}
 write(U/'NEXT_PLAN.json',next_plan)
 text='''좌우 근거 분리·중복 방지 별도 후보 결과

결론: 정상 배정을 유지하면서 알려진 잘못된 좌우 배정1건을 추가로 보류했다. 나머지12건은 해결되지 않았다. 모든 알려진 오류13건은 확정 상태로 취급하지 않는다. 이는 좌우 배정 처리 결과이며 TCN 인식률 개선 결과가 아니다.

구현
- 기존 제안 좌우와 실제 채택 좌우를 함께 저장한다.
- 임시: 라벨만 있거나 독립된 위치/추적 근거가 부족하다. 같은 이름으로 여러 프레임 이어져도 그 사실만으로 확정 승격하지 않는다.
- 확정: 현재 독립 Pose 손목 위치와 명확히 맞거나, 이전에 독립 근거로 확인한 추적과 명확히 이어진다. 알고리즘의 근거 수준이지100% 정답 인증은 아니다.
- 최근 확정 손에서 갑자기 멀리 떨어진 반대 후보가 라벨만으로 같은 슬롯을 덮어쓰려 하면 보류한다.
- 손21점의 전체 영역이 크게 겹쳐도, 확정된 반대 손 근거가 있고 자신의 확정 추적을 잇지 못할 때만 중복 위험을 차단한다. 실제 교차/겹침은 자체 확정 추적이 있으면 보호한다.
- raw 관절 좌표는 그대로 보존하고 관절을 생성/보간하지 않는다. 영상의 미래나 crop 밖 픽셀을 판단 입력으로 쓰지 않는다.

공개 검증 범위 및 결과
TRAIN899영상×전체 검출/실제 잘린 검출2조건,226362프레임/305602 raw 손 후보. 직전 배정 후보의 저장 결과와 제안을 전수 일치 검증 후 비교했다. 웹 C 모델의 단어 인식 비교는 아니다.
정상으로 검토했던441/441 배정 유지, 기하학적 교차315/315프레임 유지, 기존 비라벨 배정302738건 유지. 전체 후보305601건은 같은 배정. 새 강제배정/좌우뒤집기0, 잘못된 배정을 추가 보류1.
감사13오류: 그대로 남음12, 잘못된 좌우 보류1, 정답 좌우로 복구0. 판정불가2건은 임시로 유지.
변경 사례는 '전근' REAL03 sensor f26: 실제 왼손이 이전에 오른손으로 들어갔다. 새 후보는 최근 확정 오른손 위치와 멀어 오른손 배정을 보류. 원본 전후24~28프레임 확인, 다른 프레임 결과는 동일.

중복 방지의 실제 한계
이번 새 중복 차단 규칙은 합성 계약 검사를 통과했으나 공개 자료에서 추가 차단0건. 기존 단위에서 잡은 중복1건은 그대로 유지. 남아 있던 중복4건 중3건은 왜곡된21점 영역 겹침이 고정 기준보다 작았고,1건은 두 후보 모두 독립 확정 근거가 없었다. 결과에 맞춰 기준을 낮추지 않았다.

임시와 확정을 잘못 사용하면 생길 위험
전체 확정301908/임시1285/기존 미배정2408/신규 충돌보류1. 정상441건 중436건도 임시 상태다. 따라서 임시=틀린 손이 아니며 임시 관절을 모두 삭제하면 정상 입력을 잃는다. 좌우가 확정돼도 손가락 모양이 정확하다는 보장은 없다. 현재 C에는 이 상태정보를 연결하지 않았다.

검증
별도 scalar 계산으로 전체226362프레임의 위치·확정 추적·승격 근거·차단 결정을 검산했다. 전체21점 다각형 겹침도 별도 polygon clipping 구현으로 대조. raw 순서 반전2240프레임, prefix1114프레임, 행동 계약13개 PASS. 보호14408핀과 출력1798파일 해시 보존. Codex 동일 담당자의 별도 계산 검증이며 독립 사람 정답표는 아니다. 시각 판정은 이전456건+이번 변경1장면 재검토를 사용했다.
초기 합성 시험의6px 간격은 실제 겹침0.799576으로0.80기준 미만이었다. 양성 예시를4px로 정정하고6px는 기준 미달 보존 시험으로 유지했다. 최초 코드/실패 원인 보존, 후보 코드나 계수는 변경하지 않았다.

다음 판단
기존 C를 유지한다. 이 후보는 잘못된 근거를 확정으로 취급하지 않는 기반이지만 남은12오류를 고친 모델이 아니다. 화면 안 팔-손 연결의 추가 근거와 왜곡/중복 손 가설 구분을 먼저 검토하고, 임시 관측을 어떻게 보존할지 정한 뒤 학습 비교 계획을 동결해야 한다. 사용자와 결과 확인 전 학습하지 않는다.

학습/분류 추론/운영 변경0. 사용자·멘토·DEV·보류 자료 사용0. 원본과 기존 C/TCN/DTW/웹 보존.
'''
 with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as f:f.write(text)
 e=html.escape;rows=''.join(f"<tr><td>#{r['audit_id']}</td><td>{r['region_overlap']:.3f}</td><td>{e(r['reason'])}</td></tr>" for r in remaining)
 report=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>좌우 배정 근거 분리 후보 검증</title><style>
body{{font:17px/1.7 'Malgun Gothic',sans-serif;background:#f2f5f9;color:#172b42;margin:0}}main{{max-width:1080px;margin:auto;padding:36px 24px}}h1{{font-size:34px;line-height:1.35}}h2{{font-size:24px;margin:8px 0 18px}}section{{background:white;padding:24px;border-radius:13px;margin:20px 0;border:1px solid #dce4ed}}.cards{{display:flex;flex-wrap:wrap;gap:15px}}.card{{flex:1;min-width:200px;background:white;border-radius:13px;padding:20px}}.card b{{display:block;font-size:35px;color:#1b7263}}.note{{color:#52667a}}.warning{{border-left:5px solid #c28836}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #dce4ed;padding:12px;vertical-align:top}}img{{max-width:100%;display:block;margin:15px auto}}a{{color:#135da0}}li{{margin:10px 0}}code{{background:#edf2f8;padding:2px 5px}}details{{margin:12px 0}}summary{{cursor:pointer}}</style><main>
<p>공개 자료 입력 처리 검증 · 2026-10-18 · 기존 C 유지</p><h1>좌우 이름의 근거를 구분하고<br>잘못된 배정 1건을 추가 보류</h1><p>정상 배정과 손 교차는 유지됐다. <b>남은 오류12건은 아직 해결되지 않았다.</b> 이번 결과는 수어 인식률 개선이나 새 모델 학습 결과가 아니다.</p>
<div class="cards"><div class="card">기존 정상 배정 유지<b>441 / 441</b></div><div class="card">교차 프레임 유지<b>315 / 315</b></div><div class="card">잘못된 좌우 추가 보류<b>1건</b>정답 손 복구는0건</div></div>
<section><h2>무엇을 바꿨나</h2><table><tr><th>상태</th><th>쉽게 설명하면</th></tr><tr><td><b>임시</b></td><td>손은 보이지만 왼손인지 오른손인지 믿을 근거가 약함. 같은 이름으로 계속 따라갔다고 자동으로 확정하지 않음.</td></tr><tr><td><b>확정</b></td><td>몸 관절에서 찾은 손목 위치와 맞거나, 이전에 근거가 확인된 손의 추적과 명확히 이어짐. 실제 정답을100% 보장하는 뜻은 아님.</td></tr><tr><td><b>충돌 보류</b></td><td>확인한 손과 전혀 다른 위치의 손이 같은 이름으로 들어오려 하면 좌우 배정을 멈춤. 검출된 관절 자체는 보존.</td></tr></table><p>손 전체 영역의 겹침과 독립된 추적 근거를 함께 보는 중복 방지도 구현했다. 자신의 추적 근거가 있는 정상적인 두 손의 교차는 보호한다.</p></section>
<section><h2>실제로 달라진 장면</h2><p><b>전근 · REAL03 · f26</b>: 실제 왼손을 오른손으로 배정하던 장면이다. 새 후보는 직전 오른손과 거리가 크게 달라 오른손 배정을 보류했다. 왼손으로 복원한 것은 아니다.</p><a href="evidence/change_001.jpg"><img src="evidence/change_001.jpg" alt="전근 전후 원본 및 이전과 후보 좌우 배정"></a><p class="note">노란 선은 검출 입력 하단. 밖의 원본 영상은 오프라인 시각 확인용이며 후보 판단에 사용하지 않는다.</p></section>
<section><h2>공개 자료 전수 비교</h2><p>학습용899영상의 전체 화면 검출·잘린 화면 검출을 함께 검사했다. 총226,362프레임,305,602개 손 후보다. 직전 배정 후보와 같은 입력을 사용했다.</p><table><tr><th>비교 항목</th><th>결과</th></tr><tr><td>같은 좌우 배정 유지</td><td>305,601개 후보</td></tr><tr><td>기존 근거 기반 배정 유지</td><td>302,738건 모두 유지</td></tr><tr><td>감사에서 확인한 오류13건</td><td>1건 보류 /12건 남음 / 정답으로 복구0</td></tr><tr><td>강제 새 배정·좌우 뒤집기</td><td>0건</td></tr><tr><td>새 중복 차단의 공개 자료 추가 효과</td><td>0건 · 이전 중복 차단1건은 유지</td></tr></table></section>
<section class="warning"><h2>‘임시’를 모두 버리면 안 된다</h2><p>정상441건 중 <b>436건도 임시 상태</b>다. 임시는 틀렸다는 뜻이 아니라, 독립 확인 근거가 부족하다는 뜻이다. 관절을 모두 삭제하면 오히려 정상 입력을 잃는다.</p><p>전체 상태는 확정301,908 / 임시1,285 / 기존 미배정2,408 / 신규 충돌보류1이다. 상태를 구분하는 기반은 마련됐지만, 이를 학습에 어떻게 쓸지는 별도 검증이 필요하다.</p></section>
<section><h2>남은 오류와 다음 보완</h2><p>첫 등장·기존 임시 추적 오류는 독립된 손목 근거가 부족했다. 중복4건은 왜곡된 손 영역 때문에 겹침 조건이 부족하거나, 두 후보 모두 근거가 약했다. 기준을 느슨하게 바꾸면 정상 교차를 잘못 지울 수 있어 이번 결과에 맞춰 기준을 낮추지 않았다.</p><ol><li>화면 안에서 확인 가능한 팔과 손의 연결 근거를 더 확보할 수 있는지 검토.</li><li>왜곡된 중복 검출과 실제 포개진 두 손을 구분하는 근거 확인.</li><li>임시 관측을 보존하는 입력 방식을 정하고 결과 확인 후 별도 학습 비교 계획 수립.</li></ol><details><summary>남은12오류의 차단되지 않은 이유</summary><table><tr><th>이전 감사ID</th><th>영역겹침</th><th>근거</th></tr>{rows}</table></details></section>
<section><h2>검증과 한계</h2><p>전체 위치·추적·승격·차단 계산 및 다각형 겹침을 별도 구현으로 검산했다. 계약13개, 입력 순서 반전2,240프레임, prefix1,114프레임을 통과했다. 원본과 보호14,408파일 해시는 유지됐다.</p><p>계산 검산과 시각 검토는 Codex 동일 담당자이며 독립 사람 정답 검증은 아니다. 이미 분석한 공개 TRAIN 자료로 개발했으므로 새로운 사람에 대한 성능 증거도 아니다. 변경1건 원본 검토에서 새 잘못된 배정은 확인되지 않았지만, 기존 오류와 손가락 좌표 문제는 남아 있다.</p><p><b>기존 C·TCN·DTW·웹은 변경하지 않았고, 사용자 촬영·학습·튜닝·분류 추론은 사용하거나 실행하지 않았다.</b></p><a href="RESULT_KO.txt">텍스트 보고서</a> · <a href="RESULT.json">결과 JSON</a> · <a href="INDEPENDENT.json">독립 계산 검산</a> · <a href="../word100_label_only_audit_20261018/RESULT_KO.html">이전456건 원본 감사</a></section></main></html>'''
 with (U/'RESULT_KO.html').open('x',encoding='utf-8') as f:f.write(report)
 for target in re.findall(r'(?:href|src)="([^"]+)"',report):assert (U/target).exists(),target
 at=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
 status=('ops/word100_assignment_provenance_20261018. 공개 TRAIN899 두 조건226362프레임/305602손 후보로 좌우 임시·확정/근거출처와 중복위험 차단 후보 구현·검증 완료. '
 '이전 정상441/441·교차315/315·비라벨배정302738 보존. 전근 REAL03 sensor f26 실제왼손→R 오배정1건을 신규 보류; 정답으로 복구0, 나머지305601후보 동일. '
 '감사13오류 중12 미해결(임시), 불확실2 임시 유지. 새 중복guard 공개 추가효과0, 이전 중복차단1 유지. '
 '확정301908/임시1285/기존미배정2408/신규보류1. 정상441 중436도 임시이므로 임시관절 일괄제거 금지. '
 '별도scalar 전수/독립polygon/계약13/순서2240/prefix1114 PASS, 보호14408핀 유지. 전체변경1 원본전후6패널 검토. '
 '초기시험fixture 겹침0.799576을0.80이상으로 가정한 오류는 원파일보존 후 양성4px/음성6px로 수정; 후보계수불변. '
 'Codex 로컬 구현·동일담당자 별도계산 검산, 독립 사람/새사용자 성능검증 아님. C/TCN/DTW/웹 유지, 학습·추론0, 사용자/멘토/DEV/보류 사용0.')
 nxt=('word100_assignment_provenance_20261018/RESULT_KO.html·RESULT_KO.txt 결과를 사용자와 검토. 후보는 별도로 보존, 기존 C 유지. '
 '좌우 임시·확정 상태는 정답 여부가 아니므로 임시관측 삭제나 채널 추가 학습을 자동 진행하지 않는다. '
 '남은 첫등장/왜곡중복의 화면안 팔-손 연결 근거와 임시관측 보존 방식을 먼저 검토하고 후속 범위를 정한다. 학습은 사용자 결과 확인 및 별도 비교조건 동결 뒤 승인 범위에서 진행. '
 '현재 검산/재생 완료 결과 재실행 불필요. 사용자 자료 학습/튜닝/후보선택 금지.')
 ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=ctx.read_text('utf-8-sig');h8='### 2026-10-18 좌우 근거 분리·중복 방지 별도 후보 검증 완료';h9='### 2026-10-18 좌우 근거 후보 결과 검토·학습 대기';assert h8 not in txt
 txt=re.sub(r'(^## 8\.[^\n]*\n)',lambda m:m.group(1)+'\n'+h8+'\n'+status+'\n\n',txt,count=1,flags=re.M)
 txt=re.sub(r'(^## 9\.[^\n]*\n)',lambda m:m.group(1)+'\n'+h9+'\n'+nxt+'\n\n',txt,count=1,flags=re.M);ctx.write_text(txt,encoding='utf-8')
 sp=O/'STATE.json';state=read(sp);state['updated_date_kst']=at;state['current_stage']='WORD100_ASSIGNMENT_PROVENANCE_VALIDATED_AWAITING_REVIEW';state['next_single_action']=nxt;state['word100_assignment_provenance_20261018']={**result,'at':at,'result':str(U/'RESULT_KO.html'),'next_plan':str(U/'NEXT_PLAN.json'),'training_authorized_now':False};sp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 handoff=f'''# WordsInSilence 최신 인수인계

갱신 {at}

### 현재 목표·마지막 완료
{status}

### 다음 단일 작업
{nxt}

### 근거·실행
- 단위 research/front_keyphase_v1/ops/word100_assignment_provenance_20261018
- CONTEXT: 승인/조건/보호핀. candidate.py: 별도후보. SUMMARY/ANNOTATED_COMPARISON/CHANGES/INDEPENDENT/VISUAL_REVIEW/RESULT/NEXT_PLAN: 결과와 근거.
- exec1290 공개재생 종료0, exec6254 독립검산 종료0. 현재 실행·잠금 없음. st-gcn 자동화 삭제 유지.
- 비교대상은 직전 공개 관절배정 후보이며 운영 C의 단어 인식 비교가 아님. RAW 관측 보존/좌우 강제변경0/정답손복원0.
- 기존 C/웹/DTW, W:/sources/동결 결과 읽기 전용. 학습/추론/카메라/다운로드/외부전송/Git 없음. 사용자·멘토·DEV·보류 미사용.
''';(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
 assert read(sp)['word100_assignment_provenance_20261018']['normal441_preserved']==441
 assert h8 in ctx.read_text('utf-8') and '1285' in (ROOT/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
 hashes={str(p):sha(p) for p in U.iterdir() if p.is_file() and p.suffix in ('.py','.json','.txt','.html')}
 write(U/'FINAL_AUDIT.json',{'status':'PASS','at':at,'independent_reused':str(U/'INDEPENDENT.json'),'all_changed_associations_visually_reviewed':1,'artifact_hashes':hashes,'root_sync':{str(p):sha(p) for p in (ctx,sp,ROOT/'NEW_CHAT_HANDOFF.md')},'training':0,'production_changed':False,'stop':'Await user review before training.'})
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
