import html
from run import *
def main():
    c=read(U/'CONTEXT.json');s=read(U/'SUMMARY.json');iv=read(U/'INDEPENDENT.json');it=read(U/'INTEGRATION_CHECKS.json');ev=read(U/'VISUAL_EVIDENCE.json');at=datetime.now(timezone(timedelta(hours=9))).isoformat()
    assert iv['status']==it['status']=='PASS' and len(ev)==4 and s['normal441_preserved']==441 and s['crossing315_preserved']==315
    assert s['counts']['prior_confirmed']==s['counts']['prior_confirmed_preserved']==301908
    assert s['counts']['prior_unassigned']==s['counts']['prior_unassigned_preserved']==2411
    reviews=[]
    for e in ev:
        x=e['change'];right=x['word']=='후원';actual=1 if right else 0
        assert x['word'] in ('후원','추출')
        assert x['new_side'] is None or x['new_side']==actual
        reason=('후원: 위쪽 펼친 손은 화면 왼쪽 팔꿈치에서 이어지는 실제 오른손. 원본에서 아래쪽 손은 실제 왼손이며, 위쪽 손을 L로 계속 부르던 배정이 잘못됐음.' if right else '추출: 들어 올린 손은 실제 왼손, 실제 오른손은 아래쪽. 처음 등장 때 raw0에 붙은 R이 계속 이어지던 오류임.')
        reason+=' 현재 팔 연결 충돌 첫 프레임은 보류, 같은 관측이 다음 프레임에도 이어지자 실제 좌우로 수정. 앞뒤 원본과 현재 raw관절을 대조했으며, 새 좌표 생성이나 정상손 삭제는 없음.'
        reviews.append({'image':e['image'],'clip_id':x['clip_id'],'view':x['view'],'index':x['index'],'raw_index':x['raw_index'],'old_side':x['old_side'],'new_side':x['new_side'],'actual_side':actual,'verdict':'correct_withholding' if x['new_side'] is None else 'correct_identity_reassignment','basis':reason,'reviewer':'Codex visual inspection, same operator as implementation'})
    write(U/'VISUAL_REVIEW.json',{'status':'ALL4_CHANGES_REVIEWED_NO_NEW_IDENTITY_ERROR_FOUND','records':reviews,'source_hashes_verified':True,'viewed_current_and_adjacent_frames':True,'limits':'Public TRAIN development cases, not independent human annotation or new-user accuracy. Previously inspected broader original context reused for anatomical interpretation.'})
    result={**s,'status':'VALIDATED_ISOLATED_TRACK_RECHECK_AWAITING_REVIEW','at':at,'correct_reassignments_visually_verified':2,'correct_new_withholds_visually_verified':2,'all_changed_frames_reviewed':4,'new_regression_found_in_changed_frames':0,'old13_error_set':{'withheld':3,'wrong':10,'correct':0},'additional5_error_set':{'withheld':2,'correct':2,'wrong':1},'missing_hand_coordinates_generated':0,'training_authorized_now':False,'independent':iv['status'],'integration':it['status'],'protected_pins':len(c['pins']),'scope_limit':'Changes are4 individual hand/frame events in2 public videos. No claim that entire clips or all tracking failures were fixed.','operator':'Codex implementation, same-operator separate scalar verification and source-video inspection.'}
    write(U/'RESULT.json',result)
    next_action='word100_track_revalidation_20261018/RESULT_KO.html 결과를 사용자와 검토. 별도후보 보존·기존C/웹 유지·학습 대기. 공통 현재팔 연결2회 재확인으로 후원·추출4프레임에서2보류+2정정, 정상441·교차315 유지. 현재 팔 근거가 부족한 공항1추가오류와 기존고정감사10오류는 남아있음. 학습은 사용자 결과 확인 및 입력/비교조건을 정한 뒤 승인 범위에서만 진행. 사용자·멘토자료 학습/튜닝/후보선택 금지. 완료재생/검산 반복 불필요.'
    write(U/'NEXT_PLAN.json',{'next_single_action':next_action,'training_now':False,'production_C':'preserved','candidate':'isolated, not deployed','remaining':['First-entry identity cannot be settled without usable current arm evidence.','Long/ambiguous occlusion cannot be reconstructed by this short bounded memory.','A single usable arm is not enough to override an opposite provisional identity under this frozen rule.','Algorithmic confirmation is not anatomical ground truth; existing confirmed assignments were protected, not exhaustively human audited.'],'possible_training_followup':'After joint review only: freeze how corrected side/withheld mask enters input and compare against unchanged C using public splits; no user-data tuning.'})
    report=f'''잘못 시작한 좌우 추적 재확인 — 결과 보고
갱신 {at}

결론
현재 팔·손 연결로 임시 좌우 추적을 다시 확인하는 별도 후보를 구현하고 공개자료로 검증했다. 후원·추출의4개 프레임에서 잘못된 기존 배정을2건 보류하고2건은 실제 좌우로 정정했다. 기존 정상441건·교차315장면은 보존했다. 기존 C/TCN/DTW/웹은 그대로이고 학습은 하지 않았다.

쉽게 설명하면
처음에 왼손이라고 이름 붙였더라도, 이후 보이는 팔의 연결이 오른손을 가리키면 그 이름을 다시 확인한다. 첫 충돌에서는 잠시 보류한다. 다음 프레임에서도 같은 손의 팔 연결이 일치하면 이름을 바꾼다. 한 프레임의 흔들림, 손이 화면에서 왼쪽에 있다는 이유, 손 라벨이 여러 번 반복됐다는 이유로 바꾸지 않는다.

공통 규칙
- 직전 후보에서 근거가 확정되지 않은 임시 배정에만 재확인을 적용한다. 현재 양쪽 어깨·팔꿈치·손목이 관측범위 안이고 신뢰 기준을 통과해야 한다.
- 손목 한 점 대신 기존 검증된 팔 구간·손바닥 영역의 연결을 사용하며, 다른 검출 후보보다도 대상 손에 연결되는 근거가 뚜렷해야 한다.
- 기존과 다른 팔 연결이 발견되면 첫 프레임 보류. 같은 실제 관절 묶음이 다음 프레임에도 일치할 때 수정한다.2회/100ms/거리·경쟁 기준은 기존 공통 계수를 유지했다. 특정 단어/촬영자/프레임 예외는 없다.
- 수정된 추적도 현재 팔 근거가 끊기면100ms 뒤 효력이 만료된다. 위치 추적만 반복해서 이 시간을 연장하지 않는다. 이 짧은 유지·만료 경로는 합성 검사에서 확인했으며 이번 공개 변경4건은 현재 팔 재확인 경로였다.
- 기존 확정 배정·중복으로 보류한 후보·빈 손 슬롯을 건드리지 않는다. 손이 없으면 만들어 채우지 않는다. 두 손을 같은 좌우 칸에 중복 배정하지 않는다.

공개 검증
- 공개 TRAIN899영상 × 전체화면/상반신 검출2조건,226,362프레임/305,602손 후보. 이미 검증한 저장 관측·팔 기하 근거를 해시로 확인해 재사용했다. 새 검출 실행·학습·분류모델 추론은 없다.
-305,598후보 배정 유지.4후보만 변경:2보류+2실제 좌우 정정. 프레임별 손 후보 수이며4개 단어나4명의 성공을 뜻하지 않는다.
- 확인된 정상441/441·교차로 분류한315/315 장면 유지. 기존 알고리즘상 확정 배정301,908개와 기존 미배정/보류2,411개 모두 유지.
- 후원 REAL05 sensor f69: 실제 오른손의 기존L을 보류. f70: 같은 손의 연결 재확인 후L→R로 정정.
- 추출 REAL09 sensor f60: 실제 왼손의 기존R을 보류. f61: 같은 손의 연결 재확인 후R→L로 정정.
- 변경4장면 모두 원본 앞뒤를 대조했고, 새로 정상 손을 잘못 바꾸거나 보류한 경우는 발견되지 않았다. 각 영상의 처음부터 끝까지 모든 오류가 해결됐다는 뜻은 아니다.

남은 문제
- 직전 보고에서 추가 확인했던5오류 프레임은 이번에2정정+2보류,공항1프레임은 그대로 남았다. 공항에서는 한쪽 팔 손목이 화면 밖이어서 양쪽 연결을 비교할 근거가 부족했다.
- 그5건과 별개인 기존 고정감사13오류는 이번에 변하지 않았다: 앞선 후보에서3보류,10오류가 남아 있다. 처음 등장할 때 팔 근거가 없는 경우, 가림/왜곡된 중복 등이 포함된다. 판정불확실2건도 자동으로 정답 처리하지 않았다.
- 현재 규칙은 임시 배정의 오류 지속을 제한적으로 개선한다. 이미 알고리즘상 확정된 배정도100%정답이라는 뜻은 아니며, 이번에는 정상 배정 보호를 위해 그 부분을 바꾸지 않았다.
- 손이 완전히 가려지거나 검출되지 않으면 이 방법으로 손가락 모양을 복원할 수 없다. 반대 손목이 보이지 않는 경우까지 억지로 좌우를 바꾸지 않았다.

검증 근거와 한계
- 합성 계약19항목 통과: 정상/교차/확정배정 보호,2회 확인, 시간초과, 빈 프레임, 손 후보 순서, 중복 슬롯, 잘못된 시간 재시작, 신뢰 부족, 좌표 미생성 등.
- 별도 scalar 상태계산으로226,362프레임/305,602후보의 출력·행동종류·확인횟수·근거시작 프레임을 전수 검산했다. 시간 만료는 별도 구현의 정수 마이크로초 계산으로도 일치했다.
- 실제 연결 API와 저장결과 일치/후보순서반전1,010프레임, 변경장면 포함 앞부분 재생519프레임, 미관측 Pose좌표10,630점 교란 통과. 입력 배열은 읽기 전용으로 검사했다.
- 보호핀18,022개와 신규결과1,798개 해시 검증. 결과를 보고 계수나 단어별 조건을 변경하지 않았다. 이번 실행/검증의 미해결 오류 없음.
- 구현·시각검토 담당자는 Codex이며 수치 검산은 별도 코드로 수행했다. 독립 사람의 정답 판정이나 보지 않은 사람의 성능 검증은 아니다. 알려진 공개 TRAIN 진단 사례를 참고해 만든 후보이므로 독립 테스트 정확도로 제시하지 않는다.
- 좌우 배정 보완이 수어 인식률을 얼마나 높이는지는 아직 미검증이다. 사용자·멘토·DEV·보류 자료 사용0, 학습0, 분류추론0, C/TCN/DTW/웹 변경0.

다음 단계
결과를 사용자와 함께 확인한다. 학습 반영은 수정된 좌우와 보류 마스크를 입력에 어떻게 연결할지, 기존 C와 어떤 고정 조건으로 비교할지 정한 뒤 진행한다. 현재는 후보·검증 근거만 보존하고 학습을 대기한다.
'''
    (U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
    pictures=''
    for e,rr in zip(ev,reviews):
        x=e['change'];path=Path(e['image']).relative_to(U).as_posix();pictures+=f'<details {"open" if e["id"]==2 else ""}><summary>{html.escape(x["word"])} · f{x["index"]} — {"좌우 정정" if x["new_side"] is not None else "재확인 대기"}</summary><p>{html.escape(rr["basis"])}</p><a href="{path}"><img src="{path}" alt="원본 영상과 재확인 전후"></a></details>'
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>좌우 추적 재확인 결과</title><style>body{{margin:0;background:#eef3f8;color:#182c42;font-family:Malgun Gothic,Arial,sans-serif;line-height:1.7}}main{{max-width:1120px;margin:auto;padding:40px 25px}}h1{{font-size:34px;line-height:1.35}}.lead{{font-size:20px}}.cards,.steps{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:26px 0}}article,section,details{{background:white;padding:23px;border-radius:16px;margin:16px 0}}article strong{{font-size:29px;color:#06788c;display:block}}summary{{font-size:19px;font-weight:bold;cursor:pointer}}img{{width:100%;height:auto}}pre{{font:inherit;white-space:pre-wrap}}.note{{padding:20px;background:#fff2d8;border-radius:14px}}a{{color:#06788c}}@media(max-width:700px){{.cards,.steps{{grid-template-columns:1fr}}h1{{font-size:27px}}}}</style><main><p>WordsInSilence · 별도 관절배정 후보 · 학습 전 검증</p><h1>처음 붙인 좌우 이름을<br>현재 팔 연결로 다시 확인</h1><p class="lead">후원·추출에서 이어지던 오배정4프레임을 보완했습니다.<br>2프레임은 잠시 보류하고, 다음2프레임은 실제 좌우로 수정했습니다.</p><div class="cards"><article><strong>2건 정정 + 2건 보류</strong>프레임별 손 후보 · 두 공개영상</article><article><strong>441 / 441</strong>확인된 정상 배정 유지</article><article><strong>315 / 315</strong>교차 장면 유지</article></div><div class="steps"><article><b>① 충돌 발견</b><p>기존 이름과 현재 팔 연결이 다르면 재확인.</p></article><article><b>② 잠시 보류</b><p>한 프레임만 보고 좌우를 바꾸지 않음.</p></article><article><b>③ 같은 근거 재확인</b><p>다음 프레임에도 같은 손의 연결이 일치하면 수정.</p></article></div><div class="note">기존 C·웹·DTW는 그대로이며, 사용자 촬영은 사용하지 않았습니다. 이번 결과는 관절의 좌우 배정 보완입니다. TCN 인식률 향상은 아직 평가하지 않았습니다.</div><h2>원본과 변경 장면</h2>{pictures}<section><h2>아직 해결하지 못한 부분</h2><p>추가5오류 프레임 중 공항1프레임이 남았습니다. 한쪽 손목이 화면 밖이어서 양쪽 팔 연결의 비교 근거가 부족했습니다. 별도로 추적하던 기존13오류도 앞선3보류·10미해결 상태가 유지됩니다.</p><p>현재 팔 연결이 부족한 처음 등장과 완전 가림은 이 규칙만으로 해결되지 않습니다. 정상 배정 보호를 위해 확정 배정은 유지했습니다.</p></section><section><h2>전체 보고와 검증 범위</h2><pre>{html.escape(report)}</pre></section><p><a href="RESULT.json">결과 수치</a> · <a href="INDEPENDENT.json">별도 검산</a> · <a href="VISUAL_REVIEW.json">원본 검토</a> · <a href="NEXT_PLAN.json">다음 작업</a></p></main></html>'''
    (U/'RESULT_KO.html').write_text(page,encoding='utf-8')
    pinscheck(c['pins']);pinscheck(read(U/'OUTPUT_PINS.json'))
    write(U/'FINAL_AUDIT.json',{'status':'PASS','at':at,'candidate_sha256':sha(U/'candidate.py'),'protocol_sha256':sha(U/'CONTEXT.json'),'protected_pins':len(c['pins']),'training':0,'production_changed':False,'verified_files':{str(p):sha(p) for p in [U/'RESULT.json',U/'RESULT_KO.txt',U/'RESULT_KO.html',U/'INDEPENDENT.json',U/'VISUAL_REVIEW.json',U/'INTEGRATION_CHECKS.json',U/'verify.py',U/'render_evidence.py',*[Path(x['image']) for x in ev]]}})
    short='ops/word100_track_revalidation_20261018. 임시좌우를 현재 화면안 팔·손 연결로2회 재확인하는 별도후보 구현·검증완료. 공개TRAIN899 두조건226362프레임/305602후보 중305598동일, 후원 sensor69보류/70 L→R, 추출 sensor60보류/61 R→L:2보류+2정정,2영상4프레임. 정상441/441·교차315/315·기존확정301908·기존미배정2411 보존. 추가5오류는2정정/2보류/공항1미해결; 별도기존13오류는3보류/10미해결유지. 전체변경4원본검토·별도scalar전수검산PASS. 계약19/API순서1010/prefix519/미관측좌표10630교란PASS, 보호18022핀유지. 계수/단어별예외추가없음. 동일Codex구현·시각검토+별도계산이며 독립사람/새사용자인식검증아님. C/TCN/DTW/웹유지, 학습/추론/사용자·멘토·DEV·보류사용0. 학습은사용자결과확인후.'
    state=read(O/'STATE.json');state['current_stage']='WORD100_TRACK_REVALIDATION_VALIDATED_AWAITING_REVIEW';state['next_single_action']=next_action
    for k in ('updated_at','updated_at_kst'):
        if k in state:state[k]=at
    state[U.name]={**result,'result':str(U/'RESULT_KO.html'),'next_plan':str(U/'NEXT_PLAN.json')};write(O/'STATE.json',state)
    root=(ROOT/'WordsInSilence_PROJECT_CONTEXT.md').read_text(encoding='utf-8')
    for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','2026-10-18 처음 잘못된 좌우 추적 재확인 후보 검증 완료',short),('## 9. 다음 단일 작업','2026-10-18 좌우 추적 재확인 결과 검토·학습 대기',next_action)]:
        assert root.count(marker)==1;root=root.replace(marker,marker+'\n\n### '+title+'\n'+body+'\n',1)
    (ROOT/'WordsInSilence_PROJECT_CONTEXT.md').write_text(root,encoding='utf-8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'# WordsInSilence 최신 인수인계\n\n갱신 {at}\n\n### 현재 목표·마지막 완료\n{short}\n\n### 다음 단일 작업\n{next_action}\n\n### 근거·실행\n- 단위 research/front_keyphase_v1/ops/{U.name}\n- CONTEXT: 승인/동결조건/보호핀, candidate.py: 별도후보. RESULT_KO/RESULT/NEXT_PLAN 결과와다음단계.\n- CHANGES/ANNOTATED_COMPARISON/CROSSING_COMPARISON/VISUAL_REVIEW/INDEPENDENT/INTEGRATION_CHECKS/FINAL_AUDIT: 검증근거.\n- 공개재생 exec35151 종료0, 별도검산 exec2078 종료0, 영상표시/연계검증 종료0. 현재실행·잠금없음. st-gcn 자동화삭제유지.\n- 원본/W:/sources/과거동결읽기전용. 사용자자료학습·튜닝·후보선택금지. 학습·분류추론·운영교체 없음.\n- 변경은 개별프레임의 좌우 재배정이며 전체영상/단어정확도향상으로 보고하지 않는다.\n',encoding='utf-8')
    assert not (U/'RUNNING.json').exists();assert read(O/'STATE.json')['current_stage']=='WORD100_TRACK_REVALIDATION_VALIDATED_AWAITING_REVIEW'
    print('FINALIZED',U/'RESULT_KO.html',flush=True)
if __name__=='__main__':main()
