import html
from datetime import datetime,timezone,timedelta
from run import *
def main():
    c=read(U/'CONTEXT.json');s=read(U/'SUMMARY.json');iv=read(U/'INDEPENDENT.json');integ=read(U/'INTEGRATION_CHECKS.json');ev=read(U/'VISUAL_EVIDENCE.json')
    assert iv['status']==integ['status']=='PASS' and s['normal_preserved']==441 and s['crossing_preserved']==315
    assert s['audited_errors_new_withheld']==[89,91] and len(read(U/'CHANGES.json'))==2
    review=[]
    for e in ev:
        i=e['id'];actual=0 if i in (3,4,5,6) else 1
        if i in (3,4):basis='소복의 위쪽 주먹은 실제 왼손. raw0과 왜곡된 raw1이 모두 같은 왼손 위에 놓임. 실제 오른손은 왼쪽 팔꿈치 부근에 별도로 있음. raw0 L 유지, raw1의 잘못된 R만 보류가 타당. 사라진 실제 오른손 좌표를 복원한 것은 아님.'
        elif i in (1,2):basis='후원 원본 f60~88 추가 확인: 위쪽 펼친 손은 화면 왼쪽 팔꿈치와 이어진 실제 오른손. 아래 손은 실제 왼손. raw0을 L로 이어가던 기존 오배정이며 새 팔 근거는 R을 지지. 확정 반대손 anchor가 없어 자동 수정하지 않음.'
        elif i in (5,6):basis='추출 원본에서 들어 올린 손은 실제 왼손이고 오른손은 아래에 있음. raw0의 기존 R 오배정이 처음 잘못 잡힌 이후 이어짐. f61의 별도 실제 오른손 raw1은 기존 보류. 현재 후보는 이를 강제로 재배정하지 않음.'
        else:basis='공항 원본의 들어 올린 손은 실제 오른손, 왼손은 아래 화면 밖. raw0의 기존 L 오배정이 유지됨. 왼팔 손목은 관측범위 밖이고 확정 반대 anchor도 없어 새 중복 규칙 미적용.'
        review.append({'evidence_id':i,'clip_id':e['case']['clip_id'],'view':e['case']['view'],'index':e['case']['index'],'raw_index':e['case']['raw_index'],'actual_side':actual,'outcome':'correct_duplicate_withholding' if i in (3,4) else 'preexisting_wrong_identity_retained','basis':basis,'image':e['image']})
    write(U/'VISUAL_REVIEW.json',{'status':'ALL_HARD_CHANGES_REVIEWED_NO_NEW_WRONG_WITHHOLD_FOUND','reviewer':'Codex visual inspection; same operator as implementation, no independent human ground truth','hard_changes_reviewed':2,'all_arm_conflict_cases_reviewed':7,'newly_documented_preexisting_error_frames':5,'records':review,'extra_evidence':str(U/'evidence/support_context.jpg'),'outside_crop_pixels_used_only_for_offline_annotation':True,'candidate_outside_pixel_use':False})
    at=datetime.now(timezone(timedelta(hours=9))).isoformat()
    next_action='별도 팔 연결·왜곡중복 후보의 결과를 사용자와 검토한다. 이번 후보는 같은 소복 영상의 두 검출 조건에서 오배정2건을 보류했으며 일반 인식률 개선은 미검증. 기존 C 유지, 학습 대기. 남은 첫등장/겹침/추적 고착은 현재 팔 근거와 기존 추적이 충돌한5프레임(후원2·추출2·공항1)을 포함하여 공통 재확인 절차의 근거를 정리한 뒤 후속 범위를 결정한다. 현재 팔 근거를 곧바로 정답 라벨로 확정하거나 임시 관측을 일괄 삭제하지 않는다. 사용자 자료 학습·튜닝·후보선택 금지.'
    result={**s,'status':'VALIDATED_ISOLATED_CANDIDATE_AWAITING_USER_REVIEW','at':at,'previous_audited13_withheld':1,'new_audited13_withheld':2,'cumulative_audited13_withheld':3,'audited13_wrong_associations_remaining':10,'correct_hand_reconstruction':0,'new_withholds_are_two_views_of_one_clip':True,'newly_documented_preexisting_error_frames':5,'known_errors_not_exhaustive':True,'new_error_from_candidate_changes_found':0,'independent_status':iv['status'],'integration_status':integ['status'],'training_authorized_now':False,'limits':['TRAIN diagnostic set informed design; not heldout/test accuracy.','Two changed frames in two detector streams from one video; not two independent users.','Withholding avoids a wrong identity, does not recover a missing hand.','441 known normal and315 geometric crossing controls preserved; not every raw observation manually labeled.','Five additional existing wrong-identity frames found outside old13 audit; old13 is not exhaustive.','Visible arm preference is geometric evidence, not guaranteed ground truth.','No TCN accuracy or DTW performance measured; C untouched.'],'implementation':'Codex local','verification':'Same operator, separate scalar calculation and source RGB inspection.'}
    write(U/'RESULT.json',result);write(U/'NEXT_PLAN.json',{'next_single_action':next_action,'training_now':False,'candidate_status':'isolated, not deployed','production_C':'unchanged','do_not':['use user footage','force identity from nearest elbow','blanket discard provisional hands','claim recognition improvement from frame guards','repeat completed replay without new reason']})
    text='''화면 안 팔 연결·왜곡 중복 검출 보완 결과

결론: 별도 후보 구현·공개자료 검증 완료. 잘못된 중복 배정2건 추가 보류, 기존 정상441건·손 교차315장면 유지. 기존 C/TCN/DTW/웹은 그대로이며 새 학습은 하지 않았다.

연속 검출이면 좌우가 고정되는가?
이번 기록에서는 그렇지 않았다. 처음 화면에 들어오는 손을 반대로 배정하거나, 겹친 손을 한 손으로/같은 손을 두 후보로 검출하거나, 잘못 시작한 좌우를 추적이 그대로 이어가는 경우가 있었다. 재등장 때 다시 찾는 문제뿐 아니라 첫 배정과 중복, 잘못 이어지는 추적도 포함된다. 손이 교차해도 팔에 연결된 좌우 이름은 유지해야 하므로 화면을 좌우 반으로 나누지 않는다.

이번에 바뀐 점
1. 화면 안에서 신뢰 기준을 통과한 어깨·팔꿈치·손목 연결과 실제 손바닥 영역을 함께 비교한다. 팔꿈치만 보이거나 손목이 화면 밖이면 좌우 확정 근거로 쓰지 않는다.
2. 손목 한 점이 늘어져도 나머지 손가락 관절들이 다른 후보의 같은 손 영역에 몰려 있는지 확인한다. 기존의 두 영역 겹침 비율만 보는 방법을 보완했다.
3. 반대쪽으로 임시 배정된 후보가 확인된 다른 후보와 같은 팔에 연결되고 손가락 영역도 중복되면, 그 잘못된 임시 배정만 보류한다. 좌표 원본·기존 정상 배정·이전 추적은 보존하며 좌우를 억지로 뒤집거나 없는 손을 만들지 않는다.

공개 검증 범위와 결과
- 공개 TRAIN899개 영상 × 전체화면/상반신 검출 두 조건 =226,362프레임,305,602개 손 후보. 저장된 검출 결과의 새 배정 처리이며 새 영상추출/모델학습/인식률평가는 아니다.
- 직전 후보와305,600개 손 후보의 배정이 동일.2개만 새 보류.
- 정상으로 검토된441/441 보존. 손 교차315/315 보존. 새로 바뀐2장면 원본 앞뒤를 확인했으며 정상 손을 잘못 보류한 경우는 발견되지 않았다.
- 소복 REAL05의 전체화면 f84·상반신검출 f88: 실제 왼손 위에 두 좌표 묶음이 잡혀 하나가 오른손으로 불리던 오류를 차단했다. 같은 원본 영상의 두 관측 조건이며, 서로 다른 영상2개나 사용자2명의 성공이 아니다.
- 고정 감사13오류 중 직전1건+이번2건=3건의 잘못된 배정을 보류. 나머지10건은 미해결. 정답 손 복원0건. 중복4건 중2건은 이번에 보류됐고 무산·공항의 나머지2건은 유지됐다. 기존 판정불확실2건도 유지됐다.

추가로 확인한 기존 문제
팔 연결이 임시 좌우 배정과 다른7프레임을 모두 원본 확인했다. 그중2개가 위의 새 보류이고,5개는 기존 오배정이 계속 이어지는 장면이다: 후원2프레임(실제 오른손→L), 추출2프레임(실제 왼손→R), 공항1프레임(실제 오른손→L). 새 후보가 만든 오류는 아니다. 이5개는 기존13건 감사의 해당 프레임 밖에서 추가로 확인한 것으로,13건만이 전체 오류라는 뜻은 아니다.
후원은 손을 모으는 장면에서 기존 라벨이 계속 이어지고, 추출·공항은 첫 등장 오류 뒤에도 추적이 이어진다. 확실한 반대손 후보가 없거나 한쪽 손목이 화면 밖이어서 이번 좁은 중복 차단 규칙이 자동 수정하지 않았다. 다음에는 팔 근거와 추적의 불일치를 재확인하는 공통 절차가 필요하다. 현재 팔 기하 근거 하나를 곧바로 정답으로 채택하지 않는다.

검증과 한계
- 전체226,362프레임/305,602후보의 출력 결정을 별도 계산으로 확인.
- 변경 가능 임시후보1,285건 전부와 정상/교차 통제 및 균등표본을 포함한13,670프레임/18,569후보는 독립 scalar 기하 계산으로 검산.
- 합성 계약18항목, 실제 후보 연계·순서반전1,010프레임, 앞부분만 재생502프레임, 미관측 좌표10,630점 교란 통과.
- 보호핀16,215개와 신규결과1,798개 해시 확인. 영상표시 도구가 기존 감사 캐시에 없던 후원 원본을 찾지 못한 최초 오류는 보존했고, 기존 로컬 ZIP의 해당 영상만 읽어 크기/CRC/SHA를 확인해 새 단위에 보관했다. 후보·계수는 결과를 보고 변경하지 않았다.
- 구현자와 시각검토자는 Codex이며 별도 수치 계산 검산을 수행했다. 독립 사람 정답 검증이나 새 사용자 일반화 검증은 아니다.
- 이는 제한된 중복 관절배정 개선이다. 전체 수어 인식률이 얼마나 오를지, 학습에 넣었을 때 좋아질지는 아직 미검증이다. 사용자·멘토·DEV·보류 자료 사용0, 학습0, 분류 추론0.

다음: 이 결과를 함께 확인한 뒤 남은 오류의 공통 재확인 절차와 학습 반영 범위를 결정한다. 기존 C 유지, 현재 학습 대기.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
    cards=''.join(f'<article><strong>{a}</strong><span>{b}</span></article>' for a,b in [('2건','추가 중복 오배정 보류'),('441 / 441','확인된 정상 배정 유지'),('315 / 315','손 교차 장면 유지'),('학습 전','기존 C · 웹 유지')])
    pictures=''
    for e,r in zip(ev,review):
        if e['id'] not in (3,4,1,5,7):continue
        path=Path(e['image']).relative_to(U).as_posix();pictures+=f'<details {"open" if e["id"]==3 else ""}><summary>{html.escape(e["case"]["word"])} · {e["case"]["view"]} · f{e["case"]["index"]} — {"새 중복 차단" if e["id"] in (3,4) else "남아 있는 기존 오류"}</summary><p>{html.escape(r["basis"])}</p><a href="{path}"><img src="{path}" alt="원본과 저장 관절 앞뒤 비교"></a></details>'
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>팔 연결·중복 검출 보완 결과</title><style>body{{margin:0;background:#eef3f8;color:#172b42;font-family:Malgun Gothic,Arial,sans-serif;line-height:1.7}}main{{max-width:1120px;margin:auto;padding:48px 28px}}h1{{font-size:34px;line-height:1.3}}.lead{{font-size:20px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}}article,section,details{{background:white;border-radius:16px;padding:24px;margin:16px 0}}article strong{{display:block;color:#076f85;font-size:28px}}article span{{font-size:14px}}summary{{font-size:19px;font-weight:bold;cursor:pointer}}img{{width:100%;height:auto}}pre{{white-space:pre-wrap;font:inherit}}.note{{background:#fff1d6;padding:20px;border-radius:12px}}a{{color:#076f85}}@media(max-width:700px){{.cards{{grid-template-columns:1fr 1fr}}main{{padding:24px 15px}}h1{{font-size:27px}}}}</style><main><p>WordsInSilence · 공개 자료 관절배정 후보 검증</p><h1>겹쳐 잡힌 같은 손을<br>두 손으로 부르지 않도록</h1><p class="lead">팔의 연결과 손가락 영역을 함께 확인해 잘못된 중복 배정2건을 추가로 보류했습니다.</p><div class="cards">{cards}</div><div class="note">보류는 틀린 좌우 이름을 사용하지 않는다는 뜻입니다. 사라진 손을 복원하거나 수어 단어를 더 잘 맞췄다는 결과는 아닙니다. 두 변경은 같은 소복 영상의 서로 다른 검출 조건입니다.</div><section><h2>계속 추적해도 왜 혼동이 남나요?</h2><p>처음 좌우를 잘못 붙인 뒤 그 이름을 계속 이어갈 수 있습니다. 겹침 뒤 재등장뿐 아니라 첫 등장, 같은 손의 중복 검출, 잘못 이어지는 추적도 검토 대상입니다.</p><p>현재 관측된 팔 연결을 더 활용하되, 손목이 안 보이면 팔꿈치만으로 좌우를 강제 결정하지 않습니다. 정상적인 손 교차는 그대로 유지합니다.</p></section><h2>원본으로 확인한 변경과 남은 문제</h2>{pictures}<section><h2>전체 결과와 검증 범위</h2><pre>{html.escape(text)}</pre></section><p><a href="RESULT.json">기계 판독 결과</a> · <a href="INDEPENDENT.json">별도 계산 검산</a> · <a href="VISUAL_REVIEW.json">원본 검토 기록</a></p></main></html>'''
    (U/'RESULT_KO.html').write_text(page,encoding='utf-8')
    pinscheck(c['pins']);pinscheck(read(U/'OUTPUT_PINS.json'))
    write(U/'FINAL_AUDIT.json',{'status':'PASS','at':at,'candidate_sha256':sha(U/'candidate.py'),'protocol_sha256':sha(U/'CONTEXT.json'),'protected_pins':len(c['pins']),'production_changed':False,'training':0,'verified_files':{str(p):sha(p) for p in [U/'RESULT.json',U/'RESULT_KO.txt',U/'RESULT_KO.html',U/'INDEPENDENT.json',U/'VISUAL_REVIEW.json',U/'INTEGRATION_CHECKS.json',U/'verify.py',U/'render_evidence.py']}})
    # Synchronize current summaries without changing historical frozen units.
    summary='ops/word100_visible_arm_duplicate_20261018. 현재 화면 안 팔 연결+손바닥 영역+손가락 관절 영역 중복을 쓰는 별도 후보 구현·검증 완료. 공개 TRAIN899 두조건226362프레임/305602후보 중305600배정동일, 소복 REAL05 masked84/sensor88 중복오배정2건 추가보류(같은영상 두조건). 정상441/441·교차315/315 보존. 고정감사13오류 중 누적3보류·10미해결, 정답손복원0. 추가 팔불일치7프레임 원본검토:2새보류+기존오배정지속5(후원2·추출2·공항1) 확인; 새후보가 만든오류아님, 기존13감사는전체오류총량아님. 전체출력결정/임시1285전부 포함 기하13670프레임 별도scalar검산PASS, 계약18·연계/순서1010·prefix502·미관측좌표10630교란PASS. 보호16215핀유지. 표시원본캐시누락1은 최초오류보존후 기존로컬ZIP의 검증된해당영상만확보, 후보계수불변. C/TCN/DTW/웹유지, 학습/추론/사용자·멘토·DEV·보류사용0. Codex단일담당구현·시각검토+별도수치검산; 인식률/새사용자성능미검증.'
    state=read(O/'STATE.json');state['current_stage']='WORD100_VISIBLE_ARM_DUPLICATE_VALIDATED_AWAITING_REVIEW';state['next_single_action']=next_action
    for key in ('updated_at','updated_at_kst'):
        if key in state:state[key]=at
    state[U.name]={**result,'result':str(U/'RESULT_KO.html'),'next_plan':str(U/'NEXT_PLAN.json')};write(O/'STATE.json',state)
    context=(ROOT/'WordsInSilence_PROJECT_CONTEXT.md').read_text(encoding='utf-8')
    for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','2026-10-18 화면 안 팔 연결·왜곡 중복 후보 검증 완료',summary),('## 9. 다음 단일 작업','2026-10-18 팔 연결·중복 후보 결과 검토·학습 대기',next_action)]:
        assert context.count(marker)==1
        context=context.replace(marker,marker+'\n\n### '+title+'\n'+body+'\n',1)
    (ROOT/'WordsInSilence_PROJECT_CONTEXT.md').write_text(context,encoding='utf-8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'# WordsInSilence 최신 인수인계\n\n갱신 {at}\n\n### 현재 목표·마지막 완료\n{summary}\n\n### 다음 단일 작업\n{next_action}\n\n### 근거·실행\n- 단위 research/front_keyphase_v1/ops/{U.name}\n- CONTEXT: 승인/고정조건/보호핀, candidate.py: 별도후보. RESULT_KO/RESULT/NEXT_PLAN: 결과와 다음단계.\n- SUMMARY/CHANGES/ANNOTATED_COMPARISON/ARM_CONFLICTS/VISUAL_REVIEW/INDEPENDENT/INTEGRATION_CHECKS/FINAL_AUDIT: 검증근거.\n- 공개재생 exec15913 종료0, 별도검산 exec60928 종료0, 영상표시 exec72343 종료0. 현재실행/잠금없음. st-gcn 자동화삭제유지.\n- 학습·분류추론·운영웹교체 없음. 신규후보를 기존 C의 인식률향상으로 보고하지 않는다.\n- 원본/W:/sources/과거결과읽기전용. 사용자자료학습·튜닝·후보선택금지.\n',encoding='utf-8')
    assert read(O/'STATE.json')['current_stage']=='WORD100_VISIBLE_ARM_DUPLICATE_VALIDATED_AWAITING_REVIEW'
    assert not (U/'RUNNING.json').exists()
    print('FINALIZED',str(U/'RESULT_KO.html'),flush=True)
if __name__=='__main__':main()
