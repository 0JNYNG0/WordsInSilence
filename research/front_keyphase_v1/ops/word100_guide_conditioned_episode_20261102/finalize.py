"""Close this fixed experiment; keep the overall user goal active."""
from condition_model import *
ROOT=E.parents[3]

def main():
    assert not (E/'FINAL.json').exists()
    v=read(E/'VERIFIED.json');plan=read(E/'FROZEN.json');train=read(E/'TRAINING_RESULT.json');scope=read(E/'EXPERIMENT_SCOPE_NOTE.json');audit=read(E/'OBSERVATION_INTERPRETATION_AUDIT.json')
    assert len(train['runs'])==4 and v['verified_observer_arrays']==352 and v['status']=='FACTORS_TESTED_NOT_READY'
    assert audit['existing_face_hint_confident']==len(audit['rows'])==19
    at=clock();summary='공개FIT36영상에 준비·복귀174/대기42시점을 추가하고96애매경계를 보류한 뒤, 기존조건부손상태정보 유무×추가구간감독 유무4조건을seed42로 학습하고352새비교·별도검산을 마쳤다. 主둘다사용 조건은 주요행동554/615(90.08%), 정상순서3/21, 국소역순1/3으로 통과하지 못했다. 동일구조 대조587/615(95.45%)보다 주요행동이 저하했다. 주동작→복귀 오분류는1시점뿐이며61주동작오답 중60은 다른주동작과 혼동했다. 얼굴앞동작19오답에는 기존입력의얼굴앞관측이 모두0.8이상으로 남아있어 단순정보누락만으로 설명되지 않는다. 웹미채택/기존관측352배열·시간축·문구연결·누락차단보존. 전체goal active.'
    nxt='E완료4학습·352비교 재실행 금지. OBSERVATION_INTERPRETATION_AUDIT와EXPERIMENT_SCOPE_NOTE를 근거로 다음은 관측을 더지우는 보완이 아니라 관측과해석의 일관성을 학습하는 분리구조를 검토한다. 공개FIT직접정답만으로 손모양/신체관계 관측과 설명의 행동귀속이 어떤관계인지 계약을 만들고, 정상관측을 고정한 문맥학습과 공동갱신을 작은 고정비교로 분리한다. 기존모델확률을 무조건정답으로 복제하거나 준비복귀예측으로 주요프레임을 지우지 않는다. 실제수행과 끝자세는 동시에성립할수있으므로 배타적분류로 밀어내는지도 먼저점검한다. 검증통과범위만 목표불일치시험·안정성·전체지연·격리데모로 이어간다. H14/15·사용자/멘토자료학습선택·C/T3F·운영변경금지. 현재실행중없음.'
    lines=['이번 작업 결과 — 설명 해석과 준비·복귀 학습을 나눈 비교',at,'',
      '결론: 네 조건의 학습과352비교를 완료했다. 일부 순서 확인은 나아졌지만 주요 동작 분류가 나빠져 새 후보를 웹에 적용하지 않았다. 전체 피드백 목표는 아직 진행 중이다.','',
      '1. 무엇을 바꿨나',
      '공개 학습 영상36개(4표현×기록ID01~09)의 시간순 표본을 확인했다. 기존850정답은 보존하고 준비·복귀174개와 쉬는 자세42개를 추가했다. 애매한 경계96개는 정답으로 채우지 않았다. 모든 영상의 원래 시간축은 유지했다.',
      '같은 신경망 구조·초기값·seed42·80epoch로 네 조건을 비교했다. 기존 SignRep 특징을 사용하고 작은 설명/행동 해석 부분의 가중치를 학습했다. 새 백본 다운로드/학습은 하지 않았다.',
      '조건정보란 기존 목표표현 조건부 손상태10종 판정값이다. 목표 단어 이름이나 문장 이해만의 효과를 독립적으로 증명하는 시험은 아니다. 추가구간감독에는 새 준비·복귀와 대기 정답 및 그에 따른 클래스 가중치가 함께 포함된다.','',
      '2. 같은 구조끼리 비교한 결과',
      '조건 | 주요 행동 정답 | 정상 순서 확인 | 짧은 국소역순 발견 | 대기 정답']
    names={'guide0_context0_42':'추가 정보/구간 감독 없음','guide1_context0_42':'기존 손상태 정보만 추가','guide0_context1_42':'준비·복귀/대기 감독만 추가','guide1_context1_42':'둘 다 추가(사전 지정 후보)'}
    for tag_,r in v['conditions'].items():
        lines.append(f"{names[tag_]} | {r['normal_action_correct']}/{r['normal_action_known']} ({100*r['normal_action_correct']/r['normal_action_known']:.2f}%) | {r['normal_order']}/21 | {r['local_reverse_detected']}/3 | {r['normal_rest_correct']}/{r['normal_rest_known']}")
    lines+=['','615는 정상 조건에서 검토 정답이 있는 주요 행동 시점의 수이다. 단어 인식률이 아니다. 같은 원본의 속도 변형 등이 포함되어615개 독립 영상/사람을 뜻하지 않는다. 정상 순서21조건은 지원 순서 관계가 있는3표현의 원본/정상변형이다. 나머지1표현의 반복 횟수·언어 순서 규칙을 새로 단정하지 않았다.',
      '학습 기록ID01~09와 개발ID11~13을 분리했다. 개발 자료는 이미 여러 번 확인한 자료이며 새 독립 최종평가가 아니다. H14/15는 다시 쓰지 않았다.',
      '직전 D의 주후보는585/615,정상순서0/21이었다. 그보다 순서3개가 확인됐지만, 기존 보존 A의12/21에도 못 미친다. D와는 출력 종류가 다르므로 순수한 조건 효과는 위의 동일구조 네 조건끼리 해석해야 한다. 다른 조건5/21을 사후에 골라 주후보로 바꾸지 않았다.','',
      '3. 원인이 어떻게 더 분명해졌나',
      '추가 준비·복귀 감독은 일부 순서 확인에 도움을 줬지만 주요 행동끼리의 구분을 약하게 만들었다. 주후보의61개 주요 행동 오답 중 복귀로 잘못 분류한 것은1개이며 나머지60개는 다른 주요 행동으로 분류했다. 따라서 단순히 복귀 장면을 더 넣거나 더 많이 제외하는 방식으로 해결된다고 볼 수 없다.',
      '얼굴 앞 손 제시를 다른 행동으로 해석한19시점 모두, 입력에 남아 있던 기존 손상태 판정은 얼굴 앞 상태를0.8이상으로 제시했다. 독립 정답이 아닌 기존 모델의 힌트이므로 무조건 따라야 한다는 뜻은 아니다. 다만 정보가 사라진 문제만 있는 것이 아니라, 새 해석 부분이 이용 가능한 근거를 제대로 유지/사용하는지도 확인해야 한다는 구체적 증거다.',
      '어느 손 자세가 보이는지와 어떤 동작을 수행 중인지, 준비/복귀인지가 하나의 배타적 분류에서 경쟁한다. 이 설계가 오류의 유일한 원인이라고 확정하지 않는다. 수행 중인 움직임과 끝자세처럼 동시에 성립할 수 있는 관측을 억지로 하나만 고르게 하는지 다음 입력·구조 계약에서 확인한다.','',
      '4. 유지한 부분과 한계',
      '352개 저장 출력의 기존 관측 배열, 전체 프레임 수, 가이드 문구·시간 연결을 별도 계산으로 확인했다. 이미지 누락은 잘 맞았다고 통과시키지 않았고, 주요 팔 이동을 자른 시험의 차단도 유지했다. 정상의 명백한 역순 오경고나 역순의 정상 통과는 없었지만, 이는 대부분을 확인 보류한 결과와 함께 보아야 한다.',
      f"새 작은 판정부만의 중앙 계산시간은 약{v['conditions'][plan['primary']]['head_median_seconds']*1000:.1f}ms이다. 영상 관측 등을 포함한 전체 처리시간을 이번에 새로 재지 않았으므로 사용자 대기시간이 줄었다고 주장하지 않는다.",
      '같은 Codex가 시각 검토와 별도 검산을 수행했다. 독립 수어 전문가의 정답 검증은 아니다. 실제 수어의 모든 허용 범위나 모든 단어에 대한 성능을 증명한 결과도 아니다.',
      'C·T3-F/기존 웹/기존 완료 결과는 그대로 보존했고 사용자·멘토 촬영은 접근·학습·튜닝·선택에 쓰지 않았다. 새 HTML은 만들지 않았다.','',
      '5. 다음 작업',
      '① 공개FIT의 직접 관측 정답과 행동 설명의 대응을 점검한다. 준비/복귀, 손 자세, 실제 움직임을 서로 대체하는 정답으로 취급하지 않는다.',
      '② 잘 맞던 관측 부분을 고정한 상태에서 문맥 해석을 배우는 구조와 공동으로 갱신하는 구조를 비교한다. 보존은 파일만 그대로 두는 것이 아니라 최종 해석이 관측 근거와 충돌하지 않는지도 확인한다.',
      '③ 기존 주요 동작, 짧은 변화, 이동 누락, 정상 속도/손내림/교차에 대한 같은 기준을 통과한 경우에만 안정성·목표불일치·전체 처리시간 시험과 별도 데모로 간다.',
      '새 조건의 결과를 보고 통과 기준을 낮추거나, 보류를 정답으로 바꾸거나, 특정 단어 예외를 추가하지 않는다.']
    (E/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    exclusive(E/'FINAL.json',dict(at=at,status=v['status'],adopted=False,web_changed=False,training_runs=4,new_comparisons=352,main_correct=554,main_known=615,normal_order=3,local_reverse=1,normal_main_as_context=1,existing_face_hint_present_in_all19=True,old_observer_arrays_preserved=352,review=str(E/'TARGETS_VERIFIED.json'),verification=str(E/'VERIFIED.json'),result=str(E/'RESULT_KO.txt'),next=nxt,whole_goal_complete=False))
    write(E/'CURRENT_EXECUTION.json',dict(at=at,status='COMPLETE_NOT_ADOPTED',active=None,training_session=55688,training_exit=0,evaluation_session=83170,evaluation_exit=0,verification_session=28102,verification_exit=0,diagnosis_exit=0,next=nxt,whole_goal_complete=False))
    path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=path.read_text(encoding='utf8')
    for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','조건분리4학습·352비교 완료/미채택',summary),('## 9. 다음 단일 작업','관측과 행동해석의 충돌부터 분리',nxt)]:
        header=f'### 2026-11-02 {title}';assert header not in text;text=text.replace(marker,marker+'\n\n'+header+'\n\n'+body+'\n',1)
    path.write_text(text,encoding='utf8')
    state=read(E.parent/'STATE.json');state['current_stage']='GUIDE_CONTEXT_FACTORIAL_NOT_ADOPTED';state['next_single_action']=nxt;state['updated_date_kst']='2026-11-02';state[E.name]=dict(at=at,status=v['status'],active=None,result=str(E/'RESULT_KO.txt'),verified=str(E/'VERIFIED.json'),diagnosis=str(E/'DIAGNOSIS.json'),observation_interpretation_audit=str(E/'OBSERVATION_INTERPRETATION_AUDIT.json'),summary=summary,next=nxt,whole_goal_complete=False);write(E.parent/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n'+str(E/'RESULT_KO.txt')+'\n'+str(E/'VERIFIED.json')+'\n'+str(E/'OBSERVATION_INTERPRETATION_AUDIT.json')+'\n\n실행 종료: train55688 exit0, eval83170 exit0, verify28102 exit0. 현재 active=None. 같은Codex검산≠독립전문가정답. 새HTML없음.\n',encoding='utf8')
    for name in ['FINAL.json','CURRENT_EXECUTION.json','FROZEN.json','TRAINING_RESULT.json','EVALUATION.json','VERIFIED.json','DIAGNOSIS.json','EXPERIMENT_SCOPE_NOTE.json','OBSERVATION_INTERPRETATION_AUDIT.json']:read(E/name)
    assert summary in path.read_text(encoding='utf8') and summary in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8') and read(E.parent/'STATE.json')[E.name]['summary']==summary
    exclusive(E/'FINAL_VERIFICATION.json',dict(at=at,status='PASS',documents_agree=True,json_parsed=True,result_exists=True,models_not_adopted=True,active=None,whole_goal_complete=False))
    print('Final report and root documents synchronized; whole goal remains active')
if __name__=='__main__':main()
