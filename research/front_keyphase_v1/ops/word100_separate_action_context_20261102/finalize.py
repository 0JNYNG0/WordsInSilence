from model import *
ROOT=F.parents[3];J=O/'word100_sequence_supervision_audit_20261102'
def main():
    assert not (F/'FINAL.json').exists();v=read(F/'VERIFIED.json');diag=read(F/'DIAGNOSIS.json');rest=read(F/'REST_PRESERVATION_AUDIT.json');weak=read(J/'SEQUENCE_INPUT_AUDIT.json');at=clock()
    assert v['status']=='SEPARATE_CONTEXT_COMPONENT_NOT_READY' and v['old_observer_arrays_preserved']==176
    summary='F행동내용/과정역할2출력의 같은구조를특징고정/공동갱신2조건으로학습하고176비교·별도검산을완료했다. 主특징고정의주요행동원판정585/615는D와같았고역할해석후583/615(2시점손실),정상순서2/21·국소역순1/3으로미통과/웹미채택이다. 공동갱신은577/615·순서2/21·역순0/3. 손상태T관측176배열·문구/시간·누락차단보존과동결특징가중치일치를검산했다. 같은대기정답107시점에서기존T저장결과96개정답인데새역할판정부는30개여서,잘되던대기관측을다시못배우는문제를명확히확인했다. J단위에서FIT216입력의순서증거목록을준비했으나완전한전사정답/CTC학습으로간주하지않는다. 전체goal active.'
    nxt='F완료2학습·176비교/E완료4학습352비교재실행금지. J word100_sequence_supervision_audit_20261102의SEQUENCE_INPUT_AUDIT와F REST_PRESERVATION_AUDIT를이어받는다. 먼저잘학습된T대기관측을명시적으로보존하면서활성구간의수행/준비복귀만배우는구조와관측충돌보류계약을FIT에서점검한다. 이어216순서목록의약한지도범위를확인하고,시점별직접정답을보존한전체순서보조학습(예:CTC)의조건을결과전동결한다. CTCblank는대기/미검출/무의미/정확구간경계가아니며전체예상순서를디코딩에강제하지않는다. WORD1567의단일행동가족[5]는교대반복세부순서/횟수정답이아니므로기존양손상태·방향관측을유지하고별도로다룬다. 틀린모델확률을자동정답으로복제·DEV/H/private튜닝·주요시간삭제금지. 실제기능통과후안정성/목표불일치/전체시간/격리데모. 현재활성프로세스없음.'
    lines=['행동내용과 과정역할을 분리한 비교 결과',at,'',
      '이번 단위는 완료했지만 전체 피드백 데모 목표는 아직 진행 중이다. 새 후보를 웹에 적용하지 않았다.','',
      '무엇을 했나',
      '공개FIT36영상·216준비입력에서 직접 검토한 행동과 준비/복귀/대기를 서로 다른 출력으로 학습했다. 잘 배운 영상 특징을 고정한 조건과 그 특징도 함께 수정한 조건을 같은구조·seed42·80epoch로 비교했다. 행동 해석 readout과 과정역할은 두 조건 모두 실제 학습했다. 기존C·T3-F와원본손상태/모양/관계/이동판정은바꾸지않았다.',
      '손 자세와 행동 과정의 직접정답을 대조하여, 같은 자세가 다른 행동 과정에서 나타난 사례를 확인했다. 예를 들어 팔을 따라 움직이는 중에도 이미 손목 부근에 도달한 자세가 동시에 관측됐다. 두 사실을 서로 없애는 정답으로 바꾸지 않았다.','',
      '조건 | 주요동작 원판정 | 역할을 반영한 판정 | 정상 순서 확인 | 짧은 국소역순 발견']
    for name,r in v['conditions'].items():lines.append(f"{name} | {r['raw_action']['correct']}/615 | {r['effective_action']['correct']}/615 | {r['normal_order']}/21 | {r['local_reverse']}/3")
    lines+=['','주후보는 결과 전에 정한 frozen_encoder42다. 이전 E 주후보554/615와 달리 주요 동작 원판정585/615를 보존했지만, 이전 D585/615보다 개선된 것은 아니다. 역할을 적용할 때2시점을 잘못 제외해583/615가 됐다. 더 이전 보존 기준의 정상 순서12/21에도 아직 못 미친다.',
      '615는 검토 정답이 있는 주요 동작 시점 수이며 TCN단어 인식률이 아니다. 동일 원본의 속도/대기변형이 포함된다. 학습 기록ID01~09와 개발ID11~13을 분리했지만 개발 자료는 이미 반복 검토한 자료다. 독립 최종평가H14/15는 사용하지 않았다.','',
      '이번에 더 분명하게 확인한 문제',
      '특징을 고정하면 기존 동작 구분을 보존하는 데에는 도움이 됐다. 그러나 준비·복귀·대기를 새로 판단하는 부분이 잘 되던 관측을 제대로 활용하지 못했다.',
      '정확히 같은107대기시점을 대조했다. 이미 보존한 T손상태 모델의 저장 결과는96개를 맞혔지만, 새 역할 판정부는30개만 맞혔다. 기존 파일을 그대로 둔 것만으로는 충분하지 않다. 최종 해석에서도 그 관측을 실제로 보존하고 활용해야 한다.',
      '현재 학습은 시점별 행동/역할 정답을 중심으로 한다. 시간문맥을 쓰지 않는다는 뜻은 아니다. 다만 결과에서 요구하는 전체 동작 순서를 직접 맞추는 학습 목표와는 차이가 있다. 이 차이도 다음 보완에 포함한다.','',
      '보존과 검증',
      f"동결 대상 특징 가중치{v['exact_frozen_encoder_tensors']}개 텐서가 원본과 일치했다. 기존 관측176개 배열과 원래 시간축, 가이드 문구/시간 연결, 이미지 누락과 주요 이동 삭제 차단을 별도 계산으로 확인했다.",
      '정상 역순 오경고와 역순을 정상으로 통과시킨 사례는 이번 주후보에서0이지만, 정상 순서 확인이2/21에 불과하므로 그 사실만으로 성공이라고 하지 않는다.',
      f"작은 새 판정부만의 중앙 계산시간은 약{v['conditions']['frozen_encoder_42']['head_median_seconds']*1000:.1f}ms이다. 전체 영상분석 대기시간을 새로 재지 않았으므로 전체가 빨라졌다고 주장하지 않는다.",
      '같은 Codex가 구현·시각 정답 검토·별도 검산을 수행했다. 독립 수어 전문가 정답 검증은 아니다. 사용자·멘토 촬영 접근/학습/튜닝/선택0, 기존웹·C·T3-F변경0. 새HTML없음.','',
      '다음 작업과 준비한 자료',
      '1) 잘 맞던 대기 관측은 유지하고, 실제 활동 중 준비/수행/복귀를 해석하는 부분만 보완한다. 모델끼리 충돌하면 무조건 기존모델을 정답으로 강제하지 않고 충돌로 표시한다.',
      '2) FIT216입력에서 직접 확인된 행동들의 순서 목록을 준비했다. 원본9개 기록ID의 각 표현에서 관측된 큰 순서는 일치했다. 그러나 중간 모든 행동이 완전히 주석됐다는 뜻은 아니어서 약한 지도 후보로만 저장했다. 아직 추가 학습은 없다.',
      '3) 시점별 정답을 보존하면서 전체 순서도 학습하는 보조 손실을 검토한다. CTC는 경계가 정확히 나뉘지 않은 입력과 순서 정답의 여러 대응을 학습하는 방법이다. 우리 한국수어 피드백의 개선을 보장하지 않으므로 작은 비교를 먼저 고정한다.',
      'CTC의 blank출력은 손내림, 검출누락, 중요하지 않은 행동, 정확한 구간 경계와 다르다. 이것으로 관측을 지우지 않는다. 예상 정답 순서를 디코딩에 강제로 넣어 역순을 숨기지도 않는다. 교대반복 표현의[5]한토큰은 실제 교대순서/횟수 정답이 아니며 기존 양손상태·움직임방향 관측은 그대로 필요하다.',
      '기술 근거: https://www.cs.toronto.edu/~graves/icml_2006.pdf',
      '공식 구현 설명: https://docs.pytorch.org/docs/2.15/generated/torch.nn.CTCLoss.html',
      '4) 실제 기능 비교를 통과한 뒤에만 안정성, 목표 표현 불일치, 전체 처리시간과 별도 웹 데모를 검증한다.']
    (F/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    exclusive(F/'FINAL.json',dict(at=at,status=v['status'],adopted=False,web_changed=False,training_runs=2,comparisons=176,verified=str(F/'VERIFIED.json'),rest_audit=str(F/'REST_PRESERVATION_AUDIT.json'),next_inputs=str(J/'SEQUENCE_INPUT_AUDIT.json'),result=str(F/'RESULT_KO.txt'),next=nxt,whole_goal_complete=False))
    write(F/'CURRENT_EXECUTION.json',dict(at=at,status='COMPLETE_NOT_ADOPTED',active=None,train_session=58891,train_exit=0,evaluation_session=67551,evaluation_exit=0,verify_exit=0,diagnose_exit=0,next=nxt,whole_goal_complete=False))
    exclusive(J/'SCOPE_NOTE.json',dict(at=at,status='INPUT_AUDIT_ONLY',whole_order_training=0,scope='Coarse[5]does not encode alternating sub-actions or repetition count. No blanket CTC admission from sparse witnesses. First preserve successful resting-state observation and audit weak-sequence contract.',sources=weak['sources'],whole_goal_complete=False))
    path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=path.read_text(encoding='utf8')
    for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','분리구조2학습·176비교 완료/대기관측활용 문제 확인',summary),('## 9. 다음 단일 작업','대기관측 보존과 전체흐름 감독 점검',nxt)]:
        h=f'### 2026-11-02 {title}';assert h not in text;text=text.replace(marker,marker+'\n\n'+h+'\n\n'+body+'\n',1)
    path.write_text(text,encoding='utf8');state=read(O/'STATE.json');state['current_stage']='SEPARATE_CONTEXT_NOT_ADOPTED_SEQUENCE_INPUT_AUDIT';state['next_single_action']=nxt;state[F.name]=dict(at=at,status=v['status'],active=None,result=str(F/'RESULT_KO.txt'),verified=str(F/'VERIFIED.json'),summary=summary,whole_goal_complete=False);state[J.name]=dict(at=at,status=weak['status'],active=None,input_audit=str(J/'SEQUENCE_INPUT_AUDIT.json'),new_training=0,next=nxt,whole_goal_complete=False);write(O/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n'+str(F/'RESULT_KO.txt')+'\n'+str(F/'VERIFIED.json')+'\n'+str(F/'REST_PRESERVATION_AUDIT.json')+'\n'+str(J/'SEQUENCE_INPUT_AUDIT.json')+'\n\n실행종료: train58891exit0, eval67551exit0, verify/diagnose/Jprepare exit0. 현재 active=None. 같은Codex검산≠독립전문가정답. 새HTML없음.\n',encoding='utf8')
    for file in ['FINAL.json','CURRENT_EXECUTION.json','FROZEN.json','VERIFIED.json','DIAGNOSIS.json','REST_PRESERVATION_AUDIT.json']:read(F/file)
    assert summary in path.read_text(encoding='utf8') and summary in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8') and read(O/'STATE.json')[F.name]['summary']==summary
    exclusive(F/'FINAL_VERIFICATION.json',dict(at=at,status='PASS',json_parsed=True,documents_agree=True,results_exist=True,whole_goal_complete=False))
    print('F complete/not adopted; J input audit prepared; root documents synchronized; goal active')
if __name__=='__main__':main()
