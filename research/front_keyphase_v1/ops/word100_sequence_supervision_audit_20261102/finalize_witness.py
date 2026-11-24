from sequence_model import *
ROOT=J.parents[3]
def main():
    assert not (J/'FINAL.json').exists();v=read(J/'VERIFIED.json');diag=read(J/'DIAGNOSIS.json');prov=read(J/'PROVENANCE_AUDIT.json');boundary=read(J/'BOUNDARY_AUDIT.json');at=clock()
    assert v['status']=='WITNESS_SEQUENCE_COMPONENT_NOT_READY'
    summary='J대기관측보존/확인시점순서보조2조건80epoch학습·176개개발비교·별도검산완료. 主witness_ctc42 주요원판정585/615·역할반영583/615·정상순서2/21·짧은역순0/3, 대기96/107로F30/107에서기존T수준복구. CE대조도583/615·순서1/21·대기96/107. 전체개선미입증/웹미채택. FIT36원본충돌0이나DEV정상28조건중19에서대기/주요판정충돌. 원본7충돌은첫0~3프레임,속도변형등12조건은더긴구간도있어초기주장정정. FIT216은live입력,DEV12native는과거ROI입력/76은live로혼합된현재계보확인. 같은FIT원본36개도oldROI/live RGB가모두다름. 혼합은전부원인이라는뜻아님. 전체goal active.'
    nxt='완료J2학습176비교/F/E/D재실행금지. word100_coherent_sensor_input_20261102에서동일전체센서처리조건입력부터준비한다. 공개DEV11~13원본12개를새경로로학습FIT와같은capture/RGB/시간/좌표계약으로추출하고새입력의24고정모델진단을동결해비교한다. 기존완료점수수정/재채점은하지않는다. 전체frontend차이를묶은입력비교이며RGB만의원인으로단정금지. 같은조건에서도남는실패에한해FIT측변이/감독을보완;목표순서강제/시작구간일괄삭제금지. 기존C/T3F/웹·사용자/멘토/H14·15보존.'
    lines=['대기 관측 보존과 확인된 시점의 순서 학습 비교',at,'',
      '이번 비교는 끝났지만, 전체 피드백 시스템은 아직 완성되지 않았다. 새 모델은 웹에 적용하지 않았다.','',
      '쉬운 설명',
      '기존에는 대기 동작을 잘 맞히던 결과가 있었는데, 새 판정부가 이를 다시 판단하면서 놓쳤다. 이번에는 기존 대기 정보를 그대로 연결했다. 또 직접 확인한 주요 동작들의 순서를 함께 배우는 조건을 비교했다.',
      '순서 정답은 모든 프레임을 설명하는 완전한 정답이 아니므로, CTC 순서 학습은 직접 확인한 시점들에만 적용했다. 확인하지 않은 중간 순간에 정답을 지어 넣지 않았다.','',
      '조건 | 주요 동작 원판정 | 최종 역할 반영 | 정상 순서 확인 | 짧은 역순 발견 | 대기 정답']
    for name,r in v['conditions'].items():lines.append(f"{name} | {r['raw_action']['correct']}/615 | {r['effective_action']['correct']}/615 | {r['normal_order']}/21 | {r['local_reverse']}/3 | {r['normal_rest_correct']}/107")
    lines+=['','이전 F 주후보는 대기30/107, 주요역할반영583/615, 정상순서2/21, 짧은역순1/3이었다. 대기는 기존 T 수준96/107을 회복했지만 새로운 최고 성능은 아니다. 주요동작과 순서는 개선됐다고 할 수 없고, 짧은 역순 발견은 오히려 줄었다.',
      '615는 공개 영상의 정답이 있는 주요 시점 수이다. TCN 단어 인식률이 아니며, 같은 원본에서 만든 속도변형 등이 포함된다. 28정상조건 중 순서가 정의된3표현의21조건을 순서 분모로 쓴다. 조상의 큰 행동가족1개는 교대반복 순서나 횟수 정답을 뜻하지 않는다.',
      '특징추출 부분은 고정했고, 행동 해석 출력과 수행/비수행 시간 판정부를 실제 학습했다. 두 조건은 같은 자료·초깃값·학습횟수이며 확인시점 순서손실 유무만 다르다. 원래의 시간축과 손모양·관계·이동 근거는 보존했다.','',
      '새로 확인한 원인 후보',
      '학습 원본36개에서는 대기와 수행 판단의 충돌이 없었다. 개발 자료에서는28개 정상조건 중19개에서 충돌했다. 따라서 학습 자료를 더 오래 외우게 하는 것만으로 해결됐다고 볼 수 없다.',
      '원본 영상7개의 충돌은 맨 처음0~3프레임에 몰렸다. 그러나 재인코딩/속도변형/대기잘림의12조건에는 더 긴 앞뒤 구간도 있었다. 중간 보고에서 전체 정상조건의 충돌을 짧은 시작 문제라고 설명한 것은 범위가 너무 넓었으며 이 기록으로 바로잡는다.',
      '현재 입력 계보도 확인했다. 학습216입력은 실제 전체센서 처리 경로인데, 개발88입력 중 원본12개는 과거 상반신영역 처리값을 쓰고76개는 실제 처리 경로를 쓴다. 똑같은 학습 원본36개도 옛 입력과 현재 입력의 RGB특징이 달랐다. 이 차이는 앞선 입력 감사에도 기록돼 있었고, 이번에 현재 판정부까지 이어진 경로를 확인했다.',
      '속도변형 등 전체센서 입력에서도 실패가 있으므로 이 혼합이 유일한 원인이라고 단정하지 않는다. 다만 동일 조건으로 비교하기 전에 모델 구조나 학습 방식만 원인이라고 결론 내릴 수 없다.',
      '확인된 시작 프레임도 이미 학습 정답에 포함되어 있었다. 시작4프레임을 자동으로 삭제하거나 무조건 대기라고 처리하는 방식은 쓰지 않는다.','',
      '검증과 제한',
      '176개 저장 관측배열, 대기 확률의 정확한 보존, 원래 시간/문구 연결, 누락 보류, 주요 이동 삭제 차단과 동결 가중치를 별도 계산으로 확인했다. 기존 정답 동작2시점이 해석에서 잘못 제외된 문제도 그대로 실패로 계산했다.',
      f"새 작은 판정부 계산 중앙값은 주후보 약{v['conditions']['witness_ctc_42']['head_median_seconds']*1000:.1f}ms이며 전체 영상분석 시간은 새로 측정하지 않았다.",
      'FIT기록ID01~09/DEV11~13 분리. DEV는 반복 분석된 개발자료로 독립 최종평가가 아니다. 같은Codex의 검산이며 수어 전문가의 독립 정답 확인과 다르다.',
      '초기 import 파일명 충돌은 학습 전 실패를 보존한 뒤 수정했다. 이후 사전검증/2학습/176비교/별도검산은 완료했다. 사용자·멘토자료 접근/학습/튜닝/선택0, C·T3-F·기존웹 변경0. 명확한 전체 개선이 없어 HTML은 만들지 않았다.','',
      '다음은 입력 처리 조건부터 통일한다. 새 공개 원본12개 입력을 별도 경로에서 만들고 같은 고정모델로 진단한다. 과거 점수를 덮어쓰지 않는다. 이 비교 후에도 남는 문제에 대해서만 학습자료의 변이와 동작 구분 감독을 보완한다.']
    (J/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    exclusive(J/'FINAL.json',dict(at=at,status=v['status'],adopted=False,web_changed=False,training_runs=2,comparisons=176,fit_diagnostic_inputs=36,
        result=str(J/'RESULT_KO.txt'),verified=str(J/'VERIFIED.json'),provenance=str(J/'PROVENANCE_AUDIT.json'),next=nxt,whole_goal_complete=False))
    write(J/'CURRENT_EXECUTION.json',dict(at=at,status='COMPLETE_NOT_ADOPTED',active=None,train_session=38902,train_exit=0,evaluation_session=3915,evaluation_exit=0,verify_exit=0,provenance_session=75517,provenance_exit=0,next=nxt,whole_goal_complete=False))
    path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=path.read_text(encoding='utf8')
    for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','大기관측복구·순서보조2비교 완료/입력조건 혼합 확인',summary),('## 9. 다음 단일 작업','동일 전체센서 입력으로 원인분리',nxt)]:
        h='### 2026-11-02 '+title;assert h not in txt;txt=txt.replace(marker,marker+'\n\n'+h+'\n\n'+body+'\n',1)
    path.write_text(txt,encoding='utf8');state=read(O/'STATE.json');state['current_stage']='WITNESS_SEQUENCE_COMPLETE_INPUT_CONTRACT_AUDIT';state['updated_at']=at;state['next_single_action']=nxt
    state[J.name]=dict(at=at,status=v['status'],active=None,result=str(J/'RESULT_KO.txt'),verified=str(J/'VERIFIED.json'),summary=summary,whole_goal_complete=False);write(O/'STATE.json',state)
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n'+str(J/'RESULT_KO.txt')+'\n'+str(J/'PROVENANCE_AUDIT.json')+'\n\nJ실행모두exit0/현재active=None. 같은Codex검산≠독립전문가정답. 새HTML없음.\n',encoding='utf8')
    for f in ['FINAL.json','CURRENT_EXECUTION.json','VERIFIED.json','PROVENANCE_AUDIT.json']:read(J/f)
    assert read(O/'STATE.json')[J.name]['summary']==summary and summary in path.read_text(encoding='utf8') and summary in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
    exclusive(J/'FINAL_VERIFICATION.json',dict(at=at,status='PASS',documents_agree=True,files_exist=True,json_valid=True,whole_goal_complete=False))
    print('J finalized; full goal remains active')
if __name__=='__main__':main()
