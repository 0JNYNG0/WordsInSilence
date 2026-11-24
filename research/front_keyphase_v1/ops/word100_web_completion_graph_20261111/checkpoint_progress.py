"""Synchronize verified work and corrections, without promoting a model."""
from graph import *
O=U.parent;ROOT=O.parents[2]


def main():
 correction=dict(
  version=2,at=stamp(),supersedes_interpretation_only=True,
  prior_results_preserved=True,
  invalid_target='Treating all 9 partial-core-reversal pairs as proved whole-clip semantic order errors.',
  reason='Unreversed required-action fragments remain; source identity is not semantic action identity after direction reversal.',
  valid_targets=['Exact supplied endpoint ordering with known source indices',
                 'Selected-support-window source-role conflict',
                 'Normal direct-anchor coverage, direction-pair coverage and short arrival preservation',
                 'Whole-action direction/completion only when separately annotated'],
  current_gate='No candidate PASS merely from revised truth; normal component regression still fails.',
  scope='Separate Codex source/visual audit, not independent human/KSL semantic truth.',
  evidence=[str(U/'reviews/sequence_truth_audit.json'),str(U/'reviews/joint_correspondence_verification.json')])
 if not (U/'EVALUATION_CONTRACT_V2.json').exists():write(U/'EVALUATION_CONTRACT_V2.json',correction)
 g=read(U/'TASK_GRAPH.json')
 if not any('joint_correspondence_independent' in a['note'] for a in next(n for n in g['nodes'] if n['id']=='G2')['attempts']):
  record('G2','REWORK',dict(normal_preserved=False,false_support_reduced=False,not_abstention_only=False,three_seeds=True),
         [U/'JOINT_CORRESPONDENCE_COMPLETE.json',U/'reviews/joint_correspondence_verification.json',U/'EVALUATION_CONTRACT_V2.json'],
         'joint_correspondence_independent: Normal order alone hides 24→23 direct-anchor coverage and a fast endpoint lost by minimum duration. Not adopted. Partial reversal semantic denominator corrected, old scores preserved.')
 protections=read(U/'PROTECTIONS.json')
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protections.items())
 write(U/'PROTECTION_RECHECK.json',dict(at=stamp(),protected=len(protections),all_match=True,web_changed=False))
 msg='''웹 피드백 완성 그래프 G2 보완 중. 발표용 16:9 PNG·SVG를 만들고 확인했다. 이는 개발/검증 계획이며 구현 완료 아키텍처가 아니다.
순서 필터가 실제 다른 순서의 후보를 미리 버리는 원인을 확인했다. 새 연결 후보의 순서 수치만 보면 정상이 유지되지만, 별도 검산에서 정상 행동 구간 축소와 빠른 도착 근거 소실이 발견되어 웹에 적용하지 않았다. 현재 역할 근거는 후보 선택에만 쓰고 기존 관측 구간의 시작·끝은 보존하는 수정본을 검산 중이다.
과거 부분 역순 9쌍을 모두 확정 순서 오답으로 해석한 것은 정정했다. 뒤집지 않은 정상 동작 조각이 남고, 동작을 거꾸로 하면 방향/의미도 달라진다. 기존 결과는 보존했으며 판정 기준을 EVALUATION_CONTRACT_V2.json으로 분리했다. 이 정정만으로 후보를 통과시키지 않는다.
공개 FIT 순종·이집트 6영상의 준비/주요동작/유지/마무리와 경계 불확실성을 RGB로 보강했다. 같은 Codex 검토이며 독립 전문가 정답이 아니다. 엄지 간격 변화는 이집트 FIT에서 관측되지만 주먹/검지형/가려진 엄지와 동일시할 수 없어 다른 표현 근거를 추가 확인 중이다.
공통100 학습 기반과 C·T3-F, 현재 웹 기본값은 보존했다. 개인 자료 접근·학습·튜닝 없음. 게임 병합은 뒤로 미룬다.'''
 next_work='''다음 단일 작업은 INTERVAL_PRESERVATION 결과의 정상 직접 정답/방향쌍/짧은 도착 보존 검산이다. 통과 시 선택한 관측 구간과 행동 완료를 분리한 통합 판정을 검증한다. 엄지 모음 정답은 좁힘이 보이는 공개 FIT를 확보한 경우에만 추가 학습한다. 미확보는 보류 근거로 남기고 주먹을 모음 정답으로 대체하지 않는다. REAL17/18×4 공개 영상은 피드백 최종 확인 후보로만 보존하며 후보 고정 전 점수를 보지 않는다. 이후 G3→G4→G5→G6 진행; 실제 카메라는 사용자 시작 필요.'''
 title='2026-11-11 작업그래프 실행 중간 검산 — 평가 해석 정정·행동 구간 보존'
 ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text(encoding='utf-8-sig')
 if '### '+title not in text:
  for section,body in [('8.',msg),('9.',next_work)]:
   lines=text.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if l.startswith('## '+section));lines.insert(i+1,'\n### '+title+'\n'+body+'\n근거: `'+str(U.relative_to(ROOT)).replace('\\','/')+'`\n\n');text=''.join(lines)
  ctx.write_text(text,encoding='utf-8')
 handoff='# 최신 인수인계\n\n'+msg+'\n\n'+next_work+'\n\n현재 그래프: '+str(U/'TASK_GRAPH.json')+'\n실행: '+str(U/'CURRENT_EXECUTION.json')+'\n발표 그림: '+str(U/'presentation/web_completion_graph.png')+'\n완료 실험 재실행 금지. 코드/계산 검산과 독립 사람 정답을 구분한다.\n'
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
 state=read(O/'STATE.json');state['current_stage']=U.name;state['last_updated']=stamp();state['next_single_action']=next_work
 state[U.name]=dict(status='ACTIVE_G2_INTERVAL_AND_TRUTH_REPAIR',graph=str(U/'TASK_GRAPH.json'),execution=str(U/'CURRENT_EXECUTION.json'),web_changed=False,training_running=False,private_access=False,truth_correction=str(U/'EVALUATION_CONTRACT_V2.json'),presentation=str(U/'presentation/web_completion_graph.png'))
 write(O/'STATE.json',state)
 write(U/'CURRENT_EXECUTION.json',dict(status='G2_INTERVAL_PRESERVATION_VERIFICATION_AND_FIT_FEATURE_AUDIT',pid=None,training_running=False,web_changed=False,camera=False,private_access=False,graph=str(U/'TASK_GRAPH.json'),pending=next_work))
 history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt';old=history.read_text(encoding='utf-8-sig')
 if title not in old:history.write_text(old+'\n'+title+'\n'+msg+'\n次 작업: '+next_work+'\n',encoding='utf-8')
 print('Checkpoint synchronized; model/web protection hashes match.')


if __name__=='__main__':main()
