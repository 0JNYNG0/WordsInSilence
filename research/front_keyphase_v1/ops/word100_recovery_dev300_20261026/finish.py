from common import *
import shutil
def main():
    done=read(U/'DONE.json');check(done['evidence_pins']);d=read(U/'DIAGNOSTIC.json');assert d['input_counts']['clips']==300
    assert read(U/'CLASSIFIER_VERIFIED.json')['scores_recounted']==7500
    assert not (U/'RUNNING.json').exists();check(read(U/'FROZEN.json')['protected'])
    nxt=O/'word100_recovery_evidence_contract_20261026';nxt.mkdir()
    # Reuse the existing tested adapter; do not start another detector or duplicate the old46 study.
    src=O/'word100_public_recovery_bridge_20261026'
    pins={str(p):sha(p) for p in [U/'DONE.json',U/'DIAGNOSTIC.json',U/'FROZEN.json',src/'adapter.py',src/'INTEGRATION_CONTRACT_KO.txt',src/'VERIFIED.json',src/'CLASSIFIER_VERIFIED.json']}
    paths=[str(U/'outputs'/(r['clip_id']+'.npz')) for r in read(U/'FROZEN.json')['rows']]
    contract=dict(at=now(),status='READY_FOR_RESEARCH_ADAPTER_IMPLEMENTATION_NOT_OPERATIONAL',authorization='Continue active user-authorized autonomous improvement goal; no repeated routine approval.',
      pins=pins,public_dev_inputs=paths,reference_existing_train_bridge=str(src),
      goal='Connect existing hand-evidence adapter to the300 prepared real detections, keeping classifier input and feedback observation eligibility distinct. Produce a usable research paired analysis receipt, not another detector-only summary.',
      steps=[
       'Reuse adapter.py channel_view/compared and angle states from prior46. Generalize native75/repair provenance input adapter to current300 saved outputs. No rerun of detection or classifier scores.',
       'Keep original frozen classifier x_whole byte-exact and expose saved C13/T3F/E3 result. Recovered coordinates go to separate diagnostic/DTW evidence view. Do not erase the planned five-condition negative/mixed result.',
       'Keep every raw coordinate and source. Wrist-only never provides palm center/finger angles. Angle geometry/two-scale agreement are quality proxies, not exact anatomy. Shape uncertainty and lack of identity evidence must remain visible; no word-dependent mask or score-dependent threshold.',
       'Verify old normal/crossing coordinates and masks, per-angle independence, fixed reference denominator and unknown-not-zero under loss. Use priorTRAIN46 existing same-word public comparisons to verify downstream consumer; newDEV300 receipt is coverage/contract audit only, not a second score sweep.',
       'Quantify usable location evidence separately from shape confidence and unresolved observation. Preserve short sequences and original timing. Do not use oracle correspondence to label normal execution.',
       'Report research integration effects/remaining detector and alignment limits. Only then decide a common next training comparison using publicTRAIN; do not train now on this unverified shape input.'
      ],
      acceptance='Exact original classifier input and prior score reuse; coordinate/crossing preservation; additional position evidence does not create finger evidence; empty/unknown comparisons remain unavailable with reference scope fixed. No claim of precise corrective feedback from unvalidated shape.',
      prohibited=['operating model/web/DTW edits','user/mentor data','extra199/200 data','camera','downloads','external transfer','Git','per-word rules','redo completed inference','training before evidence contract checks'],
      scientific_limits='Already exposed development data. Same Codex numerical/image review is not independent anatomical truth. There is no promise of zero errors or all-sign vocabulary generalization.')
    write(nxt/'PREFLIGHT.json',contract)
    backup=U/'before_finish';backup.mkdir()
    state_path=O/'STATE.json';context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';handoff=ROOT/'NEW_CHAT_HANDOFF.md'
    for p in [state_path,context,handoff,U/'CURRENT_EXECUTION.json']:shutil.copy2(p,backup/p.name)
    summary='공개 DEV300 실제 검출 보완 비교 완료. MP 재검출2131손 장면/40822점, 추가YOLO48손 장면/387점(손목만29/여러점19). 원래1285373 유효 관절·프레임/교차60프레임 보존. C13 294→295/300, T3F 세 모델295/299/297→296/298/298(평균99.00→99.11%,한모델회귀), E3 300유지. YOLO의추가Top1이득0. 크기고정진단완료,7500예측별도검산PASS. 새관측2179중103(모든YOLO48포함)시각검토,일부손가락모양불확실. 운영변경0·학습0·사용자자료접근0. 전체목표계속진행.'
    next_text='word100_recovery_evidence_contract_20261026/PREFLIGHT.json에따라 기존공개46의adapter.py를 재사용해현재300보완관측을연결한다. 인식입력은기존x_whole/저장점수그대로보존, 보완손은DTW의위치/모양근거로분리한다. 손목1점으로각도생성금지,예측일치를정답인증금지,미관측시고정참고분모/unknown유지. 기존TRAIN46비교소비자/현재DEV300출처계약검증으로연구통합결과를만든다. 검출/점수재실행금지. 운영C/T3F/웹/DTW·원본·사용자/멘토/추가199/200불변. 다음학습전근거계약검증,단어별예외/점수기반임계값맞추기금지. 소규모실험종료로goal완료처리금지.'
    state=read(state_path);state['updated_date_kst']=now();state['current_stage']='AUTONOMOUS_RECOVERY_INTEGRATION_WITH_SEPARATE_RECOGNITION_AND_FEEDBACK_EVIDENCE';state['next_single_action']=next_text
    state[U.name]=dict(at=now(),status=done['status'],result=str(U/'RESULT_KO.html'),summary=summary,done=str(U/'DONE.json'),running=False,goal_complete=False)
    state[nxt.name]=dict(at=now(),status=contract['status'],preflight=str(nxt/'PREFLIGHT.json'),next=next_text,running=False)
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    text=context.read_text('utf-8-sig')
    h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
    assert text.count(h8)==1 and text.count(h9)==1
    text=text.replace(h8,h8+'\n\n### 2026-10-26 공개300 실제 재검출 효과 — 제한적 개선과 회귀 확인\n'+summary+'\n근거: research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.html, INPUTS_VERIFIED.json, CLASSIFIER_VERIFIED.json. 동일Codex별도검산·시각검토이며독립사람정답아님. prepare56293/verify3453/classifier15070 모두종료0,진행프로세스없음. 최초JSON-null재생검산실패는VERIFY_REPLAY_RECOVERY.json으로보존·복구.\n',1)
    text=text.replace(h9,h9+'\n\n### 2026-10-26 지속 개선 — 인식 입력과 피드백 관측의 분리 연결\n'+next_text+'\n',1)
    context.write_text(text,encoding='utf8')
    handoff.write_text('# WordsInSilence 최신 인수인계\n\n'+now()+'\n현재: '+summary+'\n\n다음: '+next_text+'\n\n완료단위: '+str(U)+'\n준비/검산/고정분류/별도점수검산완료. 검출과점수재실행금지. 진행실행없음. 구현/별도산술검산동일Codex,독립사람정답아님. 새unit PREFLIGHT만작성했으며구현은다음. 활성goal유지,heartbeat없음.\n',encoding='utf8')
    (U/'CURRENT_EXECUTION.json').write_text(json.dumps(dict(at=now(),running=False,preparation=dict(session=56293,exit_code=0),verification=dict(session=3453,exit_code=0),classification=dict(session=15070,exit_code=0),next_preflight=str(nxt/'PREFLIGHT.json')),ensure_ascii=False,indent=2),encoding='utf8')
    assert read(state_path)['current_stage'].startswith('AUTONOMOUS_RECOVERY_INTEGRATION')
    assert nxt.name in handoff.read_text('utf8') and U.name in context.read_text('utf8')
    check(pins);check(read(U/'FROZEN.json')['protected']);check(done['evidence_pins'])
    write(U/'SYNC_CHECK.json',dict(at=now(),status='PASS',next_preflight=str(nxt/'PREFLIGHT.json'),goal_complete=False,
       updated={str(p):sha(p) for p in [state_path,context,handoff,U/'CURRENT_EXECUTION.json']},independent_people_ground_truth=False))
    print('SYNC PASS; GOAL ACTIVE',flush=True)
if __name__=='__main__':main()
