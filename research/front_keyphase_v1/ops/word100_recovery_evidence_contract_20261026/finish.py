from common import *
import shutil

def main():
    done=read(U/'DONE.json');check(done['evidence_pins']);v=read(U/'VERIFIED.json');assert v['status']=='PASS'
    assert not (U/'RUNNING.json').exists();f=read(U/'FROZEN.json');check(f['protected'])
    write(U/'UI_CHECK.json',dict(at=now(),static_consumer_status='PASS',view_model_combinations=1278,clips=300,comparison_pairs=126,
        actual_render_verified=False,reason='cua file URL navigation denied by browser security policy. No alternate route attempted. Local Node vm validated all selection view-model combinations without browser/network.',
        script_source_sha256=sha(U/'RESULT_KO.html'),counts_do_not_mean_anatomical_truth=True))
    nextunit=O/'word100_wrist_only_training_preflight_20261026';nextunit.mkdir()
    write(nextunit/'PREFLIGHT.json',dict(at=now(),status='PUBLIC_TRAIN_INPUT_AUDIT_FIRST',goal_continues=True,
        authorization='User approved continued autonomous public-data recognition improvement; preserve original models and never train/tune/select on user recordings.',
        pins={str(p):sha(p) for p in [U/'DONE.json',U/'VERIFIED.json',U/'RESULT.json',U/'adapter.py',V/'FROZEN.json',V/'CLASSIFIER_RESULT.json',
            O/'word100_body_boundary_20261017/training_CONTEXT.json',O/'word100_body_boundary_20261017/training_core.py']},
        scope='Public TRAIN899 source/input inventory and frozen protocol first. No new training in inventory step. DEV300 already exposed is development only; extra199/200 excluded.',
        preparation=['Read original C training schedule and each saved native/masked/sensor input. Confirm exact spans, hand/body scales, source frame identity and sensor crop geometry before proposing wrist-only injection.',
            'Candidate must preserve all original visible hand/finger/arm channels and original span/scales. Only previously absent hand wrist supported by current arm/image evidence may contribute. No invented finger coordinates or new hand-size estimate.',
            'Full-source -> original ROI and realistic sensor input must remain different available-image conditions. Never fill sensor input using pixels outside its actual camera canvas. Use common recovery rule on the image each condition can actually see.',
            'Reuse cached detections only if source frame, pixel canvas, model/rule hashes and threshold settings match. Otherwise prepare new TRAIN-only inference separately; no replay of completed DEV scores.',
            'Before training freeze matched control and wrist-only candidate across3seeds, identical schedule/architecture/optimizer/selection rule and no per-word exceptions. Preserve C/T3F/E3 unchanged. Include both observed camera conditions and report gains/losses; do not tune on new evaluation results.'],
        acceptance='All899 retained, source hash/CRC consistency and view mapping proved, exact original features reconstructed, out-of-view information cannot leak into sensor input; unresolved data must stop dependent training, not silently exclude clips.',
        forbidden=['user/mentor/extra199/200','camera','downloads','external transfer','Git','operating model/web/DTW edits','per-word rules','training before input contract verification']))
    backup=U/'before_finish';backup.mkdir();context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';handoff=ROOT/'NEW_CHAT_HANDOFF.md';statep=O/'STATE.json'
    for p in (context,handoff,statep):shutil.copy2(p,backup/p.name)
    summary='공개DEV300 원래인식과 보완관측의 분리 연결 완료. 원래6113800분류값 바이트동일,580494손관절·프레임/교차60프레임 보존. YOLO손목-only29는 손가락근거0. 261손·프레임 좌우/중복근거엇갈림 표시(실제오류정답아님). 3356550각도상태·2070000비교셀 별도검산PASS. 기존TRAIN126 DTW비교를 고정분모/누락시간/각도별기록으로 전달. 새인식추론0·학습0·운영변경0·사용자자료0. 계산가능한각도434418중386101은재확인없음;정밀교정완료아님. 전체goal계속.'
    nxt='word100_wrist_only_training_preflight_20261026/PREFLIGHT.json을 따라 공개TRAIN899와 원래C의masked/sensor 학습일정·구간·정규화·가용화면을 먼저감사한다. 손가락은그대로보존하고 재확인된누락손목만활용할 공통후보의입력계약을동결한다. sensor에는화면밖원본픽셀근거를유입하지않는다. 같은조건캐시는해시검증재사용,새추론필요범위만준비한다. 대응입력검증후같은조건3seed비교를동결/진행하되기존C/T3F/E3·운영유지. DEV300이미개발용,추가199/200/사용자/멘토자료금지. 이전완료점수재계산금지. 불확실손가락을확정피드백으로승격금지. 소규모체크완료로goal완료처리금지.'
    state=read(statep);state['updated_date_kst']=now();state['current_stage']='AUTONOMOUS_WRIST_ONLY_TRAIN_INPUT_PREFLIGHT';state['next_single_action']=nxt
    state[U.name]=dict(at=now(),status=done['status'],summary=summary,result=str(U/'RESULT_KO.html'),verified=str(U/'VERIFIED.json'),running=False,goal_complete=False)
    state[nextunit.name]=dict(at=now(),status='PUBLIC_TRAIN_INPUT_AUDIT_FIRST',preflight=str(nextunit/'PREFLIGHT.json'),running=False)
    statep.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    txt=context.read_text('utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업';assert txt.count(h8)==txt.count(h9)==1
    txt=txt.replace(h8,h8+'\n\n### 2026-10-26 인식/피드백 근거 분리 연구 연결 검증\n'+summary+'\n근거: research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.html, VERIFIED.json, UI_CHECK.json. 같은Codex별도계산검산이며사람정답아님. 로컬파일브라우저정책차단으로실제렌더미검증,로컬UI분기1278조합PASS. 실행91639/3885종료0.\n',1)
    txt=txt.replace(h9,h9+'\n\n### 2026-10-26 손목 보완 인식 후보 입력 준비\n'+nxt+'\n',1);context.write_text(txt,encoding='utf8')
    handoff.write_text('# WordsInSilence 최신 인수인계\n\n'+now()+'\n현재: '+summary+'\n\n다음: '+nxt+'\n\n완료단위: '+str(U)+'\nrun91639/verify3885종료0,학습/검출/분류추론없음. 현재진행프로세스없음. 구현/별도검산Codex. UI실제렌더는file URL정책차단으로미확인,우회없음. 이전DEV300결과와현재근거연결검증을혼동금지. 다음unit은PREFLIGHT만작성. goal active,heartbeat없음.\n',encoding='utf8')
    check(done['evidence_pins']);check(f['protected'])
    write(U/'SYNC_CHECK.json',dict(at=now(),status='PASS',next=str(nextunit/'PREFLIGHT.json'),goal_complete=False,updated={str(p):sha(p) for p in (context,handoff,statep)}))
    print('SYNC PASS; NEXT TRAIN899 INVENTORY',flush=True)
if __name__=='__main__':main()
