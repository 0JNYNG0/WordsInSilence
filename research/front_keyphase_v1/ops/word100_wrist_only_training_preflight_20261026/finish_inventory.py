from common import *
from collections import Counter
import shutil

def main():
    f=read(U/'FROZEN.json');r=read(U/'INVENTORY.json');check(f['pins']);check(f['protected']);assert r['status']=='PASS' and not r['errors']
    assert not (U/'RUNNING.json').exists()
    assert {x['clip_id'] for x in r['rows']}=={x['clip_id'] for x in f['rows']} and len(r['rows'])==899
    c=Counter();targets=[];ids=Counter()
    for row in r['rows']:
        ids[str(row['signer_id'])]+=1
        assert row['signer_id'] in range(1,10)
        for view,v in row['views'].items():
            alltargets={(q['index'],s) for q in v['recovery_targets'] for s in q['sides']}
            active={(i,s) for i,s in alltargets if any(a<=i<=b for a,b in v['original_spans'].values())}
            c[view+'_active_target_hand_frames']+=len(active);c[view+'_active_target_frames']+=len({i for i,s in active});c[view+'_outside_fixed_spans']+=len(alltargets-active)
            assert c[view+'_outside_fixed_spans']>=0
            if view=='sensor':assert v['canvas']==v['original_roi'][2:]
            times={q['index']:q['frame_id'] for q in v['recovery_targets']}
            targets.append(dict(clip_id=row['clip_id'],view=view,canvas=v['canvas'],targets=[dict(index=i,frame_id=times[i],sides=sorted(s for j,s in active if j==i)) for i in sorted({i for i,s in active})]))
    assert dict(ids)==r['by_record_id'];assert len(targets)==1798
    write(U/'INVENTORY_VERIFIED.json',dict(at=now(),status='PASS',public_train_clips=899,view_records=1798,record_ids=dict(ids),active_counts=dict(c),
        arithmetic='Separate manifest/split/canvas/active-domain audit. Original feature byte equality performed in inventory.py, not claimed as a second human/data ground truth review.',
        original_features_reproduced=sum(v for k,v in r['counts'].items() if k.startswith('exact_')),
        source_crc_header_verified=True,full_source_sha_verification_pending=True,detector_inference=False,training=False))
    write(U/'PREPARE_PROTOCOL.json',dict(at=now(),status='READY_TO_IMPLEMENT_PUBLIC_RECOVERY_PREPARATION',
        pins={str(p):sha(p) for p in [U/'FROZEN.json',U/'INVENTORY.json',U/'INVENTORY_VERIFIED.json',U/'finish_inventory.py']},
        targets=targets,stage='Actual TRAIN wrist recovery not run yet.',
        protocol=[
          'Use existing full-source IMAGE-mode MP retry then selective YOLO rules/weights/constants unchanged. In sensor mode, pass only actual cropped camera pixels; full mode sees original source then applies original ROI. No source pixels outside camera can justify sensor recovery.',
          'Only run new IMAGE retries for targets inside union of original whole/context spans; outside spans cannot enter the frozen classifier input, so avoid wasted processing without changing native extraction or dt. Detector is IMAGE mode, not a temporal tracker whose hidden state depends on skipped frames.',
          'Inspect existing public46/all recovery caches before inference. Reuse only when source SHA, frame IDs, pixel canvas, original observed arrays, ROI, detector/model/rule hashes all match. No new inference for verified identical prior calls. Record reuse or mismatch reasons.',
          'All899 retained; original sources read-only. Verify video SHA after selective extraction, source frame mapping, existing wrist/hand/arm preservation, current arm and duplicate rules; emit ambiguity as unavailable. Start with label-blind diverse camera/record-ID smoke then full once exact checks pass.',
          'Create wrist-only classifier channels by copying original x_whole/x_context. For originally whole-hand absent frames, only newly observed body-relative hand-wrist xy and matching body-valid bit can be filled, requiring original valid shoulder origin and original scale. Local hand/finger coordinates/masks remain original; no finger fabrication, scale/span change or word exception.',
          'Saved original C uses both masked and sensor inputs and whole/context temporal views. Keep its four-epoch deterministic schedule unchanged and verify every old channel outside allowed sparse wrist cells byte-exact.',
          'Before training pin code/inputs/seeds20261013,20261014,20261015, model/optimizer/early-stop/selection rules. Reuse original3 C control results only if complete control inputs/training contract match exactly; otherwise do not pretend comparison is matched.',
          'Candidate3seed training only after full input and visual wrist/duplicate checks. Fixed DEV300 is development, not new final accuracy. Use prior full-source DEV caches; sensor recovery may be new inference because its available image differs. Never redo completed detector/model calls.',
          'Report sensor+masked mean/range, every seed gain/loss, input failures, observation gaps/crossing subgroups and time. No automatic operating replacement. Gate: sensor mean must exceed matched C, at least2 seeds nonregressing with1 improved, masked mean nondecreasing, no silent input exclusion; any regression remains visible. Do not alter protocol using results.'
        ],protected=f['protected'],forbidden=['user/mentor/extra199/200','camera','download','external','Git','operating C/T3F/web/DTW edits','word-specific exceptions']))
    text='공개 TRAIN899 입력 감사 완료\n\n기존 C가 사용한 899영상 × 2관측조건 × 전체/핵심2구간의 원래 입력을 모두 동일하게 재현했다. 총 '+str(sum(v for k,v in r['counts'].items() if k.startswith('exact_')))+'채널값. 실패0. 원본ZIP 항목의 CRC/크기는 확인했으며 실제 추출 시 SHA256 확인은 남아 있다.\n\n같은 원본을 전체 화면에서 보는 조건과 실제 카메라 범위가 좁은 조건은 가용 정보가 다르다. 화면이 좁은 조건의 누락을 전체 원본의 화면 밖 정보로 채우지 않는다.\n전체 화면 관측의 실제 고정구간 내 보완 대상: '+str(c['masked_active_target_frames'])+'프레임 / '+str(c['masked_active_target_hand_frames'])+'손·프레임.\n좁은 카메라 관측: '+str(c['sensor_active_target_frames'])+'프레임 / '+str(c['sensor_active_target_hand_frames'])+'손·프레임. 보완 대상이지 실제 손 검출 실패 정답 수가 아니다.\n학습에 포함되지 않는 고정구간 밖 후보18744손·프레임의 추가 IMAGE 재검출은 생략할 수 있다. 원래 시계열·학습 구간은 변경하지 않는다.\n\n다음: PREPARE_PROTOCOL.json의 실제 재검출 캐시 호환성 확인 → 필요한 공개 입력만 준비 → 손목-only 입력 및 정상/교차 보존 전수검증 → 조건 동결 후3seed 비교. 지금은 새 재검출·후보 학습을 시작하지 않았다. 기존 C·T3-F와 사용자 촬영은 그대로다.\n'
    (U/'RESULT_KO.txt').write_text(text,encoding='utf8')
    execution=read(U/'CURRENT_EXECUTION.json');execution.update(at=now(),running=False,exit_code=0,stage='TRAIN899_INPUT_AUDIT_PASS_RECOVERY_PREPARATION_PENDING');(U/'CURRENT_EXECUTION.json').write_text(json.dumps(execution,ensure_ascii=False,indent=2),encoding='utf8')
    backup=U/'before_sync';backup.mkdir();context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';handoff=ROOT/'NEW_CHAT_HANDOFF.md';sp=O/'STATE.json'
    for p in (context,handoff,sp):shutil.copy2(p,backup/p.name)
    summary='TRAIN899/1798관측 입력 감사PASS·실패0. 전체/핵심3596입력의66252760채널값 원래입력과바이트동일. 고정구간내full7285프레임/7750손, sensor20749프레임/21379손이실제재검출준비대상. 손목-only후보는새손목body좌표/마스크만채우고기존손가락·크기·구간·dt를보존하는계약. 아직새재검출/후보학습0. 앞단공개300근거연결/126비교소비자검증완료,모델/사용자자료유지. 전체goal진행중.'
    nxt='word100_wrist_only_training_preflight_20261026/PREPARE_PROTOCOL.json에따라 실제공개TRAIN재검출입력을준비한다. 먼저기존46캐시의source/frame/canvas/원래좌표/ROI/모델·규칙hash일치여부감사해같은호출재실행을피한다. full→ROI와sensor실제화면각각에서공통MP재검출→선별YOLO. label-blind smoke검산후전수. 손목-only는원래x복사후허용body좌표/valid만수정하며손가락·정규화·시간구간은불변. 입력통과후3seed조건동결/학습비교. 기존C/T3F/E3·운영/원본유지,사용자/멘토/추가199/200금지. DEV개발검증을새최종성능으로부르지않는다. 현재실행없음(12495종료0).'
    s=read(sp);s['updated_date_kst']=now();s['current_stage']='AUTONOMOUS_WRIST_ONLY_ACTUAL_INPUT_PREPARATION';s['next_single_action']=nxt;s[U.name].update(status='INPUT_AUDIT_PASS_RECOVERY_PREPARATION_PENDING',summary=summary,protocol=str(U/'PREPARE_PROTOCOL.json'),running=False,session_exit_code=0,goal_complete=False)
    sp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    body=context.read_text('utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
    body=body.replace(h8,h8+'\n\n### 2026-10-26 손목 보완 후보의 공개899 입력 감사\n'+summary+'\n근거: research/front_keyphase_v1/ops/'+U.name+'/INVENTORY.json, INVENTORY_VERIFIED.json, PREPARE_PROTOCOL.json.\n',1)
    body=body.replace(h9,h9+'\n\n### 2026-10-26 실제 손목 재관측 입력 준비\n'+nxt+'\n',1);context.write_text(body,encoding='utf8')
    handoff.write_text('# WordsInSilence 최신 인수인계\n\n'+now()+'\n현재: '+summary+'\n\n다음: '+nxt+'\n\n현재unit: '+str(U)+'\nfreeze12245/감사12495종료0. CURRENT_EXECUTION에종료기록. 실제재검출준비코드는아직미구현. 이전evidence_contract의RESULT_KO.html에공개300/126읽기전용보기완료;file URL브라우저정책차단으로시각렌더미검증,로컬1278UI분기PASS. 우회하지말것. 전체goal active,heartbeat없음. 구현/검산Codex이며사람정답아님.\n',encoding='utf8')
    # Arithmetic value inserted in human summary must match actual verified sum.
    total=sum(val for key,val in r['counts'].items() if key.startswith('exact_'))
    if total!=66252760:
        raise AssertionError(('summary count mismatch',total))
    check(f['protected']);write(U/'SYNC_CHECK.json',dict(at=now(),status='PASS',counts=dict(c),goal_complete=False,next=str(U/'PREPARE_PROTOCOL.json'),updated={str(p):sha(p) for p in (context,handoff,sp)}))
    print('AUDIT SYNC PASS',dict(c),flush=True)
if __name__=='__main__':main()
