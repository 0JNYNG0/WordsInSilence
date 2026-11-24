from pathlib import Path
import json,hashlib
Z=Path(__file__).resolve().parent;O=Z.parent;Y=O/'word100_direct_occurrence_order_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exclusive(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def main():
    frozen=read(Y/'FROZEN.json');gates=read(Y/'GATES.json');verify=read(Y/'SAVED_READOUT_VERIFIED.json')
    assert not gates['passed'] and not verify['candidate_passed']
    x=O/'word100_learned_component_extent_20261102/public_runtime_runs/fd785d173921436492c65e1703e46589/analysis'
    ck=Path('W:/WordsInSilence_derived/word100_direct_occurrence_order_20261102/runs/tied_readout/model.pt')
    files=[Y/'EVALUATION.json',Y/'GATES.json',Y/'SAVED_READOUT_VERIFIED.json',Y/'CAUSE_DESIGN_REVIEW.json',Y/'EVALUATION_PROTOCOL.json',Y/'pair_contract.py',Y/'readout_model.py',ck,
        x/'analysis.npz',x/'SOURCE_FEEDBACK.json',Z/'order_runtime.py',Z/'verify_runtime_input.py']
    plan=dict(status='FROZEN_PARTIAL_ORDER_RESEARCH_CONNECTION',
        purpose='Connect observed pair order to guide descriptions/native times without claiming whole-sign correctness.',
        selection='Tied comparator retained after direct failed; explicitly post-comparison research use, not independent improvement validation.',
        primary_direct_failed=True,old_scores_unchanged=True,new_training=0,new_backbone_inference=0,new_video_capture=0,
        saved_pair_display_checks=110,
        input_compatibility=dict(preselected='native WORD0739 recorded ID11 / dev_000',method='exact saved tensor/path comparison without a model forward'),
        new_runtime_input=str(x/'analysis.npz'),base_result=str(x/'SOURCE_FEEDBACK.json'),
        new_tied_head_forward_limit=1,new_current_state_forward_limit=1,
        existing_base_output_fields_preserved=True,individual_reference_video_comparisons=0,additional_DTW=0,
        essential_cases=['dev_034 cut with residual A/B, no completion inference','dev_003 interleaving','dev_035 missing image evidence','dev_048 mixed reverse order'],
        acceptance=['input x130/rgb769/flow6/path exact','native axes retained','all old output fields identical','no labels/participant IDs/DEV truth in runtime inference',
                    'uncertain/cut/interleaved/unsupported repeat never marked complete/correct','both observed directions kept','existing web8804/8808 unchanged'],
        runtime_scope='Adapter validation only; no new operational port and no claimed full end-to-end latency or accuracy gain.',
        prohibitions=['private/mentor/H14/H15/ID16','new thresholds/seeds/window selection','old evaluation re-inference/re-scoring','new downloads/external transmission','C/T3F/default web replacement'],
        pins={str(p):sha(p) for p in files},whole_goal_complete=False)
    exclusive(Z/'SCOPE_FROZEN.json',plan)
    report='''두 순서 판정부 비교와 검산을 마쳤습니다. 새 direct 후보는 적용하지 않았습니다.

기존 손·팔 관측과 주요 동작 판단을 함께 쓰는 tied 조건: 개발 관계 103개 중 82개를 맞는 방향으로 확정, 21개 보류.
그 의존을 제거한 direct 조건: 42개 확정, 61개 보류. 확정한 방향의 오판은 둘 다 0개지만, 보류 증가를 개선으로 보지 않았습니다.
103개는 103개의 독립 영상이 아닙니다. 반복 사용한 개발 ID11~13의 15개 원래 앞뒤 관계에 속도/누락/역순 등 변형을 포함한 수입니다.

새 후보는 두 행동의 근거를 같은 시간에 함께 잡는 경우가 늘었습니다. 학습 자료에서도 확정을 잃은 137개가 모두 이 겹침에 해당했습니다.
따라서 더 오래 같은 학습을 돌리는 대신, 기존 관측을 유지하는 조건에서 실제로 확인한 부분 순서를 설명 및 원본 시간에 연결합니다.
이것은 결과를 본 뒤 선택한 연구용 기능 연결이며, 새로운 인식률 향상이나 독립 평가 성공을 주장하지 않습니다.

별도 에이전트가 저장값 680개, 관계850개와 주석 위치59,728개를 검산했습니다. 코드/산술 검산이며 한국수어 전문가의 정오 판정은 아닙니다.
원모델과 C·T3-F, 기존 웹8804/8808은 유지했습니다. 사용자·멘토 촬영은 사용하지 않았습니다.
전체 행동 완료, 중요한 구간 누락, 필수 반복 횟수의 판정은 아직 남아 있습니다.
'''
    (Y/'RESULT_KO.txt').write_text(report,encoding='utf8')
    exclusive(Y/'FINAL.json',dict(status='COMPLETE_DIRECT_NOT_ADOPTED',gates_passed=False,failed_gates=gates['failed'],
        independent_saved_verification=str(Y/'SAVED_READOUT_VERIFIED.json'),cause_review=str(Y/'CAUSE_DESIGN_REVIEW.json'),
        new_training_runs=2,new_model_clip_inferences=680,old_model_reruns=0,production_changed=False,
        next=str(Z),whole_goal_complete=False))
    import sys;sys.path.insert(0,str(O/'word100_learned_flow_role_20261102'));import sync_flow
    sync_flow.U=Y
    sync_flow.sync('順序 판정부 비교 완료·새 후보 미채택 및 부분 순서 연결 시작'.replace('順序','순서'),
        'Y 두조건80epoch/680새입력추론·저장값별도검산완료. tied DEV82/103확정·21보류, direct42/103·61보류, 확정오류각0.103관계=15원관계+변형/반복DEV11~13. direct15gate중8미달로미채택. FIT손실137전부overlap/DEV손실41중40overlap로추가epoch근거없음. 실패결과보존후 tied의부분관측순서만Z별도연결검증. C/T3F·웹8804/8808유지,전체goalactive.',
        'Z/SCOPE_FROZEN 및agent runtime/formatter검토확인. saved110관계표시검사와dev_000 입력130/RGB769/flow6/기존DTWpath exact검증(모델추론0), 통과시 X 기존공개job fd785d...에tied새head정확히1회/current1회만실행. MP/SignRep/옛평가재실행금지. partial order는완료/누락없음아님; cut dev_034잔여근거주의. 새웹포트/HTML추가없음. 결과후docs동기화.',
        'PARTIAL_ORDER_RESEARCH_CONNECTION_PREPARING',dict(training=False,evaluating=False,subsequent_unit=str(Z)))
    print('Y complete; Z scoped',flush=True)

if __name__=='__main__':main()
