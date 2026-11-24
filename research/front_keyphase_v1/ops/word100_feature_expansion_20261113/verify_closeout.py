from finish import *

def main():
    # Refresh the overall graph pointer; old result artifacts remain untouched.
    s=read(O/'STATE.json');n=s['next_single_action']
    s['word100_web_completion_graph_20261111'].update(status='C7_C8_C9_COMPARED_G2_REWORK_NOT_ADOPTED',
        active_unit=str(C9),latest_evidence=str(C9/'GRAPH_PATCH.json'),training_running=False,
        progress='C7/C8/C9 fixed comparisons complete; C9 ordered9-time inputs versus matched mean control failed movement/hold preservation. Training-fit misses0, held-record generalization unresolved. No G2 pass or web replacement.',
        remaining=n)
    write(O/'STATE.json',s)
    doc=read(U/'DOCUMENT_SYNC.json');doc['files'][str(O/'STATE.json')]=sha(O/'STATE.json');write(U/'DOCUMENT_SYNC.json',doc)
    checks=[]
    for p,h in doc['files'].items():assert sha(p)==h;checks.append('document_hash:'+p)
    for p,h in read(U/'FROZEN.json')['protected'].items():assert sha(p)==h;checks.append('protected:'+p)
    for p in [U/'GRAPH.json',U/'NEXT_PLAN.json',U/'CURRENT_EXECUTION.json',U/'EVALUATION_COMPLETE.json',U/'RESULT_VERIFIED.json',
              U/'TIMING_CONTRACT_REVIEW.json',U/'FINAL_REPORT_REVIEW.json',U/'feedback_contract/INTEGRATION_DECISION.json',
              C9/'CLOSEOUT_LEARNING.json',C9/'GRAPH_PATCH.json',O/'avatar_multiview_review_20261113/VERIFIED.json']:
        read(p);checks.append('json_valid:'+str(p))
    report=(U/'RESULT_KO.txt').read_text('utf-8');decision=read(U/'feedback_contract/INTEGRATION_DECISION.json')
    assert '펴짐과 접힘을 합친' in report and '154개에서152개로 감소' in report
    assert 'passes seed42 and fails seeds43/44' in decision['reason']
    assert (U/'TIMING_CONTRACT_REVIEW.json').exists()
    assert '기존 그래프 작업도 함께 완료한 비교' in report and '예전3방향 시도는 삼각측량' in report
    assert not read(U/'CURRENT_EXECUTION.json')['training_running'] and not read(C9/'GRAPH_PATCH.json')['training_running']
    assert len(list((U/'runs').glob('*/COMPLETE.json')))==9
    assert len(read(C9/'TRAINING_COMPLETE.json')['rows'])==36
    assert read(U/'GRAPH.json')['project_complete'] is False
    assert s[U.name]['status']=='COMPLETE_COMPARED_NOT_ADOPTED'
    assert s['word100_web_completion_graph_20261111']['active_unit']==str(C9)
    for key,p in read(U/'GRAPH.json')['evidence'].items():assert Path(p).is_file();checks.append('graph_evidence:'+key)
    write(U/'FINAL_VERIFICATION.json',dict(status='PASS_CLOSEOUT',checks=checks,
        report_review_corrections=['extended+folded explicitly combined; extended seed43 decline disclosed','legacy retention seed42 pass vs43/44 fail distinguished','timing review file now exists'],
        training_runs=9,C9_runs=36,running_jobs=[],project_complete=False,production_changed=False,
        verifier='root closeout after separate Codex result arithmetic, source, contract, timing and prose reviews',
        report_sha256=sha(U/'RESULT_KO.txt'),document_sync_sha256=sha(U/'DOCUMENT_SYNC.json'),
        source_verified=str(U/'RESULT_VERIFIED.json'),C9_verified=str(C9/'SAVED_VERIFICATION.json'),
        human_ksl_expert_validation=False,no_training_or_inference=True))
    print('PASS_CLOSEOUT: nine fine-tuning runs and36 C9 diagnostics complete; candidate not adopted; documents synchronized.')

if __name__=='__main__':main()
