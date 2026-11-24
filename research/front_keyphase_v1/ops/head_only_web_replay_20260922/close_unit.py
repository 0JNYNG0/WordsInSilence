"""Record replay completion and pending semantic gate without model mutation."""
from evaluate import H,L,read,sha,write
from datetime import datetime,timezone,timedelta
import json

def main():
    result=read(H/'result_LOCAL_ONLY.json');audit=read(H/'INDEPENDENT_AUDIT_LOCAL_ONLY.json')
    assert audit['status']=='PASS' and audit['result_sha256']==sha(H/'result_LOCAL_ONLY.json')
    assert all(r['models']['candidate'][s]['prediction']==r['models']['d1'][s]['prediction'] for r in result['rows'] for s in ['full93','project18'])
    pending_path=L/'ops/greeting_semantic_gate_20260922/PENDING_SEMANTIC_REVIEW.json';pending=read(pending_path)
    assert pending['response'] is None and pending['indexed_clips']==0
    now=datetime.now(timezone(timedelta(hours=9)));stamp=now.isoformat();human=now.strftime('%Y-%m-%d %H:%M KST')
    record=dict(status='COMPLETE_WEB_REGRESSION_REMAINS_NO_PROMOTION',at_kst=stamp,candidate=result['candidate'],
       report=str(H/'RESULT.md'),closeout=str(H/'CLOSEOUT_LOCAL_ONLY.json'),summary=result['summary'],
       all_top1_predictions_equal_d1=True,independent_checks=19,scores_recomputed=3534,
       user_feature_records_read=36,public_feature_records_read=2,new_training=0,new_encoder_forwards=0,
       user_raw_observations_read=0,reserved_arrays_read=0,default_promoted=False)
    state_path=L/'ops/STATE.json';state=read(state_path)
    state['head_only_web_replay_20260922']=record
    state['greeting_semantic_gate_20260922']=dict(status=pending['status'],record=str(pending_path),
        user_question_sent=True,response=None,train_candidate_labels=9,development_labels=3,
        source_frames_resolved=False,new_training=0,semantic_target_id=None)
    state['updated_at']=stamp;state['updated_date_kst']=now.strftime('%Y-%m-%d')
    state['current_stage']='FROZEN_HEAD_WEB_REPLAY_COMPLETE_REGRESSIONS_REMAIN_GREETING_SEMANTIC_REVIEW_PENDING'
    state['next_single_action']='Resolve user semantic judgment for existing 인사 reference vs AIHub SEN0354 안녕하세요. Before an affirmative answer, no class merge or source training. If confirmed, resolve target native frames in existing12 sentence archives using existing adapters and preserved train01-09/dev11-13 split. Reuse completed decoding/catalog work.'
    state['autonomous_workers_current_unit']=dict(unit=H.name,evaluation_exit_code=0,independent_audit_exit_code=0,metadata_routing_exit_code=0,all_current_unit_workers_finished=True,work_paused_by_user=False)
    state['autonomous_goal_status'].update(status='active',at_kst=stamp,project_complete=False,user_requested_pause=False,
       previous_goal_turn_classification='progress: matched position/motion pair trained and independently rejected',
       progress_this_turn='New frozen head evaluated on38 existing web-domain feature records; identical top1 to d1 and regression retained. Existing greeting source semantic question presented and12 source metadata rows routed without new payload.',fresh_blocked_audit_count=0)
    state['goal_status']='INCOMPLETE: native score gain did not yield web recognition improvement; known classification regressions remain. Pending source semantic judgment is not completion or a user-requested pause. Multi-person18-word web validation,valid execution feedback,and validated Unity integration remain.'
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    root=L.parents[1]/'WordsInSilence_PROJECT_CONTEXT.md';text=root.read_text(encoding='utf-8-sig');lines=text.splitlines();lines[2]='최종 갱신: '+human;text='\n'.join(lines)+'\n'
    current=f'''### {human} 최우선 — 기존 분류층 후보 웹 대입 완료, 회귀 유지·인사 의미검토 요청

- 공개691/699 분류층 후보83212868을 고정하고 기존 사용자36/의미확인 공개2개 특징으로 새head 점수만 계산했다. 기존d1과 두분류범위 모든1위예측 동일. 과거사용자17/18(나→죄송하다), 최근 전체93은17/18(자다→서울대학교)/목표18은18/18, 공개공부성공·먹다실패1/2다. b8 대비 과거나 회귀유지; 기본교체0.
- encoder58상태동일·3534점수/76순위 독립계산19PASS, 실행/검산종료0. 기존특징재사용이며 새학습/encoder추론/원본관측열람/보류0. 개발노출1명과 촬영자미확정 공개2개 진단을 새사람일반화로 부르지 않는다. `ops/head_only_web_replay_20260922/RESULT.md`.
- 보유SEN0354 안녕하세요와 기존INSA 인사 원본을 제시하고 의미동등성을 질문했다. 현재답변없음·동일클래스병합0. 라벨기존9학습후보/3개발만연결, 보유12ZIP의선택적인덱스에해당프레임없음(원본ZIP없다는뜻아님). 다음원본프레임확인/입력구성은의미답변후. SEN0254반갑다/다른KSL34개는이번범위밖. 문의/다운로드/새화면/촬영/Unity/삭제/백업/예약0.

'''
    following=f'''### {human} 다음 단일 작업 — 보유 인사 관련 원본의 의미 판단 후 입력 준비

- `ops/greeting_semantic_gate_20260922/PENDING_SEMANTIC_REVIEW.json`의질문이진행중이다. 기존INSA 시범↔AIHub SEN0354 안녕하세요가같은의미의유효표현인지확인한다. 의미미확정인형태차이를곧다른의미로단정하거나,단순진행지시/시간경과를긍정답변으로대체하지않는다. 이미제시한두영상을다시추출/디코드하거나질문을반복하지않는다.
- 긍정이면기존보유관절ZIP에서01–09/11–13의정확한SEN0354 F프레임을해결하고검증된입력변환으로준비한다. 현재라벨메타데이터12개만연결,원본프레임/280채널새입력준비전이다. 기존선택적인덱스0건을원본부재로오해하지않는다. 원천라벨/형태계보와18목표/93출력·사람분리유지,보류10/14–18/25열람금지.
- 부정/판단어려움이면매핑null과학습미승인유지. 현재후보83212868/d1/b8재대입·결과기반재튜닝/후보재선택·기본승격/새UI/촬영/문의/다운로드/백업/삭제/Unity/예약은하지않는다. 이번단위worker모두종료, 사용자작업중단요청아님·목표active/미완료.

'''
    text=text.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+current,1)
    text=text.replace('## 9. 다음 단일 작업\n\n','## 9. 다음 단일 작업\n\n'+following,1);root.write_text(text,encoding='utf-8')
    paths=[p for p in H.iterdir() if p.is_file()]+[pending_path]
    write(H/'CLOSEOUT_LOCAL_ONLY.json',dict(status='COMPLETE_FROZEN',at_kst=stamp,record=record,
       hashes={str(p):sha(p) for p in paths},worker_exit_codes=state['autonomous_workers_current_unit'],
       next_semantic_response=None,next_native_input_prepared=False,project_complete=False))
    assert read(state_path)['current_stage']==state['current_stage'] and root.read_text(encoding='utf-8').count(current)==1
    print(dict(status='CLOSED',at_kst=stamp,semantic_response=None,project_complete=False))

if __name__=='__main__':main()
