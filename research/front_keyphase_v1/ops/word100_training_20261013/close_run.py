"""Record verified project state only. Presentation report/chart remain deferred."""
from datetime import datetime
from pathlib import Path
from common import U,BASE,ROOT,RUN,read,write,sha,pin_check

def main():
    f=pin_check();check=read(U/'METRICS_CHECK.json');inputs=read(U/'INPUT_CHECK.json')
    assert check['status']==inputs['status']==read(U/'MEDIA_CHECK.json')['status']=='PASS'
    assert read(U/'CACHE_REUSE_CHECK.json')['status']=='PASS'
    assert read(U/'EXIT.json')['exit_code']==0
    assert all(sha(p)==h for p,h in read(U/'PRODUCTION_GUARD.json').items())
    assert not (U/'COMPLETION.json').exists()
    r=read(RUN/'COMPARISON.json');a=r['arms']['whole'];b=r['arms']['whole_context']
    assert check['correct']=={'whole':a['metrics']['correct'],'whole_context':b['metrics']['correct']}
    now=datetime.now().astimezone().isoformat()
    predictions=read(RUN/'whole_context/PREDICTIONS.json');errors=[]
    for x in predictions:
        if x['correct']:continue
        errors.append(dict(clip_id=x['clip_id'],signer_id=x['signer_id'],
            expected_code=f['classes'][x['label_index']],expected_label=f['labels'][x['label_index']],
            predicted_code=f['classes'][x['prediction']] if x['prediction'] is not None else None,
            predicted_label=f['labels'][x['prediction']] if x['prediction'] is not None else None,
            regression_vs_whole=x['clip_id'] in r['regressed']))
    assert len(errors)==300-b['metrics']['correct']
    write(U/'ERROR_INDEX.json',dict(source='Saved predictions only; no new inference or tuning',candidate_errors=errors,
        improved=r['improved'],regressed=r['regressed']))
    done=dict(status='COMPLETE_VERIFIED',at_kst=now,unit='ops/'+U.name,
        implementation='Codex local implementation and execution',verification='Codex separate raw-array reconstruction and NumPy saved-logit recount; not an external reviewer',
        classes=100,train_rows=899,development_rows=300,input_frames=inputs['frames'],input_failures=inputs['failures'],
        methods={key:{k:x[k] for k in ['checkpoint_sha256','selected_epoch','epochs_executed','steps','seconds','parameter_count','metrics']} for key,x in r['arms'].items()},
        difference_pp=r['percentage_point_difference'],improved=len(r['improved']),regressed=len(r['regressed']),
        caveats=['Same300 development rows select epochs and report metrics; not an untouched final test.',
                 'Same early-stopping rule, different selected epochs36/60; comparison of training procedures, not a fixed-step proof that cropping alone caused every gain.',
                 'Different vocabulary/data from old93-class model; no direct improvement claim over its accuracy.',
                 'Live-web/new-user correctness and rejection of malformed signs untested; one seed.'],
        defect_repaired='Before training: exact integer-microsecond endpoint restored one frame in each of3 training contexts. Originals/whole/development unchanged; prior artifacts and first failure preserved.',
        checks={k:str(U/k) for k in ['INPUT_CHECK.json','MEDIA_CHECK.json','MEDIA_VISUAL_CHECK.json','CACHE_REUSE_CHECK.json','METRICS_CHECK.json','CONTEXT_BOUNDARY_REPAIR.json']},
        frozen_experiment_sha256=sha(U/'FROZEN_EXPERIMENT.json'),comparison_path=str(RUN/'COMPARISON.json'),
        production_checkpoints_preserved=True,production_model_replaced=False,
        presentation_report='Deferred by latest user instruction; report.py not executed.',
        next='Review this100-class candidate for explicit web research integration and external evaluation; no automatic production replacement or heldout use.')
    assert not (U/'RESULT_KO.txt').exists() and not (U/'ACCURACY.png').exists()
    write(U/'COMPLETION.json',done);write(U/'STATUS.json',done)
    s=read(BASE/'ops/STATE.json');s['updated_date_kst']=now;s['current_stage']='WORD100_COMPLETE_VERIFIED'
    s['word100_training_20261013']=done
    s['next_single_action']='100단어 후보의 웹 연구용 연결 범위와 기존18개 표현의 어휘 교집합 확인. 완료 학습/검증 재실행·점수에 맞춘 튜닝·보류시험·운영기본교체는 자동 실행하지 않음. 발표 자료는 사용자 지시에 따라 마지막으로 유예.'
    write(BASE/'ops/STATE.json',s)
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig')
    def replace_top(text,num,title,body):
        start=text.index('\n',text.index(f'## {num}. '))+1;old=text.index('### ',start);end=text.index('\n### ',old+4)
        assert '100단어' in text[old:old+100]
        return text[:start]+'\n'+title+'\n'+body+'\n'+text[end:]
    text=replace_top(text,8,'### 2026-10-13 AI Hub 정면 100단어 학습·검증 완료',f'''`ops/{U.name}/COMPLETION.json`, `runs/word100_tcn_v1/COMPARISON.json`. 상태 COMPLETE_VERIFIED, {now}.
학습899개(ID01~09), 개발300개(ID11~13); 100클래스 fresh MaskedSignTCN/280채널/37,636매개변수. 기존 웹 MediaPipe로 1,199영상/150,476프레임 준비, 입력 실패0.
전체 동작:277/300(92.33%), 전체·핵심 혼합:290/300(96.67%), +4.33pp. 개선17/회귀4. 촬영자11:94→98,12:88→92,13:95→100(각100개).
동일한 사전 고정 CE 선택/early-stop 규칙. 전체는48회 실행/36회 선택, 혼합은72회 실행/60회 선택. 다른 선택 epoch까지 포함한 학습 절차 비교이며, 같은 단계 수에서 구간 변경만의 인과 효과를 입증한 실험은 아님.
학습 전 경계 부동소수점 결함3건을 발견: 각각 끝1프레임 복원. 원본 관절·전체 구간·개발 입력 보존, 최초 오류와 이전 파생 파일 보존. 재추출0, 기준 완화0. 전체 입력·6원본 시각·저장 점수·선택 기록 검산 PASS. Codex 작성/별도 계산 검산, 외부 전문가 검수 아님.
개발300개는epoch선택에도 쓰였음. 독립 최종 시험/실제 웹/잘못된 수형 거절 능력은 미검증. 기존93클래스와 직접 점수 비교 금지.
원본ZIP·기존 두 체크포인트 해시 보존, 운영교체0. NIKL/KSL/사용자/멘토 및ID10·14~18 미사용. 양쪽학습1회씩 완료. 프로세스 exit0.
최신 사용자 지시로 발표용 보고서·차트 작성 유예. 기술 결과와 재개 기록만 갱신. 후보 오답10개는 ERROR_INDEX.json.
''')
    text=replace_top(text,9,'### 2026-10-13 다음 — 검증된 100단어 후보의 웹 연결 범위 확인',f'''학습·검산 완료 `ops/{U.name}/COMPLETION.json`. 다음은100개 새 어휘와 기존18개 표현의 교집합을 확인하고, 웹의 명시적 연구 모델 선택으로 연결할 범위를 정리하는 것이다. 기존 기본 모델 교체는 별도 결정이다.
현재100개 결과로 추가계수탐색/단어교체/보류시험을 자동 시작하지 않는다. 실제웹·새사용자·틀린수형 판별은 따로 검증해야 한다.
이번 관절 캐시는 어휘 확장 시 재사용한다. 학습·개발 검증을 완료했으므로 동일 실행을 반복하지 않는다. 발표 자료는 사용자가 요청한 마지막 단계로 남긴다.
''')
    p.write_text(text,'utf-8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계
갱신 {now}. AGENTS / CONTEXT 최신 §3·§8·§9 / WORKFLOW를 먼저 읽는다.

## 마지막 완료
사용자100단어 진행 승인으로 `ops/{U.name}` COMPLETE_VERIFIED. COMPLETION / INPUT_CHECK / METRICS_CHECK, `runs/word100_tcn_v1/COMPARISON.json` 참조.
AIHub-only fresh MaskedSignTCN 100클래스. 학습899(ID01~09), 개발300(ID11~13). 전체1,199영상/150,476프레임, 입력실패0.
전체 학습277/300=92.33%, 핵심 혼합290/300=96.67%, +4.33pp. 개선17/회귀4, 혼합 오답10개 ERROR_INDEX.json.
같은 사전 선택규칙, 전체48회 실행/36회 선택; 혼합72회 실행/60회 선택. 고정 단계 수의 순수 구간 효과 비교는 아님. 개발자료로epoch선택했으므로 독립 최종시험 아님.
학습 전3개 context 끝1프레임 경계 반올림 결함을 수정. 최초 오류·old context 보존, 원 관절/whole/dev 불변. 전체 입력·원본6개·저장 점수/trace 별도 검산PASS, 모델기본 해시 유지.
Codex 로컬 작성·실행·별도 알고리즘 검산, 다른 에이전트/수어 전문가 검수 아님. NIKL/KSL/사용자/멘토 및ID10·14~18 미사용.

## 실행 상태 / 다음
기존exec83125는검산오류exit1, 복구exec3399/controller44160은exit0 완료. 실행 기록 EXECUTION_SESSION.json. 새 추출기/학습기를 시작하지 않는다.
캐시 W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013, 모델 runs/word100_tcn_v1. 완료 추출·학습·개발점수·검산 재실행금지.
다음:100새어휘/기존18표현 교집합 확인 후 웹 연구용 모델 연결 범위. 운영 기본 교체·보류시험·추가튜닝 자동시작 없음. 실제 웹/새사용자/잘못된 수형 거절은 미검증.
최신 사용자: 발표용 보고서·그림은 마지막으로 미루고 프로젝트에 집중. report.py는 미실행; RESULT_KO.txt/ACCURACY.png 생성 안 함. 원영상 정합 검사용 contact sheet는 기능 검산 자료임.
어휘 확장은 기존 캐시 재사용 + 기존/새단어 자료를 함께 별도 학습. 기존93점수와 이번100점수 직접 비교 금지.

## 앞선 완료 (반복하지 않음)
word_label_time_contract_20261013:2494후보/22388학습행, 의미충돌506보류.
word_pilot12_inputs_20261013:12영상1444프레임 입력 검산, 이번에1개캐시 재사용·원ZIP바이트추가대조PASS.
word_dataset_plan_20261013:33ZIP1.181TB/270000영상/F54000/3000코드×18ID×5시점.
''','utf-8')
    state=read(BASE/'ops/STATE.json');assert state['word100_training_20261013']==read(U/'COMPLETION.json')
    assert all(Path(v).exists() for v in done['checks'].values())
    session=read(U/'EXECUTION_SESSION.json');session.update(status='COMPLETE_VERIFIED',recovery_exit_code=0,finished_at=now);write(U/'EXECUTION_SESSION.json',session)
    write(U/'CONTROLLER_PROGRESS.json',dict(status='COMPLETE_VERIFIED',exit_code=0,presentation_report='DEFERRED',finished_at=now))
    print('Verified project state saved. Presentation report/chart not generated.')

if __name__=='__main__':main()
