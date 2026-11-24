"""Render concise evidence from already saved comparison/audit; no rescore."""
import json
from pathlib import Path
from collections import Counter

U = Path(__file__).resolve().parent
def read(name): return json.loads((U/name).read_text(encoding='utf-8'))
saved, audit, bank = (read(n) for n in ('COMPARISON_RESULTS.json','PROJECTION_AUDIT.json','REFERENCE_BANK.json'))
words={w['word_id']:w for w in bank['words']}
queries={r['query'].get('legacy_alias') or r['query']['alias']:r['query'] for r in bank['development_queries']}
events={(e['query'],e['reference'],e['method'],e['reference_event']):e for e in audit['events']}
consumer=[]
for q in saved['results']:
    if q['test_role']!='development':continue
    item={'practice_word_id':q['practice_word_id'],'label':words[q['practice_word_id']]['label'],
          'query_alias':q['query'],'query_source':queries[q['query']]['source'],'references':[],
          'whole_sign_verdict':None}
    for ref in q['references']:
        entry={'reference_alias':ref['reference_alias'],'reference_source':ref['reference_source'],'methods':{}}
        for method,md in ref['methods'].items():
            phase=Counter();metrics={s:Counter() for s in (42,43,44)};stages=[];examples=[]
            for st in md['stages']:
                ev=events[(q['query'],ref['reference_alias'],method,st['reference_event']['id'])]
                phase.update(ev['weighted_states'])
                trace={(t['reference_frame'],t['query_frame']):t['status'] for t in ev['trace']}
                frames=st['seeds'][0]['frames']
                qpoints=sorted({(l['query_frame'],l['query_pts_us']) for f in frames for l in f['links']})
                stage={'reference_event_id':st['reference_event']['id'],
                    'visible_description_ko':st['reference_event']['description_ko'],
                    'reference_core_frames':st['reference_event']['core'],
                    'reference_core_seconds':[frames[0]['reference_pts_us']/1e6,frames[-1]['reference_pts_us']/1e6],
                    'projected_query_seconds_envelope':[qpoints[0][1]/1e6,qpoints[-1][1]/1e6] if qpoints else None,
                    'query_frames_with_original_pts_us':qpoints,'unknown_boundaries_not_filled':True,
                    'projection_audit':ev['weighted_states'],'source_candidate_ambiguity':st['saved_candidate_ambiguity']}
                stages.append(stage)
                for seed in st['seeds']:
                    for feature,info in seed['features'].items():
                        m=metrics[seed['seed']];m['expected']+=info['expected_reference_frames']
                        m['qualified']+=info['confidence_qualified_weight']
                        m['agreement']+=info['weighted_states'].get('ESTIMATED_FEATURE_AGREEMENT',0)
                        m['difference']+=info['weighted_states'].get('ESTIMATED_FEATURE_DIFFERENCE_CANDIDATE',0)
                    for f in seed['frames']:
                        for link in f['links']:
                            for ix,state in enumerate(link['states']):
                                if state=='ESTIMATED_FEATURE_DIFFERENCE_CANDIDATE':
                                    examples.append({'reference_event_id':stage['reference_event_id'],'seed':seed['seed'],
                                        'feature':seed['target_order'][ix],'query_frame':link['query_frame'],
                                        'query_seconds':link['query_pts_us']/1e6,'reference_frame':f['reference_frame'],
                                        'reference_seconds':f['reference_pts_us']/1e6,
                                        'query_probability':link['query_probabilities'][ix],
                                        'reference_probability':f['reference_probabilities'][ix],
                                        'projection_audit_at_this_link':trace[(f['reference_frame'],link['query_frame'])],
                                        'educational_error':None})
            entry['methods'][method]={'seed_metrics':dict(metrics),'physical_projection_audit':dict(phase),
                                     'stages':stages,'difference_examples_all':examples,'selected_best_method':False}
        item['references'].append(entry)
    consumer.append(item)
evidence={'schema':'wis_target_known_consumer_evidence_v1','scope':'Derived saved public historical DEV4 evidence',
          'queries':consumer,'all20_routing':saved['routing_results'],'educational_accuracy':None}
with (U/'CONSUMER_EVIDENCE.json').open('x',encoding='utf-8') as stream:
    json.dump(evidence,stream,ensure_ascii=False,indent=2)

text=[
'목표 단어를 지정한 공개 참고 비교 — 실행 결과', '',
'목표는 그대로다. 게임에서 고른 연습 단어의 참고를 각각 찾아, 확인 가능한 손모양·관계·순서의 근거를 비교한다. 새 단어 일반화를 이 제품 목표의 선행 관문으로 두지 않았다.', '',
'실제로 연결한 것',
'20개 단어의 FIT 참고60개와 DEV 입력20개를 모두 라우팅했다. 기존 주요 단계와 자동 정렬이 있는4단어/16영상은 구간별 비교를 실행했다. 나머지16단어는 참고를 찾되 단계 비교 미지원으로 남겼다. 입력의 정답 단어·수동 단계 라벨은 자동 비교에 쓰지 않았다.',
'기존 global_dtw/core_first 두 경로와 기존 손모양3출력(R_index_only/R_open_four/L_open_four)의 seed42/43/44를 모두 고정했다. global 경로도 순수 기하만이 아니라 당시 관측 기하와 저장된 움직임 출력을 사용한 자동 경로다. 최고점 참고·방법·seed를 선택하지 않았다.',
'원래117개 상위 단계 비교와144개 세부 구간을 구분했다. 방법·seed마다 참고1080프레임×3항목=3240셀, 총19440셀의 분모를 보존했다. 같은 종류의 단계·unknown 경계를 합치지 않았다.', '',
'부분 근거와 실제 대응의 차이',
'DEV의1080셀 중 confidence 조건을 만족한 비교 근거는 global925.5~934셀(85.69~86.48%), core_first923~935.67셀(85.46~86.64%)였다. 전체 입력이 모두 관측돼야 한다는 가상 관문이면 모두 보류됐을 근거를 부분별로 제공한다. 운영 성능 향상이나 수어 정답률을 뜻하지 않는다.',
'실행 저장 후 수동 물리 구간으로 별도 감사한 결과, DEV 참고360프레임 중 같은 관측 종류 대응은 global295.79(82.16%), core_first281.83(78.29%)였다. 경계·미확인·다른 종류 대응도 그대로 남았다. 따라서 손모양 비교 가능률을 정확한 단계 정렬률로 표시하지 않는다.', '',
'네 단어의 원본 시점과 개별 참고',
'아래 시간은 저장된 원본PTS의 대응 범위다. 범위 안 모든 프레임이 확인됐다는 뜻은 아니다. CONSUMER_EVIDENCE.json에는 세부 단계별 원프레임·시각·차이 후보와 대응 감사가 함께 있다.'
]
unsupported={
'WORD0739':'오른손이 아래 얼굴 앞에서 펼쳐지고 내려가 어깨 쪽에 머무르는 참고 단계다. 얼굴 접촉·정확한 닫힌 손모양·손바닥 방향은 원3항목으로 판정하지 못한다.',
'WORD1133':'오른손이 왼팔을 따라 움직이고 그 위에 머무르는 참고 단계다. 왼손 주먹의 정확한 구성·팔 접촉·허용 위치 편차는 미지원이다.',
'WORD1567':'엄지를 든 두 손의 교대 상승과 높은 위치 유지라는 참고 단계다. 원3항목에는 엄지 전용/정확한 주먹 분류가 없으므로 이 주요 손모양을 검증했다고 할 수 없다. 정확한 반복 횟수도 요구하지 않는다.',
'WORD2523':'양손을 모으고 벌려 유지하는 참고 단계다. 펼친 손의 일부 특징은 비교하되 손바닥 방향·접촉·언어상 필수 순서는 확정하지 않는다.'}
for item in consumer:
    text+=['',item['label']+' — 입력 '+item['query_alias'],unsupported[item['practice_word_id']],
           '입력 영상: '+item['query_source']]
    for ref in item['references']:
        for method,md in ref['methods'].items():
            spans=[s['projected_query_seconds_envelope'] for s in md['stages'] if s['projected_query_seconds_envelope']]
            met=list(md['seed_metrics'].values());phase=md['physical_projection_audit'];den=sum(phase.values())
            text.append(f"참고 {ref['reference_alias']} / {method}: 입력 {min(s[0] for s in spans):.3f}~{max(s[1] for s in spans):.3f}초; 같은 물리 종류 대응 {phase.get('same_observed_kind',0):.2f}/{den:g}; 손모양 비교 근거 {min(m['qualified'] for m in met):.2f}~{max(m['qualified'] for m in met):.2f}/{met[0]['expected']:g}셀; 차이 후보 가중치 {min(m['difference'] for m in met):.2f}~{max(m['difference'] for m in met):.2f}.")
    examples=[(r['reference_alias'],method,e) for r in item['references']
              for method,md in r['methods'].items() for e in md['difference_examples_all']]
    if examples:
        ra,method,e=sorted(examples,key=lambda z:(z[2]['query_seconds'],z[0],z[1],z[2]['seed']))[0]
        text.append(f"가장 이른 차이 후보 예시: 입력 {e['query_seconds']:.3f}초(frame{e['query_frame']}) ↔ 참고 {ra} {e['reference_seconds']:.3f}초(frame{e['reference_frame']}), {method}/seed{e['seed']}/{e['feature']}, 확률 {e['query_probability']:.3f} 대 {e['reference_probability']:.3f}. 해당 대응 감사={e['projection_audit_at_this_link']}. 실제 학습자 오류의 정답은 없다.")
    else:
        text.append('원3항목에서 고정 confidence 조건을 만족한 차이 후보는 없었다. 미지원 손모양·관계나 다른 단계 대응이 맞았다는 증거는 아니다.')
text += ['', '실제로 드러난 한계',
'잘하다 P048→P047의 core_first는 같은 물리 종류 대응4.67/22프레임, global은13.75/22였다. 두 방법 모두 원3항목 차이 후보가 없었다. 따라서 차이가 없다고 동작이 맞거나 정렬이 정확하다고 표시해서는 안 된다. 구간별 근거와 대응 한계·미지원 특징을 함께 보여야 한다.',
'손 사이 상대좌표3항목은 기존 관측 mask와 함께 원값·차이를 제공한다. 작은 수치 차이에 임의의 언어 허용 문턱을 넣거나 접촉/오류로 이름 붙이지 않았다. 주먹·엄지·정확한 접촉·방향과 실제 학습자 오답 정답은 이번에 확보되지 않았다. 새7추가4는 중심시점 저장값뿐이므로 전체 구간으로 보간하지 않았다.', '',
'검증과 보존',
'자체 안전검사28개 통과. 저장 배열 기술 제어183개 실행 통과,12개는 해당 원구간의 입력지원/confidence가 없어 수행 불가로 보존했다. 시간 변형·한 특징 변경·가리기·대응 삭제·순서 반전/겹침의 기술 검사이며, 실제 변형 영상을 다시 자동 정렬한 검증이 아니다. 실제 학습자 오류 성능도 아니다.',
'독립 저장 산술 검산47062개 통과. 원본 C·T3F·운영·이전 점수·자료 보존. 학습0/새 추론0/정렬 재탐색0/다운로드0/개인·멘토자료0/HTML0/배포0.', '',
'API와 근거',
'compare_practice(practice_word_id, query_alias, bank, runtime_store). 게임이 준 명시 단어로 FIT 참고를 선택하며 관측을 덮어쓰지 않는다. 자기 참고·없는 단어·미지원 단계·없는 경로를 구분한다. 반환은 개별 참고×방법×seed×부분 구간이며 전체 수어 정답은 null이다.',
'COMPARATOR_EXECUTION.json: 실행 전 코드·입력·72경로·모델출력 동결',
'COMPARISON_RESULTS.json / PROJECTION_AUDIT.json: 실제 비교와 실행 후 대응 감사',
'CONSUMER_EVIDENCE.json: 네 단어의 원시점·구간·개별 차이 후보',
'COMPARATOR_SELFTEST_FROZEN.json / COMPARATOR_CONTROLS.json: 자체·기술 검사',
'review/COMPARATOR_RESULTS_VERIFICATION.json: 독립 저장 산술 검산']
with (U/'COMPARATOR_RESULT_KO.txt').open('x',encoding='utf-8') as stream:
    stream.write('\n'.join(text)+'\n')
print(json.dumps({'queries':len(consumer),'report_lines':len(text)}))
