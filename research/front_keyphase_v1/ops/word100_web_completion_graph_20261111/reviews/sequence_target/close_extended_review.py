from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
d=json.loads((H/'TARGETS.json').read_text('utf-8'));checks=[]
assert d['input_manifest_sha256']==sha(H/'INPUTS.json')
for r in d['rows']:
    assert sha(r['source'])==r['source_sha256']
    for p in r['panels']:assert p['reviewed'] and sha(p['path'])==p['sha256']
    assert all(f in r['reviewed_frame_ids'] for o in r['observations'] for f in o['source_frame_ids'])
    c=r['candidate']
    if c:
        p=c['preview'];assert sha(p['path'])==p['sha256'];p['reviewed']=True
        p['review_observation']='Directly viewed the listed mapped preview frames including each join. Within-block action direction remains original. Splice jumps are visible and remain excluded from performed-action truth. Source coverage is dense; this preview itself is sampled.'
        q=c['query_to_source_frame_ids'];assert sorted(q)==list(range(r['frame_count']))
        for b in c['blocks'].values():assert b['source_frame_ids']==list(range(b['source_span'][0],b['source_span'][1]+1))
        for key in ['all_source_frames_exactly_once','each_block_internally_ascending','all_source_frames_visually_reviewed','all_descriptor_witnesses_inside_single_block','frame_count_preserved']:assert c['preservation_checks'][key]
        g=c['semantic_direction_order'];g.pop('semantic_order_negative_gold',None);g['guide_semantic_order_negative_gold']=False
        g['visible_dynamic_descriptor_order_negative_gold']=r['word_id']=='WORD0739'
        g['negative_gold_qualification']='Only the expressly defined visible dynamic occurrence pair in obedience is eligible. No unconditional guide-level or whole-sign semantic order error gold is supplied.'
        if r['word_id']=='WORD0739':
            c['suffix_join_caution']='The join from the end of B to original return jumps from a near-horizontal posture to a raised posture. This can imitate a raising event to an unmasked model. It is a splice, not a natural action; both source-adjacency and the complete within-block direction evidence must be retained in reporting. Do not score across-join differences as semantic motion gold.'
    checks.append(dict(clip_id=r['clip_id'],source_sha256_match=True,panel_hashes_match=True,reviewed_frames=len(r['reviewed_frame_ids']),total_frames=r['frame_count'],dense=r['all_source_frames_reviewed'],candidate=c is not None,dynamic_descriptor_negative_eligible=bool(c and r['word_id']=='WORD0739')))
d['finalized_at']=datetime.datetime.now().astimezone().isoformat()
write(H/'TARGETS.json',d)
write(H/'CHECK.json',dict(status='PASS_SAME_AGENT_SOURCE_HASH_FRAME_COVERAGE_AND_MAP_ARITHMETIC',rows=checks,target_file_sha256=sha(H/'TARGETS.json'),source_panels=37,current_previews=4,source_frames_reviewed=575,dense_source_frames=516,limited_dynamic_descriptor_pairs=3,guide_level_semantic_negative_gold=0,independent_human_or_expert_validation=False,new_model_inference=False,training=False))
text='''작업: 공개 DEV 완전 동작 블록 순서 진단 정답 설계 — 최종
작성/실행/직접RGB검토/산술검사: Codex sequence_target_review. 부모 Codex의 정답범위·복귀구간 검토를 반영했다. 독립 사람·수어 전문가 검증은 아니다.

완료 범위
- 순종 REAL11/12/13의141+123+146프레임, 이집트 REAL11의106프레임: 총4원본516프레임 전부 contact sheet로 직접 검토했다. REAL11 두 영상은 손 확대 각20시점도 보았다.
- 이집트 REAL12/13은4프레임 간격 전체개요59시점만 확인했다. 추가 밀집검토 전 정확한 동작 경계/잔여동작/순열 정답 없음.
- 총6원본575고유프레임, 원본검토시트37개, 현재순열미리보기4개. TARGETS.json에 모든 실제검토 source_frame_id(0시작), 원본/이미지경로·sha256, 직접관측, 경계불확실성, 정확한 query→source·source→query 배열과 접합지점을 기록했다.
- 원본6개 및시트해시,4순열의모든원본프레임각1회보존·개수보존·블록내시간순서보존·관측프레임실제검토포함 검사PASS. CHECK.json은 같은 Codex 산술검사이며 별도전문가확인과 구분한다.

최종정답범위
순종3영상은 [prefix,A,C,B,suffix]. B는 편손에서 닫으며 얼굴→가슴높이내림, C는 닫힌손의 전완을 수평에 가까운상태에서 더세우는 동적변화이다. 내부방향을 보존한 이한쌍의 C-before-B만 제한적 descriptor 진단정답이다.
REAL11: prefix0..35/A36..66/B67..79/C80..103/suffix104..140.
REAL12: prefix0..30/A31..58/B59..72/C73..101/suffix102..122.
REAL13: prefix0..44/A45..76/B77..90/C91..114/suffix115..145.
복귀는 모두 원래 global suffix에 둔다. suffix는 복귀+대기이며 아무움직임도 없다는주장은 하지 않는다. 전체 source검토에서 B의편손→닫기/얼굴→가슴 또는 C의수평→세우기를 완전히 재실행한 근거는 없었다. 초기정적자세가 남는다는 점과 새로운동적행동은 구분한다.
이집트 REAL11: prefix0..32/A33..54/B55..90/suffix91..105를 [prefix,B,A,suffix]로 매핑. source_identity_order 진단만허용. B초기정적지붕형자세가 A가이드조건을 만족할가능성이 있어 semanticnegative분모에서제외했다. 엄지마지막가림은unknown이며 주먹정답으로대체하지않았다.
순종도 A의정적편손/얼굴앞자세가 B초기상태에남으므로 모든가이드쌍을 역순오답으로세지않는다. guide-level/whole-sign semanticnegativegold는제공하지않으며 visible_dynamic_descriptor_order_negative_gold만3개이다.

접합한계
합성접합의순간이동은 실제동작/방향정답이아니다. 특히 B끝→원본suffix시작의높은손자세점프가 모델에는추가세우기로보일수있다. source-adjacency와완전한내부동작증거를함께보고해야하며 접합을가로지른차분을정답으로쓰지않는다. 자동구간/중단판정의반응도편집흔적영향을받는다. 실제사용오류성능·새사람평가·수어전체완료성능으로해석금지. 내부시간순서보존은기계적으로확인했으나 수어전문가의완전의미정답을증명한것은아니다.

노출/분리
모두이미평가에노출된공개DEV이며FIT/학습/문턱·후보선택에쓰지않는다. 요청된과거sequence_truth_audit를읽었고 그내과거모델상태에노출됐다. 현재INTERVAL결과나새예측을읽지않았으며 이번경계/순열은RGB와부모의정답논리검토로정했다. 완전blind독립평가가아니다.

실제파일/실행
이폴더내 prepare_review.py → prepare_detail.py → make_targets.py → finalize_review.py로 초기2개를만들었고, 부모검토후 prepare_obedience_extra.py → extend_review.py → close_extended_review.py로현재범위를확정했다. 초기make/finalize는중간이력이며 단독재실행하면이전범위를재생성하므로재실행금지. 현재정답은TARGETS.json과CHECK.json. 오래된순종 _permutation_preview.jpg는복귀가C에포함된초안이며현재참조는 _permutation_preview_v2.jpg이다. 최초draft한계/정정은TARGETS.revision_record에남겼다.
학습·모델추론·영상재인코딩·개인/멘토자료·다운로드·Git·원본W·기존외부결과·웹·루트기준문서변경없음.산출물은 reviews/sequence_target만. 진행중프로세스없음.

다음한작업
부모Codex가최종세제한적descriptor정답/맵을검산한뒤고정모델진단에사용할경우 원본대조·sourceorder·dynamicdescriptororder·splice영향을분리해보고한다. 이집트순열을의미오답분모로합치지않는다.
'''
(H/'HANDOFF_KO.txt').write_text(text,'utf-8')
print(json.dumps(dict(target_sha256=sha(H/'TARGETS.json'),checked_rows=len(checks),dynamic_descriptor_pairs=3,guide_semantic_negative_gold=0),ensure_ascii=False))
