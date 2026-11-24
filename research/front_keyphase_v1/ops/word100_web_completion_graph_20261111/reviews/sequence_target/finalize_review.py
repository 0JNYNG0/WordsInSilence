from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
d=json.loads((H/'TARGETS.json').read_text('utf-8'))
for r in d['rows']:
    c=r['candidate']
    if c is None:continue
    c['preview']['reviewed']=True
    c['preview']['review_observation']='Directly inspected exactly the listed preview frames, including both sides of each discontinuous join. Preserved within-block direction and visible preparation/hold/return match source observations; unnatural jumps remain. The preview is sampled, while the two source clips were viewed frame by frame.'
    inv=c['source_to_query_frame_ids'];s=c['semantic_direction_order']
    s['permuted_witness_query_frame_ids']={k:[inv[f] for f in ids] for k,ids in s['normal_witness_source_frame_ids'].items()}
    for p in r['panels']:assert sha(p['path'])==p['sha256']
    assert sha(c['preview']['path'])==c['preview']['sha256']
    assert all(f in r['reviewed_frame_ids'] for obs in r['observations'] for f in obs['source_frame_ids'])
d['finalized_at']=datetime.datetime.now().astimezone().isoformat()
write(H/'TARGETS.json',d)
check=json.loads((H/'CHECK.json').read_text('utf-8'))
check.update(target_file_sha256=sha(H/'TARGETS.json'),preview_visual_review_completed=True,all_direct_observations_are_in_reviewed_frames=True,source_frame_numbering='zero based',new_predictions_seen=False)
write(H/'CHECK.json',check)
report='''작업: 공개 DEV 완전 동작 블록 순서 진단 정답 설계
작성/실행/직접 RGB 검토/산술 검사: Codex sequence_target_review 에이전트. 독립 사람·수어 전문가 정답은 아님.

완료
- 공개 순종 REAL11 141프레임, 이집트 REAL11 106프레임 전부 원본 RGB contact sheet로 직접 검토했다. 확대 손 장면은 각20시점이다.
- 순종/이집트 REAL12,13은 전체 길이의4프레임 간격 개요129시점만 검토했다. 빠른 경계/잔여동작은 미확정이므로 이4영상의 합성 정답은 만들지 않았다.
- TARGETS.json에 원본 경로/sha256,0부터 시작하는 source_frame_id, 검토 이미지/sha256/실제 프레임목록, 경계 불확실성, 블록별 source→query 대응, 접합지점과 평가 제한을 기록했다.
- 원본6개 해시, 검토 이미지22개 해시, 두 변환의 원본프레임 각1회 보존·개수보존·블록내시간순서보존·모든 직접관측프레임의 실제검토포함을 검사했다. 변환 미리보기2개도 확인했다.

정답 범위
1. 순종: prefix0..35, A36..66, B67..79, C80..119, suffix120..140. [prefix,A,C,B,suffix]. 동적 B=편 손을 닫으며 얼굴→가슴 내리기, C=닫힌 손의 전완을 세우기 한 쌍만 C-before-B로 정의한다. A의 정적 편 손/얼굴 앞 자세가 B시작67..68에도 필연적으로 남으므로 모든 가이드쌍을 역순오답으로 세지 않는다.
2. 이집트: prefix0..32, A33..54, B55..90, suffix91..105. [prefix,B,A,suffix]. A=양손끝을 가까운 삼각형 윤곽으로 접근, B=양손끝 벌림/재배향 한 쌍을 B-before-A로 정의한다. 엄지-검지 간격의 가려진 마지막 변화는 unknown이며 주먹·검지형 정답으로 대체하지 않는다.
- 두 경우 모두 준비·유지·복귀까지 블록에 포함하여 편집 밖에는 전프레임 확인한 대기자세만 남긴다. 블록의 기계적 자르기 위치는 정확한 생리적 경계와 동일시하지 않는다.
- 불연속 접합의 순간 이동은 실제 수행한 동작/방향 정답이 아니다. 합성 진단이며 실사용 오류 성능·새 사람 평가·수어 전체 의미 완료 정답으로 해석하지 않는다.
- 과거 sequence_truth_audit.json은 요청대로 읽었으며 그 안의 과거 모델상태 필드에 노출됐다. 현재 INTERVAL 결과나 새 모델출력을 읽지 않았고 이 정답/범위는 원본RGB로 정했다. 완전 blind 독립 평가라고 주장하지 않는다.

실행 명령
python research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/reviews/sequence_target/prepare_review.py
python research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/reviews/sequence_target/prepare_detail.py
python research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/reviews/sequence_target/make_targets.py
python research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/reviews/sequence_target/finalize_review.py

미실행/변경 범위
학습·모델추론·영상 재인코딩·개인/멘토 자료 접근·다운로드·Git·원본W·기존결과·웹·루트기준문서 변경 없음. 모든 산출물은 reviews/sequence_target 내부뿐이다. 진행중 프로세스 없음.

다음 한 작업
부모 Codex가 TARGETS.json의 두 제한적 descriptor 정답·변환 인덱스를 검산하고, 고정 모델의 진단에 사용할 경우 이 scope와 splice 제한을 그대로 보존한다. REAL12/13을 평가 분모로 늘리려면 먼저 추가 밀집 RGB 검토를 완료해야 한다.
'''
(H/'HANDOFF_KO.txt').write_text(report,'utf-8')
print(json.dumps(dict(target_sha256=sha(H/'TARGETS.json'),panels=sum(len(r['panels']) for r in d['rows']),preview_count=2,unique_source_frames_reviewed=sum(len(r['reviewed_frame_ids']) for r in d['rows'])),ensure_ascii=False))
