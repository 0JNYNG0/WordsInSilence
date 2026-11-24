from pathlib import Path
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote
import json, hashlib

U=Path(__file__).resolve().parent
ROOT=U.parents[3]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
r=read(U/'evaluation/RESULT.json')
checks=read(U/'INDEPENDENT_CHECK.json'); protection=read(U/'PROTECTION_CHECK.json')
assert protection['status']=='PASS'
assert sha(U/'BLIND_REVIEW_A.json')=='dc21f8ebd7e6e2265847721a29f20ce184d722d31af99b468e336dddf67e1fe6'
assert sha(U/'BLIND_REVIEW_B.json')=='d0bcdc7de1bc01f6cde97ad465fb0b7ea0da4e91df95ce0ffed7a17a0cad3301'
for c in r['by_condition']: assert r['by_condition'][c]['total_scenes']==24
final=dict(status='COMPLETED_DIAGNOSTIC_NO_NEW_MODEL_ADOPTED',at_kst=now,unit=U.name,
 scope={'scenes':24,'word_types':9,'people':6,'clips':21,'people_role':'existing TRAIN01-06, not unseen-user test','jointly_scorable_wrist_scenes':23,'withheld_blur_scenes':1,'saved_conditions':9,'comparison_rows':216},
 results={'redetection_A8':r['by_stratum']['B1']['A']['statuses'],'temporal_B8':{c:r['by_stratum'][c]['B']['statuses'] for c in r['by_condition']},'all24':r['by_condition']},
 conclusion='Image re-detection recovers six coarse-supported wrists among A8, one outside and one withheld. B8 Linear5/8, B2 each5/8, B3 4/8 3/8 3/8. Learned temporal/RGB superiority not established; B2/B3 not adopted.',
 limitations=['Two AI blind reviewers, not expert21joint ground truth','TRAIN and availability-mask-conditioned sample; one word8/24; not population sign accuracy','Wrist boxes cannot prove finger shape; fingertip-count disagreement12','No proof of full hidden-finger restoration; original observed values may still be wrong','Range-boundary distances are not anatomical keypoint errors; small changes near bounds can change coarse categories'],
 validation={'implementation':'Codex local/evaluator agent','independent_calculation':'temporal_bundle_inventory,3522 assertions; separate scientific review216rows','protected_files_unchanged':32,'training_runs':0,'new_model_inference':0,'user_mentor_recording_access':0,'production_changes':0,'downloads':0},
 next='Keep C/T3-F and all observations/estimates separated. Prepare reliable independent hand-keypoint evaluation before more training. Small external detector-only benchmark (FreiHAND official724MB with annotations claim) requires actual archive/XYZ/K checks after separately scoped acquisition; no invented body joints. InterHand later for interacting temporal evidence; no broad model expansion from current coarse wrist labels.',
 report=str(U/'RESULT_KO.html'),running_processes=[])
write(U/'FINAL_RESULT.json',final)
next_action='실제복원24장면 진단완료. B2/B3 단순보간 대비 우월성없어 미채택, C·T3-F/운영 유지. 다음은 정밀 손관절 정답을 가진 소규모 검출평가 설계와 확보조건 확인. FreiHAND 공식724MB 평가주석 안내는 확인했으나 실제압축XYZ/K내용미확인·미다운로드. 외부손사진에 가짜몸/어깨를 넣지말고 검출부품평가부터 분리. 현재24의 AI손목상자를21관절 학습정답으로 쓰지않음. 기존TEST08/09·사용자·멘토자료 튜닝금지. 새학습/운영교체없음.'
statep=U.parent/'STATE.json'; state=read(statep)
state.update(updated_date_kst=now,current_stage='RECOVERY_IMAGE_BENCHMARK_COMPLETED_NO_NEW_MODEL_ADOPTED',next_single_action=next_action)
state[U.name]=final
write(statep,state)
contextp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'; context=contextp.read_text(encoding='utf-8-sig')
tag='### 2026-10-25 실제 복원24장면 원영상 검증 완료 — 학습 보완 우월성 미확인'
assert tag not in context
summary='''

### 2026-10-25 실제 복원24장면 원영상 검증 완료 — 학습 보완 우월성 미확인
- 새 단위 `research/front_keyphase_v1/ops/word100_recovery_image_benchmark_20261025`. 공개24장면/9표현/6명/21영상, 전원기존TRAIN01–06. 마스크조건·동결선정, 과거36진단과문맥프레임중복0. 대표적인전체수어성공률아님.
- 두 AI가 예측좌표·상대판독을 보지않고원영상손목범위동결. 23장면비교가능,R06흐림불일치1보류. 전문가21관절정답아님. 손끝개수12장면불일치로손가락정밀성능주장금지.
- 재검출A8: B1양쪽범위내6/밖1/판독보류1. 시간추정B8: 선형5/8,B2각5/8,B3각4/8·3/8·3/8. RGB보완우월성없고일부회귀. B2/B3미채택,C·T3-F/운영DTW웹유지.
- 전체24범위내 B0=3,B1=12,선형=17,B2=17/17/17,B3=16/15/15. 전체수치에는기존관측유지가포함되어시간보완순수이득과구분. 근거부족·미가용은0오차/성공으로처리하지않음.
- 재검출R02/R13은판독범위밖이며모든시간후보가그대로보존. R16은선형지원이나B3범위밖/한쪽지원. 상자까지거리는해부학관절정확오차아님; 사후범위확대없음.
- 합성19검사/독립3522검산/별도과학검토216행동일. 원관측189클립×조건보존,추정관측승격0,후보126역사핀불변,보호32불변. 새학습·모델추론·다운로드·사용자촬영접근0. 실행중없음.
- 공식FreiHAND/InterHand/HO-3D주석 조사완료. 손전용자료에몸좌표임의생성금지. 실제외부파일미확보,외부자료학습효과미측정. 근거 `RESULT_KO.html`, `FINAL_RESULT.json`, `INDEPENDENT_CHECK.json`, `SCIENTIFIC_REVIEW.json`.
'''
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
h9='## 9. 다음 단일 작업'
assert h8 in context and h9 in context
context=context.replace(h8,h8+summary,1)
context=context.replace(h9,h9+'\n\n### 2026-10-25 손목 복원 진단 이후 정밀 평가 근거\n\n'+next_action+'\n',1)
contextp.write_text(context,encoding='utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {now}

현재 목표: 실제 손을 다시 찾거나 추정한 결과를 원영상과 독립 비교해 신뢰할 수 있는 보완 방향 결정.
마지막 완료: 실제 복원 구간24장면·9표현·6명·21영상, 두 AI 눈가림 손목 판독→저장9조건216행 대조→독립검산 완료.
전원 기존TRAIN01–06이고 과거DEV07/TEST08–09는 이번진단에서제외. 새사용자평가아님. 예측을보기전에원영상범위동결.
손목비교가능23, R06흐림판독불일치1보류. 손끝보임개수12장면불일치→전문가21점GT아님, 손가락교정정확도주장금지.
재검출A8: B1양쪽범위내6/밖1/보류1. 실제시간추정B8: 선형5/8,B2각5/8,B3각4/8·3/8·3/8.
전체24양쪽범위내 B0=3/B1=12/선형17/B2=17·17·17/B3=16·15·15. 전체는기존관측보존포함이라B8을별도해석.
B2/B3우월성미입증·미채택, RGB접근전체불가능을뜻하지않음. 기존C·T3-F/운영웹DTW그대로.
재검출R02/R13범위밖은시간보완기가기존관측을보존하므로고치지못함. R16선형지원→B3밖/애매. 상자거리=정확한관절오차아님.
합성19검사·독립3522검산 및별도216행해석검산완료. 보호32파일/후보126역사핀유지,원관측189클립×조건보존,추정승격0.
이번 새학습0/모델추론0/다운로드0/사용자·멘토자료접근0. 진행중프로세스·학습잠금없음.

다음 한 작업: {next_action}

외부손주석조사: FreiHAND공식평가주석포함724MB안내. 실제ZIP내XYZ/K/scale여부미검증; 미확보. InterHand는양손시간에더적합하나대용량·분할주의. HO-3D물체가림후순위. 상세EXTERNAL_LABEL_OPTIONS.
큰문제외자율진행 원칙유지. 원본/W:/sources/과거동결결과읽기전용. Git변경·외부전송·운영교체금지. AI손목범위를학습정답으로전환하거나기존TEST결과로재튜닝하지않음.
담당: root판독A/보고, bundle_inventory판독B/독립검산, bundle_split선정/평가, refiner_review공식자료/과학검토. Claude미호출.
근거: {U/'RESULT_KO.html'}
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
class Links(HTMLParser):
 def __init__(self): super().__init__(); self.refs=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ('href','src') and v: self.refs.append(v)
l=Links(); l.feed((U/'RESULT_KO.html').read_text(encoding='utf-8')); local=[unquote(urlparse(x).path) for x in l.refs if not urlparse(x).scheme and not x.startswith('#')]
missing=[x for x in local if not (U/x).is_file()]; assert not missing,missing
assert read(statep)['current_stage']=='RECOVERY_IMAGE_BENCHMARK_COMPLETED_NO_NEW_MODEL_ADOPTED'
assert tag in contextp.read_text(encoding='utf-8')
for f in U.glob('*.json'): read(f)
write(U/'DELIVERY_CHECK.json',dict(status='PASS',at_kst=now,local_links_checked=len(local),missing_links=missing,root_documents_synchronized=True,blind_reviews_unchanged=True,protected_files=32,report_sha256=sha(U/'RESULT_KO.html'),final_result_sha256=sha(U/'FINAL_RESULT.json')))
print(json.dumps({'status':'COMPLETE','links':len(local),'unit':str(U)},ensure_ascii=False))
