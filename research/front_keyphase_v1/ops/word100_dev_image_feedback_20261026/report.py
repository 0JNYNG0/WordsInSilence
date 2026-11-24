import prepare as R
from pathlib import Path
import json,html,numpy as np
from collections import defaultdict,Counter
Z=R.O/'word100_core_image_evidence_20261026'
def clean(x):
 if isinstance(x,list):return [clean(y) for y in x]
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def main():
 f=R.read(R.U/'FROZEN.json');v=R.read(R.U/'INTEGRATION_VERIFIED.json');fv=R.read(R.U/'FEEDBACK_VERIFIED.json');p=R.read(R.U/'PREPARED.json');res=R.read(R.U/'PAIRS_DONE.json');im=R.read(R.U/'VERIFIED_IMAGES.json');assert fv['queries']==300;meta={r['clip_id']:r for r in p['rows']};samples={cid:R.load(r['output']) for cid,r in meta.items()};bm=R.read(R.B/'PREP_CONTEXT.json');labels={r['label_index']:r['canonical_label'] for r in bm['ordered_rows'] if r['role']=='development_only'};data={};sources=Counter()
 for row in res['rows']:
  z=R.load(row['pair']);q=samples[row['query']];r=samples[row['reference']];core=np.flatnonzero(r['core']);a=np.floor(z['phase_old']).astype(int);b=np.ceil(z['phase_old']).astype(int);n=len(r['core']);qs=np.where(q['state_after'][a]==q['state_after'][b],q['state_after'][a],6);qs=np.concatenate([qs,np.zeros((n,2,8),np.uint8)],-1);rs=np.concatenate([r['state_after'],np.zeros((n,2,8),np.uint8)],-1)
  item=dict(reference=row['reference'],reference_id=meta[row['reference']]['source']['signer_id'],time=np.round(r['timestamp_us'][core]/1e6,3).tolist(),query_time=np.round(np.interp(z['phase_old'][core],np.arange(len(q['core'])),q['timestamp_us']/1e6),3).tolist(),difference=clean(np.round(z['point_delta'][core],4).tolist()),raw=z['point_ok'][core].astype(int).tolist(),before=z['point_corroborated_before'][core].astype(int).tolist(),after=z['point_corroborated_after'][core].astype(int).tolist(),query_state=qs[core].tolist(),reference_state=rs[core].tolist(),yolo_audit=np.zeros((len(core),2,15),np.uint8).tolist(),counts=row['counts'])
  if row['query'] not in data:
   preds={k:dict(**vv['predictions'][row['query']],predicted_label=labels[vv['predictions'][row['query']]['prediction']],top5_labels=[labels[i] for i in vv['predictions'][row['query']]['top5']]) for k,vv in f['scores'].items()};data[row['query']]=dict(word=row['word_code'],label=row['label'],query=row['query'],query_id=meta[row['query']]['source']['signer_id'],predictions=preds,references=[])
  data[row['query']]['references'].append(item)
  rr=R.read(R.U/'receipts'/Path(row['baseline_receipt']).name)
  for hand in rr['source_separated_point_evidence']:
   for angle in hand['angles']:
    for e in angle['states']:sources[e['code']]+=e['cells']
 data=sorted(data.values(),key=lambda r:(r['word'],r['query_id']));assert len(data)==300 and all(len(x['references'])==2 for x in data);cc=res['counts'];rate=lambda a,b:round(a/b*100,2) if b else None
 recognition={k:dict(correct=sum(v['correct'] for v in s['predictions'].values()),top5=sum(v['top5_correct'] for v in s['predictions'].values()),total=300,source=s['source'],sha256=s['sha256']) for k,s in f['scores'].items()}
 result=dict(at=R.now(),status='VERIFIED_PUBLIC_DEV_RESEARCH_INTEGRATION',scope=dict(words=100,videos=300,recorded_ids=[11,12,13],pairs=600,development_only=True),counts=cc,source_counts=dict(sources),image_counts=im['counts'],latency=im['latency'],recognition=recognition,recognition_inference=0,training=0,operating_changed=False,coverage=dict(point_before=rate(cc['point_before'],cc['raw_point_angles']),point_after=rate(cc['point_after'],cc['raw_point_angles']),sequence_before=rate(cc['sequence_before'],cc['raw_sequence_angles']),sequence_after=rate(cc['sequence_after'],cc['raw_sequence_angles'])),verifiers=[R.sha(R.U/'INTEGRATION_VERIFIED.json'),R.sha(R.U/'FEEDBACK_VERIFIED.json'),R.sha(R.U/'VERIFIED_IMAGES.json')],normal_tolerance=None,semantic_accuracy=None)
 R.write(R.U/'SUMMARY.json',result);R.write(R.U/'VIEW_DATA.json',data)
 text=f'''공개 개발자료에서 인식과 관측근거 피드백 연결

공개100단어, 기록ID11·12·13의300영상과600고정비교쌍을 사용했다. 각 영상은 다른 두 기록ID의 같은단어 참고와 각각 비교했다. 이미 개발과정에 사용한 자료로, 새로운 사람에 대한 최종 평가가 아니다.

원래 좌표·좌우배정·정규화·시간대응·차이값·분모를 모두 보존했다. C와T3-F의 저장된 원본 인식 결과를 정확한영상ID로 연결했다. 이번에는 학습이나 인식모델 재추론을 하지 않았다. 인식 결과 C {recognition['C']['correct']}/300, T3-F {recognition['T3F']['correct']}/300은 이전 저장결과이며 이번 보완으로 올라간 인식률이 아니다.

이번에 바뀐 점은 손가락별 현재 이미지 재확인 근거와 피드백 설명이다. 원본에서 비교 가능한 순간각도 {cc['raw_point_angles']:,}칸 중 이미지근거 {cc['point_before']:,}→{cc['point_after']:,}({result['coverage']['point_before']}→{result['coverage']['point_after']}%). 연속 비교 가능한 {cc['raw_sequence_angles']:,}칸 중 {cc['sequence_before']:,}→{cc['sequence_after']:,}({result['coverage']['sequence_before']}→{result['coverage']['sequence_after']}%). 비교칸은 한 시점의 한 손가락각도이며 영상수·정답수·관절정확도가 아니다.

이전 근거 철회 {cc['point_withdrawn']} / {cc['sequence_withdrawn']}. 이전 이미지근거가0인 것은 기존DTW비교를 하지 않았다는 뜻이 아니라, 이번에 요구하는 전체화면·확대·원위치 재확인 기록이 없었다는 뜻이다. 기존 몸·손의 연속 근거 {cc['legacy_sequence_quality']:,}칸도 원래 정의로 별도 보존했다. 새로운 이미지근거가 없다는 이유로 이미 계산한 차이나 팔의 관측을 지우지 않았다.

피드백은 촬영 영상의 좌표 부족, 참고 영상의 좌표 부족, 재확인 미실시, 관측결과 불확실, 양쪽 현재이미지 근거를 구분한다. 보간 양끝과 참고 영상의 원인을 각각 기록해 혼합 원인도 숨기지 않는다. 참고자료가 부족한 것을 사용자 잘못으로 설명하지 않고, 보이지 않는 각도를0차이나 성공으로 기록하지 않는다.

추가검출은 전체프레임 {im['counts']['new_full']:,}회·확대 {im['counts']['new_crop']:,}회이며 저장전체결과 {im['counts']['cached_full']:,}회를 재사용했다. 현재영상 전체와 손 주변 확대는 동일MediaPipe 기반이므로 두 결과의 일치가 독립정답은 아니다. YOLO의 기존 제안은 원본 연구기록에 보존되며 이번에는 새YOLO호출이나 정밀각도 근거 채택을 하지 않았다. 추가전체검출1회평균 {im['latency']['full']['mean_ms']:.1f}ms, 손확대1회평균 {im['latency']['crop']['mean_ms']:.1f}ms였다. 이는기본검출을제외한추가비용이다. 이 시험은 오프라인이며 웹캠실시간FPS를 뜻하지 않는다.

검증: 300입력/600경로·원본수치 완전일치, 새 이미지출력 별도수식 검산, 한각도/한손 근거숨김통제, 600피드백의 출처별시간구간 재구성,300개 저장인식결과 연결을 검사했다. 같은Codex의 별도산술검산이며 독립사람의21점정답 또는 수어전문가판정은 아니다.

남은 문제: 가려진 손가락의 실제3D모양, 정상수어 허용범위, 구체적교정방향은 아직 검증되지 않았다. 같은모델이 같은오류를 공유할 수 있다. 원본 보존으로 새좌표변형은 없지만 원래의검출오류까지 해결했다는뜻은아니다. 기존 C·T3-F·E3/운영웹·DTW·원본보존,사용자·멘토·추가199/200접근0. 전체목표계속.
'''
 (R.U/'RESULT_KO.txt').write_text(text,encoding='utf8')
 # Reuse the locally checked two-reference explorer. Adapt its declared scope and saved classification banner.
 old=(Z/'RESULT_KO.html').read_text('utf8');start=old.index('<script>');script=old[start:];a=script.index('const DATA=');b=script.index(';\nconst labels=',a);script=script[:a]+'const DATA='+json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</',r'<\/')+script[b:]
 script=script.replace("q.label+' · '+q.word,i","q.label+' · ID '+q.query_id,i")
 oldline="$('query').textContent='비교 영상: '+q.query+' (기록ID '+q.query_id+')';"
 newline="$('query').textContent='비교 영상: '+q.query+' | 기존 C: '+q.predictions.C.predicted_label+' | 기존 T3-F: '+q.predictions.T3F.predicted_label+' (저장 결과, 새 인식 아님)';"
 assert oldline in script;script=script.replace(oldline,newline)
 css=old[old.index('<style>'):old.index('</style>')+8]
 head='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공개 개발자료 인식·피드백 연구 연결</title>'+css
 body=f'''<main><header><div class="small">운영 모델 보존 · 공개 DEV 연구용</div><h1>어떤 단어인지, 어디까지 확인했는지</h1><p>기존 인식 결과와 참고별 동작 차이를 연결했습니다. 관측이 불확실한 부분은 따로 표시합니다.</p><div class="stats"><div class="stat"><strong>300영상 · 600쌍</strong>100단어, 기록ID11·12·13</div><div class="stat"><strong>{result['coverage']['point_after']}%</strong>계산 가능 각도 중 이미지 재확인 근거<br>정답률·관절정확도가 아닙니다</div><div class="stat"><strong>기존 모델 유지</strong>새학습·인식재추론·운영변경0</div></div></header><section><h2>두 참고를 각각 살펴보기</h2><div class="controls"><select id="word"></select><select id="side"><option value="0">촬영자의 왼손</option><option value="1">촬영자의 오른손</option></select><select id="channel"></select></div><p id="query" class="small"></p><p class="small">파란선: 관측 좌표 차이 · 초록표시: 두 영상에 MP현재이미지 재확인근거 · 회색: 좌표비교 불가. 작은차이는 올바른수어판정이 아닙니다. 위치는 어깨너비 기준, 각도는 화면상의 2차원 각도입니다.</p><div id="cards"></div></section><section><h2>사용자에게 어떻게 설명할까?</h2><p><b>수행 영상에서 손이 안 보임:</b> “이 시간대는 손 모양을 확인하기 어려워요.”</p><p><b>참고 영상에서 안 보임:</b> “참고 자료의 근거가 부족해요.” 사용자 동작을 수정하라고 요구하지 않습니다.</p><p><b>관측이 서로 다름:</b> 차이값과 불확실한 근거를 함께 제시합니다. 손을 잘못 움직였다고 단정하지 않습니다.</p><p><b>두 영상 모두 현재 이미지 근거가 있음:</b> 관측 차이를 설명할 근거가 늘어난 것입니다. 정확한 교정각도나 전체수어정답 판정은 아직 검증하지 않았습니다.</p></section><section><h2>검증 결과와 남은 문제</h2><pre>{html.escape(text)}</pre></section></main>'''
 (R.U/'RESULT_KO.html').write_text(head+body+script,encoding='utf8');R.write(R.U/'REPORT_CHECK.json',dict(at=R.now(),html_sha256=R.sha(R.U/'RESULT_KO.html'),data_sha256=R.sha(R.U/'VIEW_DATA.json'),summary_sha256=R.sha(R.U/'SUMMARY.json'),label_map_sha256=R.sha(R.B/'PREP_CONTEXT.json'),queries=300,pairs=600,browser_rendered=False));print(text,flush=True)
if __name__=='__main__':main()
