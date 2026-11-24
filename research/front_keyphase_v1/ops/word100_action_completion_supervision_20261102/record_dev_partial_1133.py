"""Serialize root's completed direct review of all 13 DEV1133 views/19 pages.

This records visual findings already made, not an automatic edit-type labeler.
No model output was consulted. Existing source reviews are read-only.
"""
from pathlib import Path
from collections import Counter
import json, hashlib
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SPEC={11:([49,51,52],[65,69,73]),12:([63,64,65],[77,79,82]),13:([50,52,54],[73,77,81])}
FINDINGS={
 'prefix':(0,'partial','남은 화면에는 손올림과 왼쪽 위팔 부근 접근이 보인다. 그 위치에서 팔을 따라 손목까지 이어지는 표적 이동은 보이지 않는다.'),
 'suffix':(0,'partial','손목 부근 관계에서 시작해 유지·마무리한다. 작은 몸 흔들림과 팔내림은 남아 있지만 위팔에서 손목까지 이동한 과정의 증거는 아니다.'),
 'middle_splice':(0,'partial','위팔 부근과 손목 부근 모습은 각각 남았지만 그 사이 이동 화면이 빠졌다. 각 남은 조각 안에서도 별도의 온전한 위팔→손목 이동은 보이지 않는다.'),
 'speed_0_8':(1,'yes','시간 간격만 달라진 화면에서 위팔 부근의 오른손이 왼팔을 따라 손목까지 이동하는 연결 과정이 보인다.'),
 'speed_1_25':(1,'yes','시간 간격만 달라진 화면에서 위팔 부근의 오른손이 왼팔을 따라 손목까지 이동하는 연결 과정이 보인다.'),
 'intact_context_trim':(1,'yes','앞뒤 여백이 줄었지만 위팔 부근 출발, 팔을 따라 이동, 손목 부근 도착이 모두 연속된 화면에 남았다.'),
 'missing_rgb':(None,'uncertain','출발 관계와 이후 손목 관계는 보이지만 이동에 해당하는 원프레임52~65가 회색이다. 그 사이 과정을 관측했다고도, 실제로 수행하지 않았다고도 판단하지 않는다.')}

def main():
 dest=U/'REVIEW_DEV_PARTIAL_1133.json';assert not dest.exists()
 man=read(U/'DEV_PARTIAL_VIEWS_MANIFEST.json');rows=[];pages={}
 for v in man['views']:
  if not v['query_id'].startswith('NIA_SL_WORD1133') or v['construction_kind_audit_only']=='native':continue
  kind=v['construction_kind_audit_only'];label,chain,reason=FINDINGS[kind]
  fm=v['ordered_frame_map'];by={f['source_native_frame']:f for f in fm}
  def witness(fs):
   return [dict(source_frame=f,output_index=by[f]['output_index'],source_timestamp_us=by[f]['source_native_timestamp_us'],view_timestamp_us=by[f]['view_timestamp_us']) for f in fs if f in by and by[f]['image_available']]
  for p in v['review_pages']:
   assert sha(p['path'])==p['sha256'];pages[p['path']]=p['sha256']
  for f in fm:assert sha(f['source_frame_path'])==f['source_frame_sha256']
  setup,terminal=SPEC[v['recorded_person_id']];motion=[]
  if label==1:
   fs=list(range(52,66));ws=witness(fs);assert len(ws)==14
   motion=[dict(source_frames=fs,output_indices=[w['output_index'] for w in ws],label=1,
      reason='The retained continuous images directly show along-arm travel, not just two endpoint poses.')]
  rows.append(dict(view_id=v['view_id'],query_id=v['query_id'],physical_query=v['physical_query'],pixel_view_key=v['pixel_view_key'],
   reviewed_pages=v['review_pages'],reviewed_output_indices=list(range(len(fm))),
   observed_process_evidence=label,visible_chain=chain,evidence_reason=reason,
   setup_witnesses=witness(setup),terminal_witnesses=witness(terminal),motion_witnesses=motion,
   remaining_evidence_by_fragment=[dict(native_interval=[lo,hi],output_interval=[by[lo]['output_index'],by[hi]['output_index']],
     reviewed_complete_query_process=label,note=reason) for lo,hi in v['retained_contiguous_native_fragments']],
   seam_output_indices=v['seam_output_indices'],seam_process_label=None,
   whole_word_correctness=None,learner_error=None,required_repetitions=None,
   review_method='Root direct tools.view_image review of every page, including both speed conditions; physical evidence, not independent KSL expert truth.'))
 assert len(rows)==13 and len(pages)==19,(len(rows),len(pages))
 counts=Counter(str(r['observed_process_evidence']) for r in rows);assert counts=={'0':9,'1':3,'None':1},counts
 out=dict(status='DIRECT_DEVELOPMENT_PARTIAL_PHYSICAL_REVIEW',rows=rows,manifest_sha256=sha(U/'DEV_PARTIAL_VIEWS_MANIFEST.json'),
  reviewer='Codex root; same system, not independent human/KSL expert',views=13,pages_actually_viewed=19,
  observed_process_counts=dict(counts),speed_pixel_reuse=False,
  limitation='Speed sheets show the same ordered source images with different timestamps; this is label review, not model speed invariance validation.',
  model_predictions_used=False,new_training=0,new_inference=0,whole_goal_complete=False)
 dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print('DEV1133 recorded 13 views, 19 actually viewed pages; 3 supported, 9 partial, 1 unknown')
if __name__=='__main__':main()
