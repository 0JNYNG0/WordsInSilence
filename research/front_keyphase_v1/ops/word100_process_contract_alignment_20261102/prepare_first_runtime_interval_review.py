"""Fixed first runtime interval review sheets; no inference or score-based choice."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import av,numpy as np,json,hashlib,traceback,textwrap
from PIL import Image,ImageDraw,ImageFont

U=Path(__file__).resolve().parent;AB=U.parent/'word100_action_completion_supervision_20261102'
OUT=Path('W:/WordsInSilence_derived/word100_process_contract_alignment_20261102/first_runtime_interval_sheets')
CONTEXT=4;PER_PAGE=24;COLS=4;CW=440;IH=248;CH=286;HEADER=94
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def make_records():
 scored=read(AB/'PROCESS_AUTO_EVALUATION.json')
 meta={r['query_id']:r for r in read(AB/'PROCESS_AUTO_NATIVE_MANIFEST.json')['rows'] if r['role']=='fit'}
 bindings={r['query_id']:r for r in read(AB/'PROCESS_RUNTIME_SUPERVISION_BINDING.json')['rows'] if r['role']=='fit'}
 originals={r['id']:r for r in read(AB/'PROCESS_TARGETS.json')['rows']}
 rows=[];axes={}
 for r in scored['automatic_rows']:
  if r['role']!='fit':continue
  m=meta[r['query_id']];assert r['person'] in range(1,10) and m['person']==r['person']
  assert r['clip_id']==f"NIA_SL_{m['word_id']}_REAL{r['person']:02d}_F"
  assert sha(m['input'])==m['input_sha256']
  if m['input'] not in axes:
   with np.load(m['input'],allow_pickle=False) as z:axes[m['input']]={k:z[k].copy() for k in ['frame_id','timestamp_us','image_available']}
  a=axes[m['input']];ii=m['proposal']['output_indices']
  assert a['frame_id'].tolist()==m['frame_ids'] and a['timestamp_us'].tolist()==m['timestamp_us']
  assert r['source_frame_ids']==a['frame_id'][ii].tolist()
  assert r['source_timestamps_us']==a['timestamp_us'][ii].tolist()
  candidates=r['qualification']['interval_candidates']
  # Geometry only; motion/process/setup/terminal probability values are not read.
  order=[(int(c['end']),int(c['start'])) for c in candidates]
  assert order==sorted(order)
  first=candidates[0] if candidates else None
  item=dict(review_id=r['query_id']+'__first_runtime_interval',query_id=r['query_id'],
   case_id=r['query_id']+'__runtime_proposal',clip_id=m['clip_id'],word_id=m['word_id'],person=m['person'],role='fit',
   physical_query=m['physical_query'],source=m['source'],source_sha256=m['source_sha256'],
   input=m['input'],input_sha256=m['input_sha256'],source_resolution=m['actual_sensor_extent'][2:],
   fixed_proposal_indices=ii,first_candidate_ordinal=0 if first else None,selection_rule='First chronological saved candidate, never maximum score.',
   nominal_output_indices=None,nominal_full_native_indices=None,nominal_source_frames=None,nominal_source_timestamps_us=None,
   display_axis=[],review_pages=[],observed_process_evidence=None,review_status='pending_direct_nominal_span_review' if first else 'no_candidate_unknown',
   no_candidate_is_not_negative=True,model_scores_exposed=False,context_is_not_label_scope=True,
   whole_word_correctness=None,learner_error=None,required_repetitions=None)
  if first:
   local=first['output_indices'];assert local==list(range(first['start'],first['end']+1)) and len(local)>=2
   full=[ii[j] for j in local];assert full==list(range(full[0],full[-1]+1))
   item.update(nominal_output_indices=local,nominal_full_native_indices=full,
    nominal_source_frames=a['frame_id'][full].tolist(),nominal_source_timestamps_us=a['timestamp_us'][full].tolist())
   display=list(range(max(0,full[0]-CONTEXT),min(len(a['frame_id'])-1,full[-1]+CONTEXT)+1))
   item['display_axis']=[dict(display_index=k,full_native_index=j,source_frame=int(a['frame_id'][j]),
     source_timestamp_us=int(a['timestamp_us'][j]),image_available=bool(a['image_available'][j]),
     nominal=full[0]<=j<=full[-1],proposal_output_index=(ii.index(j) if j in ii else None),
     nominal_start=j==full[0],nominal_end=j==full[-1]) for k,j in enumerate(display)]
  # Keep one original explicit observed chain as a separate positive span.
  # Never copy the whole-proposal positive onto the shorter first candidate.
  b=bindings[m['query_id']];t=originals[m['query_id']]
  yes=[s for s in b['sequence_bindings'] if s['supports_observed_process_positive'] and s['original_annotation']['visible_chain']=='yes']
  item['existing_direct_positive_span_candidates']=[]
  if yes:
   link=yes[0];maps=[x for x in link['required_mapped_axis'] if x is not None]
   lo=min(x['output_index'] for x in maps);hi=max(x['output_index'] for x in maps)
   pidx=list(range(lo,hi+1));fidx=[ii[j] for j in pidx];native=a['frame_id'][fidx].tolist()
   seen=t.get('reviewed_native_frames')
   if seen is None:seen=sorted({f for page in t['reviewed_pages'] for f in page['frames']})
   assert set(native)<=set(seen) and a['image_available'][fidx].all()
   item['existing_direct_positive_span_candidates']=[dict(case_id=item['case_id'],output_indices=pidx,label=1,
     source_frames=native,source_timestamps_us=a['timestamp_us'][fidx].tolist(),
     original_sequence_index=link['source_sequence_index'],original_sequence=link['original_annotation'],
     source_annotation=t['source_annotation'],source_annotation_sha256=t['source_annotation_sha256'],
     runtime_binding_sha256=sha(AB/'PROCESS_RUNTIME_SUPERVISION_BINDING.json'),
     label_provenance='Existing direct visible-chain yes, exact original source pixels and continuous range retained. Not a copy of whole-view1; not a new direct review.',
     whole_word_correctness=None,learner_error=None,required_repetitions=None)]
  rows.append(item)
 assert len(rows)==45 and len({r['query_id'] for r in rows})==45 and len({r['clip_id'] for r in rows})==36
 return rows

def main():
 dest=U/'FIRST_RUNTIME_INTERVAL_REVIEW_MANIFEST.json'
 assert not dest.exists() and not OUT.exists(),'Do not overwrite completed or partial sheets; inspect first failure.'
 sources=[Path(__file__),U/'SCOPE.json',U/'SOURCE_EVIDENCE_REUSE_CONTRACT.json',
  AB/'PROCESS_LABEL_PROTOCOL.json',AB/'PHASE_LABEL_ADDENDUM.json',AB/'PROCESS_AUTO_PROPOSAL_PLAN.json',
  AB/'PROCESS_AUTO_EVALUATION.json',AB/'PROCESS_AUTO_NATIVE_MANIFEST.json',AB/'PROCESS_RUNTIME_SUPERVISION_BINDING.json',AB/'PROCESS_TARGETS.json']
 pins={str(p.resolve()):sha(p) for p in sources}
 rows=make_records()
 write(U/'FIRST_RUNTIME_INTERVAL_REVIEW_INTENT.json',dict(at=now(),pins=pins,rows=rows,
  choice='Exactly chronological candidate0 per FIT source-query; score values not exported.',
  context_margin_frames=CONTEXT,nominal_label_scope='Only exact nominal frames; context may clarify a visible state but cannot supply missing movement.',
  no_candidate_label=None,new_inference=0,new_training=0,existing_artifacts_changed=False))
 OUT.mkdir(parents=True);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
 totalpages=0
 for clip in sorted({r['clip_id'] for r in rows}):
  group=[r for r in rows if r['clip_id']==clip];m=group[0]
  assert sha(m['source'])==m['source_sha256']
  wanted={x['source_frame'] for r in group for x in r['display_axis']}
  frames={};times={};origin=None
  with av.open(m['source']) as c:
   stream=c.streams.video[0]
   assert [stream.codec_context.width,stream.codec_context.height]==m['source_resolution']
   for i,f in enumerate(c.decode(video=0)):
    assert f.pts is not None
    if origin is None:origin=f.pts
    if i in wanted:
     rgb=f.to_ndarray(format='rgb24');assert [rgb.shape[1],rgb.shape[0]]==m['source_resolution']
     frames[i]=rgb;times[i]=round(float((f.pts-origin)*stream.time_base)*1e6)
  assert set(frames)==wanted
  for r in group:
   for e in r['display_axis']:
    f=e['source_frame'];assert times[f]==e['source_timestamp_us']
    sample=frames[f][::16,::16].astype(float);available=bool(sample.std()>1 or sample.max()-sample.min()>4)
    assert available==e['image_available']
    e['decoded_rgb_sha256']=hashlib.sha256(np.ascontiguousarray(frames[f]).tobytes()).hexdigest()
   for pno,start in enumerate(range(0,len(r['display_axis']),PER_PAGE),1):
    page=r['display_axis'][start:start+PER_PAGE];height=HEADER+CH*((len(page)+COLS-1)//COLS)
    sheet=Image.new('RGB',(CW*COLS,height),(245,247,250));draw=ImageDraw.Draw(sheet)
    lo,hi=r['nominal_source_frames'][0],r['nominal_source_frames'][-1]
    draw.text((10,8),r['clip_id']+' | '+r['physical_query'],font=font,fill=(15,23,42))
    draw.text((10,34),f'JUDGE BLUE NOMINAL f{lo}..f{hi} ONLY | gray CONTEXT is reference, not missing process evidence | page {pno}',font=font,fill=(15,23,42))
    draw.text((10,61),'Full original frame; native f/time shown. NOMINAL START / END mark the actual target window.',font=small,fill=(51,65,85))
    for cell,e in enumerate(page):
     x=(cell%COLS)*CW;y=HEADER+(cell//COLS)*CH;nom=e['nominal']
     color=(36,99,235) if nom else (107,114,128)
     draw.rectangle((x+2,y+2,x+CW-3,y+CH-3),outline=color,width=3)
     tag='NOMINAL' if nom else 'CONTEXT ONLY'
     if e['nominal_start']:tag+=' START'
     if e['nominal_end']:tag+=' END'
     draw.text((x+8,y+7),f'{tag} | f{e["source_frame"]} | {e["source_timestamp_us"]/1e6:.3f}s',font=small,fill=color)
     image=Image.fromarray(frames[e['source_frame']]);image.thumbnail((CW-12,IH),Image.Resampling.LANCZOS)
     sheet.paste(image,(x+(CW-image.width)//2,y+32))
    name=r['review_id']+f'_p{pno}.jpg';path=OUT/name
    sheet.save(path,quality=94,subsampling=0)
    r['review_pages'].append(dict(path=str(path),sha256=sha(path),display_indices=[e['display_index'] for e in page],
       source_frames=[e['source_frame'] for e in page],nominal_source_frames=[e['source_frame'] for e in page if e['nominal']],
       nominal_output_indices=[e['proposal_output_index'] for e in page if e['nominal']]))
    totalpages+=1
  assert sha(m['source'])==m['source_sha256']
 for p,h in pins.items():assert sha(p)==h,p
 counts=dict(queries=45,source_clips=36,selected_nominal_spans=sum(bool(r['nominal_source_frames']) for r in rows),
  no_candidate_unknown=sum(r['nominal_source_frames'] is None for r in rows),pages=totalpages,
  nominal_frame_slots=sum(len(r['nominal_source_frames'] or []) for r in rows),
  context_frame_slots=sum(sum(not x['nominal'] for x in r['display_axis']) for r in rows),
  explicit_old_positive_span_links=sum(len(r['existing_direct_positive_span_candidates']) for r in rows))
 write(dest,dict(status='FIXED_FIRST_NOMINAL_INTERVALS_PREPARED_PENDING_DIRECT_REVIEW',at=now(),rows=rows,counts=counts,
  pins=pins,intent_sha256=sha(U/'FIRST_RUNTIME_INTERVAL_REVIEW_INTENT.json'),context_margin_frames=CONTEXT,
  score_fields_used_for_choice=False,score_fields_shown=False,labels_not_generated_from_model=True,
  prior_results_known_not_blinded_human_review=True,independent_human_expert=False,
  whole_word_correctness=None,learner_error=None,required_repetitions=None,new_inference=0,new_training=0,
  existing_AB_files_changed=False,root_documents_changed=False,whole_goal_complete=False))
 print(json.dumps(counts))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'FIRST_RUNTIME_INTERVAL_PREPARATION_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise

