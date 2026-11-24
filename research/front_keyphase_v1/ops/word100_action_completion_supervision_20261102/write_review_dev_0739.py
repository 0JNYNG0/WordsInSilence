"""Serialize direct review of six public development sheets, no predictions."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

A=Path(__file__).resolve().parent
M=A/'DEV_PROCESS_REVIEW_MANIFEST.json'
OUT=A/'REVIEW_DEV_0739.json'
reviewer='Codex /root/predicate_contract_review; direct RGB contact-sheet review, not independent human/KSL expert gold'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Notes selected after all native images on both pages for each clip were viewed.
notes={
 11:dict(face=[59,63,66,68], boundary=[71,73], torso=[75,78,81,83],
         stable=[[59,66],[78,81]], motion=[68,78], setup=[66,68], terminal=[78,81],
         reason='f59–68에서 오른손이 입·코 앞에 있고, f69–78의 인접 원본 그림에서는 턱과 목 높이를 지나 가슴 부근으로 내려오는 과정이 보인다. f78 이후 손목·주먹 방향과 팔꿈치가 바뀌는 다른 움직임도 있으므로 전체를 정지라고 부르지 않는다.'),
 12:dict(face=[43,49,55,60], boundary=[62,63], torso=[68,70,75,80],
         stable=[[45,56],[70,75]], motion=[60,70], setup=[55,60], terminal=[70,75],
         reason='f43–60에서 오른손이 얼굴 앞에 유지되다가 f61–70 사이에 얼굴 아래에서 어깨·가슴 부근으로 내려온다. f70–75에는 그 상체 관계가 유지된다. 전체화면의 위치가 아니라 그 수행자의 얼굴과 상체에 대한 관계를 보았다.'),
 13:dict(face=[68,72,76,78], boundary=[81,82], torso=[86,89,93,102],
         stable=[[68,76],[89,93]], motion=[78,89], setup=[76,78], terminal=[89,93],
         reason='머리를 조금 숙인 상태에서도 f68–78의 오른손은 얼굴 앞이다. f79–89에서 손이 턱 아래를 거쳐 가슴 부근으로 연속해서 내려오며, 이후에는 손목 방향과 팔꿈치가 다시 바뀐다. 다른 사람과 같은 화면 높이일 것을 요구하지 않았다.')
}

manifest=json.loads(M.read_text(encoding='utf-8'))
sources=[r for r in manifest['rows'] if r['word_id']=='WORD0739']
assert len(sources)==3 and {r['person'] for r in sources}=={11,12,13}
rows=[]
for s in sources:
    n=notes[s['person']]
    assert s['role']=='development' and s['physical_query']=='right_hand_descends_face_to_torso'
    fi={r['frame']:r for r in s['frames']}
    pages=[]
    for p in s['pages']:
        assert sha(p['path'])==p['sha256']
        pages.append(dict(p,actual_sha256=p['sha256'],viewed_with='tools.view_image',reviewer=reviewer))
    assert len(pages)==2
    viewed=[f for p in pages for f in p['frames']]
    assert viewed==list(range(s['display_bounds'][0],s['display_bounds'][1]+1))
    relations=[]
    for frames,a,b,reason in [
        (n['face'],1,0,'오른손이 입·코/얼굴 앞에 보이며 상체 부근 관계는 아니다. 정확한 얼굴 접촉 위치는 판정하지 않았다.'),
        (n['torso'],0,1,'오른손이 얼굴 영역 아래 가슴·어깨 부근에 보인다. 고정 화면 좌표나 정확한 높이 기준을 적용하지 않았다.'),
        (n['boundary'],None,None,'손이 턱·목·어깨 경계로 지나가는 시점이다. 넓은 얼굴/상체 관계를 단정하기 어려워 두 관계는 미확인으로 두며, 인접 시점의 실제 하강 근거는 별도로 유지한다.')
    ]:
        for f in frames:
            z=fi[f];assert f in viewed and sha(z['path'])==z['sha256']
            relations.append({'frame':f,'timestamp_us':z['timestamp_us'],'setup_relation':a,'terminal_relation':b,
                              'reason':reason,'source_image':{'path':z['path'],'sha256':z['sha256']}})
    relations.sort(key=lambda r:r['frame'])
    intervals=[]
    for (lo,hi),label in [(n['stable'][0],0),(n['motion'],1),(n['stable'][1],0)]:
        frames=list(range(lo,hi+1));assert set(frames)<=set(viewed)
        intervals.append({'start':lo,'end':hi,'label':label,
          'reason':('모든 인접 원본 그림에서 손이 얼굴 앞에서 상체 쪽으로 내려오는 순차 변화가 보인다. 구간의 진행 근거이며 내부 모든 프레임에 정답을 복제하지 않는다.' if label else
                    '인접 원본 그림에서 같은 넓은 얼굴/상체 관계를 유지하며, 질문한 얼굴→상체 하강은 이 짧은 구간에서 보이지 않는다. 몸 흔들림이나 손목 방향 변화까지 없다는 뜻은 아니다.'),
          'reviewed_native_frames':frames,'timestamps_us':[fi[lo]['timestamp_us'],fi[hi]['timestamp_us']]})
    r={k:s[k] for k in ['id','clip_id','word_id','person','role','physical_query','source','source_sha256']}
    r.update(reviewer=reviewer,reviewed_pages=pages,reviewed_full_size_frames=[],reviewed_native_frames=viewed,
      predicate_definitions={
       'setup_relation':'Right hand visibly at the face region, including mouth/nose/lower-face; no exact point contact required.',
       'terminal_relation':'Right hand visibly at upper torso/chest-shoulder region below the facial area. An ambiguous neck/shoulder border is unknown, not forced into either region.',
       'motion_interval':'Visible downward hand-to-body relational change from the face toward the upper torso; interval progress does not itself certify terminal arrival or full language execution.'},
      relations=relations,motion_intervals=intervals,
      sequence_links=[{'setup_frames':n['setup'],'motion_interval':n['motion'],'terminal_frames':n['terminal'],
        'visible_chain':'yes','reason':n['reason'],'linguistic_sufficiency':None,'whole_clause_truth':None}],
      review_bounds_sufficient=True,
      missing_context='None needed for this narrow visible face-to-upperbody chain. No exact onset/offset, required hold, repetition, or whole-guide correctness is judged.',
      unlisted_label='unknown',dense_labels_generated=False,exact_onset_offset=None,
      whole_word_correctness=None,required_repetition_count=None,learner_error=None,
      review_note=n['reason'],repeated_development=True,independent_final_test=False)
    rows.append(r)
result={'status':'DIRECT_PUBLIC_DEVELOPMENT_REVIEW_COMPLETE_NOT_EXPERT_GOLD',
  'created_at':datetime.now(timezone.utc).isoformat(),'reviewer':reviewer,
  'manifest':{'path':str(M),'sha256':sha(M)},'manifest_sha256':sha(M),
  'protocols':{str(A/k):sha(A/k) for k in ['PROCESS_LABEL_PROTOCOL.json','PHASE_LABEL_ADDENDUM.json']},
  'counts':{'queries':3,'source_clips':3,'pages':6,'reviewed_native_frames':sum(len(r['reviewed_native_frames']) for r in rows),
     'relation_points':sum(len(r['relations']) for r in rows),'motion_intervals':9,'visible_chains':3,'border_relation_points_unknown':6},
  'scope':'Same physical rubric as FIT; development IDs11–13 only, no prediction inspection, threshold edits, or training. Repeated development, not independent final evaluation.',
  'limitations':['Same Codex non-expert physical annotations; no independent KSL educator correctness gold.',
     'Sparse relation labels and net interval judgments are separate; intervals are not dense frame labels.',
     'Visible process support is not proof that every important sign action was completed or that the whole sign was correct.',
     'Neck/chin boundaries remain unknown; ordinary body-height and wrist-orientation differences do not create a signer error.'],
  'rows':rows,'model_predictions_used':False,'new_inference':0,'new_training':0,
  'private_accessed':False,'holdout14_15_accessed':False,'fit_labels_changed':False,'whole_goal_complete':False}
assert not OUT.exists()
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'counts':result['counts']},ensure_ascii=False))
