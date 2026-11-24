"""Serialize root's actual review of the 27 new views, not automatic cut labels."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Recorded after all retained frames of all27 views were visually examined.
NOTES={
 1:([46,48,49],[60,64,76], '37~45에서 손을 올리고46~49에서 왼쪽 위팔 부근에 놓는다. 60~76에서는 이미 손목 부근인 관계가 유지된다.'),
 2:([46,48,49],[62,66,73], '33~42 대기·손올림 뒤43~49에서 위팔로 접근한다. 62~73은 손목 부근 유지이며 몸과 팔이 조금 내려가도 팔을 따라 다시 이동하는 과정은 없다.'),
 3:([50,51,52],[64,68,76], '38~47에서 손을 올리고48~52는 왼쪽 위팔로 향한다. 64~76에는 손목 부근 관계와 작은 자세 유지 변화만 남는다.'),
 4:([74,75,76],[89,93,100], '62~71의 준비 후72~76에서 반대쪽 위팔에 손이 놓인다. 89~100은 손목 부근 상태로 시작하며 작은 손·팔 변화가 새 위팔→손목 이동을 이루지는 않는다.'),
 5:([44,46,47],[60,64,72], '32~41의 손올림 후42~47은 왼쪽 위팔 부근 접근이다. 60~72는 원위 팔·손목 부근 유지로, 새로운 팔 따라 이동은 보이지 않는다.'),
 6:([40,41,42],[53,57,61], '30~39에서 손을 들어40~42 왼쪽 위팔에 놓는다. 53~62는 이미 손목 부근이며63~66에서는 양손이 떨어지며 내려가는 마무리가 보인다. 마무리 이동을 위팔→손목 이동으로 세지 않았다.'),
 7:([44,46,47],[62,66,74], '35~43 손올림·상대 팔로 접근,44~47 위팔 부근이다. 62~74는 손목 부근 유지다. 준비에서 팔을 향해 가는 것과 팔을 따라 손목으로 내려가는 것은 다르다.'),
 8:([70,71,72],[88,93,101], '60~69의 접근 후70~72는 위팔/어깨 옆에 손이 있다. 88~101은 손목 부근에 머물며 몸통이 조금 움직인다. 몸 흔들림은 별도 수어 오류로 표시하지 않는다.'),
 9:([44,46,47],[60,64,68], '38~43 손을 들고44~47 위팔 부근을 보인다. 60~68에서는 이미 손목 옆이며69~71의 작은 마무리 변화까지 보아도 위팔에서 손목으로 연결된 새 이동은 없다.')
}
def main():
    dest=U/'REVIEW_PARTIAL_1133.json';assert not dest.exists()
    man=read(U/'PARTIAL_VIEWS_MANIFEST.json');rows=[];pages={}
    for v in man['views']:
        if not v['query_id'].startswith('NIA_SL_WORD1133') or v['construction_kind_audit_only'] not in ['prefix','suffix','middle_splice']:continue
        person=v['recorded_person_id'];setup,terminal,note=NOTES[person]
        ids={m['source_native_frame']:m['output_index'] for m in v['ordered_frame_map']}
        witness=lambda fs:[dict(source_frame=f,output_index=ids[f]) for f in fs if f in ids]
        for p in v['review_pages']:
            assert sha(p['path'])==p['sha256'];pages[p['path']]=p['sha256']
        kind=v['construction_kind_audit_only']
        reason={'prefix':'보이는 것은 손올림과 반대 위팔 부근 접근이다. 그 위치에서 팔을 따라 손목 부근까지 이어지는 이동은 이 보존 조각에 없다.',
                'suffix':'보존된 화면이 이미 손목 부근에서 시작한다. 유지 또는 마무리만으로 출발 관계에서부터의 이동 과정을 관측했다고 볼 수 없다.',
                'middle_splice':'위팔 부근 관계와 손목 부근 관계는 각각 보이지만, 연결점 사이 원래 프레임은 화면에 없다. 두 조각 내부에도 별도의 온전한 위팔→손목 이동은 보이지 않는다.'}[kind]
        remaining=[]
        for lo,hi in v['retained_contiguous_native_fragments']:
            remaining.append(dict(native_interval=[lo,hi],output_interval=[ids[lo],ids[hi]],
                observed_relation='proximal arm approach' if any(lo<=f<=hi for f in setup) else 'distal forearm/wrist and possible finish',
                reviewed_complete_query_process=0,note=note))
        rows.append(dict(view_id=v['view_id'],query_id=v['query_id'],physical_query=v['physical_query'],pixel_view_key=v['pixel_view_key'],
            reviewed_pages=v['review_pages'],reviewed_output_indices=list(range(len(v['ordered_frame_map']))),
            observed_process_evidence=0,visible_chain='partial',evidence_reason=reason+' '+note,
            remaining_evidence_by_fragment=remaining,setup_witnesses=witness(setup),motion_witnesses=[],terminal_witnesses=witness(terminal),
            seam_process_label=None,seam_output_indices=v['seam_output_indices'],whole_word_correctness=None,learner_error=None,required_repetitions=None,
            note='Zero means this shown material does not contain the directly supported linked physical process, not a judgment about the person or deleted images.'))
    assert len(rows)==27 and len(pages)==35
    out=dict(at=datetime.now(timezone.utc).isoformat(),status='DIRECT_REVIEW_OF_RETAINED_PHYSICAL_PROCESS',rows=rows,
        manifest_sha256=sha(U/'PARTIAL_VIEWS_MANIFEST.json'),reviewer='Root Codex tools.view_image; same system, no independent KSL expert',
        method='All35 pages visually inspected. ID01 splice page1 was already seen directly in the preceding render check; unchanged hash and current visible context reused. All other pages opened in this goal turn.',
        page_count=35,view_count=27,registered_ids=list(range(1,10)),observed_process_counts={'0':27},
        not_an_automatic_edit_type_label=True,source_videos_changed=False,new_inference=0,new_training=0,whole_goal_complete=False)
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Recorded27 explicitly reviewed process-observation labels,35 pages')
if __name__=='__main__':main()
