"""Record actual root visual inspection; preserve three prior direct reviews."""
from prepare_review import U,AG,read,sha,exclusive
from datetime import datetime,timezone
reasons={
2:'f61~72에서 오른손은 왼손목·아래팔 끝의 같은 부근에 얹혀 있다. 작은 동반 자세 정리는 보이지만 왼팔 길이를 따라 손목 쪽으로 쓸어가는 별도 이동은 보이지 않는다.',
3:'f64~74에서 오른손바닥과 왼손목 부근의 겹침 관계가 유지된다. 팔 길이를 따라 이동하는 쓸기 동작은 이 창 안에서 관찰되지 않는다.',
5:'f60~68에서 오른손은 왼손목 가까운 부위의 접촉·겹침을 유지한다. 양팔의 작은 정리와 별개로 오른손의 팔길이 방향 이동은 보이지 않는다.',
6:'f54~60에서 오른손이 왼손목 위 같은 부근에 머문다. 작은 공동 위치 변화가 있으나 오른손이 왼팔을 따라 따로 이동하는 모습은 보이지 않는다.',
8:'f89~93에서 오른손을 왼손목 위에 둔 배치가 거의 유지된다. 이 창 안에서는 팔을 따라 쓸어가는 변화가 보이지 않는다.',
9:'f59~68에서 오른손과 왼손목의 관계가 유지된다. 끝쪽 손가락·엄지 자세 변화가 조금 있으나 이는 이번 질문인 아래팔 길이 방향 쓸기 이동은 아니다.',
11:'f66~75에서 오른손은 왼손목 부근에 얹힌 상태를 유지한다. 뒤쪽에서 양손과 팔이 조금 함께 내려오지만 오른손이 왼팔 길이를 따라 따로 쓸어가는 이동은 보이지 않는다.',
12:'f79~84에서 오른손은 왼주먹 바로 위의 손목 부근에 머문다. 손과 팔이 미세하게 정리되는 동안 같은 상대 배치가 유지되고 별도 쓸기 이동은 보이지 않는다.',
13:'f71~82에서 양팔·손의 묶음이 조금 낮아지지만 오른손은 왼손목 가까운 부근에 계속 놓여 있다. 공동 하강과 구분되는 아래팔 길이 방향 쓸기 이동은 보이지 않는다.'}

def main():
    q=read(U/'WORD1133_REVIEW_QUEUE.json')['rows']
    old={r['clip_id']:r for r in read(AG/'LATE_VISUAL_REVIEW.json')['rows']}
    rows=[]
    for r in q:
        assert sha(r['sheet']['path'])==r['sheet']['sha256']
        reuse=r['reused_verified_review'];person=r['person']
        if reuse:
            a=old[r['clip_id']]
            assert a['source_sha256']==r['source_sha256'] and a['local_motion_presence']==0
            assert a['actually_viewed_native_frames']==r['native_frames']
            assert a['native_timestamps_us']==r['timestamp_us']
            reason=a['reason_ko'];reviewer=a['reviewer']
        else:
            reason=reasons[person];reviewer='/root; Codex visual review, not independent KSL expert'
        rows.append(dict(**{k:r[k] for k in ['clip_id','word_id','role','person','source','source_sha256','input','input_sha256','sheet','native_frames','timestamp_us']},
            queries=[dict(target_index=0,presence=0,reason_ko=reason,witness_frames=r['native_frames'],
                observability='Visible relative hand/forearm placement throughout fixed native window; no label outside window')],
            actually_viewed=True,all_native_frames_viewed=True,reviewer=reviewer,
            reused_verified_review=reuse,new_sheet_review_count=0 if reuse else 1,
            previous_review=dict(path=str(AG/'LATE_VISUAL_REVIEW.json'),sha256=sha(AG/'LATE_VISUAL_REVIEW.json')) if reuse else None,
            whole_clause_truth=None,educational_error=None,model_scores_read=False,old_stable_labels_copied=False))
    exclusive(U/'WORD1133_REVIEW.json',dict(status='DIRECT_FIXED_WINDOW_REVIEW_COMPLETE',rows=rows,
        at=datetime.now(timezone.utc).isoformat(),new_sheets_viewed=9,reused_prior_direct_reviews=3,
        new_native_frame_cells=sum(len(r['native_frames']) for r in rows if not r['reused_verified_review']),
        limit='Interval physical occurrence absence only; not whole-word error or exact pixel immobility. Same Codex, not an independent KSL expert.',
        extraction=0,model_inference=0,training=0))
    print('1133 direct reviews recorded: 9 new, 3 reused; no model execution')
if __name__=='__main__':main()
