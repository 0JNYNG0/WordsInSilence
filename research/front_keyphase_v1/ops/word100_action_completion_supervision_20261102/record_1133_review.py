"""Persist root's actual review of all18 native-frame WORD1133 FIT pages."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# These are manual visual judgments made after viewing every native-frame page,
# not values calculated from prior predictions, timestamps or motion endpoints.
NOTES={
1:dict(setup=[46,48,49],terminal=[60,64,70,76],before=[37,40],middle=54,motion=[49,58],other=[[37,39],[62,76]],reason='오른손이 왼쪽 위팔 부근에서 아래 전완을 따라 손목 쪽으로 내려온다. 도착 후 손목 부근 관계를 유지한다.'),
2:dict(setup=[46,48,49],terminal=[62,66,70,73],before=[33,36],middle=54,motion=[49,60],other=[[33,36],[64,73]],reason='왼팔도 올라오는 도입 뒤, 오른손이 위팔 쪽에서 전완을 따라 손목 부근으로 이동한다. 손목 부근에서 모양을 유지하는 뒤 구간과 구별된다.'),
3:dict(setup=[50,51,52],terminal=[64,68,72,76],before=[38,40],middle=57,motion=[52,61],other=[[38,40],[67,76]],reason='오른손이 왼쪽 위팔 가까이 올라간 뒤, 왼팔을 따라 손목 쪽으로 내려온다. 몸이 약간 기울어도 마지막 손목 부근 관계는 유지된다.'),
4:dict(setup=[74,75,76],terminal=[89,93,97,100],before=[62,64],middle=82,motion=[76,87],other=[[62,64],[90,100]],reason='왼팔 위쪽에 놓인 오른손이 전완을 따라 손목 가까이 이동하고 그 관계가 남는다. 좌표가 아닌 본인 왼팔과의 관계로 보았다.'),
5:dict(setup=[44,46,47],terminal=[60,64,68,72],before=[32,34],middle=51,motion=[47,58],other=[[32,34],[61,70]],reason='위팔 부근에서 시작한 오른손이 왼전완을 따라 손목 쪽으로 옮겨 간다. 뒤의 약한 자세 변화는 같은 손목 부근 관계에서 유지로 보인다.'),
6:dict(setup=[40,41,42],terminal=[53,55,59,61],before=[30,32],middle=46,motion=[42,51],other=[[30,32],[55,60],[64,66]],extra_absent=[66],reason='위팔 부근에서 손목으로 이어지는 이동과 뒤 손목 부근 유지가 보인다. 64~66프레임은 팔을 내리며 서로 떨어지는 마무리로, 다시 팔을 따라 손목으로 가는 이동은 아니다.'),
7:dict(setup=[44,46,47],terminal=[62,66,70,74],before=[35,37],middle=52,motion=[47,60],other=[[35,38],[64,74]],reason='도입에서 오른손이 올라온 뒤 왼팔을 따라 손목 쪽으로 옮겨 간다. 35~38은 위팔 쪽으로 접근하는 도입이고, 64~74는 손목 부근의 유지여서 표적 이동과 구별했다.'),
8:dict(setup=[70,71,72],terminal=[88,93,97,101],before=[60,63],middle=78,motion=[72,86],other=[[60,64],[90,101]],reason='오른손이 왼쪽 위팔 부근에서 전완을 따라 손목 부근으로 내려온다. 다른 수행자보다 늦은 시점이어도 실제 관계 변화로 보았고 초기 들어올림을 표적 이동으로 세지 않았다.'),
9:dict(setup=[44,46,47],terminal=[60,64,68],before=[38,40],middle=53,motion=[47,59],other=[[38,41],[64,69]],reason='팔을 들어올린 뒤 오른손이 왼전완을 따라 손목 쪽으로 옮겨 간다. 두 팔이 함께 낮아지는 작은 자세 변화 속에서도 도착 관계는 남아 있다.'),
}
def main():
    dest=U/'REVIEW_1133.json';assert not dest.exists()
    manifest=read(U/'PROCESS_REVIEW_MANIFEST.json');rows=[]
    for p in manifest['rows']:
        if p['word_id']!='WORD1133':continue
        v=NOTES[p['person']];frames={r['frame']:r for r in p['frames']};relations=[]
        for page in p['pages']:assert sha(page['path'])==page['sha256']
        def point(f,a,b,why):
            assert f in frames
            relations.append(dict(frame=f,timestamp_us=frames[f]['timestamp_us'],setup_relation=a,terminal_relation=b,reason=why))
        for f in v['before']:point(f,0,0,'오른손이 아직 왼팔의 출발 관계나 손목 부근 관계에 있지 않다.')
        for f in v['setup']:point(f,1,0,'오른손이 왼쪽 위팔/근위 팔 부근에 있으며 아직 손목 부근이 아니다. 접촉 깊이는 판정하지 않았다.')
        point(v['middle'],0,None,'위팔 부근 출발 관계에서는 벗어났다. 전완 이동 중이며 손목 부근 도착 여부의 경계는 보류한다.')
        for f in v['terminal']:point(f,0,1,'오른손이 왼쪽 손목/원위 전완 부근에 있다. 이 관계 존재가 전체 수어 성공을 뜻하지 않는다.')
        for f in v.get('extra_absent',[]):point(f,0,0,'마무리하며 손들이 분리되어 두 물리 관계 모두 보이지 않는다.')
        motion=[]
        for bounds,label in [(v['motion'],1)]+[(z,0) for z in v['other']]:
            assert all(f in frames for f in range(bounds[0],bounds[1]+1))
            motion.append(dict(start=bounds[0],end=bounds[1],label=label,
                start_us=frames[bounds[0]]['timestamp_us'],end_us=frames[bounds[1]]['timestamp_us'],
                reason='인접한 모든 원프레임에서 오른손이 왼팔을 따라 손목 쪽으로 이어져 이동한다.' if label else '이 구간은 도입/유지/마무리이며, 왼팔을 따라 손목으로 가는 표적 이동은 보이지 않는다. 픽셀 정지를 요구하지 않는다.'))
        rows.append(dict(id=p['id'],clip_id=p['clip_id'],word_id=p['word_id'],person=p['person'],source_sha256=p['source_sha256'],
            physical_query=p['physical_query'],reviewed_pages=p['pages'],reviewed_all_native_frames=True,
            predicate_definitions=dict(setup_relation='Right hand beside the proximal left arm/upper arm in the observed approach.',terminal_relation='Right hand beside the distal left forearm/wrist; projected body relationship, not verified skin contact.'),
            relations=sorted(relations,key=lambda x:x['frame']),motion_intervals=motion,
            sequence_links=[dict(setup_frames=v['setup'],motion_interval=v['motion'],terminal_frames=v['terminal'],visible_chain='yes',reason=v['reason'])],
            review_bounds_sufficient=True,missing_context=None,whole_word_correctness=None,required_extent=None,
            reviewer='Codex root visual review; not independent human/KSL expert',rationale=v['reason']))
    assert len(rows)==9
    out=dict(at=datetime.now(timezone.utc).isoformat(),status='DIRECT_FIT_PHYSICAL_PROCESS_REVIEW_COMPLETE',rows=rows,
        reviewed_pages=18,registered_ids=list(range(1,10)),new_inference=0,new_training=0,
        labels_are_sparse=True,unlisted_is_unknown=True,interval_targets_are_not_dense_frame_labels=True,
        pins={str(U/n):sha(U/n) for n in ['PROCESS_REVIEW_MANIFEST.json','PROCESS_LABEL_PROTOCOL.json','PHASE_LABEL_ADDENDUM.json']},whole_goal_complete=False)
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Recorded direct review:',len(rows),'processes,',sum(len(r['relations']) for r in rows),'relation points,',sum(len(r['motion_intervals']) for r in rows),'motion intervals')
if __name__=='__main__':main()
