"""Separate arithmetic/archive verification. Same Codex, not independent human truth."""
from pathlib import Path
import json,hashlib,struct,zlib,collections
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()
def run():
    frozen=read(U/'FROZEN.json');done=read(U/'DONE.json');index=read(U/'INDEX.json');diagnostic=read(U/'DIAGNOSTICS.json')
    assert len(done['rows'])==len(frozen['rows'])==100
    refs={r['clip_id']:r for r in frozen['rows']};members=collections.defaultdict(list)
    for entry in index['members']:members[entry['clip_id']].append(entry)
    checked=0;totals=collections.Counter();group_details=[]
    with open(frozen['archive'],'rb') as archive:
        for r in done['rows']:
            cid=r['clip_id'];src=refs[cid];assert src['signer_id']==2 and src['role']=='train_candidate'
            assert digest(r['arrays'])==r['arrays_sha'];a=np.load(r['arrays']);n=r['frames']
            present={m['frame'] for m in members[cid]}
            assert len(present)==len(members[cid])
            assert sorted(set(range(n))-present)==r['missing_frames']
            assert sorted(present-set(range(n)))==r['extra_frames']
            for m in members[cid]:
                archive.seek(m['offset']);head=archive.read(30);assert head[:4]==b'PK\x03\x04'
                nl,el=struct.unpack_from('<HH',head,26);name=archive.read(nl).decode('utf-8');archive.read(el)
                assert name==m['name'];compressed=archive.read(m['compressed']);raw=zlib.decompress(compressed,wbits=-15)
                assert len(raw)==m['uncompressed'] and zlib.crc32(raw)&0xffffffff==m['crc32']
                assert hashlib.sha256(raw).hexdigest()==r['payload_sha256'][str(m['frame'])]
                obj=json.loads(raw);p=obj['people'];p=p[0] if isinstance(p,list) else p
                t=m['frame'];checked+=1
                if t>=n:continue
                for side,key in enumerate(('hand_left','hand_right')):
                    xy=np.array(p[key+'_keypoints_2d']).reshape(21,3)
                    assert np.array_equal(xy,a['official'][t,side])
                    xyz=np.array(p[key+'_keypoints_3d']).reshape(21,4)
                    assert np.array_equal(xyz,a['official_xyz'][t,side])
                assert np.array_equal(np.array(p['pose_keypoints_2d']).reshape(25,3),a['official_pose'][t])
            if r['missing_frames']:assert np.isnan(a['official'][r['missing_frames']]).all()
            v=np.isfinite(a['official']).all(-1)&(a['official'][...,2]>0)&((a['official'][...,0]!=0)|(a['official'][...,1]!=0))
            assert np.array_equal(v,a['official_valid'])
            mp=np.load(src['candidate']);base=np.load(src['baseline'])
            assert np.array_equal(mp['xy_px'][:,33:75].reshape(n,2,21,2),a['mp_xy'],equal_nan=True)
            expected_mp=mp['observed'][:,33:75].reshape(n,2,21)&np.isfinite(a['mp_xy']).all(-1)
            assert np.array_equal(expected_mp,a['mp_valid'])
            assert np.array_equal(base['xy_px'][:,33:75].reshape(n,2,21,2),a['baseline_xy'],equal_nan=True)
            delta=np.sqrt(np.sum((a['official'][...,:2]-a['mp_xy'])**2,-1));delta[~(v&expected_mp)]=np.nan
            assert np.allclose(delta,a['distance'],equal_nan=True)
            assert abs(np.nanmedian(delta[:,:,0])-r['wrist_disagreement_px']['median'])<1e-9
            assert int(v.sum())==r['official_valid_points']
            assert int(expected_mp.sum())==r['mp_valid_points']
            assert int((v&~expected_mp).sum())==r['provided_but_mp_absent_points']
            assert np.array_equal(a['frame_id'],np.arange(n))
            assert np.array_equal(a['timestamp_us'],mp['timestamp_us'])
            for key in ['frames','official_members','official_valid_points','total_hand_points','mp_valid_points','provided_but_mp_absent_points','confidence_exactly_one']:totals[key]+=r[key]
            totals['missing_frames']+=len(r['missing_frames']);totals['extra_frames']+=len(r['extra_frames'])
    assert checked==totals['official_members']==done['crc_checked']==9565
    assert totals['frames']-totals['missing_frames']+totals['extra_frames']==checked
    visual=read(U/'VISUAL_MANIFEST.json');assert len(visual['clips'])==12 and visual['frame_count']==89
    for v in visual['clips']:
        r=next(r for r in done['rows'] if r['clip_id']==v['clip_id'])
        assert r['video_check']['exact_mp_frames'];assert digest(r['video'])==r['source_sha']
        assert (U/v['sheet']).exists()
        for p in v['frames']:assert (U/p['full_image']).exists()
    for p,h in frozen['protected'].items():assert digest(p)==h,p
    for p,h in frozen['pins'].items():assert digest(p)==h,p
    exact=[r['clip_id'] for r in done['rows'] if not r['missing_frames'] and not r['extra_frames']]
    high=[r['clip_id'] for r in done['rows'] if r['wrist_disagreement_px']['median']>40]
    for g in diagnostic['camera_groups']:
        rr=[r for r in diagnostic['rows'] if r['clip_id'] in g['clips']]
        group_details.append(dict(intrinsics=g['intrinsics'],clips=len(rr),high_disagreement=sum(r['clip_id'] in high for r in rr),first_wrist_z_min=min(r['first_left_wrist_z'] for r in rr),first_wrist_z_max=max(r['first_left_wrist_z'] for r in rr)))
    result=dict(status='ARITHMETIC_AND_ARCHIVE_VERIFIED_NOT_GROUND_TRUTH_APPROVAL',verifier='Separate local verifier by the same Codex; no independent human point annotations',clips=100,crc_rechecked=checked,totals=dict(totals),exact_frame_sets=len(exact),missing_clips=sum(bool(r['missing_frames']) for r in done['rows']),tail_missing_clips=diagnostic['missing_contiguous_tails'],extra_frame_clips=sum(bool(r['extra_frames']) for r in done['rows']),high_disagreement_clips=high,high_disagreement_definition='diagnostic >40 pixel median versus MP; NOT error-rate threshold',camera_groups=group_details,protected_files=len(frozen['protected']),visual_base_frames=89,visual_extra_frames=sum(len(v['frames']) for v in diagnostic['extra_visuals']),limits=['No 21-point expert ground truth','One record ID only','Correspondence issues unresolved','Stored repeated_hand_frame_pairs counts all-NaN absent frames too; not interpreted as frozen hand detections'],pins={str(U/k):digest(U/k) for k in ['FROZEN.json','INDEX.json','DONE.json','DIAGNOSTICS.json','VISUAL_MANIFEST.json']})
    with (U/'VERIFIED.json').open('x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2)
    print('VERIFIED',checked,'files',100,'clips',len(exact),'exact',len(high),'high',flush=True)
if __name__=='__main__':run()
