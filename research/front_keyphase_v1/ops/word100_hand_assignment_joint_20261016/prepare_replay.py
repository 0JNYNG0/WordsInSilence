"""Prepare research-only replay from the already audited v2 harness."""
from pathlib import Path
U=Path(__file__).resolve().parent
old=U.with_name('word100_hand_assignment_repair_20261016')
s=(old/'replay.py').read_text('utf-8')
s=s.replace('import json,sys,time','import json,sys,time,importlib.util\nsys.dont_write_bytecode=True')
s=s.replace("from assignment import HandAssigner", """from assignment import HandAssigner
V2=U.with_name('word100_hand_assignment_repair_20261016')
spec=importlib.util.spec_from_file_location('frozen_v2_assignment',V2/'assignment.py')
v2module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v2module;spec.loader.exec_module(v2module)
""")
s=s.replace("capture=read(P/'RAW_CAPTURE_RESULT.json');ctx=read(P/'RAW_CAPTURE_CONTEXT.json')", """capture=read(P/'RAW_CAPTURE_RESULT.json');ctx=read(P/'RAW_CAPTURE_CONTEXT.json')
    broad=read(U/'BROADENED_RAW_CAPTURE_RESULT.json')
    broadctx=read(U/'BROADENED_RAW_CAPTURE_CONTEXT.json')
    original_rows=capture['rows']
    rows_source=[]
    for r in original_rows:
        rows_source.append(dict(r,raw_dir=str(P/'raw_capture'/r['clip_id']/r['mode']),cohort='original_train01'))
    for r in broad['rows']:
        rawfile=next(Path(p) for p in r['output_sha256'] if Path(p).name=='raw.npz')
        rows_source.append(dict(r,raw_dir=str(rawfile.parent),cohort='additional_train02_03'))
    capture=dict(rows=rows_source)
    assert len(rows_source)==40
    for p,h in broadctx['pins'].items():assert sha256_file(p)==h,p
    frozen=read(U/'CANDIDATE_FREEZE.json')
    for p,h in frozen['pins'].items():assert sha256_file(p)==h,p
""")
s=s.replace("files=[U/x for x in", "files=[U/'BROADENED_TRAIN_PLAN.json',U/'BROADENED_RAW_CAPTURE_CONTEXT.json',U/'CANDIDATE_FREEZE.json']+[U/x for x in")
s=s.replace("'TRAIN12/native+crop24'", "'TRAIN20/native+crop40'")
s=s.replace("TRAIN12/native+crop24", "TRAIN20/native+crop40")
s=s.replace("cid,mode=rec['clip_id'],rec['mode'];d=P/'raw_capture'/cid/mode", "cid,mode=rec['clip_id'],rec['mode'];d=Path(rec['raw_dir'])")
s=s.replace("candidate=HandAssigner();trace=[];counts=Counter();changed=[]", """candidate=HandAssigner();trace=[];counts=Counter();changed=[]
        previous=v2module.HandAssigner();prev_array=np.full((old.n_frames,2),-1,np.int8)
        previous_changes=[];previous_counts=Counter()
        cached_v2=V2/'replay'/cid/mode/'assignment.npz'
        if cached_v2.exists():
            with np.load(cached_v2,allow_pickle=False) as z: cached_previous=z['sides']
        else: cached_previous=None
""")
s=s.replace("diag.update(frame=f,timestamp_us=int(old.timestamp_us[f]));trace.append(diag)", """diag.update(frame=f,timestamp_us=int(old.timestamp_us[f]));trace.append(diag)
            if cached_previous is not None:
                prev_array[f]=cached_previous[f]
            else:
                prev_sides,_=previous.assign(hands,labels,scores,raw['pose_xy_px'][f],raw['pose_usable'][f],int(old.timestamp_us[f]),w,h)
                for i,s in prev_sides.items():prev_array[f,i]=-1 if s is None else s
""")
s=s.replace("counts[category]+=1;counts['raw_detections']+=1", """counts[category]+=1;counts['raw_detections']+=1
                prev=int(prev_array[f,i]);new=-1 if s is None else int(s)
                pk='unchanged' if prev==new else 'newly_assigned' if prev<0 else 'dropped' if new<0 else 'side_changed'
                previous_counts[pk]+=1
                if pk!='unchanged':previous_changes.append(dict(frame=f,raw_index=i,previous_side=prev,new_side=new,category=pk,reason=diag['reasons'][i]))
""")
s=s.replace("'handassoc3_'", "'handassoc4_'")
s=s.replace("write_new(dest/'trace.json',trace);write_new(dest/'changes.json',changed)", """write_new(dest/'trace.json',trace);write_new(dest/'changes.json',changed)
        np.savez_compressed(dest/'previous_assignment.npz',sides=prev_array)
        write_new(dest/'changes_vs_previous.json',previous_changes)
""")
s=s.replace("'baseline_replay_exact':True,'channels':channels", "'raw_dir':str(d),'cohort':rec['cohort'],'counts_vs_previous':dict(previous_counts),'baseline_replay_exact':True,'channels':channels")
s=s.replace("for p,h in {**pins,**protected}.items()", "for p,h in {**pins,**protected,**broadctx['pins'],**frozen['pins']}.items()")
s=s.replace("'counts':dict(total),'rows':rows", "'counts':dict(total),'counts_vs_previous':dict(sum((Counter(r['counts_vs_previous']) for r in rows),Counter())),'rows':rows")
compile(s,str(U/'replay.py'),'exec')
with (U/'replay.py').open('x',encoding='utf-8') as f:f.write(s)
print('Prepared40-condition replay; no detector/classifier/training executed')
