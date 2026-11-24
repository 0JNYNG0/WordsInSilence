"""Training-metadata adjudication overlay and bounded, non-model temporal audit."""
import collections
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

U = Path(__file__).resolve().parent
OPS = U.parent
ROOT = OPS.parents[2]
JOIN = OPS/'word_dataset_plan_20261013/FRONT_VIDEO_JOIN.jsonl'
VOCAB = OPS/'word_vocabulary_manifest_20261013/VOCABULARY.json'
PILOT = Path('W:/WordsInSilence/data/AIHub_103/pilot12_inputs_20261013/RESULT.json')
CASES = [('WORD0077', 1, 3), ('WORD0085', 2, 4), ('WORD0111', 1, 8), ('WORD1185', 1, 1)]

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(name, d):
    (U/name).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def train(r):
    return r['role']=='train_candidate' and r['metadata_video_join_eligible']

def classify(rows):
    g, labels = collections.defaultdict(list), collections.defaultdict(set)
    for r in rows:
        if not train(r): continue
        assert 1 <= int(re.search(r'_REAL(\d+)_F', r['clip_id']).group(1)) <= 9
        assert len(r['labels'])==1
        name=r['labels'][0].strip()  # Only boundary whitespace, never spelling/suffix normalization.
        g[r['word_code']].append(r)
        labels[name].add(r['word_code'])
    out=[]
    for c, rr in sorted(g.items()):
        names=collections.Counter(r['labels'][0].strip() for r in rr)
        reasons=[]
        if len(names)!=1: reasons.append('mixed_labels')
        if any(len(labels[n])>1 for n in names): reasons.append('cross_code_label')
        assert len(rr) in (8,9)
        out.append(dict(code=c, candidate=not reasons, labels=dict(names), training_clips=len(rr),
                        reasons=reasons, semantic_form_verified=False))
    return out

def stats(x):
    a=np.asarray(x, dtype=float)
    return dict(n=len(a), sum=float(a.sum()), min=float(a.min()), median=float(np.median(a)),
                p95=float(np.quantile(a,.95)), max=float(a.max()))

def main():
    assert not (U/'ANALYSIS.json').exists(), 'Do not overwrite a completed audit'
    pins={str(p):sha(p) for p in [JOIN,VOCAB,PILOT]}
    write('SCOPE.json', dict(authorization='User requested concrete label/form review, temporal policy and processing budget.',
        author='Codex local', source_hashes=pins, source_read_only=True,
        allowed='Training-only metadata overlay; eight preselected training video contact sheets; reuse twelve existing observations; planning. No model or detector inference.',
        selection='0077 REAL01/03 spelling;0085 REAL02/04 differing gloss;0111 REAL01/08 differing gloss;1185/1194 REAL01 shared gloss.',
        label_rule='Keep original raw labels; derived outer strip only. Keep 8 eligible rows instead of discarding entire code. Require one derived label per code and no label shared with other training codes.',
        semantic_limit='Images establish visible differences only; cannot certify gloss meanings or regional equivalence.',
        forbidden=['training','recognition inference','MediaPipe rerun','heldout media decode','split changes','source edits','camera','network upload','downloads','default model changes','Git'],
        output_budget_bytes=50_000_000, full_frame_media_reads_max=8))
    rows=[json.loads(s) for s in JOIN.read_text('utf-8').splitlines()]
    tr=[r for r in rows if train(r)]
    decisions=classify(rows)
    assert decisions==classify(tr), 'Non-training rows affected decisions'
    selected={d['code'] for d in decisions if d['candidate']}
    old={d['code'] for d in read(VOCAB)['codes'] if d['metadata_candidate']}
    overlay=[dict(clip_id=r['clip_id'], code=r['word_code'], role=r['role'],
                  raw_label=r['labels'][0], derived_label=r['labels'][0].strip(),
                  duration_seconds=r['duration_metadata'], intervals=r['intervals'], video_entries=r['video_entries'])
             for r in tr if r['word_code'] in selected]
    with (U/'TRAINING_CANDIDATES_V2.jsonl').open('x',encoding='utf-8') as f:
        for r in overlay: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write('VOCABULARY_V2.json', decisions)
    changes=[dict(clip_id=r['clip_id'],raw=r['labels'][0],derived=r['labels'][0].strip()) for r in tr if r['labels'][0]!=r['labels'][0].strip()]
    write('FORMAT_OVERLAY.json', changes)
    examples=[]
    for c in ['WORD0077','WORD0085','WORD0111','WORD0188','WORD1185','WORD1194']:
        examples.append(dict(code=c, rows=[dict(clip_id=r['clip_id'],raw_labels=r['labels'],intervals=r['intervals']) for r in tr if r['word_code']==c]))
    write('CONCRETE_CASES.json',examples)
    bounds=[]
    for r in overlay:
        a,b=r['intervals'][0]; dur=r['duration_seconds']
        assert len(r['intervals'])==1 and 0 <= a < b <= dur
        bounds.append(dict(clip_id=r['clip_id'],duration=dur,start=a,end=b,core=b-a,
                           context_start=max(0,a-.3),context_end=min(dur,b+.3)))
    dur=stats([r['duration'] for r in bounds]); core=stats([r['core'] for r in bounds])
    ctx=stats([r['context_end']-r['context_start'] for r in bounds])
    write('TRAIN_TIME_BOUNDS.json',bounds)
    pilot=read(PILOT); pilot_audit=[]
    for c in pilot['clips']:
        assert sha(c['observation'])==c['observation_sha256']
        with np.load(c['observation']) as f:
            ts=f['timestamp_us']; fid=f['frame_id']; use=f['usable']
            sec=ts/1e6; a,b=c['intervals'][0]
            m=(sec>=a)&(sec<=b)
            cm=(sec>=max(0,a-.3))&(sec<=min(float(sec[-1]),b+.3))
            ix=np.flatnonzero(cm)
            assert len(ix) and np.all(np.diff(ix)==1) and np.all(cm[m])
            anyhand=use[:,33:75].any(1)
            pilot_audit.append(dict(clip_id=c['clip_id'],full=len(ts),core=int(m.sum()),
                context=int(cm.sum()),context_frame_ids=fid[ix].tolist(),
                missing_in_core=int((m&~anyhand).sum()),missing_in_context=int((cm&~anyhand).sum()),
                timestamps_retained=True,contiguous_original_rows=True,
                observed_hand_span_full=bool(anyhand[0] and anyhand[-1])))
    write('PILOT_TIME_AUDIT.json',pilot_audit)
    frames=pilot['totals']['full']['frames']; timing=pilot['totals']['timing']; caches=pilot['totals']['cache_bytes']
    # Explicit 30fps assumption from pilot, not a claim to have probed all media.
    estimated_frames=dur['sum']*30
    sizes={role:sum(r['video_entries'][0]['bytes'] for r in rows if r['word_code'] in selected and r['role']==role and r['metadata_video_join_eligible']) for role in sorted({r['role'] for r in rows})}
    estimates=dict(training_seconds=dur['sum'],fps_assumed=30,estimated_frames=estimated_frames,
        pose_hours=estimated_frames/frames*timing['extract_seconds']/3600,
        full_rgb_pipeline_hours=estimated_frames/frames*pilot['elapsed_seconds']/3600,
        coordinates_and_channels_bytes=estimated_frames/frames*(caches['observation']+caches['channels']),
        all_rgb_cache_bytes=estimated_frames/frames*caches['rgb'],
        interpretation='Linear extrapolation from 12 clips of REAL01 only; no HDD contention/long-clip/multi-person throughput guarantee.')
    write('BUDGET.json',dict(estimates=estimates, eligible_video_bytes_by_role=sizes,
        disk_snapshot={d:dict(total=shutil.disk_usage(d).total,free=shutil.disk_usage(d).free) for d in ['C:/','W:/']},
        fixed_caps=dict(next_chunk_clips=300,next_chunk_wall_seconds=7200,workers=1,
            temporary_video_bytes=2_000_000_000,total_coordinate_cache_bytes=20_000_000_000,
            rgb_cache_bytes=2_000_000_000,incremental_workspace_bytes=1_000_000_000,
            min_free_external_bytes=500_000_000_000,training_runs_this_unit=0),
        storage_policy='Keep ZIPs. Stream/read a single selected F member or use bounded temporary cache; coordinate cache on W. No full F expansion, no all-frame RGB archive. Caps are stop/review boundaries, not delete-original rules.',
        training_time='Unmeasured. Freeze after a bounded training-only runtime benchmark and input verification; do not reuse small93-class timings as a 2494-class estimate.'))
    media=[]
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
    index={r['clip_id']:r for r in tr}
    for case_no,(code,s1,s2) in enumerate(CASES):
        codes=[code,'WORD1194' if code=='WORD1185' else code]
        sheet=Image.new('RGB',(1400,570),'white'); draw=ImageDraw.Draw(sheet)
        for rowno,(co,sid) in enumerate(zip(codes,[s1,s2])):
            cid=f'NIA_SL_{co}_REAL{sid:02d}_F'; r=index[cid]; e=r['video_entries'][0]
            zp=Path(e['archive']); before=(zp.stat().st_size,zp.stat().st_mtime_ns)
            with zipfile.ZipFile(zp) as z: payload=z.read(e['member']) # zipfile verifies this member CRC only
            assert len(payload)==e['bytes']
            a,b=r['intervals'][0]; targets=[max(0,a-.3),a,a+(b-a)*.25,(a+b)/2,a+(b-a)*.75,b,min(r['duration_metadata'],b+.3)]
            best=[None]*len(targets); dist=[float('inf')]*len(targets); n=0; last=-1
            with av.open(io.BytesIO(payload)) as container:
                for frame in container.decode(video=0):
                    t=float(frame.pts*frame.time_base); assert t>last; last=t; n+=1
                    for j,target in enumerate(targets):
                        if abs(t-target)<dist[j]:
                            best[j]=(t,frame.to_image().resize((200,190))); dist[j]=abs(t-target)
            assert before==(zp.stat().st_size,zp.stat().st_mtime_ns)
            y=rowno*285
            draw.text((6,y+4),cid+' | '+repr(r['labels'][0])+f' | core {a:.3f}–{b:.3f}s',font=font,fill='black')
            for j,(t,img) in enumerate(best):
                sheet.paste(img,(j*200,y+40)); draw.text((j*200+4,y+235),f'{t:.3f}s',font=font,fill='black')
            media.append(dict(clip_id=cid,raw_label=r['labels'][0],frames_decoded=n,
                sha256=hashlib.sha256(payload).hexdigest(),member_crc_checked=True,
                archive_stat_unchanged=True,archive=e['archive'],member=e['member'],
                targets=targets,actual_times=[t for t,img in best],sheet=f'case_{case_no+1}.jpg'))
        sheet.save(U/f'case_{case_no+1}.jpg',quality=90)
    write('MEDIA_REVIEW_INPUTS.json',media)
    assert pins=={p:sha(p) for p in pins}
    write('ANALYSIS.json',dict(status='ANALYZED_PENDING_SEPARATE_VERIFICATION_AND_VISUAL_REVIEW',source_hashes=pins,
        old_candidates=len(old), revised_candidates=len(selected),added_codes=sorted(selected-old),removed_codes=sorted(old-selected),
        revised_training_clips=len(overlay),people_histogram=dict(collections.Counter(d['training_clips'] for d in decisions if d['candidate'])),
        remaining_review_codes=3000-len(selected),raw_format_rows=len(changes),
        remaining_reason_overlap=dict(collections.Counter(x for d in decisions for x in d['reasons'])),
        duration_seconds=dur,core_seconds=core,context_seconds=ctx,
        core_fraction=core['sum']/dur['sum'],context_fraction=ctx['sum']/dur['sum'],
        decoded_training_clips=len(media),pilot_cache_clips_reused=12,detector_inference=0,training=0,default_model_changed=False))
    print(json.dumps(dict(candidates=len(selected),clips=len(overlay),duration=dur,core_fraction=core['sum']/dur['sum'],budget=estimates),ensure_ascii=False))

if __name__=='__main__': main()
