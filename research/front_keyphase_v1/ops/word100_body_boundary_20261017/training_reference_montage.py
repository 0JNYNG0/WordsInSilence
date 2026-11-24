"""100 fixed TRAIN references: sparse full-frame source montage, no detector/model."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import sys
import time
import numpy as np
import av
from PIL import Image,ImageDraw,ImageFont
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('body_fallback_evidence_drawing',U/'evidence_fallback.py')
draw=importlib.util.module_from_spec(spec);spec.loader.exec_module(draw)


def main():
    started=time.perf_counter()
    output=U/'training_reference_montage_RESULT.json'
    assert not output.exists()
    context=draw.read(U/'PREP_CONTEXT.json')
    ids=set(context['fixed_reference_ids'])
    rows=sorted([r for r in context['ordered_rows'] if r['role']=='train_candidate' and r['clip_id'] in ids],key=lambda r:r['word_code'])
    assert len(rows)==100 and len(ids)==100 and all(1<=r['signer_id']<=9 for r in rows)
    forecasts={r['clip_id']:r for r in draw.read(U/'LEGACY_PREFLIGHT.json')['rows'] if r['role']=='train_candidate'}
    triage={r['clip_id']:r for r in draw.read(U/'training_geometry_triage_RESULT.json')['rows']}
    pinfiles=[Path(__file__),U/'evidence_fallback.py',U/'PREP_CONTEXT.json',U/'LEGACY_PREFLIGHT.json',U/'training_geometry_triage_RESULT.json']
    pins={str(p):draw.sha(p) for p in pinfiles}
    tw,th=416,234;rowh=286;head=76
    pages=[];items=[]
    for page in range(10):
        sheet=Image.new('RGB',(tw*5,head+10*rowh),(16,22,32));sd=ImageDraw.Draw(sheet)
        sd.text((12,7),f'고정 TRAIN 대표 100개 / {page+1:02d} / 원본 전체 영상 + legacy 관측',font=draw.font(25),fill='white')
        sd.text((12,43),'노랑 하단선 / 선 아래 제외영역 / 초록 왼손·하늘 오른손 / 빨강 제외점 | 4핵심샘플 + 1경계샘플',font=draw.font(19),fill=(208,220,232))
        for offset,row in enumerate(rows[page*10:page*10+10]):
            cid=row['clip_id'];rec=draw.read(row['legacy_record_path'])
            assert rec['role']=='train_candidate'
            video=Path(row['retained_video'])
            assert video.is_file() and draw.sha(video)==rec['source_sha256']
            op=Path(rec['observation']);assert draw.sha(op)==rec['observation_sha256']
            pins[str(video)]=rec['source_sha256'];pins[str(op)]=rec['observation_sha256']
            with np.load(op,allow_pickle=False) as z:
                xy=z['xy_px'];usable=z['usable'];ts=z['timestamp_us']
            a,b=row['intervals'][0];core=np.flatnonzero((ts>=round(a*1e6))&(ts<=round(b*1e6)))
            bottom=int(forecasts[cid]['bottom'])
            selected=np.linspace(core[0],core[-1],4).round().astype(int).tolist()
            risk=[]
            if cid in triage:
                for side,h in triage[cid]['hands'].items():
                    for e in h['evidence_frames']:
                        risk.append((0 if e['kind']=='native_palm_observed_but_entire_hand_removed' else 1,e['frame_id'],side,e['kind']))
            if risk:
                _,extra,side,kind=sorted(risk)[0]
            else:
                wrist=np.array([33,54]);valid=usable[core][:,wrist]
                distance=np.where(valid,np.abs(xy[core][:,wrist,1]-bottom),np.inf)
                i,j=np.unravel_index(np.argmin(distance),distance.shape)
                extra=int(core[i]);side=('left','right')[j];kind='closest_visible_wrist_to_boundary'
            selected.append(extra)
            wanted=set(selected);frames={}
            with av.open(str(video)) as container:
                stream=container.streams.video[0];stream.thread_type='AUTO'
                for i,f in enumerate(container.decode(video=0)):
                    if i in wanted:frames[i]=f.to_ndarray(format='rgb24')
                    if i>=max(wanted):break
            assert set(frames)==wanted
            y=head+offset*rowh
            sd.text((8,y+4),f'{page*10+offset+1:03d}  {row["canonical_label"]} | {cid} | core {a:.3f}~{b:.3f}s | 하단{bottom}px',font=draw.font(21),fill='white')
            for col,index in enumerate(selected):
                title=f'f{index} {ts[index]/1e6:.3f}s'+(' / 경계확인' if col==4 else ' / 핵심구간')
                tile=draw.draw_frame(frames[index],xy[index],usable[index],bottom,title,size=(tw,th))
                sheet.paste(tile,(col*tw,y+38))
            items.append(dict(clip_id=cid,label=row['canonical_label'],role='train_candidate',
                              page=page+1,row_on_page=offset+1,source=str(video),source_sha256=rec['source_sha256'],
                              observation=str(op),observation_sha256=rec['observation_sha256'],bottom=bottom,
                              selected_frames=selected,extra_frame_reason=dict(side=side,kind=kind,frame=extra),
                              core_seconds=[a,b],review_status='PENDING_ACTUAL_IMAGE_INSPECTION'))
        path=U/f'training_reference_page_{page+1:02d}.jpg'
        assert not path.exists();sheet.save(path,quality=94,subsampling=0)
        pages.append(dict(page=page+1,path=str(path),sha256=draw.sha(path)))
        print(json.dumps({'page':page+1,'path':str(path)},ensure_ascii=False),flush=True)
    result=dict(status='100_FIXED_TRAIN_REFERENCE_MONTAGES_CREATED_PENDING_VISUAL_REVIEW',
                created_utc=datetime.now(timezone.utc).isoformat(),pages=pages,rows=items,source_pins=pins,
                sampled_images=500,unique_source_clips=100,detector=0,training=0,development_arrays=0,
                seconds=time.perf_counter()-started,
                limits='Sparse full-frame source RGB with legacy landmarks. Not continuous video review, not fresh sensor detector verification, not linguistic semantic-preservation proof.')
    for p,h in pins.items():assert draw.sha(p)==h
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
