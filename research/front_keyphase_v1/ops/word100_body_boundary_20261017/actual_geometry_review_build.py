"""Saved TRAIN observations and exact original RGB only; no detector/model calls."""
from pathlib import Path
import sys, json, zipfile, shutil
import numpy as np
import av
from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
sys.path.insert(0, str(U))
from evidence_fallback import read, sha, font, draw_frame
from training_geometry_triage import hand_info

IDS = ['NIA_SL_WORD0651_REAL08_F','NIA_SL_WORD0772_REAL02_F',
       'NIA_SL_WORD0185_REAL04_F','NIA_SL_WORD1085_REAL08_F','NIA_SL_WORD2029_REAL08_F']

def observe(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k:z[k] for k in ('xy_px','usable','observed','timestamp_us','frame_id')}
    d['valid'] = d['usable'] & d['observed'] & np.isfinite(d['xy_px']).all(-1)
    return d

def main():
    context_path=U/'PREP_CONTEXT.json'; triage_path=U/'actual_input_triage_RESULT.json'
    context=read(context_path); actual=read(triage_path)
    rows={r['clip_id']:r for r in context['ordered_rows'] if r['role']=='train_candidate'}
    targets={r['clip_id']:r for r in actual['rows'] if r['clip_id'] in IDS}
    assert len(targets)==5
    for cid in IDS:
        row,item=rows[cid],targets[cid]; record_path=Path(item['record_path']);record=read(record_path)
        assert sha(record_path)==item['record_sha256']
        pins={str(context_path):sha(context_path),str(triage_path):sha(triage_path),str(record_path):sha(record_path)}
        source=U/'review_sources'/(cid+'.mp4')
        if not source.exists():
            sd=U/'actual_geometry_review_sources';sd.mkdir(exist_ok=True)
            source=sd/(cid+'.mp4')
            if not source.exists():
                e=record['source']
                with zipfile.ZipFile(e['archive']) as z:
                    info=z.getinfo(e['member'])
                    assert info.CRC==e['crc32'] and info.file_size==e['bytes']
                    with z.open(info) as src, source.open('xb') as dst:shutil.copyfileobj(src,dst,1<<20)
        assert source.stat().st_size==record['source']['bytes'] and sha(source)==record['source']['sha256']
        pins[str(source)]=sha(source)
        obs={}
        for mode in ('native','masked','sensor'):
            p=Path(record['views'][mode]['observation']);expected=record['outputs'][str(p.resolve())]
            assert sha(p)==expected;pins[str(p)]=expected;obs[mode]=observe(p)
        ts=obs['native']['timestamp_us'];n=len(ts);bottom=item['bottom_pixels']
        a,b=item['core_seconds'];core=(ts>=round(a*1e6))&(ts<=round(b*1e6));ci=np.flatnonzero(core)
        overview=sorted(set([0,n-1,*np.linspace(ci[0],ci[-1],8).round().astype(int).tolist()]))
        focus=set()
        for side,h in item['hands'].items():
            for e in h['native_to_mask_geometry']['evidence_frames']:
                k=e['frame_id'];focus.update(range(max(0,k-3),min(n,k+4)))
        for loss in item['zero_hand_intervals']['mask_has_hand_but_sensor_has_none']['runs']:
            focus.update(range(max(0,loss['start_index']-3),min(n,loss['end_index']+4)))
        focus=sorted(focus);selected=set(overview)|set(focus)
        rgb={};pts=[]
        with av.open(str(source)) as video:
            for i,f in enumerate(video.decode(video=0)):
                pts.append(round(float(f.pts*f.time_base)*1e6))
                if i in selected:rgb[i]=f.to_ndarray(format='rgb24')
        pts=np.asarray(pts,np.int64);assert len(pts)==n and selected==set(rgb)
        assert np.max(np.abs((pts-pts[0])-(ts-ts[0])))<=1
        images=[]
        def sheet(ids,name,mode_list):
            columns=3 if len(mode_list)==1 else 2;tw=640;th=360;cap=44
            count=len(ids)*len(mode_list)
            canvas=Image.new('RGB',(columns*tw,80+((count+columns-1)//columns)*(th+cap)),(16,22,32));d=ImageDraw.Draw(canvas)
            d.text((12,7),f"{item['label']} / {cid} / boundary {bottom}px / {name}",font=font(24),fill='white')
            d.text((12,43),'전체 원본 | 노랑 하단·붉은 제외영역 | 초록 L·하늘 R | 관측없음 ≠ 실제손없음',font=font(19),fill=(215,222,235))
            for idx,(i,mode) in enumerate((i,mode) for i in ids for mode in mode_list):
                x=(idx%columns)*tw;y=80+(idx//columns)*(th+cap)
                o=obs[mode];tile=draw_frame(rgb[i],o['xy_px'][i],o['valid'][i],bottom,f"f{i} {ts[i]/1e6:.3f}s / {mode}")
                canvas.paste(tile,(x,y))
                counts=lambda md: '/'.join(str(int(obs[md]['valid'][i,s:s+21].sum())) for s in (33,54))
                d.text((x+7,y+th+3),f"L/R점 native {counts('native')} → mask {counts('masked')} → sensor {counts('sensor')}",font=font(16),fill='white')
                d.text((x+7,y+th+23),'핵심 주석 구간' if core[i] else '주석 구간 밖',font=font(15),fill=(196,203,216))
            p=U/f'actual_geometry_review_{cid}_{name}.png';assert not p.exists();canvas.save(p);images.append(p)
        sheet(overview,'overview',['native'])
        for offset in range(0,len(focus),9):sheet(focus[offset:offset+9],f'focus_{offset//9+1}',['native'])
        # Explicit source-view sensor disagreement, no fresh detector run.
        comparison=[]
        for loss in item['zero_hand_intervals']['mask_has_hand_but_sensor_has_none']['runs']:
            if loss['duration_seconds']>=.16:
                comparison += [max(0,loss['start_index']-1),loss['start_index'],(loss['start_index']+loss['end_index'])//2,loss['end_index'],min(n-1,loss['end_index']+1)]
        for offset in range(0,len(comparison),5):sheet(comparison[offset:offset+5],f'sensor_{offset//5+1}',['native','sensor'])
        # Recalculate the missing legacy row to distinguish filtered legacy results from a measured change.
        lp=Path(row['legacy_record_path']);lr=read(lp);op=Path(lr['observation']);assert sha(op)==lr['observation_sha256']
        pins[str(lp)]=sha(lp);pins[str(op)]=sha(op);old=observe(op)
        oldvalid=old['valid'];oxy=old['xy_px'];ow=record['calibration']['source_width']
        om=oldvalid&(oxy[:,:,0]>=0)&(oxy[:,:,0]<ow)&(oxy[:,:,1]>=0)&(oxy[:,:,1]<bottom)
        ots=old['timestamp_us'];ofid=old['frame_id'];oc=(ots>=round(a*1e6))&(ots<=round(b*1e6))
        shoulders=oc&oldvalid[:,11]&oldvalid[:,12]
        scale=float(np.median(np.linalg.norm(oxy[shoulders,11]-oxy[shoulders,12],axis=1)))
        center=np.median((oxy[shoulders,11]+oxy[shoulders,12])*.5,axis=0)
        legacy_geometry={side:hand_info(start,oxy,oldvalid,om,oc,ots,ofid,bottom,center,scale)[0] for side,start in [('left',33),('right',54)]}
        frames=[]
        for i in sorted(selected):
            frames.append({'frame_id':int(i),'timestamp_us':int(ts[i]),'source_pts_us':int(pts[i]),'in_core':bool(core[i]),
                           'valid_hand_points':{mode:{side:int(obs[mode]['valid'][i,start:start+21].sum()) for side,start in [('left',33),('right',54)]} for mode in obs}})
        meta={'clip_id':cid,'label':item['label'],'role':'train_candidate','bottom_pixels':bottom,'boundary_method':item['boundary_method'],
              'boundary_delta_pixels':item['boundary_delta_pixels'],'source':str(source),'source_pins':pins,'triage_row':item,
              'legacy_geometry_recomputed':legacy_geometry,'frame_count':n,'core_seconds':[a,b],
              'overview_frames':overview,'focus_frames':focus,'frames':frames,
              'PTS_max_relative_error_us':int(np.max(np.abs((pts-pts[0])-(ts-ts[0])))),
              'images':{str(p):sha(p) for p in images},'new_detector_runs':0,'classifier_runs':0,'training_runs':0,
              'limits':'Sparse full-clip samples with prioritized event windows, original RGB and stored assigned observations only. No automatic semantic pass or language-expert judgement.'}
        for p,h in pins.items():assert sha(p)==h
        out=U/f'actual_geometry_review_{cid}.json'
        with out.open('x',encoding='utf-8') as f:json.dump(meta,f,ensure_ascii=False,indent=2,allow_nan=False)
        print(json.dumps({'clip_id':cid,'images':[p.name for p in images],'legacy_moving':{k:v['d_moving_hand_near_boundary'] for k,v in legacy_geometry.items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
