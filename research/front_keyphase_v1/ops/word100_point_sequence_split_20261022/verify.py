"""Separate arithmetic check of new overlay; reuses only previously verified descriptors.
Same Codex author, not another agent or semantic expert. Does not call split_support.
"""
import json,hashlib,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import study as D
U=D.U;B=D.B;read=D.read;load=D.load
def gap(a,b):
    # Independent interval centre / radius expression.
    ac=(a['lo']+a['hi'])/2;bc=(b['lo']+b['hi'])/2
    ar=(a['hi']-a['lo'])/2;br=(b['hi']-b['lo'])/2
    return np.maximum(np.abs(ac-bc)-ar-br,0.)

def main():
    assert not (U/'VERIFY.json').exists() and not (U/'RUNNING.json').exists()
    f=read(U/'FROZEN.json');D.check(f['pins']);D.check(f['protected'])
    rows,es,cs,dom,anchors,peers=D.prepare(f);r=read(U/'RESULT.json');n=0;nc=0
    for cid in rows:
        refs=[x for x in rows if rows[x]['word_code']==rows[cid]['word_code']]
        for rid in refs:
            p=load(U/'calibration'/(cid+'__'+rid+'.npz'));ids=[x for x in refs if x not in (cid,rid)]
            assert p['peer_ids'].tolist()==ids and len(ids)==(8 if rid==cid else 7)
            masks=np.stack([peers[x,rid]['valid']&anchors[rid]['valid'] for x in ids])
            count=np.count_nonzero(masks,axis=0);assert np.array_equal(count,p['count']);support=count>=5
            assert np.array_equal(support,p['support'])
            ds=np.stack([gap(peers[x,rid],anchors[rid]) for x in ids])
            bound=np.full(p['component_bound'].shape,np.nan)
            for c in range(5):
                b=np.where(masks,ds[...,c],-np.inf).max(axis=0);bound[...,c]=np.where(support,b,np.nan)
            assert np.allclose(bound,p['component_bound'],equal_nan=True,atol=1e-10,rtol=1e-10);nc+=1
    summaries={}
    for folder in ('known','cross','missing'):
        summaries[folder]=0
        for meta in r[folder]:
            z=load(U/folder/(meta['name']+'.npz'));o=load(B/folder/(meta['name']+'.npz'))
            for key in ('expected','point_comparable','point_bad','native_expected','native_unknown','path','source_index','query_phase_index','query_span','query_collapsed','context_unavailable','anchor_incomplete','query_incomplete','peer_insufficient','query_span_bad','query_rest_bad'):
                assert np.array_equal(z[key],o[key],equal_nan=True),(meta['name'],key)
            assert np.array_equal(z['sequence_comparable'],o['comparable'])
            assert np.array_equal(z['sequence_unknown'],o['unknown'])
            assert np.array_equal(z['point_unknown'],o['expected'] & ~o['point_comparable'])
            assert np.array_equal(z['point_within'],o['point_comparable'] & ~o['point_bad'])
            q=D.query(meta,o,es,cs,dom);cal=load(U/'calibration'/(meta['query']+'__'+meta['reference']+'.npz'))
            d=gap(q,anchors[meta['reference']]);ok=o['comparable'];ex=d-cal['component_bound']
            assert np.allclose(z['component_distance'],np.where(ok[...,None],d,np.nan),equal_nan=True,atol=1e-10,rtol=1e-10)
            assert np.allclose(z['component_excess'],np.where(ok[...,None],ex,np.nan),equal_nan=True,atol=1e-10,rtol=1e-10)
            bad=ok[...,None] & (ex>1e-6)
            assert np.array_equal(z['component_bad'],bad)
            assert np.array_equal(z['component_within'],ok[...,None] & ~bad)
            for key,ix in [('level_bad',slice(0,3)),('change_bad',slice(3,5)),('split_any_bad',slice(0,5))]:assert np.array_equal(z[key],bad[...,ix].any(-1))
            expect=np.zeros(o['expected'].shape,np.uint8)
            expect[o['expected'] & ~o['point_comparable'] & ~ok]=1
            expect[o['point_comparable'] & ~ok]=2;expect[~o['point_comparable'] & ok]=3;expect[o['point_comparable'] & ok]=4
            assert np.array_equal(z['evidence_state'],expect)
            assert not (z['sequence_unknown'][...,None]&z['component_within']).any()
            for g,sl in D.C.GROUPS.items():
                for key,value in meta['groups'][g].items():
                    if key=='states':assert value=={str(i):int((expect[:,:,sl]==i).sum()) for i in (1,2,3,4)}
                    elif key=='components_bad':assert value==[int(bad[:,:,sl,c].sum()) for c in range(5)]
                    else:assert value==int(z[key][:,:,sl].sum())
            assert meta['whole_action_correct'] is None;n+=1;summaries[folder]+=1
        print('verified',folder,summaries[folder],flush=True)
    D.check(f['pins']);D.check(f['protected'])
    D.write(U/'VERIFY.json',{'at':D.now(),'status':'PASS','cases':summaries,'total':n,'calibrations':nc,'protected':len(f['protected']),
        'checks':['exact old instantaneous flags and domain/native masks','separate five component interval gaps and excluded-peer max bounds','same sequence support as frozen parent','point-only evidence never erased or upgraded','unknown not within','rest/crossing coordinates and assignments inherited unchanged','all stored group counts'],
        'independence':'Same Codex, separately coded new-layer arithmetic using interval centres/radii; shares previously verified descriptor replay. No second-agent or expert review claimed.',
        'result_sha256':D.sha(U/'RESULT.json')})
    print('VERIFY PASS',n,nc,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'FIRST_VERIFY_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
