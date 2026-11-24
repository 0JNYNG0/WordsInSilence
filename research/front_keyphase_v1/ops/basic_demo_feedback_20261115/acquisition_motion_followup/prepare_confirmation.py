from motion_review import *
import zipfile,importlib.util

def main():
    V=D/'confirmation';V.mkdir(exist_ok=True);(V/'media').mkdir(exist_ok=True)
    assert not(V/'FROZEN.json').exists()
    inv=read(U.parent/'demo_scope_portability_review_20261113/INVENTORY.json')['targets'];rows=[]
    for word,people in [('WORD1290',[18]),('WORD0099',[14,15])]:
        for person in people:
            r=next(v for v in inv[word] if v['record_id']==person);p=V/'media'/f'{r["clip_id"]}.mp4'
            with zipfile.ZipFile(r['archive'])as z:
                info=z.getinfo(r['member']);assert f'{info.CRC:08x}'==r['crc32'];b=z.read(r['member'])
            assert len(b)==r['bytes'];p.write_bytes(b)
            rows.append(dict(clip_id=r['clip_id'],word=word,person=person,role='confirmation',path=str(p),sha256=sha(p),
                archive=r['archive'],member=r['member'],source_interval=r['metadata']['intervals'][0]))
    prior=read(U.parent/'word100_public_g4_20261114/FROZEN.json')
    write(V/'FROZEN.json',dict(sources=rows,model_pins=prior['model_pins'],
        motion_model=[sha(D/'runs'/f'mlp32_{seed}.pt')for seed in [42,43,44]],
        bridge_sha=sha(D/'check_motion_bridge.py'),apple_classifier_sha=sha(U/'apple_classifier.pt'),
        scope='Gratitude18 held from this motion candidate; apple14/15 absent from appended classifier fit/dev. Not claimed project-wide never accessed.',
        criteria='Gratitude two visually verified arrivals; no count completion for constant/missing/first-only. Apple: report actual rank and clause observation readiness, no full pass.',
        source_truth='Inspect native RGB before model predictions; no private data or network media acquisition.'))
    p=U.parent/'word100_public_g4_20261114/prepare.py';s=importlib.util.spec_from_file_location('basic_current_confirmation',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.U=V;m.main()

if __name__=='__main__':main()
