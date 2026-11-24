from pathlib import Path
import tarfile, json, hashlib, shutil
U = Path(__file__).resolve().parent
src = U/'downloads/hamer_demo_data.tar.gz'
out = U/'private_assets/hamer'
out.mkdir(exist_ok=True, parents=True)
rows=[]
with tarfile.open(src,'r|gz') as t:
    for m in t:
        row={'name':m.name,'bytes':m.size,'type':str(m.type)}
        wanted = Path(m.name).name in {'hamer.ckpt','model_config.yaml','mano_mean_params.npz'}
        if wanted:
            dest = (out/m.name).resolve()
            if not dest.is_relative_to(out.resolve()) or not m.isfile():
                raise RuntimeError('unsafe archive member')
            if dest.exists(): raise RuntimeError('Refuse overwrite: '+str(dest))
            dest.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256()
            with t.extractfile(m) as fi, dest.open('wb') as fo:
                while block:=fi.read(8*1024*1024):
                    h.update(block);fo.write(block)
            if dest.stat().st_size!=m.size: raise RuntimeError('truncated member')
            row.update(extracted=str(dest),sha256=h.hexdigest())
            print('EXTRACTED',m.name,m.size,flush=True)
        rows.append(row)
(U/'WEIGHT_MEMBERS.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('ARCHIVE_READ_COMPLETE',len(rows),flush=True)
