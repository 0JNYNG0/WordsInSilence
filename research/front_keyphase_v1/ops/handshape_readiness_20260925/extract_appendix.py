"""Read the already-held NIKL handshape appendix with the standard library only.

No download, no new dependency, no model. The PDF is the copy this project already has
(DICTIONARY_SOURCE.json). Text is CID encoded, so the embedded ToUnicode CMaps are parsed
and applied. Output is the raw table rows for the requested headwords, nothing more.
"""
import hashlib, json, pathlib, re, sys, zlib

H = pathlib.Path(__file__).resolve().parent
PDF = H.parent / 'word_semantic_review_20260922' / 'nikl_handshape_appendix2_2017.pdf'
PDF_SHA = 'de6077e2142c514c84e34503432511a7ad506e8fe423448253ecd72a2864b593'


def cmaps(raw):
    """Merge every embedded ToUnicode CMap; report any conflicting code."""
    table, conflicts = {}, 0
    for s in re.findall(rb'stream\r?\n(.*?)endstream', raw, re.S):
        try:
            d = zlib.decompress(s)
        except Exception:
            continue
        if b'beginbfchar' not in d and b'beginbfrange' not in d:
            continue
        for blk in re.findall(rb'beginbfchar(.*?)endbfchar', d, re.S):
            for src, dst in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                k, v = int(src, 16), bytes.fromhex(dst.decode()).decode('utf-16-be', 'replace')
                if k in table and table[k] != v:
                    conflicts += 1
                table[k] = v
        for blk in re.findall(rb'beginbfrange(.*?)endbfrange', d, re.S):
            for lo, hi, dst in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
                a, b = int(lo, 16), int(hi, 16)
                base = int(dst, 16)
                for i in range(a, b + 1):
                    v = chr(base + i - a)
                    if i in table and table[i] != v:
                        conflicts += 1
                    table[i] = v
    return table, conflicts


def pages(raw, table):
    """Yield decoded text per content stream, lines split on the text-matrix y position."""
    for s in re.findall(rb'stream\r?\n(.*?)endstream', raw, re.S):
        try:
            d = zlib.decompress(s)
        except Exception:
            continue
        if b'TJ' not in d and b'Tj' not in d:
            continue
        rows, y = {}, None
        for m in re.finditer(rb'([\d.\-]+)\s+([\d.\-]+)\s+Tm|<([0-9A-Fa-f]+)>\s*\]?\s*TJ|<([0-9A-Fa-f]+)>\s*Tj', d):
            if m.group(2) is not None:
                y = round(float(m.group(2)), 1)
            else:
                hx = (m.group(3) or m.group(4)).decode()
                txt = ''.join(table.get(int(hx[i:i + 4], 16), '') for i in range(0, len(hx), 4))
                if y is not None:
                    rows.setdefault(y, []).append(txt)
        yield '\n'.join(''.join(v) for _, v in sorted(rows.items(), reverse=True))


def main(words):
    raw = PDF.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PDF_SHA, 'PDF identity mismatch'
    table, conflicts = cmaps(raw)
    hits = {w: [] for w in words}
    npages = 0
    for i, text in enumerate(pages(raw, table)):
        npages += 1
        for line in text.split('\n'):
            parts = line.split()
            if len(parts) < 9 or not parts[0].isdigit():
                continue
            headword = parts[3]
            if headword in hits:
                hits[headword].append({'page0': i, 'seq': int(parts[0]), 'row': line})
    out = {'pdf_sha256': PDF_SHA, 'pages_scanned': npages, 'cmap_entries': len(table),
           'cmap_conflicts': conflicts,
           'columns': '순 대분류 중분류 표제어 분류명 수형 수위 한손/두손 동형/이형 (header verified on page 149)',
           'searched_headwords': list(words), 'hits': hits,
           'downloads': 0, 'new_dependencies': 0, 'training_runs': 0}
    with (H / 'APPENDIX_SCAN.json').open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('pages', npages, '| cmap', len(table), 'conflicts', conflicts)
    for w in words:
        print(f'  {w:<8} rows={len(hits[w])}')
        for h in hits[w][:4]:
            print('      ', h['row'])


if __name__ == '__main__':
    main(sys.argv[1:])
