import json, re, sys
import pypdfium2 as pdfium
PDF = r'C:\MyUnityProjects\WordsInSilence\research\front_keyphase_v1\ops\word_semantic_review_20260922\nikl_handshape_appendix2_2017.pdf'
classes = json.load(open(sys.argv[1], encoding='utf-8'))
want = set(classes)
doc = pdfium.PdfDocument(PDF)
rowpat = re.compile(r'^\s*(\d{2,6})\s+(\S+\s*\S*)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d{4})\s+(\d{2})\s+([12])\s+([01])\s*$')
hits = {}
for i in range(len(doc)):
    tp = doc[i].get_textpage(); txt = tp.get_text_bounded(); tp.close()
    for line in txt.split('\n'):
        m = rowpat.match(line.strip())
        if not m: continue
        head = m.group(4)
        if head in want:
            hits.setdefault(head, []).append(
                {'seq': int(m.group(1)), 'form': m.group(5), 'handshape': m.group(6),
                 'location': m.group(7), 'hands': int(m.group(8)), 'same_diff': int(m.group(9))})
doc.close()
found = {w: v for w, v in hits.items() if v}
codes = {}
for w, v in found.items():
    s = sorted({r['handshape'] for r in v})
    codes[w] = s
single = [w for w, s in codes.items() for _ in [0] if len(s) == 1]
multi = {w: s for w, s in codes.items() if len(s) > 1}
missing = [w for w in classes if w not in found]
out = {'classes_total': len(classes), 'found': len(found), 'missing': missing,
       'single_code': len(single), 'multi_code': multi, 'per_word': found, 'codes': codes}
json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('classes', len(classes), '| appendix rows found for', len(found), '| missing', len(missing))
print('single handshape code:', len(single), '| multiple codes:', len(multi))
from collections import Counter
dist = Counter(s[0] for w, s in codes.items() if len(s) == 1)
print('code distribution (single-code words):', dict(sorted(dist.items(), key=lambda x: -x[1])))
print('missing sample:', missing[:15])
print('multi sample:', dict(list(multi.items())[:8]))
