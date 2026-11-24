"""Minimal fix: anchor on the four trailing numeric columns instead of a fixed token count.

Row layout: 순 | 대분류(2 tokens, ends 수어) | 중분류 | 표제어 | 분류명(1+ tokens) | 수형 수위 손 동형이형
Only the 표제어/분류명 split assumes a single-token headword; everything else is anchored.
"""
import json, re, sys
import pypdfium2 as pdfium
PDF = r'C:\MyUnityProjects\WordsInSilence\research\front_keyphase_v1\ops\word_semantic_review_20260922\nikl_handshape_appendix2_2017.pdf'
classes = set(json.load(open(sys.argv[1], encoding='utf-8')))
TAIL = re.compile(r'^(?P<head>.+?)\s+(?P<sh>\d{4})\s+(?P<loc>\d{2})\s+(?P<hands>[12])\s+(?P<sd>[01])\s*$')
doc = pdfium.PdfDocument(PDF)
hits, rows_total = {}, 0
for i in range(len(doc)):
    tp = doc[i].get_textpage(); txt = tp.get_text_bounded(); tp.close()
    for line in txt.split('\n'):
        m = TAIL.match(line.strip())
        if not m:
            continue
        toks = m.group('head').split()
        if len(toks) < 5 or not toks[0].isdigit():
            continue
        if not (len(toks) > 2 and toks[2].endswith('수어')):
            continue
        rows_total += 1
        headword = toks[4] if len(toks) > 4 else None   # 순 대분류(2) 중분류 표제어
        if headword in classes:
            hits.setdefault(headword, []).append(
                {'seq': int(toks[0]), 'form': ' '.join(toks[5:]) or None,
                 'handshape': m.group('sh'), 'location': m.group('loc'),
                 'hands': int(m.group('hands')), 'same_diff': int(m.group('sd')),
                 'printed': i + 128})
doc.close()
codes = {w: sorted({r['handshape'] for r in v}) for w, v in hits.items()}
json.dump({'rows_parsed_total': rows_total, 'per_word': hits, 'codes': codes},
          open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('parsed table rows:', rows_total)
print('classes with rows:', len(hits), '/', len(classes))
print('single code:', sum(1 for s in codes.values() if len(s) == 1),
      '| multi code:', sum(1 for s in codes.values() if len(s) > 1))
