"""The minimal operational change, ready to apply - and NOT applied.

  verify    show whether the guard is present and whether the file matches a known state
  apply     insert the guard into src/wis_lab/capture/landmarker.py, keeping a .bak
  rollback  restore the file from the .bak

The inserted guard is exactly the candidate that was evaluated (post-filter form): a hand keeps a
side only when that side's pose wrist is its OWN nearest usable wrist; otherwise it stays
unassigned through the path the code already has. Nothing is discarded on a distance threshold.

Run: py -3.13 ops/side_assignment_candidate_20260925/operational_change.py verify
"""
import hashlib, pathlib, shutil, sys

H = pathlib.Path(__file__).resolve().parent
TARGET = H.parents[1] / 'src' / 'wis_lab' / 'capture' / 'landmarker.py'
BAK = TARGET.with_suffix('.py.bak_side_assignment')

ANCHOR = """            for i in range(n):
                if i not in out:
                    out[i] = (None, scores[i])  # ambiguous: do not force a side
"""

GUARD = """            if len(pose_ref) > 1:
                # A hand keeps a side only when that side's wrist is its OWN nearest usable wrist.
                # Otherwise it stays unassigned: the greedy pass above has no distance cap, so a
                # second detection of one physical hand used to be pushed onto the idle side and
                # then read as that hand. Nothing is discarded here - the side is simply not
                # claimed, and the mask records the hand as not observed, which is what is true.
                for i, (s, sc) in list(out.items()):
                    if s is None:
                        continue
                    nearest = min(pose_ref,
                                  key=lambda t: float(np.linalg.norm(wrists[i] - pose_ref[t])))
                    if nearest != s:
                        out[i] = (None, sc)
"""

UNPATCHED_SHA = '138f12c6cbe12c94e07abf1751409f1411e1c98b812c4d6cea5c33df9c9158e8'
PATCHED_SHA   = '0c80f1030b29b267c15bebadd8541e81f1001fdd572eaa16bc8bca0ec04548b9'


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def state():
    src = TARGET.read_text(encoding='utf-8')
    return ('APPLIED' if 'OWN nearest usable wrist' in src else 'NOT_APPLIED'), src


def main():
    cmd = (sys.argv[1] if len(sys.argv) > 1 else 'verify').lower()
    st, src = state()
    if cmd == 'verify':
        print(f'target      {TARGET}')
        print(f'state       {st}')
        print(f'sha256      {sha(TARGET)}')
        print(f'backup      {"present" if BAK.exists() else "none"}')
        print(f'anchor found {ANCHOR in src}')
        return 0
    if cmd == 'apply':
        if st == 'APPLIED':
            print('already applied; nothing to do'); return 0
        if ANCHOR not in src:
            print('ERROR: anchor not found - the file has changed. Do not apply blindly.'); return 1
        shutil.copy2(TARGET, BAK)
        TARGET.write_text(src.replace(ANCHOR, ANCHOR + GUARD, 1), encoding='utf-8')
        print(f'applied. backup at {BAK}\nnew sha256 {sha(TARGET)}')
        print('NOW RUN THE REGRESSION CHECKS listed in OPERATIONAL_CHANGE.json')
        return 0
    if cmd == 'rollback':
        if not BAK.exists():
            print('ERROR: no backup to restore from'); return 1
        shutil.copy2(BAK, TARGET)
        print(f'rolled back from {BAK}\nsha256 {sha(TARGET)}')
        return 0
    print('usage: operational_change.py [verify|apply|rollback]')
    return 1


if __name__ == '__main__':
    sys.exit(main())
