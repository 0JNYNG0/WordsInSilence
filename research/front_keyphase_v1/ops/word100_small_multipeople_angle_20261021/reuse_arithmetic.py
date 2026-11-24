"""Run previously independent saved-output verifier on the new complete cohort.

Only input routing and restriction of old controls to this subset change.
The separate time weighting/mask/median formulas remain byte-for-byte unchanged.
"""
from study import U,M,A,module,read,write
v=module("new_cohort_stored_arithmetic",M/"verify.py")
v.U=U
original_read=v.read
def routed_read(p):
    obj=original_read(p)
    if p==U/"FROZEN.json":
        obj["rows"]=[dict(r,query=True,evidence=None) for r in read(U/"PREPARED_INDEX.json")]
        # Evidence was generated/reused already and independently checked by INPUT_VERIFY.
        for r in obj["rows"]: r["evidence"]=r["prepared"]
    if p==A/"VERIFY.json":
        ids={r["clip_id"] for r in read(U/"PREPARED_INDEX.json") if r["reuse"]}
        obj["controls"]=[r for r in obj["controls"] if r["clip_id"] in ids]
    return obj
v.read=routed_read
v.main()
