"""Standalone scientific figure and static report validation, no browser automation."""
import json
from pathlib import Path
from html.parser import HTMLParser
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from run import U, read, load, write
plt.rcParams["font.family"]="Malgun Gothic"
plt.rcParams["axes.unicode_minus"]=False

rows=read(U/"INDEX.json")
chosen=next(r for r in rows if any(a["compared_median_deg"] is not None for a in read(U/"queries"/(r["clip_id"]+".json"))["rows"]))
cid=chosen["clip_id"];q=load(U/"prepared"/(cid+".npz"));v=load(U/"queries"/(cid+".npz"))
t=(q["timestamp_us"]-q["timestamp_us"][0])/1e6
fig,axes=plt.subplots(2,1,figsize=(13,8),layout="constrained",sharex=True)
cmap=ListedColormap(["#b9c3cd","#e7b45d","#168377"])
norm=BoundaryNorm([-.5,.5,4.5,9.5],3)
names=[f"{f}{j}" for f in ("엄지","검지","중지","약지","소지") for j in (1,2,3)]
for h,ax in enumerate(axes):
    arr=v["votes"][:,h].T.astype(float)
    arr[:,~q["core"]]=np.nan
    im=ax.imshow(arr,aspect="auto",origin="upper",cmap=cmap,norm=norm,
                 extent=(t[0],t[-1],14.5,-.5),interpolation="nearest")
    ax.set_yticks(range(15),names,fontsize=9)
    ax.set_title(("왼손" if h==0 else "오른손")+" · 각도마다 다른 참고 근거",loc="left")
    ax.set_facecolor("#f3f5f7")
axes[-1].set_xlabel("원본 영상 시간 (초)")
fig.suptitle(f'{chosen["label"]} / 공개 촬영자 {chosen["signer_id"]}\n초록: 5명 이상 비교 · 노랑: 1~4명 · 회색: 참고 근거 없음 · 흰색: 주석 구간 밖',fontsize=14)
fig.savefig(U/"ANGLE_TIMELINE.png",dpi=150)
plt.close(fig)

class Audit(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.refs=[];self.external=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if "id" in d:self.ids.append(d["id"])
        for k in ("href","src"):
            if k in d:
                value=d[k]
                if value.startswith(("https:","http:")):self.external.append(value)
                elif not value.startswith("#"):self.refs.append(value)

h=(U/"RESULT_KO.html").read_text("utf8");p=Audit();p.feed(h)
assert len(p.ids)==len(set(p.ids))
for ref in p.refs:assert (U/ref).exists(),ref
assert not p.external
start=h.index("<script>",h.index("report_data.js"))+len("<script>")
end=h.index("</script>",start)
(U/"report_ui_check.js").write_text(h[start:end],encoding="utf8")
dataset=json.loads((U/"report_data.js").read_text("utf8").removeprefix("const DATA=").removesuffix(";"))
assert len(dataset)==136
for d in dataset:
    assert len(d["rows"])==30 and len(d["ts"])==len(d["votes"])==len(d["state"])==len(d["core"])
write(U/"STATIC_REPORT_CHECK.json",dict(status="PASS",ids=p.ids,links=p.refs,external_resources=0,
    dataset_clips=136,figure=str(U/"ANGLE_TIMELINE.png"),
    browser_preview="Blocked by browser file: URL policy; no workaround attempted. Static structure/data and separate scientific figure checked only."))
print("STATIC PASS",cid)
