import json, re, sys
import numpy as np, pandas as pd
rw = pd.DataFrame(json.load(open(sys.argv[1])))
rl = pd.read_csv(sys.argv[2], parse_dates=["discovered"], low_memory=False)
ALIAS = [(r"^lockbit.*","lockbit"),(r"^(alphv|blackcat).*","alphv"),(r"^ransomhub.*","ransomhub"),(r"^everest.*","everest")]
def ng(g):
    g=str(g).strip().lower()
    for p,c in ALIAS:
        if re.match(p,g): return c
    return g
RWMAP={"hiveleak":"hive","killsecurity":"killsec","ragroup":"raworld","bashe":"apt73","ransomedvc":"ransomed","dunghill_leak":"dunghill","unsafeleak":"unsafe"}
rw["group"]=rw["group_name"].map(ng).replace(RWMAP)
rw["discovered"]=pd.to_datetime(rw["discovered"],errors="coerce")
rw["key"]=rw["post_title"].astype(str).str.lower().str.replace(r"^https?://(www\.)?","",regex=True).str.replace(r"[^a-z0-9]","",regex=True)
rw=rw.dropna(subset=["discovered"]).sort_values("discovered").drop_duplicates(["group","key"])
lo,hi=pd.Timestamp("2022-01-01"),pd.Timestamp(sys.argv[4])
rw=rw[(rw.discovered>=lo)&(rw.discovered<hi)]
rl=rl[(rl.discovered>=lo)&(rl.discovered<hi)].copy()
print(f"claims {lo.date()}..{hi.date()} (excl.): ransomware.live",len(rl),"ransomwatch",len(rw))
print("groups: rl",rl.group.nunique(),"rw",rw.group.nunique(),"common",len(set(rl.group)&set(rw.group)))
print("rw groups not in rl (top):", rw[~rw.group.isin(rl.group)].group.value_counts().head(15).to_dict())
print("rl groups not in rw (top):", rl[~rl.group.isin(rw.group)].group.value_counts().head(15).to_dict())
# victim-level overlap: same group and same normalized name
rl["key"]=rl["victim"].astype(str).str.lower().str.replace(r"^https?://(www\.)?","",regex=True).str.replace(r"[^a-z0-9]","",regex=True)
k_rw=set(zip(rw.group,rw.key)); k_rl=set(zip(rl.group,rl.key))
print("share of rl claims matched in rw (same group+name):", round(np.mean([x in k_rw for x in zip(rl.group,rl.key)]),3))
print("share of rw claims matched in rl:", round(np.mean([x in k_rl for x in zip(rw.group,rw.key)]),3))
def qh(d):
    q=d.groupby([d.discovered.dt.to_period("Q"),"group"]).size()
    sh=q/q.groupby(level=0).transform("sum")
    return (sh**2).groupby(level=0).sum()*1e4, q.groupby(level=0).sum(), d.groupby(d.discovered.dt.to_period("Q")).group.nunique(), q.groupby(level=0).apply(lambda s: s.droplevel(0).idxmax())
h1,n1,g1,l1=qh(rl); h2,n2,g2,l2=qh(rw)
t=pd.DataFrame({"claims_rl":n1,"claims_rw":n2,"groups_rl":g1,"groups_rw":g2,"hhi_rl":h1.round(0),"hhi_rw":h2.round(0),"leader_rl":l1,"leader_rw":l2})
print(t.to_string())
from scipy.stats import spearmanr, pearsonr
print("pearson hhi:", pearsonr(t.hhi_rl,t.hhi_rw), "spearman:", spearmanr(t.hhi_rl,t.hhi_rw))
per=np.where(t.index<pd.Period("2024Q1","Q"),"2022-2023","2024+")
print(t.groupby(per)[["hhi_rl","hhi_rw"]].mean().round(0))
d=pd.DataFrame({"log_hhi":np.log(t.hhi_rw.values),"t":np.arange(len(t))})
import statsmodels.formula.api as smf
m=smf.ols("log_hhi~t",d).fit(cov_type="HAC",cov_kwds={"maxlags":2}); print("rw trend %/yr", (np.exp(4*m.params.t)-1)*100, m.pvalues.t)
d["log_hhi"]=np.log(t.hhi_rl.values); m=smf.ols("log_hhi~t",d).fit(cov_type="HAC",cov_kwds={"maxlags":2}); print("rl same window trend %/yr", (np.exp(4*m.params.t)-1)*100, m.pvalues.t)
t.to_csv(sys.argv[3])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.6))
x = t.index.to_timestamp()
ax.plot(x, t.hhi_rl, marker="o", ms=3, color="#2a6fdb", label="ransomware.live")
ax.plot(x, t.hhi_rw, marker="s", ms=3, color="#e0703a", label="ransomwatch")
ax.axhline(1800, color="#b5453d", lw=0.8, ls=":")
ax.set_ylabel("Quarterly HHI"); ax.set_title("Quarterly HHI on two leak-site trackers")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
for ext in ("png", "pdf"):
    fig.savefig(sys.argv[3].replace(".csv", f".{ext}").replace("tables", "figures"), bbox_inches="tight", dpi=300)
