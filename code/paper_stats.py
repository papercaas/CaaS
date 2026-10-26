"""Extra statistics for the paper, computed on the cleaned dataset produced by the notebook."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr

D = Path(__file__).parent
df = pd.read_csv(D / "data/victims_clean.csv.gz", parse_dates=["discovered"], low_memory=False)
df["month"] = df["discovered"].dt.to_period("M")
df["quarter"] = df["discovered"].dt.to_period("Q")
df["year"] = df["discovered"].dt.year
out = {}

def hhi(s):
    sh = s / s.sum()
    return float((sh ** 2).sum() * 10_000)

# 1. Yearly overview
yr = df.groupby("year").agg(victims=("victim_key", "size"), groups=("group", "nunique"))
def groups_for_half(g):
    c = g["group"].value_counts()
    return int((c.cumsum() / c.sum() < 0.5).sum() + 1)
yr["groups_for_50pct"] = df.groupby("year").apply(groups_for_half)
yr["hhi"] = df.groupby("year")["group"].apply(lambda s: hhi(s.value_counts()))
yr["top_group"] = df.groupby("year")["group"].agg(lambda s: s.value_counts().index[0])
yr["top_share"] = df.groupby("year")["group"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
out["yearly"] = yr.round(3).reset_index().to_dict("records")

# 2. Quarterly HHI: trend and robustness
qs = df.groupby(["quarter", "group"]).size()
big = df["group"].value_counts()
big = big[big >= 50].index
rob = pd.DataFrame({
    "hhi_all": qs.groupby(level=0).apply(hhi),
    "hhi_first_claims": df[df["first_claim"]].groupby(["quarter", "group"]).size().groupby(level=0).apply(hhi),
    "hhi_groups_ge50": df[df["group"].isin(big)].groupby(["quarter", "group"]).size().groupby(level=0).apply(hhi),
    "hhi_excl_clop": df[df["group"] != "clop"].groupby(["quarter", "group"]).size().groupby(level=0).apply(hhi),
})
rob["t"] = np.arange(len(rob))
rob["log_hhi"] = np.log(rob["hhi_all"])
m = smf.ols("log_hhi ~ t", data=rob).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
out["hhi_trend"] = {"coef_per_quarter": m.params["t"], "pct_per_year": (np.exp(4 * m.params["t"]) - 1) * 100,
                    "p": m.pvalues["t"], "n_quarters": len(rob)}
rob["period"] = np.where(rob.index < pd.Period("2024Q1", "Q"), "2022-2023", "2024-2026")
out["hhi_by_period"] = rob.groupby("period")[["hhi_all", "hhi_first_claims", "hhi_groups_ge50", "hhi_excl_clop"]].mean().round(0).to_dict()
out["hhi_quarterly_robust"] = rob.drop(columns=["t", "log_hhi"]).round(0).reset_index().astype({"quarter": str}).to_dict("records")

# 3. Brand survival by entry cohort
obs_end = df["discovered"].max()
life = df.groupby("group").agg(first=("discovered", "min"), last=("discovered", "max"), victims=("victim_key", "size"))
life = life[life["victims"] >= 3].copy()
life["dur_m"] = ((life["last"] - life["first"]).dt.days.clip(lower=1)) / 30.44
life["exited"] = (obs_end - life["last"]).dt.days > 90
life = life[life["first"] >= pd.Timestamp("2022-02-01")]
life["cohort"] = np.where(life["first"] < pd.Timestamp("2024-01-01"), "2022-2023", "2024-2026")
km = {}
for c, g in life.groupby("cohort"):
    k = KaplanMeierFitter().fit(g["dur_m"], g["exited"])
    km[c] = {"n": len(g), "exits": int(g["exited"].sum()), "median_months": float(k.median_survival_time_),
             "surv_12m": float(k.predict(12))}
a, b = life[life.cohort == "2022-2023"], life[life.cohort == "2024-2026"]
lr = logrank_test(a["dur_m"], b["dur_m"], a["exited"], b["exited"])
km["logrank_p"] = float(lr.p_value)
kall = KaplanMeierFitter().fit(life["dur_m"], life["exited"])
km["all"] = {"n": len(life), "median_months": float(kall.median_survival_time_),
             "ci": kall.confidence_interval_median_survival_time_.values.tolist() if hasattr(kall, "confidence_interval_median_survival_time_") else None}
out["survival"] = km

# 4. New-entrant share: victims by brands whose first post was < 12 months earlier
first = df.groupby("group")["discovered"].transform("min")
df["young_brand"] = (df["discovered"] - first).dt.days < 365
out["young_brand_share_by_year"] = df[df["year"] >= 2023].groupby("year")["young_brand"].mean().round(3).to_dict()

# 5. LockBit around Operation Cronos
ev = pd.Period("2024-02", "M")
lb = df[df["group"] == "lockbit"].groupby("month").size().reindex(pd.period_range("2023-02", "2025-02", freq="M"), fill_value=0)
tot = df.groupby("month").size().reindex(lb.index)
rel = np.array([(p - ev).n for p in lb.index])
out["lockbit"] = {
    "mean_monthly_pre6": float(lb[(rel >= -6) & (rel < 0)].mean()),
    "mean_monthly_post1_6": float(lb[(rel >= 1) & (rel <= 6)].mean()),
    "mean_monthly_post7_12": float(lb[(rel >= 7) & (rel <= 12)].mean()),
    "share_pre6": float(lb[(rel >= -6) & (rel < 0)].sum() / tot[(rel >= -6) & (rel < 0)].sum()),
    "share_post1_6": float(lb[(rel >= 1) & (rel <= 6)].sum() / tot[(rel >= 1) & (rel <= 6)].sum()),
    "share_post7_12": float(lb[(rel >= 7) & (rel <= 12)].sum() / tot[(rel >= 7) & (rel <= 12)].sum()),
    "peak_month_post": str(lb[rel > 0].idxmax()), "peak_value_post": int(lb[rel > 0].max()),
}
# raw reposts by lockbit (same victim posted twice by lockbit) before vs after
raw = []
for f in sorted((D / "data/raw").glob("*.json")):
    raw += json.loads(f.read_text())
r = pd.DataFrame(raw)
r = r[r["group"].astype(str).str.lower().str.startswith("lockbit")].copy()
r["d"] = pd.to_datetime(r["discovered"], errors="coerce", utc=True).dt.tz_localize(None)
r["k"] = r["victim"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
r = r.sort_values("d")
r["dup"] = r.duplicated("k")
r["post"] = r["d"] >= pd.Timestamp("2024-02-20")
out["lockbit_raw_repost_share"] = r.groupby("post")["dup"].mean().round(3).to_dict()

# 6. ITS on total monthly victims with and without Clop; placebo
def its(data, shift=0):
    d = data.copy()
    d["t"] = d["rel"] + shift
    d["post"] = (d["t"] >= 0).astype(int)
    d["t_post"] = d["t"] * d["post"]
    res = smf.ols("y ~ t + post + t_post", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return {"post": res.params["post"], "p_post": res.pvalues["post"], "t_post": res.params["t_post"], "p_t_post": res.pvalues["t_post"]}
win = pd.period_range("2023-02", "2025-02", freq="M")
res_its = {}
for name, sub in {"all": df, "excl_clop": df[df["group"] != "clop"]}.items():
    y = sub.groupby("month").size().reindex(win, fill_value=0)
    d = pd.DataFrame({"y": np.log(y.values), "rel": [(p - ev).n for p in win]})
    res_its[name] = its(d)
    res_its[name + "_placebo_-6"] = its(d[d["rel"] < 0], shift=6)
out["its_log_total"] = {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in res_its.items()}

# 7. Italy and EU
eu = set("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split())
known = df.dropna(subset=["country"])
it = known[known["country"] == "IT"]
out["italy"] = {
    "total": len(it), "share_world": round(len(it) / len(known), 4),
    "share_of_eu": round(len(it) / known["country"].isin(eu).sum(), 4),
    "rank_world": int(known["country"].value_counts().index.get_loc("IT") + 1),
    "rank_eu": int(known[known["country"].isin(eu)]["country"].value_counts().index.get_loc("IT") + 1),
    "top_groups": it["group"].value_counts().head(6).to_dict(),
    "top_sectors": it["sector"].value_counts(normalize=True).head(5).round(3).to_dict(),
    "by_year": it.groupby("year").size().to_dict(),
}
out["country_missing_by_year"] = df.groupby("year")["country"].apply(lambda s: s.isna().mean()).round(3).to_dict()
out["sector_missing_by_year"] = df.groupby("year")["sector"].apply(lambda s: s.isna().mean()).round(3).to_dict()

# 8. Sector specialisation vs size
spec = pd.read_csv(D / "tables/sector_specialisation.csv", index_col=0)
rho, p = spearmanr(spec["victims_with_sector"], spec["js_divergence"])
out["specialisation"] = {"n_groups": len(spec), "median_js": float(spec["js_divergence"].median()),
                         "iqr_js": spec["js_divergence"].quantile([.25, .75]).round(3).tolist(),
                         "spearman_size_vs_js": float(rho), "p": float(p),
                         "top_sectors_market": df["sector"].value_counts(normalize=True).head(6).round(3).to_dict()}

# 9. Repost stats
out["reposts"] = {"share_claims_not_first": round(1 - df["first_claim"].mean(), 4),
                  "share_victims_multi_group": round(df["claimed_by_multiple_groups"].mean(), 4),
                  "n_victims": len(df), "n_groups": int(df["group"].nunique()),
                  "period": [str(df["discovered"].min().date()), str(df["discovered"].max().date())]}

(D / "tables/paper_stats.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
