"""Step-2 robustness analyses for the paper, computed on the cleaned dataset produced by the notebook.

1. Static concentration (CR_k on 1/5/10/20% of groups, Gini) for comparison with Müller and Yannikos (2026)
2. Brand lineages: HHI and survival with documented rebrands merged (strict and broad definitions)
3. Survival sensitivity to the exit threshold (60/90/180 days) and to the minimum brand size
4. Pooled event study of seven law-enforcement operations, with randomisation inference
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times

D = Path(__file__).parent
TAB = D / "tables"; FIG = D / "figures"
df = pd.read_csv(D / "data/victims_clean.csv.gz", parse_dates=["discovered"], low_memory=False)
df["month"] = df["discovered"].dt.to_period("M")
df["quarter"] = df["discovered"].dt.to_period("Q")
df["year"] = df["discovered"].dt.year
OBS_END = df["discovered"].max()
START = pd.Timestamp("2022-01-01")
out = {}

def hhi(counts):
    sh = counts / counts.sum()
    return float((sh ** 2).sum() * 10_000)

# ---------------------------------------------------------------- 1. CR_k
def cr_k(s, frac):
    c = s.value_counts()
    k = max(1, int(round(frac * len(c))))
    return k, float(c.head(k).sum() / c.sum())

def gini(s):
    x = np.sort(s.value_counts().values.astype(float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))

crk = {"whole_period": {"n_groups": int(df["group"].nunique()), "gini": gini(df["group"])}}
for f in (0.01, 0.05, 0.10, 0.20):
    k, v = cr_k(df["group"], f)
    crk["whole_period"][f"top{int(f*100)}pct"] = {"k": k, "share": round(v, 4)}
# fixed k = 15 (Müller and Yannikos: "15 groups exceed 50%")
crk["whole_period"]["cr15"] = round(float(df["group"].value_counts().head(15).sum() / len(df)), 4)
crk["whole_period"]["groups_for_50pct"] = int((df["group"].value_counts().cumsum() / len(df) < 0.5).sum() + 1)
rows = []
for y, g in df.groupby("year"):
    r = {"year": int(y), "n_groups": int(g["group"].nunique()), "gini": round(gini(g["group"]), 3)}
    for f in (0.01, 0.05, 0.10, 0.20):
        k, v = cr_k(g["group"], f)
        r[f"k_{int(f*100)}pct"] = k; r[f"cr_{int(f*100)}pct"] = round(v, 4)
    r["cr3"] = round(float(g["group"].value_counts().head(3).sum() / len(g)), 4)
    rows.append(r)
crk["yearly"] = rows
out["cr_k"] = crk
pd.DataFrame(rows).to_csv(TAB / "crk_yearly.csv", index=False)

# ---------------------------------------------------------------- 2. Lineages
# Rebrands of the same operation, as documented in public reporting (sources in the paper).
# strict: confirmed by government advisories or by several independent analyses
# broad: adds attributions made with moderate confidence, contested links and code purchases
LINEAGE_STRICT = {
    "blacksuit": "royal",       # FBI/CISA AA23-061A (update 7 Aug 2024)
    "knight": "cyclops",        # Knight = rebranded Cyclops (Symantec 2024)
    "blacklock": "eldorado",    # BlackLock = Eldorado rebrand (Resecurity 2025; Infosecurity 2025)
    "worldleaks": "hunters",    # Hunters International -> World Leaks (Group-IB 2025)
}
LINEAGE_BROAD = {
    **LINEAGE_STRICT,
    "chaos": "royal",           # Chaos likely formed by ex-BlackSuit members (Cisco Talos 2025, moderate confidence)
    "hunters": "hive",          # Hunters International linked to Hive (Group-IB; denied by the group)
    "ransomhub": "cyclops",     # RansomHub built on purchased Knight code (Symantec 2024)
    "lynx": "incransom",        # Lynx reuses INC code (Unit 42 2024)
    "sinobi": "incransom",      # Sinobi likely a Lynx rebrand (eSentire, Blackpoint 2025)
}

def resolve(mapping):
    def f(g):
        seen = set()
        while g in mapping and g not in seen:
            seen.add(g); g = mapping[g]
        return g
    return f

lin = {}
qhhi = {}
for name, mp in {"brand": {}, "strict": LINEAGE_STRICT, "broad": LINEAGE_BROAD}.items():
    gcol = df["group"].map(resolve(mp))
    q = df.assign(g=gcol).groupby(["quarter", "g"]).size().groupby(level=0).apply(hhi)
    qhhi[name] = q
    d = pd.DataFrame({"log_hhi": np.log(q.values), "t": np.arange(len(q))})
    m = smf.ols("log_hhi ~ t", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    per = np.where(q.index < pd.Period("2024Q1", "Q"), "2022-2023", "2024-2026")
    lin[name] = {
        "n_units": int(gcol.nunique()),
        "hhi_mean_2022_2023": round(float(q[per == "2022-2023"].mean()), 0),
        "hhi_mean_2024_2026": round(float(q[per == "2024-2026"].mean()), 0),
        "hhi_max_2024_2026": round(float(q[per == "2024-2026"].max()), 0),
        "trend_pct_per_year": round(float((np.exp(4 * m.params["t"]) - 1) * 100), 1),
        "trend_p": float(m.pvalues["t"]),
    }
out["lineage_hhi"] = lin
pd.DataFrame(qhhi).round(0).rename_axis("quarter").to_csv(TAB / "hhi_quarterly_lineages.csv")

# ---------------------------------------------------------------- 3. Survival sensitivity
def survival(groups, exit_days, min_victims):
    life = df.assign(g=groups).groupby("g").agg(first=("discovered", "min"), last=("discovered", "max"),
                                                victims=("victim_key", "size"))
    life = life[(life["victims"] >= min_victims) & (life["first"] >= START + pd.DateOffset(months=1))].copy()
    life["dur_m"] = (life["last"] - life["first"]).dt.days.clip(lower=1) / 30.44
    life["exited"] = (OBS_END - life["last"]).dt.days > exit_days
    k = KaplanMeierFitter().fit(life["dur_m"], life["exited"])
    ci = median_survival_times(k.confidence_interval_)
    a = life[life["first"] < pd.Timestamp("2024-01-01")]; b = life[life["first"] >= pd.Timestamp("2024-01-01")]
    ka = KaplanMeierFitter().fit(a["dur_m"], a["exited"]); kb = KaplanMeierFitter().fit(b["dur_m"], b["exited"])
    return {"n": len(life), "exits": int(life["exited"].sum()),
            "median_months": round(float(k.median_survival_time_), 1),
            "median_ci95": [round(float(v), 1) for v in ci.values.ravel()],
            "surv_12m": round(float(k.predict(12)), 3),
            "median_2022_2023": round(float(ka.median_survival_time_), 1),
            "median_2024_2026": round(float(kb.median_survival_time_), 1),
            "logrank_p": round(float(logrank_test(a["dur_m"], b["dur_m"], a["exited"], b["exited"]).p_value), 3)}

rows = []
for unit, mp in {"brand": {}, "strict": LINEAGE_STRICT, "broad": LINEAGE_BROAD}.items():
    g = df["group"].map(resolve(mp))
    for ed in (60, 90, 180):
        for mv in (3, 5, 10):
            if unit != "brand" and mv != 3:
                continue
            rows.append({"unit": unit, "exit_days": ed, "min_victims": mv, **survival(g, ed, mv)})
sens = pd.DataFrame(rows)
sens.to_csv(TAB / "survival_sensitivity.csv", index=False)
out["survival_sensitivity"] = rows

# ---------------------------------------------------------------- 4. Multi-takedown event study
# Dates of the public announcement / seizure of the leak site (sources in the paper).
TAKEDOWNS = {
    "hive":         ("Hive",          "2023-01-26"),
    "ragnarlocker": ("RagnarLocker",  "2023-10-19"),
    "alphv":        ("ALPHV/BlackCat","2023-12-19"),
    "lockbit":      ("LockBit",       "2024-02-20"),
    "dispossessor": ("Dispossessor",  "2024-08-12"),
    "8base":        ("8Base",         "2025-02-10"),
    "blacksuit":    ("BlackSuit",     "2025-07-24"),
}
H = 6
ev_months_ = [pd.Period(d, "M") for _, d in TAKEDOWNS.values()]
months = pd.period_range("2022-01", "2026-09", freq="M")
cnt = df.groupby(["month", "group"]).size().unstack(fill_value=0).reindex(months, fill_value=0)
fc = df[df["first_claim"]]
first_post = df.groupby("group")["discovered"].min()

def window_delta(series, m0):
    """Mean of months +1..+H minus mean of months -H..-1 (month 0 excluded)."""
    i = months.get_loc(m0)
    if i - H < 0 or i + H >= len(months):
        return np.nan
    return float(series.iloc[i + 1:i + H + 1].mean() - series.iloc[i - H:i].mean())

series = {
    "log_total": np.log(cnt.sum(axis=1)),
    "log_total_excl_clop": np.log(cnt.drop(columns="clop").sum(axis=1)),
    "log_active_groups": np.log((cnt > 0).sum(axis=1)),
}
def share_gains(m0):
    """Share-point gains in first claims, months +1..+H vs -H..-1, and whether each gainer is a new brand
    (first post less than H months before the event, or after it)."""
    mo = fc["month"]
    a = fc[(mo >= m0 - H) & (mo < m0)]["group"].value_counts(normalize=True)
    b = fc[(mo > m0) & (mo <= m0 + H)]["group"].value_counts(normalize=True)
    idx = a.index.union(b.index)
    dd = (b.reindex(idx, fill_value=0) - a.reindex(idx, fill_value=0)) * 100
    gains = dd[dd > 0]
    new = first_post.reindex(gains.index) >= m0.to_timestamp() - pd.DateOffset(months=H)
    return gains, new

ev_rows = []
for tgt, (label, date) in TAKEDOWNS.items():
    m0 = pd.Period(date, "M")
    i = months.get_loc(m0)
    pre = slice(i - H, i); post = slice(i + 1, i + H + 1)
    tot = cnt.sum(axis=1)
    others = tot - cnt[tgt]
    r = {"target": label, "date": date,
         "target_claims_pre6_per_month": round(float(cnt[tgt].iloc[pre].mean()), 1),
         "target_claims_post6_per_month": round(float(cnt[tgt].iloc[post].mean()), 1),
         "target_share_pre6": round(float(cnt[tgt].iloc[pre].sum() / tot.iloc[pre].sum()), 4),
         "target_share_post6": round(float(cnt[tgt].iloc[post].sum() / tot.iloc[post].sum()), 4),
         "d_log_total": round(window_delta(series["log_total"], m0), 3),
         "d_log_others": round(window_delta(np.log(others), m0), 3),
         "d_log_total_excl_clop": round(window_delta(series["log_total_excl_clop"], m0), 3)}
    gains, new = share_gains(m0)
    r["gains_pp_total"] = round(float(gains.sum()), 1)
    r["gains_share_new_brands"] = round(float(gains[new].sum() / gains.sum()), 3)
    r["top_gainer"] = gains.idxmax(); r["top_gainer_pp"] = round(float(gains.max()), 1)
    ev_rows.append(r)
ev = pd.DataFrame(ev_rows)
ev.to_csv(TAB / "takedown_events.csv", index=False)

# randomisation inference: compare the mean delta over the 7 events with the mean over 7 random months
rng = np.random.default_rng(20261024)
ev_months = ev_months_
ri = {}
for name, s in series.items():
    all_d = pd.Series({m: window_delta(s, m) for m in months}).dropna()
    obs = float(np.mean([window_delta(s, m) for m in ev_months]))
    pool = all_d[~all_d.index.isin(ev_months)].values
    draws = np.array([rng.choice(pool, size=len(ev_months), replace=False).mean() for _ in range(20_000)])
    ri[name] = {"mean_delta_events": round(obs, 3), "mean_delta_all_months": round(float(all_d.mean()), 3),
                "p_two_sided": round(float(np.mean(np.abs(draws - draws.mean()) >= abs(obs - draws.mean()))), 3),
                "n_placebo_months": int(len(pool))}
# baseline: share of gains going to new brands around every month with a full window
base = []
for m in months[H:len(months) - H]:
    if m in ev_months_:
        continue
    g, n = share_gains(m)
    base.append(g[n].sum() / g.sum())
ri["gains_share_new_brands"] = {
    "mean_events": round(float(ev["gains_share_new_brands"].mean()), 3),
    "mean_other_months": round(float(np.mean(base)), 3),
    "p_two_sided": round(float(np.mean([abs(np.mean(rng.choice(base, len(ev_months_), replace=False)) - np.mean(base))
                                        >= abs(ev["gains_share_new_brands"].mean() - np.mean(base)) for _ in range(20_000)])), 3)}
out["takedown_events"] = ev_rows
out["takedown_randomisation"] = ri

# event-time path of log total claims (relative to month -1), events vs placebo envelope
def path(s, m0):
    i = months.get_loc(m0)
    if i - H < 0 or i + H >= len(months):
        return None
    v = s.iloc[i - H:i + H + 1].values
    return v - v[H - 1]
s = series["log_total"]
ev_paths = np.array([path(s, m) for m in ev_months])
pl_paths = np.array([p for m in months if m not in ev_months and (p := path(s, m)) is not None])
draw_means = np.array([pl_paths[rng.choice(len(pl_paths), len(ev_months), replace=False)].mean(axis=0)
                       for _ in range(5_000)])
lo, hi = np.percentile(draw_means, [2.5, 97.5], axis=0)
x = np.arange(-H, H + 1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.fill_between(x, lo, hi, color="#cfd8e6", label="95% range, 7 random months")
for p in ev_paths:
    ax.plot(x, p, color="#9aa5b1", lw=0.7)
ax.plot(x, ev_paths.mean(axis=0), color="#e0703a", lw=2, marker="o", ms=3, label="Mean of 7 takedowns")
ax.axvline(0, color="#b5453d", ls="--", lw=1); ax.axhline(0, color="#333", lw=0.6)
ax.set_xlabel("Months relative to the operation"); ax.set_ylabel("log monthly claims (vs month −1)")
ax.set_title("Market-wide claims around seven law-enforcement operations")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"figS_takedown_event_study.{ext}", bbox_inches="tight", dpi=300)
out["event_path_mean"] = dict(zip(map(int, x), np.round(ev_paths.mean(axis=0), 3).tolist()))

(TAB / "robustness_step2.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
