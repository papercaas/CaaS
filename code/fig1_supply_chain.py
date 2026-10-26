import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans"})
stages = [
 ("Developers and RaaS operators", "Build encryptors, leak sites, phishing kits\nand stealers; run the affiliate programme",
  "Sells: RaaS programme, PhaaS kits, infostealers", "Paid via: subscription or a share of each ransom"),
 ("Initial access brokers", "Breach networks with stolen credentials,\nexposed RDP/VPN or unpatched edge devices",
  "Sells: ready-made access to corporate networks", "Paid via: one-off sale per access"),
 ("Affiliates", "Hands-on-keyboard intrusion: lateral\nmovement, data theft, encryption",
  "Buys: tools from developers, access from brokers", "Keeps: the larger share of each ransom"),
 ("Extortion and publication", "Negotiation portal and data leak site;\nvictims who do not pay are named",
  "Leak-site posts are the data analysed in this paper", "Pressure: double and triple extortion"),
 ("Money laundering", "Mixers, cross-chain bridges, exchanges\nturn ransoms into usable funds",
  "Sells: cash-out and laundering services", "Paid via: commission on laundered funds"),
]
fig, ax = plt.subplots(figsize=(8.6, 6.2)); ax.set_xlim(0, 11.5); ax.set_ylim(0, 10.4); ax.axis("off")
h, gap, top = 1.45, 0.5, 9.9
for i, (t, d, r1, r2) in enumerate(stages):
    y = top - i * (h + gap) - h
    hl = i == 3
    ax.add_patch(FancyBboxPatch((0.1, y), 5.3, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                 lw=1.6 if hl else 1.0, ec="#2a6fdb" if hl else "#999", fc="#e5eefb" if hl else "white"))
    ax.text(0.35, y + h - 0.32, t, fontsize=10, weight="bold", va="center")
    ax.text(0.35, y + 0.52, d, fontsize=8.5, va="center", color="#333")
    ax.text(5.8, y + h / 2 + 0.22, r1, fontsize=8.5, va="center")
    ax.text(5.8, y + h / 2 - 0.22, r2, fontsize=8.5, va="center", color="#555")
    if i < len(stages) - 1:
        ax.annotate("", xy=(2.75, y - gap + 0.02), xytext=(2.75, y - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color="#888", lw=1))
ax.text(0.1, 0.05, "Money flows back up the chain: affiliates pay operators a share of each ransom and brokers a price per access.",
        fontsize=8, color="#444")
fig.savefig("figures/Fig1.pdf", bbox_inches="tight")
fig.savefig("figures/Fig1.png", bbox_inches="tight", dpi=300)
