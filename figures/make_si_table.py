"""Generate si_table_novel.tex (longtable of all 104 novel candidates) from
novel_mondrian.csv. Run from the manuscript/ directory."""
import re
import pandas as pd

df = pd.read_csv("../v4_results/novel_mondrian.csv")
tier_order = {"A1": 0, "A2": 1, "B": 2}
df = df.sort_values(["tier", "dist_to_pv_centre"],
                    key=lambda s: s.map(tier_order) if s.name == "tier" else s)

def sub(f):
    return re.sub(r"(\d+)", r"$_{\1}$", f)

rows = []
for _, r in df.iterrows():
    rows.append(
        f"{sub(r.formula)} & {r.e_class} & {r.bg_pred:.2f} & "
        f"[{max(r.bg_ci_lo, 0):.2f}, {r.bg_ci_hi:.2f}] & "
        f"[{max(r.mondrian_lo, 0):.2f}, {r.mondrian_hi:.2f}] & "
        f"{r.tier} & {r.mondrian_tier} \\\\")

with open("si_table_novel.tex", "w") as f:
    f.write("\n".join(rows) + "\n")
print(f"wrote si_table_novel.tex ({len(rows)} rows)")
