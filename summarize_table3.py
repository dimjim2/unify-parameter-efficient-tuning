from pathlib import Path
import csv

results = [
    {
        "Method": "Prefix, $l=200$",
        "Params": "3.6\\%",
        "Paper_R1": 43.40,
        "Reproduced_R1": 43.1582,
        "Paper_R2": 20.46,
        "Reproduced_R2": 20.1669,
        "Paper_RL": 35.51,
        "Reproduced_RL": 35.2358,
        "Paper_BLEU": 35.6,
        "Reproduced_BLEU": 34.6,
    },
    {
        "Method": "SA (attn), $r=200$",
        "Params": "3.6\\%",
        "Paper_R1": 42.01,
        "Reproduced_R1": 43.8399,
        "Paper_R2": 19.30,
        "Reproduced_R2": 20.7052,
        "Paper_RL": 34.40,
        "Reproduced_RL": 35.8797,
        "Paper_BLEU": 35.3,
        "Reproduced_BLEU": 35.3,
    },
    {
        "Method": "SA (ffn), $r=200$",
        "Params": "2.4\\%",
        "Paper_R1": 43.21,
        "Reproduced_R1": 43.3394,
        "Paper_R2": 19.98,
        "Reproduced_R2": 20.0882,
        "Paper_RL": 35.08,
        "Reproduced_RL": 35.2254,
        "Paper_BLEU": 35.6,
        "Reproduced_BLEU": 35.8,
    },
    {
        "Method": "PA (attn), $r=200$",
        "Params": "3.6\\%",
        "Paper_R1": 43.58,
        "Reproduced_R1": 43.6400,
        "Paper_R2": 20.31,
        "Reproduced_R2": 20.3583,
        "Paper_RL": 35.34,
        "Reproduced_RL": 35.4919,
        "Paper_BLEU": 35.6,
        "Reproduced_BLEU": 35.4,
    },
    {
        "Method": "PA (ffn), $r=200$",
        "Params": "2.4\\%",
        "Paper_R1": 43.93,
        "Reproduced_R1": 43.9515,
        "Paper_R2": 20.66,
        "Reproduced_R2": 20.5833,
        "Paper_RL": 35.63,
        "Reproduced_RL": 35.6944,
        "Paper_BLEU": 36.4,
        "Reproduced_BLEU": 36.3,
    },
]

out = Path("results")
out.mkdir(exist_ok=True)


def clean_terminal(text):
    return (
        text.replace("$", "")
        .replace("\\%", "%")
    )


# --------------------------------------------------
# MASTER CSV
# --------------------------------------------------

csv_path = out / "table3_all.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# TERMINAL: XSUM
# --------------------------------------------------

print("\nTABLE 3 — XSUM")
print("(Paper / Reproduced)\n")

print(
    f"{'Method':<24}"
    f"{'# Params':>10}"
    f"{'R-1':>18}"
    f"{'R-2':>18}"
    f"{'R-L':>18}"
)

print("-" * 88)

for r in results:
    print(
        f"{clean_terminal(r['Method']):<24}"
        f"{clean_terminal(r['Params']):>10}"
        f"{r['Paper_R1']:>7.2f} / {r['Reproduced_R1']:<7.2f}"
        f"{r['Paper_R2']:>7.2f} / {r['Reproduced_R2']:<7.2f}"
        f"{r['Paper_RL']:>7.2f} / {r['Reproduced_RL']:<7.2f}"
    )


# --------------------------------------------------
# TERMINAL: MT
# --------------------------------------------------

print("\nTABLE 3 — WMT16 EN→RO")
print("(Paper / Reproduced)\n")

print(
    f"{'Method':<24}"
    f"{'# Params':>10}"
    f"{'BLEU':>18}"
)

print("-" * 52)

for r in results:
    print(
        f"{clean_terminal(r['Method']):<24}"
        f"{clean_terminal(r['Params']):>10}"
        f"{r['Paper_BLEU']:>7.1f} / {r['Reproduced_BLEU']:<7.1f}"
    )


# --------------------------------------------------
# MARKDOWN: XSUM
# --------------------------------------------------

xsum_md = [
    "| Method | # Params | R-1 | R-2 | R-L |",
    "|---|---:|---:|---:|---:|",
]

for r in results:
    xsum_md.append(
        f"| {clean_terminal(r['Method'])} "
        f"| {clean_terminal(r['Params'])} "
        f"| {r['Paper_R1']:.2f} / {r['Reproduced_R1']:.2f} "
        f"| {r['Paper_R2']:.2f} / {r['Reproduced_R2']:.2f} "
        f"| {r['Paper_RL']:.2f} / {r['Reproduced_RL']:.2f} |"
    )

(out / "table3_xsum.md").write_text("\n".join(xsum_md) + "\n")


# --------------------------------------------------
# MARKDOWN: MT
# --------------------------------------------------

mt_md = [
    "| Method | # Params | BLEU |",
    "|---|---:|---:|",
]

for r in results:
    mt_md.append(
        f"| {clean_terminal(r['Method'])} "
        f"| {clean_terminal(r['Params'])} "
        f"| {r['Paper_BLEU']:.1f} / {r['Reproduced_BLEU']:.1f} |"
    )

(out / "table3_mt.md").write_text("\n".join(mt_md) + "\n")


# --------------------------------------------------
# LATEX: XSUM
# --------------------------------------------------

xsum_tex = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\caption{XSum results for the insertion forms from Table~3. Each entry reports Paper / Reproduced.}
\label{tab:table3_xsum}
\begin{tabular}{@{}lcccc@{}}
\toprule
Method & \# Params & R-1 & R-2 & R-L \\
\midrule
"""

for r in results:
    xsum_tex += (
        f"{r['Method']} "
        f"& {r['Params']} "
        f"& {r['Paper_R1']:.2f} / {r['Reproduced_R1']:.2f} "
        f"& {r['Paper_R2']:.2f} / {r['Reproduced_R2']:.2f} "
        f"& {r['Paper_RL']:.2f} / {r['Reproduced_RL']:.2f} \\\\\n"
    )

xsum_tex += r"""\bottomrule
\end{tabular}
\end{table}
"""

(out / "table3_xsum.tex").write_text(xsum_tex.strip() + "\n")


# --------------------------------------------------
# LATEX: MT
# --------------------------------------------------

mt_tex = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{WMT16 English--Romanian results for the insertion forms from Table~3. BLEU is reported as Paper / Reproduced.}
\label{tab:table3_mt}
\begin{tabular}{@{}lcc@{}}
\toprule
Method & \# Params & BLEU \\
\midrule
"""

for r in results:
    mt_tex += (
        f"{r['Method']} "
        f"& {r['Params']} "
        f"& {r['Paper_BLEU']:.1f} / {r['Reproduced_BLEU']:.1f} \\\\\n"
    )

mt_tex += r"""\bottomrule
\end{tabular}
\end{table}
"""

(out / "table3_mt.tex").write_text(mt_tex.strip() + "\n")


print("\nSaved:")
print("  results/table3_all.csv")
print("  results/table3_xsum.md")
print("  results/table3_mt.md")
print("  results/table3_xsum.tex")
print("  results/table3_mt.tex")
