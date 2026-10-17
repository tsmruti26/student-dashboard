"""Creates static preview images (images/) and the written report (SUBMISSION_REPORT.md).
Run:  python make_assets.py      (needs matplotlib + seaborn; the dashboard itself does not)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import charts as C

OUT = Path(__file__).parent
IMG = OUT / "images"; IMG.mkdir(exist_ok=True)
df = C.load_data(); s = C.stats(df)
sns.set_theme(style="whitegrid")
pal = {"0": "#4c78a8", "1": "#f6c85f", "2": "#f28e2b", "3": "#d62728"}


def save(fig, name):
    fig.tight_layout(); fig.savefig(IMG / name, dpi=130); plt.close(fig)

# Q1
f, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df, x="G2", y="G3", hue="failures_str", palette=pal, hue_order=list(pal), alpha=.7, ax=ax)
m, b = np.polyfit(df.G2, df.G3, 1); xs = np.array([0, 19]); ax.plot(xs, m * xs + b, "k--", label="Linear fit")
ax.set_title("Q1: Period-2 grade vs final grade"); ax.legend(title="Past failures"); save(f, "q1_scatter.png")
# Q2
f, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x="failures_str", y="G3", order=list(pal), palette=pal, hue="failures_str", legend=False, ax=ax)
sns.stripplot(data=df, x="failures_str", y="G3", order=list(pal), color="k", alpha=.25, size=3, ax=ax)
ax.set_title("Q2: Final grade by past failures"); ax.set_xlabel("Past failures"); save(f, "q2_box_failures.png")
# Q3 (static stand-in for the interactive sunburst)
g = (df.groupby(["school", "address", "sex"]).agg(n=("G3", "size"), pr=("Result", lambda x: (x == "Pass").mean() * 100)).reset_index())
g["seg"] = g.school + " / " + g.address + " / " + g.sex + " (n=" + g.n.astype(str) + ")"
g = g.sort_values("pr")
f, ax = plt.subplots(figsize=(8, 5)); ax.barh(g.seg, g.pr, color=plt.cm.RdYlGn(g.pr / 100))
ax.set_xlim(0, 100); ax.set_title("Q3: Pass rate by segment (static version of the sunburst)"); ax.set_xlabel("Pass rate %"); save(f, "q3_segments.png")
# Q4
f, axs = plt.subplots(1, 2, figsize=(11, 5))
sns.boxplot(data=df, x="AbsenceBand", y="G3", order=C.ABS_ORDER, ax=axs[0], color="#9ecae1")
sns.stripplot(data=df, x="AbsenceBand", y="G3", order=C.ABS_ORDER, color="k", alpha=.25, size=3, ax=axs[0])
axs[0].set_title("Q4a: Final grade by absence band")
sns.histplot(data=df, x="G3", hue="Result", bins=21, multiple="stack", palette={"Pass": "#2ca02c", "Fail": "#d62728"}, ax=axs[1])
axs[1].set_title("Q4b: Distribution of final grades"); save(f, "q4_absences.png")
# Q5
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[C.NUMERIC].corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, annot_kws={"size": 7}, ax=ax)
ax.set_title("Q5: Correlation heatmap"); save(f, "q5_heatmap.png")
# Wrong vs right
f, ax = plt.subplots(figsize=(6.5, 5))
sns.boxplot(data=df, x="studytime_str", y="G3", order=list("1234"), color="#9ecae1", ax=ax)
sns.stripplot(data=df, x="studytime_str", y="G3", order=list("1234"), color="k", alpha=.25, size=3, ax=ax)
ax.set_title("APPROPRIATE: box plot of G3 by study time"); ax.set_xlabel("Study time (1=<2h ... 4=>10h)"); save(f, "wrong_viz_1_correct.png")
mt = df.groupby("studytime").G3.mean()
f, ax = plt.subplots(figsize=(6.5, 5)); bars = ax.bar(mt.index.astype(str), mt.values, color="#d62728")
ax.set_ylim(9.5, 11.6); ax.bar_label(bars, fmt="%.1f"); ax.set_title("MISLEADING: truncated-axis bar chart"); ax.set_xlabel("Study time (1=<2h ... 4=>10h)")
ax.set_ylabel("Mean G3"); save(f, "wrong_viz_2_misleading.png")

F, R = C.findings(df), C.recommendations(df)
fl = "\n".join(f"{i}. **{h}.** {t}" for i, (h, t) in enumerate(F, 1))
rl = "\n".join(f"{i}. **{h}.** {t}" for i, (h, t) in enumerate(R, 1))

report = f"""# Data Detective Challenge - Group 5: Student Academic Performance Analytics

*Team members / roll numbers: 5, 23, 40, 58 (add names).*

## 1. Dataset and source
- **Dataset:** Student Performance - `data/student-mat.csv` (mathematics course).
- **Source:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/320/student+performance (CC BY 4.0).
- **Reference:** P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* FUBUTEC 2008, pp. 5-12.

## 2. Dataset Discovery Summary
| Item | Finding |
|---|---|
| Records / attributes | {s['n']} students x {s['cols']} attributes |
| Numerical (16) | age, Medu, Fedu, traveltime, studytime, failures, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2, G3 (several are 1-5 ordinal scales) |
| Categorical (17) | school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic |
| Temporal | No dates; the grade sequence G1 -> G2 -> G3 (periods 1-3) acts as a time axis |
| Geographical | Only `school` (2 schools) and `address` (urban/rural); no coordinates, so a choropleth/Folium map is not appropriate |
| Missing values | 0 |
| Duplicates | 0 |
| Outliers | `absences`: {s['abs_outliers']} values above the IQR fence (max {s['abs_max']}); {s['n_zero']} students ({s['zero_pct']:.1f}%) have G3 = 0 - anomalous because {s['zero_abs0']:.0f}% of them have 0 absences |
| Cleaning done | Mapped codes to labels (sex, address), added `Result` (pass if G3 >= 10), `AbsenceBand`, string versions of ordinal columns for plotting; no rows removed (optional sidebar switch excludes G3 = 0) |
| Patterns | G1/G2 vs G3 (r = {s['r_g1']:.2f}/{s['r_g2']:.2f}); failures vs G3 (r = {s['r_fail']:.2f}); urban/rural, internet and parental education gaps; weak lifestyle effects |

## 3. Five analytical questions and visualization justification
| # | Analytical question | Variables | Visualization | Why appropriate |
|---|---|---|---|---|
| Q1 | Do earlier grades predict the final grade? | G2, G3, failures | Scatter plot + fit line | Two continuous variables: scatter shows strength, direction and outliers (G3 = 0); colour adds a third variable |
| Q2 | Do past failures lower final grades? | failures, G3 | Box plot with points | Compares a numeric distribution across few ordered groups; shows median, spread, outliers and group sizes |
| Q3 | Which school / address / sex segments have the lowest pass rate? | school, address, sex, Result | Sunburst | Hierarchical categories: angle = segment size, colour = pass rate, click to drill down |
| Q4 | How do absences relate to grades and what does the grade distribution look like? | absences, G3, Result | Box plot by absence band + stacked histogram | Banding tames a skewed variable; histogram exposes the zero-grade spike and pass/fail split |
| Q5 | Which numeric factors move together with G3? | 16 numeric columns | Correlation heatmap | Shows all pairwise relationships at once and ranks the strongest drivers |

## 4. Source code
`app.py` (Streamlit dashboard), `charts.py` (data prep, charts, findings), `make_assets.py` (static images + this report).

## 5. Interactive dashboard
Run `streamlit run app.py`. It has title/objective, dataset description, 5 KPI cards, 7 sidebar filters (+ zero-grade switch), 5 analytical visualizations (plus the wrong-viz tab), the five findings and three recommendations.

## 6. Dashboard screenshots
Static previews of every dashboard visual are in `images/` (`q1_scatter.png` ... `q5_heatmap.png`). **Before submitting, also take 2-3 real screenshots of the running dashboard and save them in `images/` as `dashboard_1.png`, etc.**

## 7. Five major findings
{fl}

## 8. Three actionable recommendations
{rl}

## 9. Wrong Visualization Challenge
**Question:** Does more weekly study time lead to higher final grades?

| Correct (`images/wrong_viz_1_correct.png`) | Misleading (`images/wrong_viz_2_misleading.png`) |
|---|---|
| Box plot of G3 per study-time level | Bar chart of mean G3 with y-axis starting at 9.5 |

- **Why the first is appropriate:** it shows the whole distribution (median, spread, outliers, every student), so the heavy overlap between groups is visible and the weak relationship (r = {s['r_study']:.2f}) is honest.
- **Why the second is misleading:** bars encode value by length, so cutting the axis at 9.5 makes a difference of about {s['study3p'] - s['study1']:.1f} grade points (level 1: {s['study1']:.1f}, level 3-4: {s['study3p']:.1f}) look like a several-fold jump; means also hide spread and group size (only 27 students are in level 4).
- **Effect on interpretation:** a reader would conclude that study time is a powerful lever and might spend resources on it, while the data show most of the variation is explained by earlier grades and past failures.

## 10. AI Critique + Team Justification
*Tool used: Claude (Anthropic). Visualization critiqued: Q1 scatter plot (G2 vs G3). Replace/extend with your own run of an AI tool if your instructor requires it, and make sure every member can explain these decisions.*

| # | AI recommendation | Decision | Team justification |
|---|---|---|---|
| 1 | Use a sequential colour scale for number of failures (ordered variable) rather than unrelated colours | ACCEPTED | Failures are ordered 0-3; we now use blue -> yellow -> orange -> red so severity reads intuitively |
| 2 | Annotate the G3 = 0 cluster because it distorts the fit line | ACCEPTED | These {s['n_zero']} students are anomalies (all 0 absences); the annotation and the sidebar switch let viewers judge their influence |
| 3 | Add a confidence band to the regression line | PARTIALLY ACCEPTED | We added a linear fit line but not the band: the zero-grade cluster breaks regression assumptions so a band would imply false precision; the sidebar switch removes the cluster |
| 4 | Replace the scatter with a hexbin / density plot to avoid overplotting | REJECTED | With 395 points and integer grades, overplotting is mild; points keep hover details (G1, absences, school) and are easier to explain to non-technical peers |

## 11. Two-minute Data Story (about 280 words, speak slowly)
**What did we discover?** We studied {s['n']} Portuguese secondary students. Two out of three pass maths ({s['pass_rate']:.0f}%), but failure is predictable. A student's period-2 grade almost fixes the final grade (correlation {s['r_g2']:.2f}), and {s['g1_low_fail']:.0f}% of students who score below 10 in period 1 fail the year. Past class failures push the average final grade from {s['fail_means'][0]:.1f} to about {s['fail_means'][1]:.0f} after one failure. We also found a trap in the data: {s['n_zero']} students have a final grade of zero and none of them has any recorded absence, which makes "perfect attendance" look bad. These are probably dropouts, not model students. Meanwhile, study time and nightlife barely matter statistically.

**Why does it matter?** Schools usually react at the end of the year, when it is too late. The warning signs - low first-period marks, earlier failures, rural homes without internet - are visible in September. And if the attendance records are unreliable, any attendance policy built on them will mislead.

**What action should be taken?** One: build an early-warning list after period 1 (G1 below 10 or any past failure) and offer tutoring immediately. Two: audit the zero-grade records and fix dropout and attendance tracking. Three: target support at rural and no-internet students and give career counselling to students who do not plan higher education. Our dashboard lets any teacher filter by school, age or failures and see who needs help.

*Note: these are associations from one dataset, not proof of cause.*
"""
(OUT / "SUBMISSION_REPORT.md").write_text(report, encoding="utf-8")
print("done:", sorted(p.name for p in IMG.iterdir()))
