"""Shared data loading, chart builders and findings (single source of truth
for the dashboard, the static images and the written report)."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = Path(__file__).parent / "data" / "student-mat.csv"
NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
           "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]
FAIL_COLORS = {"0": "#4c78a8", "1": "#f6c85f", "2": "#f28e2b", "3": "#d62728"}
ABS_ORDER = ["0", "1-5", "6-10", "11+"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")
    df["sex"] = df["sex"].map({"F": "Female", "M": "Male"})
    df["address"] = df["address"].map({"U": "Urban", "R": "Rural"})
    df["Result"] = np.where(df["G3"] >= 10, "Pass", "Fail")
    df["failures_str"] = df["failures"].astype(str)
    df["studytime_str"] = df["studytime"].astype(str)
    df["AbsenceBand"] = pd.cut(df["absences"], [-1, 0, 5, 10, 1000], labels=ABS_ORDER).astype(str)
    return df


def _layout(fig, h=430):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), template="plotly_white")
    return fig


def fig_q1(df):
    """Q1 - Do earlier grades predict the final grade? (scatter + fit line)"""
    fig = px.scatter(df, x="G2", y="G3", color="failures_str", opacity=0.7,
                     color_discrete_map=FAIL_COLORS,
                     category_orders={"failures_str": ["0", "1", "2", "3"]},
                     labels={"G2": "Period-2 grade (G2)", "G3": "Final grade (G3)",
                             "failures_str": "Past failures"},
                     hover_data=["G1", "absences", "school"],
                     title="Q1: Period-2 grade vs final grade")
    if len(df) >= 3 and df["G2"].nunique() > 1:
        m, b = np.polyfit(df["G2"], df["G3"], 1)
        xs = np.array([df["G2"].min(), df["G2"].max()])
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Linear fit",
                                 line=dict(color="black", dash="dash")))
    if (df["G3"] == 0).any():
        fig.add_annotation(x=df.loc[df["G3"] == 0, "G2"].median(), y=0.3,
                           text="G3 = 0 cluster", showarrow=True, ay=-45, ax=40)
    return _layout(fig)


def fig_q2(df):
    """Q2 - Do past failures change final grades? (box plot)"""
    fig = px.box(df, x="failures_str", y="G3", color="failures_str", points="all",
                 color_discrete_map=FAIL_COLORS,
                 category_orders={"failures_str": ["0", "1", "2", "3"]},
                 labels={"failures_str": "Past class failures", "G3": "Final grade (G3)"},
                 title="Q2: Final grade by number of past failures")
    fig.update_layout(showlegend=False)
    return _layout(fig)


def fig_q3(df):
    """Q3 - Which segments have the lowest pass rate? (sunburst)"""
    path = ["school", "address", "sex"]
    g = (df.groupby(path).agg(n=("G3", "size"),
                              pass_rate=("Result", lambda s: (s == "Pass").mean() * 100))
         .reset_index())
    fig = px.sunburst(g, path=path, values="n", color="pass_rate",
                      color_continuous_scale="RdYlGn", range_color=(0, 100),
                      labels={"pass_rate": "Pass rate %"},
                      title="Q3: Segment size (angle) and pass rate (colour)")
    return _layout(fig, 480)


def fig_q4(df):
    """Q4 - How do absences relate to grades? (box + histogram)"""
    box = px.box(df, x="AbsenceBand", y="G3", points="all", color="AbsenceBand",
                 category_orders={"AbsenceBand": ABS_ORDER},
                 labels={"AbsenceBand": "Absences (days)", "G3": "Final grade (G3)"},
                 title="Q4a: Final grade by absence band")
    box.update_layout(showlegend=False)
    hist = px.histogram(df, x="G3", color="Result", nbins=21, barmode="stack",
                        color_discrete_map={"Pass": "#2ca02c", "Fail": "#d62728"},
                        labels={"G3": "Final grade (G3)"},
                        title="Q4b: Distribution of final grades")
    return _layout(box), _layout(hist)


def fig_q5(df):
    """Q5 - Which numeric factors move together with G3? (correlation heatmap)"""
    corr = df[NUMERIC].corr().round(2)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    aspect="auto", title="Q5: Correlation heatmap of numeric variables")
    return _layout(fig, 600)


def fig_wrong(df):
    """Wrong-visualization challenge: does study time raise grades?"""
    means = df.groupby("studytime")["G3"].mean().reset_index()
    wrong = px.bar(means, x="studytime", y="G3", text=means["G3"].round(1),
                   labels={"studytime": "Weekly study time (1=<2h ... 4=>10h)",
                           "G3": "Mean final grade"},
                   title="MISLEADING: bar chart with truncated y-axis")
    wrong.update_yaxes(range=[9.5, 11.6])
    right = px.box(df, x="studytime_str", y="G3", points="all", color="studytime_str",
                   category_orders={"studytime_str": ["1", "2", "3", "4"]},
                   labels={"studytime_str": "Weekly study time (1=<2h ... 4=>10h)",
                           "G3": "Final grade (G3)"},
                   title="APPROPRIATE: box plot showing full distribution")
    right.update_layout(showlegend=False)
    return _layout(right), _layout(wrong)


def stats(df):
    """All numbers used in findings / report."""
    s = {}
    s["n"], s["cols"] = len(df), df.shape[1] - 4  # minus engineered columns
    s["pass_rate"] = (df["G3"] >= 10).mean() * 100
    s["mean_g3"] = df["G3"].mean()
    s["r_g1"], s["r_g2"] = df["G1"].corr(df["G3"]), df["G2"].corr(df["G3"])
    s["fail_means"] = df.groupby("failures")["G3"].mean().round(2).to_dict()
    s["fail_counts"] = df["failures"].value_counts().sort_index().to_dict()
    z = df[df["G3"] == 0]
    s["n_zero"], s["zero_pct"] = len(z), len(z) / len(df) * 100
    s["zero_abs0"] = (z["absences"] == 0).mean() * 100
    s["abs0_mean"] = df.loc[df["absences"] == 0, "G3"].mean()
    s["abs1_5_mean"] = df.loc[df["absences"].between(1, 5), "G3"].mean()
    nz = df[df["G3"] > 0]
    s["abs0_mean_nz"] = nz.loc[nz["absences"] == 0, "G3"].mean()
    s["rural"] = df.loc[df["address"] == "Rural", "G3"].mean()
    s["urban"] = df.loc[df["address"] == "Urban", "G3"].mean()
    s["net_no"] = df.loc[df["internet"] == "no", "G3"].mean()
    s["net_yes"] = df.loc[df["internet"] == "yes", "G3"].mean()
    s["high_no"] = df.loc[df["higher"] == "no", "G3"].mean()
    s["high_yes"] = df.loc[df["higher"] == "yes", "G3"].mean()
    s["n_high_no"] = int((df["higher"] == "no").sum())
    s["medu1"] = df.loc[df["Medu"] == 1, "G3"].mean()
    s["medu4"] = df.loc[df["Medu"] == 4, "G3"].mean()
    s["r_study"] = df["studytime"].corr(df["G3"])
    s["r_walc"] = df["Walc"].corr(df["G3"])
    s["r_goout"] = df["goout"].corr(df["G3"])
    s["study1"] = df.loc[df["studytime"] == 1, "G3"].mean()
    s["study3p"] = df.loc[df["studytime"] >= 3, "G3"].mean()
    s["romantic_yes"] = df.loc[df["romantic"] == "yes", "G3"].mean()
    s["romantic_no"] = df.loc[df["romantic"] == "no", "G3"].mean()
    s["abs_outliers"] = int((df["absences"] > df["absences"].quantile(.75) +
                             1.5 * (df["absences"].quantile(.75) - df["absences"].quantile(.25))).sum())
    s["abs_max"] = int(df["absences"].max())
    low = df[df["G1"] < 10]
    s["g1_low_n"] = len(low)
    s["g1_low_fail"] = (low["G3"] < 10).mean() * 100
    s["g1_ok_fail"] = (df[df["G1"] >= 10]["G3"] < 10).mean() * 100
    s["fail_any_fail"] = (df[df["failures"] >= 1]["G3"] < 10).mean() * 100
    s["fail_any_n"] = int((df["failures"] >= 1).sum())
    s["r_fail"] = df["failures"].corr(df["G3"])
    return s


def findings(df):
    s = stats(df)
    return [
        ("Earlier grades are the strongest signal",
         f"G2 and G3 correlate at r = {s['r_g2']:.2f} (G1: r = {s['r_g1']:.2f}). Among students with G1 < 10, "
         f"{s['g1_low_fail']:.0f}% fail the year versus {s['g1_ok_fail']:.0f}% of those with G1 >= 10."),
        ("Past failures are the strongest non-grade risk factor",
         f"Mean final grade falls from {s['fail_means'][0]:.1f} (0 failures) to {s['fail_means'][1]:.1f}, "
         f"{s['fail_means'][2]:.1f} and {s['fail_means'][3]:.1f} for 1, 2 and 3 failures (r = {s['r_fail']:.2f} with G3)."),
        ("A hidden 'zero-grade' cluster distorts the absence story",
         f"{s['n_zero']} students ({s['zero_pct']:.1f}%) have G3 = 0 and {s['zero_abs0']:.0f}% of them show 0 absences, "
         f"so the 0-absence band looks weak (mean {s['abs0_mean']:.1f}) while 1-5 absences scores {s['abs1_5_mean']:.1f}. "
         f"Without the zero-grade records the 0-absence mean is {s['abs0_mean_nz']:.1f}. These look like dropouts or missing records, not truly perfect attendance."),
        ("Background and access gaps are visible",
         f"Mean G3: urban {s['urban']:.1f} vs rural {s['rural']:.1f}; internet at home {s['net_yes']:.1f} vs none {s['net_no']:.1f}; "
         f"mother's education level 4 vs 1: {s['medu4']:.1f} vs {s['medu1']:.1f}; wants higher education {s['high_yes']:.1f} "
         f"vs not {s['high_no']:.1f} (only {s['n_high_no']} students, so treat with care)."),
        ("Study time and lifestyle matter less than expected",
         f"Study time correlates only r = {s['r_study']:.2f} with G3 (level 1: {s['study1']:.1f} vs level 3-4: {s['study3p']:.1f}); "
         f"weekend alcohol r = {s['r_walc']:.2f}, going out r = {s['r_goout']:.2f}. Correlation is not causation."),
    ]


def recommendations(df):
    s = stats(df)
    return [
        ("Build an early-warning list after the first period",
         f"Flag every student with G1 < 10 ({s['g1_low_n']} students) or at least one past failure ({s['fail_any_n']} students; "
         f"{s['fail_any_fail']:.0f}% of them fail the year) for tutoring and mentoring before period 2."),
        ("Fix attendance / dropout tracking",
         f"Audit the {s['n_zero']} zero-grade records, separate 'withdrawn' from 'enrolled', and record attendance "
         "reliably so absence analysis is trustworthy; follow up with those students and families."),
        ("Target support at access gaps",
         f"Provide study spaces, connectivity and parent-engagement programmes for rural and no-internet students "
         f"and run career counselling for the few students not planning higher education."),
    ]
