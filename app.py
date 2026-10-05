import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Student Performance Analytics", layout="wide")

# ---------- Load data ----------
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload student-mat.csv (optional)", type="csv")


@st.cache_data
def load(f):
    data = pd.read_csv(f if f is not None else "student-mat.csv", sep=None, engine="python")
    data.columns = data.columns.str.strip()
    return data


try:
    raw = load(up)
except FileNotFoundError:
    st.error("Place student-mat.csv next to app.py or upload it in the sidebar.")
    st.stop()

if "G3" not in raw.columns:
    st.error("Wrong file: this must be the UCI student-mat.csv (needs G1, G2, G3 columns).")
    st.stop()

df = raw.copy()
df["Pass"] = np.where(df["G3"] >= 10, "Pass", "Fail")
df["Studytime"] = df["studytime"].map({1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"})
ORDER = ["<2h", "2-5h", "5-10h", ">10h"]

# ---------- Filters ----------
st.sidebar.header("Filters")
school = st.sidebar.multiselect("School", df.school.unique(), df.school.unique())
sex = st.sidebar.multiselect("Sex", df.sex.unique(), df.sex.unique())
age = st.sidebar.slider("Age", int(df.age.min()), int(df.age.max()),
                        (int(df.age.min()), int(df.age.max())))
internet = st.sidebar.multiselect("Internet at home", df.internet.unique(), df.internet.unique())
d = df[df.school.isin(school) & df.sex.isin(sex) & df.age.between(*age) & df.internet.isin(internet)]
if d.empty:
    st.warning("No records match the filters.")
    st.stop()

# ---------- Header ----------
st.title("Student Academic Performance Analytics")
st.markdown("**Objective:** find what drives final grades (G3, scale 0-20) so a college can "
            "identify at-risk students early and target support. "
            "**Dataset:** UCI Student Performance - 395 students, 33 attributes "
            "(demographic, family, lifestyle, absences, grades G1/G2/G3).")

k = st.columns(5)
k[0].metric("Students", len(d))
k[1].metric("Avg final grade", f"{d.G3.mean():.2f}")
k[2].metric("Pass rate (G3>=10)", f"{(d.G3 >= 10).mean():.0%}")
k[3].metric("Avg absences", f"{d.absences.mean():.1f}")
k[4].metric("Avg past failures", f"{d.failures.mean():.2f}")

t1, t2, t3, t4, t5 = st.tabs(["Discovery", "Dashboard", "Findings & Recommendations",
                              "Wrong Visualization", "AI Critique"])

# ---------- Discovery ----------
with t1:
    st.subheader("Dataset Discovery Summary")
    n_num = len(raw.select_dtypes(include="number").columns)
    n_cat = raw.shape[1] - n_num
    st.write(f"**Records:** {raw.shape[0]} | **Attributes:** {raw.shape[1]} | "
             f"**Numerical:** {n_num} | **Categorical:** {n_cat}")
    st.write("**Temporal/Geographic:** no date or location columns. G1 -> G2 -> G3 "
             "(three period grades) act as the time dimension; `school` (GP/MS) and "
             "`address` (Urban/Rural) are the closest geographic variables.")
    st.write(f"**Missing values:** {int(raw.isna().sum().sum())} | "
             f"**Duplicate rows:** {int(raw.duplicated().sum())}")
    q1, q3 = raw.absences.quantile([.25, .75])
    out_abs = int((raw.absences > q3 + 1.5 * (q3 - q1)).sum())
    st.write(f"**Outliers (IQR rule):** {out_abs} students with unusually high absences; "
             f"{int((raw.G3 == 0).sum())} students with G3 = 0 (likely dropouts/no-shows).")
    st.write("**Potential relationships:** G1/G2 strongly predict G3; past failures and "
             "absences relate to lower grades; study time, higher-education ambition and "
             "parental education may lift grades.")
    st.dataframe(raw.describe().T)

# ---------- Dashboard ----------
with t2:
    st.subheader("Analytical Questions & Visualizations")
    st.caption("Q1 Distribution -> Histogram | Q2 Study time effect -> Box plot | "
               "Q3 Absences vs grade -> Scatter | Q4 Which variables correlate -> Heatmap | "
               "Q5 Grade progression -> Line chart")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Q1. How are final grades distributed?**")
        st.plotly_chart(px.histogram(d, x="G3", nbins=21, color="Pass", marginal="box"),
                        use_container_width=True, key="q1")
    with c2:
        st.markdown("**Q2. Does weekly study time change final grades?**")
        st.plotly_chart(px.box(d, x="Studytime", y="G3", color="Studytime",
                               category_orders={"Studytime": ORDER}),
                        use_container_width=True, key="q2")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Q3. Do absences hurt final grades?**")
        fig = px.scatter(d, x="absences", y="G3", color="sex", opacity=.7,
                         hover_data=["school", "failures", "studytime"])
        m, b = np.polyfit(d.absences, d.G3, 1)
        xs = np.array([d.absences.min(), d.absences.max()])
        fig.add_scatter(x=xs, y=m * xs + b, mode="lines", name="Trend", line=dict(color="black"))
        st.plotly_chart(fig, use_container_width=True, key="q3")
    with c4:
        st.markdown("**Q4. Which factors correlate with G3?**")
        cols = ["G1", "G2", "G3", "failures", "absences", "studytime", "Medu", "Fedu",
                "goout", "Dalc", "Walc", "health", "age"]
        st.plotly_chart(px.imshow(d[cols].corr().round(2), text_auto=True,
                                  color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                        use_container_width=True, key="q4")
    st.markdown("**Q5. How do grades progress G1 -> G2 -> G3 by number of past failures?**")
    grp = d.assign(Failures=d.failures.clip(upper=3).astype(str).replace({"3": "3+"}))
    prog = grp.groupby("Failures")[["G1", "G2", "G3"]].mean().reset_index().melt(
        "Failures", var_name="Period", value_name="Avg grade")
    st.plotly_chart(px.line(prog, x="Period", y="Avg grade", color="Failures", markers=True),
                    use_container_width=True, key="q5")

# ---------- Findings ----------
with t3:
    st.subheader("Five Major Findings")
    r = df[["G1", "G2", "G3"]].corr()
    f0, f1 = df[df.failures == 0].G3.mean(), df[df.failures >= 1].G3.mean()
    hi = df.groupby("higher").G3.mean()
    lo_s, hi_s = df[df.studytime == 1].G3.mean(), df[df.studytime >= 3].G3.mean()
    st.markdown(f"""
1. **Earlier grades predict the final grade.** Correlation G1-G3 = {r.loc['G1','G3']:.2f}, G2-G3 = {r.loc['G2','G3']:.2f}.
2. **Past failures are a strong risk flag.** Avg G3 is {f0:.1f} with no failures vs {f1:.1f} with 1+ failures.
3. **Study time helps modestly.** Students studying <2h average {lo_s:.1f} vs {hi_s:.1f} for 5h+.
4. **Absences show a weak/mixed link with G3** (corr = {df.absences.corr(df.G3):.2f}); zero-grade students distort the pattern.
5. **Ambition matters.** Students wanting higher education average {hi.get('yes', float('nan')):.1f} vs {hi.get('no', float('nan')):.1f} for those who do not.
""")
    st.subheader("Three Actionable Recommendations")
    st.markdown("""
1. **Early-warning system:** flag students with low G1/G2 or 1+ past failures and give mentoring before the final term.
2. **Structured study support:** run supervised study sessions for students studying under 2 hours a week.
3. **Aspiration and attendance programs:** career counselling to raise motivation for higher education, plus follow-up on students with a G3 of 0 or very high absences.
""")

# ---------- Wrong viz ----------
with t4:
    st.subheader("Wrong Visualization Challenge")
    st.markdown("**Question:** Does study time affect final grades?")
    a, b = st.columns(2)
    with a:
        st.markdown("**Correct: Box plot of G3 by study time**")
        st.plotly_chart(px.box(d, x="Studytime", y="G3", color="Studytime",
                               category_orders={"Studytime": ORDER}),
                        use_container_width=True, key="wrong_correct")
    with b:
        st.markdown("**Wrong: Pie chart of total G3 by study time**")
        st.plotly_chart(px.pie(d.groupby("Studytime", as_index=False).G3.sum(),
                               names="Studytime", values="G3"),
                        use_container_width=True, key="wrong_pie")
    st.markdown("""
- **Why the first is appropriate:** it compares a numeric variable across ordered categories and shows median, spread and outliers.
- **Why the second is misleading:** summing grades makes the biggest group (2-5h) look best simply because it has the most students; a pie shows parts of a whole, not performance, and hides spread.
- **Effect on interpretation:** a reader would wrongly conclude 2-5h students perform far better, leading to wrong policy decisions.
""")

# ---------- AI critique ----------
with t5:
    st.subheader("AI as a Critic")
    st.markdown("""
Paste a screenshot of one chart (e.g. Q3 scatter) into an AI tool with the prompt:
*"Critique this visualization for clarity, accuracy and design."* Then fill the table below in your submission.

| AI Recommendation | Decision | Team Justification |
|---|---|---|
| (e.g. Add a trend line / reduce overplotting) | ACCEPTED / PARTIALLY ACCEPTED / REJECTED | (your reasoning) |
| (e.g. Remove G3 = 0 outliers) | ... | ... |
""")