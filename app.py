
import streamlit as st
import charts as C

st.set_page_config(page_title="Student Performance Dashboard", page_icon="🎓", layout="wide")


@st.cache_data
def get_data():
    return C.load_data()


full = get_data()

st.title("🎓 Student Academic Performance Analytics")
st.markdown("**Objective:** find which academic, family and lifestyle factors are linked to final "
            "grades in secondary-school mathematics, so schools can spot at-risk students early.")
with st.expander("Dataset / problem description", expanded=False):
    st.markdown(
        "- **Source:** UCI Machine Learning Repository - *Student Performance* (P. Cortez & A. Silva, 2008), "
        "`student-mat.csv`, two Portuguese secondary schools, licence CC BY 4.0.\n"
        "- **Size:** 395 students x 33 attributes (demographics, family, school support, lifestyle, absences).\n"
        "- **Target:** `G3` = final grade (0-20); a student **passes at G3 >= 10**. `G1`, `G2` are period-1 and period-2 grades.\n"
        "- **Problem:** which factors explain failure, and which students should be supported early?")

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
schools = st.sidebar.multiselect("School", sorted(full["school"].unique()), default=sorted(full["school"].unique()))
sexes = st.sidebar.multiselect("Sex", sorted(full["sex"].unique()), default=sorted(full["sex"].unique()))
addrs = st.sidebar.multiselect("Address", sorted(full["address"].unique()), default=sorted(full["address"].unique()))
amin, amax = int(full["age"].min()), int(full["age"].max())
ages = st.sidebar.slider("Age", amin, amax, (amin, amax))
fails = st.sidebar.multiselect("Past failures", sorted(full["failures"].unique()), default=sorted(full["failures"].unique()))
study = st.sidebar.multiselect("Study time (1=<2h, 4=>10h)", sorted(full["studytime"].unique()), default=sorted(full["studytime"].unique()))
net = st.sidebar.multiselect("Internet at home", sorted(full["internet"].unique()), default=sorted(full["internet"].unique()))
excl_zero = st.sidebar.checkbox("Exclude G3 = 0 records (possible dropouts)", value=False)

df = full[full["school"].isin(schools) & full["sex"].isin(sexes) & full["address"].isin(addrs)
          & full["age"].between(*ages) & full["failures"].isin(fails)
          & full["studytime"].isin(study) & full["internet"].isin(net)]
if excl_zero:
    df = df[df["G3"] > 0]

if df.empty:
    st.warning("No students match the current filters. Please widen your selection.")
    st.stop()

# ---------------- KPIs ----------------
k = st.columns(5)
k[0].metric("Students", f"{len(df)}", f"{len(df) - len(full)} vs all" if len(df) != len(full) else None)
k[1].metric("Average final grade", f"{df['G3'].mean():.2f} / 20")
k[2].metric("Pass rate (G3 >= 10)", f"{(df['G3'] >= 10).mean() * 100:.1f}%")
k[3].metric("At-risk (G2 < 10)", f"{(df['G2'] < 10).mean() * 100:.1f}%")
k[4].metric("Avg absences", f"{df['absences'].mean():.1f}")

# ---------------- Visualisations ----------------
t1, t2, t3, t4, t5, t6 = st.tabs(["Q1 Grades trend", "Q2 Failures", "Q3 Segments", "Q4 Absences",
                                  "Q5 Correlations", "Wrong-viz challenge"])
with t1:
    st.plotly_chart(C.fig_q1(df))
    st.caption("Reading: points hug the diagonal - a student's period-2 grade almost fixes the final grade. "
               "The row of points at G3 = 0 are students who 'collapsed' to zero.")
with t2:
    st.plotly_chart(C.fig_q2(df))
    st.caption("Reading: median grade drops with every past failure. Groups with 2-3 failures are small (see points).")
with t3:
    st.plotly_chart(C.fig_q3(df))
    st.caption("Reading: click a slice to drill down school -> address -> sex. Red = low pass rate, green = high.")
with t4:
    box, hist = C.fig_q4(df)
    c1, c2 = st.columns(2)
    c1.plotly_chart(box)
    c2.plotly_chart(hist)
    st.caption("Reading: the 0-absence box is dragged down by G3 = 0 records. Tick 'Exclude G3 = 0' in the sidebar to see the difference.")
with t5:
    if len(df) < 10:
        st.info("Select at least 10 students to compute a reliable correlation matrix.")
    else:
        st.plotly_chart(C.fig_q5(df))
        st.caption("Reading: G1/G2 dominate; failures is the strongest negative factor; lifestyle variables are weak. Grey cells mean a filter removed all variation.")
with t6:
    good, bad = C.fig_wrong(df)
    c1, c2 = st.columns(2)
    c1.plotly_chart(good)
    c2.plotly_chart(bad)
    st.markdown("The truncated bar chart makes a ~1-grade gap look huge; the box plot shows the groups overlap heavily "
                "(study time correlates only r = 0.10 with G3).")

# ---------------- Findings & recommendations (full dataset) ----------------
st.divider()
a, b = st.columns(2)
with a:
    st.subheader("Five major findings")
    for i, (h, t) in enumerate(C.findings(full), 1):
        st.markdown(f"**{i}. {h}.** {t}")
with b:
    st.subheader("Three actionable recommendations")
    for i, (h, t) in enumerate(C.recommendations(full), 1):
        st.markdown(f"**{i}. {h}.** {t}")
st.caption("Findings use the full dataset (not the filtered view). Associations are not proof of cause.")
