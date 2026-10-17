# Data Detective Challenge - Group 5: Student Academic Performance Analytics

*Team members / roll numbers: 5, 23, 40, 58 (add names).*

## 1. Dataset and source
- **Dataset:** Student Performance - `data/student-mat.csv` (mathematics course).
- **Source:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/320/student+performance (CC BY 4.0).
- **Reference:** P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* FUBUTEC 2008, pp. 5-12.

## 2. Dataset Discovery Summary
| Item | Finding |
|---|---|
| Records / attributes | 395 students x 33 attributes |
| Numerical (16) | age, Medu, Fedu, traveltime, studytime, failures, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2, G3 (several are 1-5 ordinal scales) |
| Categorical (17) | school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic |
| Temporal | No dates; the grade sequence G1 -> G2 -> G3 (periods 1-3) acts as a time axis |
| Geographical | Only `school` (2 schools) and `address` (urban/rural); no coordinates, so a choropleth/Folium map is not appropriate |
| Missing values | 0 |
| Duplicates | 0 |
| Outliers | `absences`: 15 values above the IQR fence (max 75); 38 students (9.6%) have G3 = 0 - anomalous because 100% of them have 0 absences |
| Cleaning done | Mapped codes to labels (sex, address), added `Result` (pass if G3 >= 10), `AbsenceBand`, string versions of ordinal columns for plotting; no rows removed (optional sidebar switch excludes G3 = 0) |
| Patterns | G1/G2 vs G3 (r = 0.80/0.90); failures vs G3 (r = -0.36); urban/rural, internet and parental education gaps; weak lifestyle effects |

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
1. **Earlier grades are the strongest signal.** G2 and G3 correlate at r = 0.90 (G1: r = 0.80). Among students with G1 < 10, 73% fail the year versus 10% of those with G1 >= 10.
2. **Past failures are the strongest non-grade risk factor.** Mean final grade falls from 11.2 (0 failures) to 8.1, 6.2 and 5.7 for 1, 2 and 3 failures (r = -0.36 with G3).
3. **A hidden 'zero-grade' cluster distorts the absence story.** 38 students (9.6%) have G3 = 0 and 100% of them show 0 absences, so the 0-absence band looks weak (mean 8.4) while 1-5 absences scores 11.6. Without the zero-grade records the 0-absence mean is 12.6. These look like dropouts or missing records, not truly perfect attendance.
4. **Background and access gaps are visible.** Mean G3: urban 10.7 vs rural 9.5; internet at home 10.6 vs none 9.4; mother's education level 4 vs 1: 11.8 vs 8.7; wants higher education 10.6 vs not 6.8 (only 20 students, so treat with care).
5. **Study time and lifestyle matter less than expected.** Study time correlates only r = 0.10 with G3 (level 1: 10.0 vs level 3-4: 11.4); weekend alcohol r = -0.05, going out r = -0.13. Correlation is not causation.

## 8. Three actionable recommendations
1. **Build an early-warning list after the first period.** Flag every student with G1 < 10 (142 students) or at least one past failure (83 students; 63% of them fail the year) for tutoring and mentoring before period 2.
2. **Fix attendance / dropout tracking.** Audit the 38 zero-grade records, separate 'withdrawn' from 'enrolled', and record attendance reliably so absence analysis is trustworthy; follow up with those students and families.
3. **Target support at access gaps.** Provide study spaces, connectivity and parent-engagement programmes for rural and no-internet students and run career counselling for the few students not planning higher education.

## 9. Wrong Visualization Challenge
**Question:** Does more weekly study time lead to higher final grades?

| Correct (`images/wrong_viz_1_correct.png`) | Misleading (`images/wrong_viz_2_misleading.png`) |
|---|---|
| Box plot of G3 per study-time level | Bar chart of mean G3 with y-axis starting at 9.5 |

- **Why the first is appropriate:** it shows the whole distribution (median, spread, outliers, every student), so the heavy overlap between groups is visible and the weak relationship (r = 0.10) is honest.
- **Why the second is misleading:** bars encode value by length, so cutting the axis at 9.5 makes a difference of about 1.3 grade points (level 1: 10.0, level 3-4: 11.4) look like a several-fold jump; means also hide spread and group size (only 27 students are in level 4).
- **Effect on interpretation:** a reader would conclude that study time is a powerful lever and might spend resources on it, while the data show most of the variation is explained by earlier grades and past failures.

## 10. AI Critique + Team Justification
*Tool used: Claude (Anthropic). Visualization critiqued: Q1 scatter plot (G2 vs G3). Replace/extend with your own run of an AI tool if your instructor requires it, and make sure every member can explain these decisions.*

| # | AI recommendation | Decision | Team justification |
|---|---|---|---|
| 1 | Use a sequential colour scale for number of failures (ordered variable) rather than unrelated colours | ACCEPTED | Failures are ordered 0-3; we now use blue -> yellow -> orange -> red so severity reads intuitively |
| 2 | Annotate the G3 = 0 cluster because it distorts the fit line | ACCEPTED | These 38 students are anomalies (all 0 absences); the annotation and the sidebar switch let viewers judge their influence |
| 3 | Add a confidence band to the regression line | PARTIALLY ACCEPTED | We added a linear fit line but not the band: the zero-grade cluster breaks regression assumptions so a band would imply false precision; the sidebar switch removes the cluster |
| 4 | Replace the scatter with a hexbin / density plot to avoid overplotting | REJECTED | With 395 points and integer grades, overplotting is mild; points keep hover details (G1, absences, school) and are easier to explain to non-technical peers |

## 11. Two-minute Data Story (about 280 words, speak slowly)
**What did we discover?** We studied 395 Portuguese secondary students. Two out of three pass maths (67%), but failure is predictable. A student's period-2 grade almost fixes the final grade (correlation 0.90), and 73% of students who score below 10 in period 1 fail the year. Past class failures push the average final grade from 11.2 to about 8 after one failure. We also found a trap in the data: 38 students have a final grade of zero and none of them has any recorded absence, which makes "perfect attendance" look bad. These are probably dropouts, not model students. Meanwhile, study time and nightlife barely matter statistically.

**Why does it matter?** Schools usually react at the end of the year, when it is too late. The warning signs - low first-period marks, earlier failures, rural homes without internet - are visible in September. And if the attendance records are unreliable, any attendance policy built on them will mislead.

**What action should be taken?** One: build an early-warning list after period 1 (G1 below 10 or any past failure) and offer tutoring immediately. Two: audit the zero-grade records and fix dropout and attendance tracking. Three: target support at rural and no-internet students and give career counselling to students who do not plan higher education. Our dashboard lets any teacher filter by school, age or failures and see who needs help.

*Note: these are associations from one dataset, not proof of cause.*
