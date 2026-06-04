import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Internship Selection Analytics",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/Internship_Selection_Dataset.csv")

df = load_data()

st.title("📊 Internship Selection Analytics Dashboard")
st.markdown("Deep Exploratory Data Analysis of Internship Selection Dataset")

# =======================
# Sidebar Filters
# =======================

st.sidebar.header("Filters")

college = st.sidebar.multiselect(
    "College Tier",
    options=df["college_tier"].unique(),
    default=df["college_tier"].unique()
)

training = st.sidebar.multiselect(
    "Placement Training",
    options=df["placement_training"].unique(),
    default=df["placement_training"].unique()
)

df = df[
    (df["college_tier"].isin(college)) &
    (df["placement_training"].isin(training))
]

# =======================
# KPIs
# =======================

selected_rate = round(df["selected"].mean()*100,2)

col1,col2,col3,col4 = st.columns(4)

col1.metric("Students", len(df))
col2.metric("Selection Rate", f"{selected_rate}%")
col3.metric("Average CGPA", round(df["CGPA"].mean(),2))
col4.metric("Avg Coding Score", round(df["coding_test_score"].mean(),2))

st.divider()

# =======================
# Selection Distribution
# =======================

c1,c2 = st.columns(2)

with c1:
    fig = px.pie(
        df,
        names="selected",
        title="Selection Distribution"
    )
    st.plotly_chart(fig,use_container_width=True)

with c2:
    fig = px.histogram(
        df,
        x="CGPA",
        color="selected",
        nbins=30,
        title="CGPA Distribution"
    )
    st.plotly_chart(fig,use_container_width=True)

# =======================
# Top Metrics Analysis
# =======================

st.subheader("Performance Metrics")

numeric_cols = [
    'CGPA',
    'skills_score',
    'projects_count',
    'internships_done',
    'communication_score',
    'aptitude_score',
    'coding_test_score',
    'resume_score',
    'hackathons_participated',
    'certifications_count',
    'linkedin_activity_score',
    'github_score',
    'soft_skills_score',
    'interview_score',
    'consistency_score',
    'backlogs'
]

metric = st.selectbox(
    "Select Metric",
    numeric_cols
)

fig = px.box(
    df,
    x="selected",
    y=metric,
    color="selected",
    title=f"{metric} vs Selection"
)

st.plotly_chart(fig,use_container_width=True)

# =======================
# College Tier Analysis
# =======================

st.subheader("College Tier Analysis")

tier_analysis = (
    df.groupby("college_tier")
    .agg(
        Students=("student_id","count"),
        Selection_Rate=("selected","mean"),
        Avg_CGPA=("CGPA","mean")
    )
    .reset_index()
)

tier_analysis["Selection_Rate"] *= 100

fig = px.bar(
    tier_analysis,
    x="college_tier",
    y="Selection_Rate",
    color="college_tier",
    title="Selection Rate by College Tier"
)

st.plotly_chart(fig,use_container_width=True)

# =======================
# Placement Training Impact
# =======================

st.subheader("Placement Training Impact")

training_df = (
    df.groupby("placement_training")
    .selected.mean()
    .reset_index()
)

training_df["selected"] *= 100

fig = px.bar(
    training_df,
    x="placement_training",
    y="selected",
    color="placement_training",
    title="Selection Rate by Placement Training"
)

st.plotly_chart(fig,use_container_width=True)

# =======================
# Correlation Heatmap
# =======================

st.subheader("Correlation Analysis")

corr = df[numeric_cols + ["selected"]].corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Correlation Heatmap"
)

st.plotly_chart(fig,use_container_width=True)

# =======================
# Key Insights
# =======================

st.subheader("Automated Insights")

best_cgpa = df.groupby("selected")["CGPA"].mean()

st.success(
    f"""
    • Selected students average CGPA: {best_cgpa[1]:.2f}

    • Non-selected students average CGPA: {best_cgpa[0]:.2f}

    • Overall selection rate: {selected_rate}%

    • Highest impact variables can be observed from correlation matrix.

    • Placement training and coding scores significantly influence selection outcomes.
    """
)

st.dataframe(df.head(100))
