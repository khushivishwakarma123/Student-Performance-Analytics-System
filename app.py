
import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Analytics System",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e1e7ef;
        padding: 16px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 2. LOAD CSV USING PYSPARK
# --------------------------------------------------

@st.cache_data
def load_student_data(uploaded_bytes=None):
    base_folder = Path(__file__).parent
    temporary_path = None

    try:
        if uploaded_bytes is not None:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".csv",
                delete=False
            ) as temp_file:
                temp_file.write(uploaded_bytes)
                temporary_path = temp_file.name

            csv_path = temporary_path
        else:
            csv_path = str(base_folder / "students.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                "students.csv was not found in C:\\spark_lab."
            )

        spark = (
            SparkSession.builder
            .appName("Student Analytics Dashboard")
            .master("local[*]")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("ERROR")

        try:
            spark_df = spark.read.csv(
                csv_path,
                header=True,
                inferSchema=True
            )

            required = [
                "StudentID",
                "Name",
                "StudyHours",
                "Attendance",
                "Marks"
            ]

            missing = [
                name for name in required
                if name not in spark_df.columns
            ]

            if missing:
                raise ValueError(
                    "CSV is missing columns: "
                    + ", ".join(missing)
                )

            # PySpark transformation: calculate result
            spark_df = spark_df.withColumn(
                "Result",
                when(col("Marks") >= 35, "Pass")
                .otherwise("Fail")
            )

            data = spark_df.toPandas()

        finally:
            spark.stop()

        # Validate and clean numeric values
        numeric_columns = [
            "StudentID",
            "StudyHours",
            "Attendance",
            "Marks"
        ]

        for column in numeric_columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

        data["Name"] = data["Name"].astype("string").str.strip()

        data = data.dropna(
            subset=[
                "Name", "StudyHours", "Attendance", "Marks"
            ]
        )

        data = data[
            data["Name"].ne("")
            & data["Marks"].between(0, 100)
            & data["Attendance"].between(0, 100)
            & data["StudyHours"].ge(0)
        ].copy()

        # Grade assignment
        data["Grade"] = data["Marks"].apply(
            lambda marks:
                "A" if marks >= 90 else
                "B" if marks >= 80 else
                "C" if marks >= 70 else
                "D" if marks >= 60 else
                "E" if marks >= 35 else
                "F"
        )

        return data

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


# --------------------------------------------------
# 3. HEADER
# --------------------------------------------------

st.title("📊 Student Performance Analytics System")

st.write(
    "Analyze student marks, attendance, study hours, "
    "grades, and examination results using PySpark."
)

st.caption("Technology: Python | PySpark | Streamlit | Pandas | Plotly")

st.divider()


# --------------------------------------------------
# 4. SIDEBAR: UPLOAD AND REFRESH
# --------------------------------------------------

st.sidebar.title("⚙️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload a student CSV file",
    type=["csv"],
    help=(
        "Required columns: StudentID, Name, StudyHours, "
        "Attendance, Marks"
    )
)

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    st.sidebar.success("Uploaded: " + uploaded_file.name)
else:
    uploaded_bytes = None
    st.sidebar.info("Using the default students.csv file.")

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# --------------------------------------------------
# 5. LOAD DATA
# --------------------------------------------------

try:
    df = load_student_data(uploaded_bytes)

except Exception as error:
    st.error(f"Unable to load data: {error}")
    st.info(
        "Check your CSV columns and make sure students.csv "
        "exists in the project folder."
    )
    st.stop()

if df.empty:
    st.warning("No valid student records were found.")
    st.stop()


# --------------------------------------------------
# 6. FILTERS
# --------------------------------------------------

st.sidebar.subheader("🔎 Filter Students")

selected_result = st.sidebar.selectbox(
    "Examination Result",
    ["All", "Pass", "Fail"]
)

min_marks = st.sidebar.slider(
    "Minimum Marks",
    0, 100, 0
)

max_marks = st.sidebar.slider(
    "Maximum Marks",
    min_marks, 100, 100
)

filtered_df = df[
    df["Marks"].between(min_marks, max_marks)
].copy()

if selected_result != "All":
    filtered_df = filtered_df[
        filtered_df["Result"] == selected_result
    ]

st.sidebar.caption(
    f"{len(filtered_df)} of {len(df)} students displayed"
)


# --------------------------------------------------
# 7. SUMMARY METRICS
# --------------------------------------------------

st.subheader("📌 Performance Overview")

total = len(filtered_df)

passed = int((filtered_df["Result"] == "Pass").sum())
failed = int((filtered_df["Result"] == "Fail").sum())

pass_percentage = (
    passed / total * 100 if total else 0
)

average_marks = filtered_df["Marks"].mean()
average_attendance = filtered_df["Attendance"].mean()

topper = (
    filtered_df.loc[filtered_df["Marks"].idxmax()]
    if total else None
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total Students", total)
m2.metric(
    "Average Marks",
    f"{average_marks:.2f}" if total else "N/A"
)
m3.metric("Passed", passed)
m4.metric("Failed", failed)

m5, m6, m7 = st.columns(3)

m5.metric(
    "Pass Percentage",
    f"{pass_percentage:.1f}%"
)

m6.metric(
    "Average Attendance",
    f"{average_attendance:.2f}%"
    if total else "N/A"
)

m7.metric(
    "Topper",
    str(topper["Name"]) if topper is not None else "N/A",
    f'{topper["Marks"]:g} marks'
    if topper is not None else None
)

st.divider()


# --------------------------------------------------
# 8. PERFORMANCE CHARTS
# --------------------------------------------------

st.subheader("📈 Student Performance Charts")

if filtered_df.empty:
    st.warning("No students match the selected filters.")

else:
    left, right = st.columns(2)

    with left:
        marks_chart = px.bar(
            filtered_df.sort_values(
                "Marks", ascending=False
            ),
            x="Name",
            y="Marks",
            color="Result",
            text="Marks",
            title="Marks by Student",
            color_discrete_map={
                "Pass": "#16a34a",
                "Fail": "#ef4444"
            }
        )

        marks_chart.update_layout(
            yaxis_range=[0, 100],
            xaxis_title="Student",
            yaxis_title="Marks",
            xaxis_tickangle=-30
        )

        st.plotly_chart(
            marks_chart,
            use_container_width=True
        )

    with right:
        result_counts = (
            filtered_df["Result"]
            .value_counts()
            .reindex(["Pass", "Fail"], fill_value=0)
            .rename_axis("Result")
            .reset_index(name="Students")
        )

        result_chart = px.pie(
            result_counts,
            names="Result",
            values="Students",
            title="Pass vs Fail Distribution",
            hole=0.45,
            color="Result",
            color_discrete_map={
                "Pass": "#16a34a",
                "Fail": "#ef4444"
            }
        )

        st.plotly_chart(
            result_chart,
            use_container_width=True
        )

    study_chart = px.scatter(
        filtered_df,
        x="StudyHours",
        y="Marks",
        color="Result",
        size="Attendance",
        hover_name="Name",
        title="Study Hours vs Marks",
        color_discrete_map={
            "Pass": "#16a34a",
            "Fail": "#ef4444"
        }
    )

    study_chart.update_layout(
        yaxis_range=[0, 100],
        xaxis_title="Study Hours",
        yaxis_title="Marks"
    )

    st.plotly_chart(
        study_chart,
        use_container_width=True
    )

    grade_counts = (
        filtered_df["Grade"]
        .value_counts()
        .reindex(["A", "B", "C", "D", "E", "F"], fill_value=0)
        .rename_axis("Grade")
        .reset_index(name="Students")
    )

    grade_chart = px.bar(
        grade_counts,
        x="Grade",
        y="Students",
        title="Grade Distribution",
        text="Students"
    )

    st.plotly_chart(
        grade_chart,
        use_container_width=True
    )


# --------------------------------------------------
# 9. STUDENTS NEEDING IMPROVEMENT
# --------------------------------------------------

st.divider()

st.subheader("🎯 Students Needing Improvement")

st.caption(
    "Students scoring below 50 marks are listed here "
    "as a simple academic-support indicator."
)

needs_improvement = filtered_df[
    filtered_df["Marks"] < 50
].sort_values("Marks")

if needs_improvement.empty:
    st.success(
        "No students below 50 marks in the current selection."
    )
else:
    st.dataframe(
        needs_improvement[
            [
                "StudentID", "Name", "StudyHours",
                "Attendance", "Marks", "Grade", "Result"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# 10. STUDENT RECORDS
# --------------------------------------------------

st.divider()

st.subheader("📋 Student Records")

display_columns = [
    "StudentID", "Name", "StudyHours",
    "Attendance", "Marks", "Grade", "Result"
]

display_df = filtered_df[display_columns].sort_values(
    "Marks", ascending=False
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 11. DOWNLOAD CSV AND EXCEL REPORTS
# --------------------------------------------------

st.divider()

st.subheader("📥 Export Reports")

download1, download2 = st.columns(2)

csv_bytes = display_df.to_csv(
    index=False
).encode("utf-8")

with download1:
    st.download_button(
        "Download Student Results (CSV)",
        data=csv_bytes,
        file_name="student_results.csv",
        mime="text/csv",
        use_container_width=True
    )

excel_buffer = io.BytesIO()

summary_df = pd.DataFrame({
    "Metric": [
        "Total Students",
        "Passed",
        "Failed",
        "Pass Percentage",
        "Average Marks",
        "Average Attendance"
    ],
    "Value": [
        total,
        passed,
        failed,
        round(pass_percentage, 2),
        round(average_marks, 2) if total else 0,
        round(average_attendance, 2) if total else 0
    ]
})

with pd.ExcelWriter(
    excel_buffer,
    engine="openpyxl"
) as writer:
    display_df.to_excel(
        writer,
        sheet_name="Student Records",
        index=False
    )

    summary_df.to_excel(
        writer,
        sheet_name="Performance Summary",
        index=False
    )

with download2:
    st.download_button(
        "Download Report (Excel)",
        data=excel_buffer.getvalue(),
        file_name="student_performance_report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Student Performance Analytics System | "
    "Powered by PySpark, Python, Streamlit, Pandas and Plotly"
)
