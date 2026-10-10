
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, when, count

# 1. Start Spark
spark = SparkSession.builder \
    .appName("Student Performance Analysis") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. Read student data
df = spark.read.csv(
    "students.csv",
    header=True,
    inferSchema=True
)

print("\n===== ALL STUDENTS =====")
df.show()

# 3. Add Pass/Fail column
result = df.withColumn(
    "Result",
    when(col("Marks") >= 35, "Pass").otherwise("Fail")
)

print("\n===== STUDENT RESULTS =====")
result.show()

# 4. Calculate average marks
print("\n===== AVERAGE MARKS =====")
df.select(
    avg("Marks").alias("Average_Marks")
).show()

# 5. Find students scoring 60 or more
print("\n===== STUDENTS WITH MARKS >= 60 =====")
df.filter(col("Marks") >= 60).show()

# 6. Count Pass and Fail students
print("\n===== PASS AND FAIL COUNT =====")
result.groupBy("Result") \
    .agg(count("*").alias("Total_Students")) \
    .show()

# 7. Average marks by result
print("\n===== AVERAGE MARKS BY RESULT =====")
result.groupBy("Result") \
    .agg(avg("Marks").alias("Average_Marks")) \
    .show()

# 8. Find the topper
print("\n===== TOPPER =====")
df.orderBy(col("Marks").desc()) \
    .select("Name", "Marks") \
    .show(1)

# 9. Calculate average attendance
print("\n===== AVERAGE ATTENDANCE =====")
df.select(
    avg("Attendance").alias("Average_Attendance")
).show()

# 10. Analyze marks by study hours
print("\n===== STUDY HOURS ANALYSIS =====")
df.groupBy("StudyHours") \
    .agg(avg("Marks").alias("Average_Marks")) \
    .orderBy("StudyHours") \
    .show()

# 11. Stop Spark
spark.stop()
