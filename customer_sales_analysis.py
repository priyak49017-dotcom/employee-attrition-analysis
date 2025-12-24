import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/raw_transactions.csv")

print("RAW DATA")
print(df.head())

print("\nDATA INFO")
print(df.info())
# Handle missing values safely
df["experience"] = df["experience"].fillna(df["experience"].median())
df["salary"] = df["salary"].fillna(df["salary"].mean())

# Convert joining_date safely
df["joining_date"] = pd.to_datetime(df["joining_date"], errors="coerce")

# Fill missing dates with most frequent date
df["joining_date"] = df["joining_date"].fillna(df["joining_date"].mode()[0])

# Remove duplicates
df = df.drop_duplicates()
# STEP 7: Feature Engineering

# Convert attrition Yes/No into numerical form
df["attrition_flag"] = df["attrition"].map({
    "Yes": 1,
    "No": 0
})

# Create experience level categories
df["experience_level"] = pd.cut(
    df["experience"],
    bins=[0, 2, 5, 10],
    labels=["Junior", "Mid", "Senior"]
)

print("\nFEATURE ENGINEERING DONE")
print(df[["attrition", "attrition_flag", "experience", "experience_level"]].head())
attrition_by_dept = df.groupby("department")["attrition_flag"].mean()
avg_salary_by_dept = df.groupby("department")["salary"].mean()
performance_vs_attrition = df.groupby("attrition")["performance_rating"].mean()
import os
os.makedirs("visuals", exist_ok=True)

# Attrition Rate by Department
attrition_by_dept.plot(kind="bar", title="Attrition Rate by Department")
plt.ylabel("Attrition Rate")
plt.tight_layout()
plt.savefig("visuals/attrition_by_department.png")
plt.show()

# Average Salary by Department
avg_salary_by_dept.plot(kind="bar", title="Average Salary by Department")
plt.ylabel("Salary")
plt.tight_layout()
plt.savefig("visuals/salary_by_department.png")
plt.show()

# Performance vs Attrition
performance_vs_attrition.plot(kind="bar", title="Performance vs Attrition")
plt.ylabel("Performance Rating")
plt.tight_layout()
plt.savefig("visuals/performance_attrition.png")
plt.show()
#scatterplot
plt.figure()
for status in df["attrition"].unique():
    subset = df[df["attrition"] == status]
    plt.scatter(
        subset["experience"],
        subset["salary"],
        label=status,
        alpha=0.7
    )
    

plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Experience vs Salary by Attrition")
plt.legend()
plt.tight_layout()
plt.savefig("visuals/experience_vs_salary_scatter.png")
plt.show()
plt.close()

#box plot
plt.figure()
df.boxplot(column="salary", by="attrition")
plt.title("Salary Distribution by Attrition")
plt.suptitle("")  # removes default title
plt.xlabel("Attrition")
plt.ylabel("Salary")
plt.tight_layout()
plt.savefig("visuals/salary_boxplot_attrition.png")
plt.show()
plt.close()
print("\n===== EMPLOYEE ATTRITION INSIGHTS =====")

print("Overall Attrition Rate:",
      round(df["attrition_flag"].mean() * 100, 2), "%")

high_attrition_dept = attrition_by_dept.idxmax()
print("Department with Highest Attrition:", high_attrition_dept)

low_perf_attrition = performance_vs_attrition["Yes"]
print("Avg Performance of Employees Who Left:", low_perf_attrition)

print("======================================")