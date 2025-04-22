# Importing the Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setting Plot Styles
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Loading the Dataset
file_path = "Tobacco_Retail_Dealer_and_Electronic_Cigarette_Retail_Dealer_Caps_by_Community_District_20250407.csv"
df = pd.read_csv(file_path)

# Displaying First 5 Rows
print("First 5 Rows of the Dataset:")
print(df.head())

# Objective 1: Analyze Total Licenses Issued Per Borough
license_totals = df.groupby("Borough")[["Active Tobacco Retail Dealer Licenses",
                                       "Active Electronic Cigarette Retail Dealer Licenses"]].sum().reset_index()

license_totals.plot(x='Borough', kind='bar', stacked=True, color=['#a0522d', '#800080'])
plt.title("Total Active Licenses by Borough")
plt.ylabel("Number of Active Licenses")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 2: Find Districts Exceeding the Cap Limit
over_limit = df[
    (df["Active Tobacco Retail Dealer Licenses"] > df["Tobacco Retail Dealer Cap"]) | 
    (df["Active Electronic Cigarette Retail Dealer Licenses"] > df["Electronic Cigarette Retail Dealer Cap"])
]
print("\nDistricts Exceeding License Cap Limits:")
print(over_limit[["Community District Name", "Borough"]])

# Objective 3: Calculate License Utilization Percentage
df["Tobacco Utilization %"] = np.round((df["Active Tobacco Retail Dealer Licenses"] / df["Tobacco Retail Dealer Cap"]) * 100, 2)
df["E-Cigarette Utilization %"] = np.round((df["Active Electronic Cigarette Retail Dealer Licenses"] / df["Electronic Cigarette Retail Dealer Cap"]) * 100, 2)

print("\nLicense Utilization % (First 5 rows):")
print(df[["Community District Name", "Tobacco Utilization %", "E-Cigarette Utilization %"]].head())

# Objective 4: Visualize Utilization by Community District
plt.figure(figsize=(12,6))
sns.histplot(df["Tobacco Utilization %"], kde=True, color='brown', label='Tobacco')
sns.histplot(df["E-Cigarette Utilization %"], kde=True, color='purple', label='E-Cigarette')
plt.title("Distribution of License Utilization % by Community District")
plt.xlabel("Utilization %")
plt.legend()
plt.show()

# Objective 5: Compare Average Caps and Licenses Across Boroughs
avg_stats = df.groupby("Borough")[["Tobacco Retail Dealer Cap", "Active Tobacco Retail Dealer Licenses",
                                   "Electronic Cigarette Retail Dealer Cap", "Active Electronic Cigarette Retail Dealer Licenses"]].mean()

avg_stats.plot(kind='bar', color=['#deb887', '#a0522d', '#dda0dd', '#800080'])
plt.title("Average Caps vs Active Licenses by Borough")
plt.ylabel("Average Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 6: Identify Top 5 Most Regulated Districts
df["Total Cap"] = df["Tobacco Retail Dealer Cap"] + df["Electronic Cigarette Retail Dealer Cap"]
top_districts = df.sort_values("Total Cap", ascending=False).head(5)
print("\nTop 5 Most Regulated Districts by Total Cap:")
print(top_districts[["Community District Name", "Total Cap"]])

# Objective 7: Borough-wise Correlation Between Cap and Active Licenses
def calculate_corr(sub_df):
    return sub_df["Tobacco Retail Dealer Cap"].corr(sub_df["Active Tobacco Retail Dealer Licenses"])

correlation = df.groupby("Borough").apply(calculate_corr).reset_index(name="Tobacco Correlation")
print("\nCorrelation Between Cap and Active Licenses in Each Borough:")
print(correlation)

# Objective 8: Highlight Districts with No Available Licenses
zero_avail = df[(df["TRD Available Under Cap"] == 0) & (df["ECD Available Under Cap"] == 0)]
plt.figure(figsize=(10,6))
sns.countplot(data=zero_avail, y="Borough", palette="Set2")
plt.title("Districts with No Available Licenses by Borough")
plt.xlabel("Number of Districts")
plt.ylabel("Borough")
plt.show()

# Objective 9: Heatmap of Feature Correlations
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Objective 10: Summary Statistics Using NumPy
tobacco_mean = np.mean(df["Tobacco Utilization %"])
ecig_mean = np.mean(df["E-Cigarette Utilization %"])
print(f"\nAverage Tobacco Utilization: {tobacco_mean:.2f}%")
print(f"Average E-Cigarette Utilization: {ecig_mean:.2f}%")

# Objective 11: Multi-Bar Chart - Borough Wise Cap Comparison
caps = df.groupby("Borough")[["Tobacco Retail Dealer Cap", "Electronic Cigarette Retail Dealer Cap"]].sum().reset_index()

caps.plot(
    x='Borough',
    kind='bar',
    color=['#8B4513', '#4B0082'],
    figsize=(10,6)
)
plt.title("Total License Caps by Borough")
plt.ylabel("Number of Caps")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 12 (Improved): Line Graph - Avg Caps vs Active Licenses by Borough

# Grouping by Borough to reduce clutter
avg_caps_line = df.groupby("Borough")[["Tobacco Retail Dealer Cap", "Active Tobacco Retail Dealer Licenses"]].mean().reset_index()

# Sorting by Cap
avg_caps_line = avg_caps_line.sort_values("Tobacco Retail Dealer Cap")

# Line Plot
plt.figure(figsize=(10,6))
plt.plot(avg_caps_line["Borough"], avg_caps_line["Tobacco Retail Dealer Cap"], 
         marker='o', linestyle='--', color='brown', label='Avg Cap')
plt.plot(avg_caps_line["Borough"], avg_caps_line["Active Tobacco Retail Dealer Licenses"], 
         marker='s', linestyle='-', color='orange', label='Avg Active Licenses')

# Labels and Aesthetics
plt.title("Average Tobacco Caps vs Active Licenses by Borough")
plt.xlabel("Borough")
plt.ylabel("Average Count")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()


# Objective 13: Subplots for Tobacco and E-Cigarette Caps vs Active Licenses
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Tobacco Graph
sns.barplot(data=df, x="Borough", y="Tobacco Retail Dealer Cap", ax=axs[0], color="#a0522d", label="Cap")
sns.barplot(data=df, x="Borough", y="Active Tobacco Retail Dealer Licenses", ax=axs[0], color="#deb887", alpha=0.7, label="Active")
axs[0].set_title("Tobacco Cap vs Active Licenses")
axs[0].legend()

# E-Cigarette Graph
sns.barplot(data=df, x="Borough", y="Electronic Cigarette Retail Dealer Cap", ax=axs[1], color="#4B0082", label="Cap")
sns.barplot(data=df, x="Borough", y="Active Electronic Cigarette Retail Dealer Licenses", ax=axs[1], color="#dda0dd", alpha=0.7, label="Active")
axs[1].set_title("E-Cigarette Cap vs Active Licenses")
axs[1].legend()

plt.tight_layout()
plt.show()

# Objective 14: Pie Chart - Borough Share of Tobacco Caps
cap_share = df.groupby("Borough")["Tobacco Retail Dealer Cap"].sum()

plt.figure(figsize=(8,8))
plt.pie(cap_share, labels=cap_share.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
plt.title("Share of Total Tobacco Caps by Borough")
plt.axis('equal')
plt.show()


# Objective 15: Scatter Plot - Cap vs Active Licenses
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="Tobacco Retail Dealer Cap", y="Active Tobacco Retail Dealer Licenses", hue="Borough", style="Borough", s=100)
plt.title("Relationship: Cap vs Active Licenses (Tobacco)")
plt.xlabel("Tobacco Retail Dealer Cap")
plt.ylabel("Active Licenses")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
