import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, desc, month, year
import pandas as pd


# Initializing the Spark
spark = SparkSession.builder.appName("OnlineRetailAnalysis").getOrCreate()

# Data Extraction and data Loading
df = pd.read_excel("C:\\Users\\Oleg\\B142\\Online Retail.xlsx")
df.to_csv("Online_Retail.csv", index=False)  # Convert to CSV for easier loading with PySpark
df = spark.read.csv("Online_Retail.csv", header=True, inferSchema=True)

# Data Cleaning
df_cleaned = df.dropna().dropDuplicates()

# Show the first few rows
df_cleaned.show()


# Top 10 Selling Products
top_products = df_cleaned.groupBy("Description").agg(_sum("Quantity").alias("TotalQuantity")).orderBy(desc("TotalQuantity")).limit(10)
top_products.show()

# Sales by Country
sales_by_country = df_cleaned.groupBy("Country").agg(_sum("Quantity").alias("TotalQuantity")).orderBy(desc("TotalQuantity"))
sales_by_country.show()


#  Revenue Calculation
df_cleaned = df_cleaned.withColumn("Revenue", col("Quantity") * col("UnitPrice"))
revenue_by_product = df_cleaned.groupBy("Description").agg(_sum("Revenue").alias("TotalRevenue")).orderBy(desc("TotalRevenue")).limit(10)
revenue_by_product.show()

revenue_by_country = df_cleaned.groupBy("Country").agg(_sum("Revenue").alias("TotalRevenue")).orderBy(desc("TotalRevenue"))
revenue_by_country.show()

#  Monthly Sales Trends
monthly_sales = df_cleaned.withColumn("Year", year("InvoiceDate")).withColumn("Month", month("InvoiceDate"))
monthly_sales_trend = monthly_sales.groupBy("Year", "Month").agg(_sum("Quantity").alias("TotalQuantity"), _sum("Revenue").alias("TotalRevenue")).orderBy("Year", "Month")
monthly_sales_trend.show()

# Customer Segmentation
customer_segmentation = df_cleaned.groupBy("CustomerID").agg(_sum("Quantity").alias("TotalQuantity"), _sum("Revenue").alias("TotalRevenue")).orderBy(desc("TotalRevenue"))
customer_segmentation.show()

# Filter top customers for segmentation
top_customers = customer_segmentation.limit(10)
top_customers.show()

# Convert PySpark DataFrame to Pandas DataFrame
monthly_sales_trend_pd = monthly_sales_trend.toPandas()

# Plot Monthly Sales Trend
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales_trend_pd, x="Month", y="TotalRevenue", hue="Year", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.show()
