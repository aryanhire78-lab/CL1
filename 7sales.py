
# Data Loading, Storage and File Formats

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Create sample dataset
data = {
    'Order_ID': [101,102,103,104,105,105,106,107],
    'Customer_Name': ['Alice','Bob','Charlie','David',
                      'Eva','Eva','Frank',None],
    'Product_Category': ['Electronics','Furniture','electronics',
                         'Clothing','Furniture','Furniture',
                         'Clothing','Electronics'],
    'Quantity': [2,1,3,np.nan,5,5,2,4],
    'Unit_Price': [500,1200,300,50,450,450,80,700],
    'Order_Date': ['2025-01-10','2025-01-12','2025-01-15',
                   '2025-01-18','2025-01-20','2025-01-20',
                   '2025-01-22','2025-01-25']
}

df = pd.DataFrame(data)

# 3. Save data in different formats
df.iloc[:3].to_csv('sales.csv', index=False)
df.iloc[3:6].to_excel('sales.xlsx', index=False)
df.iloc[6:].to_json('sales.json', orient='records')

# 4. Load files
csv = pd.read_csv('sales.csv')
excel = pd.read_excel('sales.xlsx')
json = pd.read_json('sales.json')

# 5. Combine datasets
df = pd.concat([csv, excel, json], ignore_index=True)

print("Dataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

# 6. Data cleaning
df['Customer_Name'] = df['Customer_Name'].fillna('Unknown')
df['Quantity'] = df['Quantity'].fillna(df['Quantity'].median())

df.drop_duplicates(inplace=True)

df['Product_Category'] = df['Product_Category'].str.title()

# 7. Data transformation
df[['First_Name', 'Last_Name']] = (
    df['Customer_Name'].str.split(' ', n=1, expand=True)
    .reindex(columns=[0, 1])
)

df['Total_Sales'] = df['Quantity'] * df['Unit_Price']
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

print("\nTransformed Data:")
print(df.head())

# 8. Data analysis
print("\nSummary:")
print(df.describe())

print("\nTotal Sales:", df['Total_Sales'].sum())
print("Average Order Value:", df['Total_Sales'].mean())

category_sales = df.groupby('Product_Category')['Total_Sales'].sum()

print("\nCategory Sales:")
print(category_sales)

# 9. Data visualization
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
category_sales.plot(kind='bar')
plt.title("Sales by Category")
plt.ylabel("Total Sales")

plt.subplot(1,3,2)
category_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title("Sales Distribution")
plt.ylabel("")

plt.subplot(1,3,3)
sns.boxplot(x='Product_Category', y='Total_Sales', data=df)
plt.title("Sales Boxplot")

plt.tight_layout()
plt.show()
