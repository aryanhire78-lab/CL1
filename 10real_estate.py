
# Data Wrangling on Real Estate Market

# 1. Import libraries
import pandas as pd
import numpy as np

# 2. Create Real Estate Dataset
data = {
    'Property ID ': [1,2,3,4,5,6,7,8],
    'Neighborhood': ['Downtown','Suburb','Downtown','Rural',
                     'Suburb','Downtown','Suburb','Rural'],
    'Property-Type': ['Residential','Residential','Commercial',
                      'Residential','Residential','Residential',
                      'Commercial','Residential'],
    'Square Footage': [1500,2200,np.nan,1800,2500,1200,3000,1600],
    'Year Built': [2005,2010,1995,2020,np.nan,2018,2015,1980],
    'Sale Price ($)': [350000,420000,800000,310000,460000,
                       3000000,950000,290000]
}

df = pd.DataFrame(data)
df.to_csv('RealEstate_Prices.csv', index=False)

# 3. Create Demographics Dataset
demo = pd.DataFrame({
    'Neighborhood': ['Downtown','Suburb','Rural'],
    'Avg_Income': [85000,95000,65000],
    'Schools_Rating': [8,9,6]
})

demo.to_csv('Neighborhood_Demographics.csv', index=False)

# 4. Load dataset and clean column names
df = pd.read_csv('RealEstate_Prices.csv')

df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r'[^a-z0-9]+', '_', regex=True)
              .str.strip('_'))

print("Columns:", df.columns.tolist())

# 5. Handle missing values
df['square_footage'] = df['square_footage'].fillna(
    df['square_footage'].median())

df['year_built'] = df['year_built'].fillna(
    df['year_built'].mode()[0])

# 6. Merge datasets
demo = pd.read_csv('Neighborhood_Demographics.csv')
demo.columns = demo.columns.str.strip().str.lower()

df = pd.merge(df, demo, on='neighborhood', how='left')

print("\nMerged Data:")
print(df.head())

# 7. Filter data
filtered = df[
    (df['property_type'] == 'Residential') &
    (df['year_built'] >= 2000)
].copy()

print("\nFiltered Data:")
print(filtered)

# 8. Encode categorical variables
encoded = pd.get_dummies(
    filtered,
    columns=['neighborhood', 'property_type'],
    drop_first=True
)

print("\nEncoded Data:")
print(encoded.head())

# 9. Aggregate data
summary = df.groupby('neighborhood').agg(
    Average_Price=('sale_price', 'mean'),
    Average_Area=('square_footage', 'mean'),
    Property_Count=('property_id', 'count')
)

print("\nSummary Statistics:")
print(summary)

# 10. Identify and remove outliers using IQR
Q1 = df['sale_price'].quantile(0.25)
Q3 = df['sale_price'].quantile(0.75)
IQR = Q3 - Q1

df_final = df[df['sale_price'].between(
    Q1 - 1.5*IQR,
    Q3 + 1.5*IQR
)]

print("\nOriginal Shape:", df.shape)
print("After Removing Outliers:", df_final.shape)

# 11. Save cleaned dataset
df_final.to_csv('Cleaned_RealEstate.csv', index=False)

print("\nData Wrangling Completed Successfully!")
