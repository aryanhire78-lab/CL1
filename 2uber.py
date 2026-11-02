import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("uber.csv")
print("Initial Shape:", df.shape)

df = df.drop(['Unnamed: 0', 'key'], axis=1, errors='ignore')
df = df.dropna()

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
df = df.drop('pickup_datetime', axis=1)

df = df[
    (df['pickup_latitude'].between(-90, 90)) &
    (df['dropoff_latitude'].between(-90, 90)) &
    (df['pickup_longitude'].between(-180, 180)) &
    (df['dropoff_longitude'].between(-180, 180)) &
    (df['fare_amount'] > 0) &
    (df['passenger_count'].between(1, 6))
]


df['distance_km'] = np.sqrt(
    (df['dropoff_latitude'] - df['pickup_latitude'])**2 + 
    (df['dropoff_longitude'] - df['pickup_longitude'])**2
) * 111  

plt.figure(figsize=(6, 3))
sns.boxplot(x=df['fare_amount'])
plt.title("Boxplot of Fare Amount (Before Outlier Removal)")
plt.show()

def remove_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

for col in ['fare_amount', 'distance_km']:
    df = remove_outliers_iqr(df, col)

df = df[df['distance_km'] > 0]
print("Shape after Preprocessing & Outlier Removal:", df.shape)

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()

X = df[['distance_km', 'passenger_count', 'hour', 'day', 'month', 'dayofweek']]
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

print("\n--- Model Evaluation Results ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"{name:18} -> R2 Score: {r2:.4f} | RMSE: {rmse:.4f}")