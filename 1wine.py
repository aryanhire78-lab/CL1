import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print(df.head())

X = df.drop('Customer_Segment', axis=1)
y = df['Customer_Segment']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Wine_Class'] = y

print("\nExplained Variance Ratio of each component:", pca.explained_variance_ratio_)
print(f"Total Variance Captured by 2 Components: { sum(pca.explained_variance_ratio_) * 100:.2f}%")

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Principal Component 1', 
    y='Principal Component 2', 
    hue='Wine_Class', 
    palette='Set1', 
    data=pca_df, 
    s=80
)
plt.title('PCA on Wine Dataset (2 Principal Components)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)')
plt.grid(True)
plt.show()