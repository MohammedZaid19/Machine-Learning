import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:/Downloads/ev_market_2026.csv')

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe())

quality_counts = df['brand'].value_counts()
plt.figure(figsize=(8,6))
plt.bar(quality_counts.index, quality_counts, color='red')
plt.title('count plot as quality')
plt.xlabel('brand name')
plt.ylabel('count')
plt.show()

sns.set_palette("Pastel1")
plt.figure(figsize=(6,10))
sns.pairplot(df)
plt.suptitle("hi lol")
plt.show()

sns.boxplot(x='model',y='year',data=df)
plt.show()

# 1D Kernel Density plot
sns.kdeplot(data=df, x='weight_kg', fill=True)
plt.show()

# 2D Kernel Density Plot
sns.kdeplot(data=df, x='year', y='annual_sales_units', fill=True, thresh=0, cmap="mako")
plt.show()

# Grouped Comparisons
sns.kdeplot(data=df, x='year', hue='warranty_years', fill=True, alpha=0.5)
plt.show()