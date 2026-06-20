import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')
df = pd.read_csv('D:/Downloads/archive (4)/mail_data.csv')
# print(df.head())
# print(df.shape)
# print(df.info())
print(df.isnull().sum())
# print(df.describe())
# print(df.describe().T)
print(df.columns.tolist())
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())

# category = df['Category'].value_counts()
# plt.figure(figsize=(8,6))
# plt.bar(category.index,category,color='blue')
# plt.title("Email Spam")
# plt.xlabel("Category")
# plt.ylabel("Messages")
# plt.show()

# plt.figure(figsize=(10,8))
# sns.swarmplot(x="Category",y="Message",data=df,palette='viridis')
# plt.title("Swarmplot for email spam detection")
# plt.xlabel("Category")
# plt.ylabel("Messages")
# plt.show()



sns.boxplot(x='Category',y='Message',data=df)
# plt.suptitle('Pair plot for dataframe')
plt.show()