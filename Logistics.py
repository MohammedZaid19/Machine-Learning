# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

# Create a dataset
data = {
    "Hours_Studied":[1,2,3,4,5,6,7,8,9,10],
    "Pass":[0,0,0,0,1,1,1,1]
}
df = pd.DataFrame(data)
print("Dataset : ")
print(df)

# Separate features of X and target y
X = df[["Hours_Studied"]]
y = df["Pass"]

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
print("\nTraining Data Size : ",len(X_train))
print("\nTesting Data Size : ",len(X_test))

# create a model
model = LogisticRegression()

# Train model
model.fit(X_train,y_train)
print("\nModel Trained Successfully : ")

# Make predictions on test data
y_pred = model.predict(X_test)