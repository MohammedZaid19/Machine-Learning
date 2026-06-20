
# Step 1 : Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2 : Create Dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Step 3 : Separate Features (X) and Target (y)
X = df[["Hours_Studied"]]
y = df["Pass"]

# Step 4 : Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

# Step 5 : Create Model
model = LogisticRegression()

# Step 6 : Train Model
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# Step 7 : Make Predictions on Test Data
y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

# Step 8 : Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 9 : User Prediction
hours = float(input("\nEnter Hours Studied: "))

new_data = pd.DataFrame({
    "Hours_Studied": [hours]
})

prediction = model.predict(new_data)
probability = model.predict_proba(new_data)

print("\nPrediction Probability")
print("Fail Probability:", round(probability[0][0] * 100, 2), "%")
print("Pass Probability:", round(probability[0][1] * 100, 2), "%")

if prediction[0] == 1:
    print("\nPrediction: PASS")
else:
    print("\nPrediction: FAIL")

# Step 10 : Display Learned Parameters
print("\nModel Parameters")
print("Coefficient (Slope):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])

print("\nLogistic Regression Equation:")
print(
    f"z = ({model.coef_[0][0]:.4f} × Hours_Studied) + ({model.intercept_[0]:.4f})"
)

print("\nProbability Equation:")
print("P(Pass) = 1 / (1 + e^(-z))")