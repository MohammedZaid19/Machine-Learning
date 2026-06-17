# step 1 Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

# Step 2 Create Dataset
data = {
    "Hours Studied" : [1,2,3,4,5,6,7,8,9,10],
    "Marks" : [20,25,35,40,50,60,70,75,85,95]
}
df = pd.DataFrame(data)
print("Dataset : ")
print(df)

# Step 3 : Separate features (X) and Target (Y)
X = df[["Hours Studied"]]
y = df["Marks"]

# Step 4 : Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("\nTraining Data Size:",len(X_train))
print("\nTesting Data Size:",len(X_test))

# Step 5 : Create Model
model = LinearRegression()

# Step 6 : Train Model
model.fit(X_train, y_train)
print("\nModel Trained Successfully!")

# Step 7 : Make Predictions on Test Data
y_pred = model.predict(X_test)

print("\nActual Marks : ")
print(y_test.values)
print("\nPredicted Marks : ")
print(y_pred)

# Step 8 : Evaluate Model
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("\nModel Evaluation")
print("Mean Squared Error (MSE) : ",mse)
print("R2 Score : ",r2)

# Step 9 : Predict for New Data
hours = float(input("\nEnter Hours Studied : "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks for {hours} hours of Study : {predicted_marks[0]:.2f}")

# Step 10 Display Learned Equation
print("\nModel Parameters")
print("Slope (Coefficient) : ",model.coef_[0])
print("Intercept : ",model.intercept_)