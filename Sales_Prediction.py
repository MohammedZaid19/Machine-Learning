# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# Load the dataset
df = pd.read_csv('D:/Downloads/archive (2)/SalesPredict.csv')
print(df)

# Separate features X and target y
X = df[["TV"]]
y = df["Sales"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
print("\nTraining Data Size : ",len(X_train))
print("\nTesting Data Size : ",len(X_test))

# Create a model
model = LinearRegression()

# Train model
model.fit(X_train,y_train)
print("\nModel Trained Successfully")

# Make predictions on test data
y_pred = model.predict(X_test)
print("\nActual Sales")
print(y_test.values)
print("\nPredicted Sales")
print(y_pred)

# Evaluate Model
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("\nModel Evaluation : ")
print("\nMean Squared Error : ",mse)
print("\nR2 Score : ",r2)

# User prediction
tv= float(input("\nEnter TV Advertising Budget : "))

# Create Dataframe with the same features
new_data = pd.DataFrame({"TV":[tv]})
predicted_sales = model.predict(new_data)
print(f"Predicted Sales for {tv} of experience : {predicted_sales[0]:.2f}")

# Display learned equation
print("\nModel Parameters")
print("Slope (Coefficient) : ",model.coef_[0])
print("Intercept : ",model.intercept_)
print("\nLinear Regression Equation")
print(f"Sales = {model.coef_[0]:.4f}*TV + {model.intercept_:.4f}")