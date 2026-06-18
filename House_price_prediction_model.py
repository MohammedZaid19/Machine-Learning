# Step 1 import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# Step 2 Load the dataset
df = pd.read_csv('D:/Downloads/house_size_price.csv')
print("Dataset : ")
print(df)

# Step 3 Separate features X and target Y
X = df[["House_Size"]]
y = df["House_Price"]

# Step 4 split Data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("\nTraining Data Size : ",len(X_train))
print("\nTesting Data Size : ",len(X_test))

# Step 5 create model
model = LinearRegression()

# Step 6 Train Model
model.fit(X_train,y_train)
print("\nModel Trained Successfully")

# Step 7 Make predictions on test data
y_pred = model.predict(X_test)
print("\nActual Price")
print(y_test.values)
print("\nPredicted Price")
print(y_pred)

# Step 8 Evaluate model
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("\nModel Evaluation : ")
print("\nMean Squared Error : ",mse)
print("\nR2 Score : ",r2)

# user prediction
house = float(input("\nEnter the House Size in terms of Sqft : "))

# Step 9 Create Dataframe with the same feature
new_data = pd.DataFrame({"House_Size":[house]})
predicted_price = model.predict(new_data)
print(f"Predicted Price for {house} house size : {predicted_price[0]:.2f}")

# Step 10 Display learned equation
print("\nModel Parameters : ")
print("Slope (Coefficient) : ",model.coef_[0])
print("Intercept : ",model.intercept_)
print("\nLinear Regression Equation")
print(f"Price = {model.coef_[0]:.4f} * House_Size + {model.intercept_:.4f}")