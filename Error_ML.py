import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error

# defining sample data
y_true = np.array([3.0,-0.5,2.0,7.0,5.0])
y_pred = np.array([2.5,0.0,2.0,8.0,4.2])

# Method 1 using scikit-learn (Recommended)
# Mean_Absolute_Error (MAE)
mae_sklearn = mean_absolute_error(y_true, y_pred)

# Mean Squared Error (MSE)
mse_sklearn = mean_squared_error(y_true, y_pred)

rmse_sklearn = np.sqrt(mse_sklearn)
print("Scikit-learn results")
print(f"MAE: {mae_sklearn:.4f}")
print(f"MSE: {mse_sklearn:.4f}")
print(f"RMSE: {rmse_sklearn:.4f}\n")