import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load the new dataset
df_augmented = pd.read_csv("data/augmented_mixtures.csv")

# Split input (first 6 columns) and output (remaining columns)
X = df_augmented.iloc[:, :6]  # First 6 columns (binary fluid presence)
y = df_augmented.iloc[:, 6:]  # Remaining columns (marker values)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale outputs
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

# Train MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(100,), random_state=42, max_iter=500)
mlp.fit(X_train, y_train_scaled)

# Predict and evaluate
y_pred_scaled = mlp.predict(X_test)
mse = mean_squared_error(y_test_scaled, y_pred_scaled)

print(mse)