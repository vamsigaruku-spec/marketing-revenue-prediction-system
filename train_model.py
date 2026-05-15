import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# LOAD DATA
# =========================

train_data = pd.read_csv("data/train.csv")

# =========================
# DROP UNUSED COLUMNS
# =========================

train_data = train_data.drop(columns=["id", "date"])

# =========================
# HANDLE MISSING VALUES
# =========================

train_data = train_data.fillna(train_data.mean(numeric_only=True))

# =========================
# ENCODE CATEGORICAL DATA
# =========================

label_encoders = {}

categorical_columns = [
    "region",
    "channel",
    "product_category",
    "customer_segment"
]

for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# =========================
# TARGET COLUMN
# =========================

TARGET = "sales_revenue"

# =========================
# FEATURES + TARGET
# =========================

X = train_data.drop(columns=[TARGET])
y = train_data[TARGET]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODELS
# =========================

models = {
    "KNN": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Linear Regression": LinearRegression()
}

results = []

best_model = None
best_score = -999
best_model_name = ""

# =========================
# TRAIN MODELS
# =========================

for name, model in models.items():

    print("\n==========================")
    print(f"TRAINING MODEL: {name}")
    print("==========================")

    # RANDOM FOREST
    if name == "Random Forest":

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

    # OTHER MODELS
    else:

        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

    # METRICS
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    # SAVE RESULTS
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"MAE: {mae}")

    print(f"RMSE: {rmse}")

    print(f"R2 Score: {r2}")

    # BEST MODEL
    if r2 > best_score:

        best_score = r2

        best_model = model

        best_model_name = name

# =========================
# RESULTS DATAFRAME
# =========================

results_df = pd.DataFrame(results)

print("\n==========================")
print("FINAL MODEL COMPARISON")
print("==========================")

print(results_df)

# =========================
# SAVE FILES
# =========================

joblib.dump(best_model, "model/best_model.pkl")

joblib.dump(scaler, "model/scaler.pkl")

joblib.dump(results_df, "model/model_results.pkl")

joblib.dump(label_encoders, "model/label_encoders.pkl")

print("\n==========================")
print(f"BEST MODEL SAVED: {best_model_name}")
print("==========================")