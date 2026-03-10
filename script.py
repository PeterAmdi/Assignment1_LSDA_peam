import mlflow
import os

# You will probably need these
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# This are for example purposes. You may discard them if you don't use them.
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


### TODO -> HERE YOU CAN ADD ANY OTHER LIBRARIES YOU MAY NEED ###
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import argparse

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.sklearn.autolog()

direction_to_degrees = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }

def wind_to_sincos(X):
    """Conversion from string direction to degrees"""
    degrees = np.array([direction_to_degrees.get(d, np.nan) for d in X[:, 0]]) * (np.pi / 180)
    return np.column_stack([np.sin(degrees), np.cos(degrees)])

def main():
    parser = argparse.ArgumentParser()
    # Model Selection
    parser.add_argument("--model_type", type=str, default="grb", choices=["grb", "svr"])
    
    # GRB Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    
    # SVR Hyperparameters
    parser.add_argument("--svr_c", type=float, default=1.0)
    parser.add_argument("--svr_epsilon", type=float, default=0.1)
    
    args = parser.parse_args()

    # We switch the experiment name based on the user's input
    if args.model_type == "svr":
        mlflow.set_experiment("Support_Vector_Regression-Experiment")
    else:
        mlflow.set_experiment("Gradient_Boosting-Experiment")
    # - Data Loading -
    # (power.csv and weather.csv have to be in the same folder as the script.py file)
    power = pd.read_csv("power.csv", parse_dates=["time"], index_col="time")
    weather = pd.read_csv("weather.csv", parse_dates=["time"], index_col="time")
    joined_dfs = weather.join(power).sort_index()
    joined_dfs["Total"] = joined_dfs["Total"].interpolate(method="time").ffill().bfill()

    X = joined_dfs[["Speed", "Direction"]]
    y = joined_dfs["Total"]
    split_index = int(len(X) * 0.80)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # -- Preprocessing --
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
        ]), ['Speed']),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('sincos', FunctionTransformer(wind_to_sincos)),
        ]), ['Direction']),
    ])

    # --- Model Selection Logic ---
    if args.model_type == "grb":
        model = GradientBoostingRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=42
        )
    else:
        model = SVR(kernel='rbf', C=args.svr_c, epsilon=args.svr_epsilon)

    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

    # ---- MLflow Run ----
    with mlflow.start_run(run_name=f"Run_{args.model_type.upper()}"):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        # Log manual artifacts (plots)
        plt.figure(figsize=(10, 4))
        plt.plot(y_test.values, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.title(f"Model: {args.model_type.upper()}")
        plt.legend()
        plot_path = f"plot_{args.model_type}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        
        print(f"Finished {args.model_type} training. MAE: {mean_absolute_error(y_test, predictions):.2f}")

if __name__ == "__main__":
    main()
