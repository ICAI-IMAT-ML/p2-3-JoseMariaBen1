from src.lab_2_3_LinearRegression import LinearRegressor, evaluate_regression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
anscombe = sns.load_dataset("anscombe")

# Anscombe's quartet consists of four datasets
# TODO: Construct an array that contains, for each entry, the identifier of each dataset
datasets = ['I','II','III','IV']

models = {}
results = {"R2": [], "RMSE": [], "MAE": []}
for dataset in datasets:

    # Filter the data for the current dataset
    # TODO
    data = anscombe[anscombe['dataset'] == dataset]

    # Create a linear regression model
    # TODO
    model = LinearRegressor()

    # Fit the model
    # TODO
    X = data['x']  # Predictor, make it 1D for your custom model
    if np.ndim(X) > 1:
        X = X.reshape(1, -1)
    y = data['y']  # Response
    model.fit_simple(X, y)

    # Create predictions for dataset
    # TODO
    y_pred = model.predict(X)

    # Store the model for later use
    models[dataset] = model

    # Print coefficients for each dataset
    print(
        f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}"
    )

    evaluation_metrics = evaluate_regression(y, y_pred)

    # Print evaluation metrics for each dataset
    print(
        f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}"
    )
    results["R2"].append(evaluation_metrics["R2"])
    results["RMSE"].append(evaluation_metrics["RMSE"])
    results["MAE"].append(evaluation_metrics["MAE"])
print(anscombe, datasets, models, results)