import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def plot_linear_weights(X, model):
    feature_names = X.columns
    feature_weights = model.coef_

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(feature_names, feature_weights)
    ax.axhline(y=0, color="k")

    _ = ax.set(xlabel="", ylabel="Weight", title="title")
    plt.xticks(rotation=90)

    return None


def update_summary(
    model_name, group, w_sdoh, r2, mse, feature_names_list, summary=None
):
    # If no summary provided, create a new DataFrame
    if summary is None:
        summary = pd.DataFrame(
            columns=["model_name", "group", "w_sdoh", "r2", "mse", "feature_names"]
        )
    feature_names_str = ", ".join(feature_names_list)
    # Create a new row with the provided values
    new_row = {
        "model_name": model_name,
        "w_sdoh": w_sdoh,
        "group": group,
        "r2": r2,
        "mse": mse,
        "feature_names": feature_names_str,
    }

    # Append the new row to the summary DataFrame
    summary = pd.concat([summary, pd.DataFrame(new_row, index=[0])], ignore_index=True)

    return summary


def process_features(df, numeric_cols=None, cat_cols=None):
    """
    Function to process the features for modeling. If numeric_cols
    are passed in, standard scaling will be applied. If cat_cols are
    passed in, one-hot encoding will be applied with k-1 levels.

    params
    ------
    df: pd.DataFrame
        Dataframe containing the features to process
    numeric_cols: list, optional
        List of column names for numeric features e.g.
        ['patiepnt_safety_score', 'avg_payment_amount_py']
    cat_cols: list, optional
        List of column names for categorical features e.g.
        ['hospital_type', 'hospital_ownership']

    returns
    -------
    X_processed: pd.DataFrame
        Dataframe containing the processed features
    """
    if numeric_cols is None and cat_cols is None:
        raise ValueError("At least one of numeric_cols or cat_cols must be provided.")

    if numeric_cols is not None:
        X_numeric = df[numeric_cols]

        # Standardize and center the numeric features
        scaler = StandardScaler()
        numerical_features_scaled = scaler.fit_transform(X_numeric)
        X_scaled = pd.DataFrame(numerical_features_scaled, columns=X_numeric.columns)
    else:
        X_scaled = pd.DataFrame()

    if cat_cols is not None:
        X_cat = df[cat_cols]
        X_encoded = pd.get_dummies(X_cat, drop_first=True)
    else:
        X_encoded = pd.DataFrame()

    # Concatenate the processed numeric and categorical features
    X_scaled.reset_index(drop=True, inplace=True)
    X_encoded.reset_index(drop=True, inplace=True)
    X_processed = pd.concat([X_scaled, X_encoded], axis=1)

    return X_processed


def fit_linear_model(X, y, model_name, group, sdoh, summary_table=None):
    """
    Quick fx for fitting a linear model w/ train test split

    params
    ------
    X: pd.DataFrame
        Features to use for model fitting that have already been
        scaled and encoded
    y: pd.Series
        Target variable
    model_name: str
        Name of the model to use for id in summary table
    sdoh: bool
        Whether or not the model includes SDOH features
    summary_table: pd.DataFrame
        Table to append model results to, if any. will make a
        new table if none provided

    returns
    -------
    linear_model: sklearn.linear_model.LinearRegression
        Trained linear regression model
    summary_table: pd.DataFrame
        Table of model results for test fit
    """

    # Create train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create an instance of the LinearRegression model & fit to train
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Predict using the trained model
    y_pred = linear_model.predict(X_test)

    # Get statistics on the model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} mse: {np.round(mse,3)}, r2: {np.round(r2, 2)}")

    summary_table = update_summary(
        model_name, group, sdoh, r2, mse, X.columns, summary=summary_table
    )

    return linear_model, summary_table
