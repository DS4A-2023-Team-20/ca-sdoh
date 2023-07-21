import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import statsmodels.api as sm
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


def update_summary_v2(model_name, race, r2, mse, feature_names_list, summary=None):
    # If no summary provided, create a new DataFrame
    if summary is None:
        summary = pd.DataFrame(
            columns=["model_name", "ract", "r2", "mse", "feature_names"]
        )
    feature_names_str = ", ".join(feature_names_list)
    # Create a new row with the provided values
    new_row = {
        "model_name": model_name,
        "race": race,
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
        X_encoded = pd.get_dummies(X_cat, drop_first=True, dtype=int)
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
    using the sk learn linear

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

    # make sure indices are set up correctly
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

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


def fit_linear_model_sm(X, y, model_name, race=None, summary_table=None):
    """
    Code to fit a linear model using statsmodels OLS
    to allow for better summary analysis and significance
    testing of the features
    """

    # make sure indices are set up correctly
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    # Create train/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Add constant term to the design matrix for intercept estimation
    X_train = sm.add_constant(X_train)

    # Create an instance of the OLS model & fit to train
    linear_model = sm.OLS(y_train, X_train)
    linear_model = linear_model.fit()

    # Add constant term to the test data
    X_test = sm.add_constant(X_test)

    # Predict using the trained model
    y_pred = linear_model.predict(X_test)

    # Get statistics on the model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} test mse: {np.round(mse,3)}, r2: {np.round(r2, 3)}")

    summary_table = update_summary_v2(
        model_name, race, r2, mse, X.columns, summary=summary_table
    )

    summary = linear_model.summary()

    return linear_model, summary


def get_significant_variables(model):
    """
    Function to extract the significant variables from a fitted
    statsmodels linear regression model.
    """

    # Extract the p-values from the summary table
    p_values = model.pvalues[1:]

    # Extract the va riable names from the summary table
    variable_names = model.params.index[1:]

    # Create a boolean array indicating significant variables (p-value < 0.05)
    significant_variables = p_values < 0.05

    # Get the names of the significant variables
    significant_variable_names = variable_names[significant_variables]

    return significant_variable_names


def plot_feature_weights_horizontal_sm(
    model, ax=None, title="Feature Weights", color="cornflowerblue", color_sig=True
):
    """
    Plots the feature weights of a linear model
    from stas model in a horizontal bar plot.
    """
    ax
    # Get the feature names and corresponding coefficients
    feature_names = model.params.index[1:]  # Exclude the bias term
    coefficients = model.params[1:]  # Exclude the bias term

    # # Sort feature names and coefficients based on absolute coefficient values
    sorted_indices = np.argsort(np.abs(coefficients))
    feature_names_sorted = feature_names[sorted_indices]
    coefficients_sorted = coefficients[sorted_indices]

    if color_sig:
        # Get the significant variables
        significant_variables = get_significant_variables(model)
        # Set the color for significant and non-significant variables
        colors = [
            color if var in significant_variables else "gray" for var in feature_names
        ]
        colors = [colors[i] for i in sorted_indices]
    else:
        colors = "gray"

    # Plot the feature weights
    if ax == None:
        fig, ax = plt.subplots()
    ax.barh(feature_names_sorted, coefficients_sorted, color=colors)

    # Set labels and title
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    ax.set_title(title)


def plot_single_model_results(results_df, ax, palette="Set2"):
    sns.boxplot(
        data=results_df, x="model_name", y="rmse", width=0.5, ax=ax[0], palette=palette
    )
    sns.pointplot(
        data=results_df,
        x="model_name",
        y="rmse",
        color="black",
        join=False,
        ax=ax[0],
    )

    sns.boxplot(
        data=results_df, x="model_name", y="r2", width=0.5, ax=ax[1], palette=palette
    )
    sns.pointplot(
        data=results_df,
        x="model_name",
        y="r2",
        color="black",
        join=False,
        ax=ax[1],
    )

    ax[0].set(
        ylabel="Error (RMSE)",
        xlabel="",
        title=f"{results_df.model_type.iloc[0]} Model Error",
    )
    ax[1].set(
        ylabel="Fit (Adj. $R^2$)",
        xlabel="",
        title=f"{results_df.model_type.iloc[0]} Model Fit",
    )
