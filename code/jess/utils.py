import matplotlib.pyplot as plt
import pandas as pd


def plot_linear_weights(X, model):
    feature_names = X.columns
    feature_weights = model.coef_

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(feature_names, feature_weights)
    ax.axhline(y=0, color="k")

    _ = ax.set(xlabel="", ylabel="Weight", title="title")
    plt.xticks(rotation=90)

    return None


def update_summary(model_name, w_sdoh, r2, mse, feature_names_list, summary=None):
    # If no summary provided, create a new DataFrame
    if summary is None:
        summary = pd.DataFrame(
            columns=["model_name", "w_sdoh", "r2", "mse", "feature_names"]
        )
    feature_names_str = ", ".join(feature_names_list)
    # Create a new row with the provided values
    new_row = {
        "model_name": model_name,
        "w_sdoh": w_sdoh,
        "r2": r2,
        "mse": mse,
        "feature_names": feature_names_str,
    }

    # Append the new row to the summary DataFrame
    summary = pd.concat([summary, pd.DataFrame(new_row, index=[0])], ignore_index=True)

    return summary
