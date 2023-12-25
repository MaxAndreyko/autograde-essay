import time

import numpy as np
import pandas as pd

from preprocess import prep_test_data
from utils import load_model, read_data


PRED_PATH = "data/predictions.csv"
TEST_PATH = "data/test_set.tsv"
MODEL_SAVE_PATH = "autograde_essay/models/rf_model.pkl"
REPO_PATH = "https://github.com/MaxAndreyko/autograde-essay/"


def export_pred(test_data: pd.DataFrame, pred: np.array, export_path: str) -> None:
    """Adds prediction column to test dataset and saves as .csv file

    Args:
        test_data (pd.DataFrame): Raw test dataset
        pred (np.array): Prediction vector
        export_path (str): Path where to export concatenated dataset
    """
    pred_df = pd.DataFrame(list(pred), columns=["pred"])
    df = pd.concat([test_data, pred_df], axis=1)
    df.to_csv(export_path)


def main():
    """Main inference function"""
    start = time.time()
    print("================ Test Data is being download ... ================ ")
    test_data = read_data(TEST_PATH, REPO_PATH)
    print("Data donwloaded successfully!\n")

    print("================ Preparing data started ... ================ ")
    X = prep_test_data(test_data)
    print("Data preparation finished.\n")

    print("Model loading ...")
    model = load_model(MODEL_SAVE_PATH)
    print("Loading finished.\n")

    print("Model predicting...")
    pred = model.predict(X)
    print("Model saving predictions ...")
    export_pred(test_data, pred, PRED_PATH)
    print(
        f"""
    ****** All done!
    ****** Predictions saved to: {MODEL_SAVE_PATH}
          """
    )
    stop = time.time()
    print(f"Time spent (min): {(stop - start) / 60}")


if __name__ == "__main__":
    main()
