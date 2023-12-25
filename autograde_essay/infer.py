import time

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from preprocess import prep_test_data
from utils import load_model, read_data


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


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main inference function"""
    start = time.time()
    print("================ Test Data is being download ... ================ ")
    test_data = read_data(cfg["path"]["test"], cfg["repo"])
    print("Data donwloaded successfully!\n")

    print("================ Preparing data started ... ================ ")
    X = prep_test_data(test_data)
    print("Data preparation finished.\n")

    print("Model loading ...")
    model = load_model(cfg["path"]["save"])
    print("Loading finished.\n")

    print("Model predicting...")
    pred = model.predict(X)
    print("Model saving predictions ...")
    export_pred(test_data, pred, cfg["path"]["pred"])
    print(
        f"""
    ****** All done!
    ****** Predictions saved to: {cfg["path"]["save"]}
          """
    )
    stop = time.time()
    print(f"Time spent (min): {(stop - start) / 60}")


if __name__ == "__main__":
    main()
