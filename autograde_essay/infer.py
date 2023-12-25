import logging
import time

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from preprocess import prep_test_data
from utils import load_model, read_data


log = logging.getLogger(__name__)


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


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main inference function"""
    start = time.time()
    log.info("================ Test Data is being download ... ================ ")
    test_data = read_data(cfg["path"]["test"], cfg["repo"])
    log.info("Data donwloaded successfully!\n")

    log.info("================ Preparing data started ... ================ ")
    X = prep_test_data(test_data, cfg)
    log.info("Data preparation finished.\n")

    log.info("Model loading ...")
    model = load_model(cfg["path"]["save"])
    log.info("Loading finished.\n")

    log.info("Model predicting...")
    pred = model.predict(X)
    log.info("Model saving predictions ...")
    export_pred(test_data, pred, cfg["path"]["pred"])
    log.info(
        f"""
    ****** All done!
    ****** Predictions saved to: {cfg["path"]["save"]}
          """
    )
    stop = time.time()
    log.info(f"Time spent (min): {(stop - start) / 60}")


if __name__ == "__main__":
    main()
