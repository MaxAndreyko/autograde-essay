import logging
import time
import os
import base64
import json

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier

from metrics import calc_metrics
from preprocess import prep_train_data
from utils import read_data, save_model


log = logging.getLogger(__name__)


def train_model(
    params: dict, x_train: np.array, y_train: np.array
) -> RandomForestClassifier:
    """Trains Random Forest Classifier models

    Args:
        params (dict): Hyperparameters for RFC model
        x_train (np.array): Train data
        y_train (np.array): Train data answers

    Returns:
        RandomForestClassifier: Trained RFC model
    """
    model = RandomForestClassifier(**params["params"])
    step = params["log"]["step"]
    for n in range(
        params["params"]["n_estimators"], params["log"]["n_estimators"] + step, step
    ):
        model.fit(x_train, y_train)
        log.info(
            f"ESTIMATORS: {n} METRICS: {calc_metrics(y_train, model.predict(x_train))}"
        )
        model.n_estimators += 10
    return model


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main train function"""
    start = time.time()
    log.info("================ Train Data is being download ... ================ ")
    train_data = read_data(cfg["path"]["train"], cfg["repo"], cfg["path"]["creds"])
    log.info("Data donwloaded successfully!\n")

    log.info("================ Preparing data started ... ================ ")
    X_train, y_train = prep_train_data(train_data, cfg)
    log.info("Data preparation finished.\n")

    log.info("================ Model training started ================")
    trained_model = train_model(cfg["model"], X_train, y_train)
    log.info("Model has been fitted.\n")

    log.info(f"Metrics: {calc_metrics(y_train, trained_model.predict(X_train))}\n")

    log.info("Saving model ...")
    save_model(trained_model, cfg["path"]["save"])
    log.info("Model saved. \n")

    stop = time.time()
    log.info(f"Time spent (min): {(stop - start) / 60}")


if __name__ == "__main__":
    main()
