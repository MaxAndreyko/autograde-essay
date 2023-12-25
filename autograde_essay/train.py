import time

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier

from metrics import calc_metrics
from preprocess import prep_train_data
from utils import read_data, save_model


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
    model = RandomForestClassifier(**params)
    model.fit(x_train, y_train)
    return model


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main train function"""
    start = time.time()
    print("================ Train Data is being download ... ================ ")
    train_data = read_data(cfg["path"]["train"], cfg["repo"])
    print("Data donwloaded successfully!\n")

    print("================ Preparing data started ... ================ ")
    X, y = prep_train_data(train_data)
    print("Data preparation finished.\n")

    print("================ Model training started ================")
    trained_model = train_model(cfg["model"]["params"], X, y)
    print("Model has been fitted.\n")

    print(f"Metrics: {calc_metrics(y, trained_model.predict(X))}\n")

    print("Saving model ...")
    save_model(trained_model, cfg["path"]["save"])
    print("Model saved. \n")

    stop = time.time()
    print(f"Time spent (min): {(stop - start) / 60}")


if __name__ == "__main__":
    main()
