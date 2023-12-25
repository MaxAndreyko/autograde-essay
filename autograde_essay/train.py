import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from metrics import calc_metrics
from preprocess import prep_train_data
from utils import read_data, save_model


TRAIN_PATH = "data/training_set_rel3.tsv"
MODEL_PARAMS = {"n_estimators": 200, "random_state": 0, "max_depth": 12}
MODEL_SAVE_PATH = "autograde_essay/models/rf_model.pkl"
REPO_PATH = "https://github.com/MaxAndreyko/autograde-essay/"


def train_model(
    params: dict, x_train: np.array, y_train: np.array
) -> RandomForestClassifier:
    model = RandomForestClassifier(**params)
    model.fit(x_train, y_train)
    return model


def main():
    start = time.time()
    print("================ Train Data is being download ... ================ ")
    train_data = read_data(TRAIN_PATH, REPO_PATH)
    print("Data donwloaded successfully!\n")

    print("================ Preparing data started ... ================ ")
    X, y = prep_train_data(train_data)
    print("Data preparation finished.\n")

    print("================ Model training started ================")
    trained_model = train_model(MODEL_PARAMS, X, y)
    print("Model has been fitted.\n")

    print(f"Metrics: {calc_metrics(y, trained_model.predict(X))}\n")

    print("Saving model ...")
    save_model(trained_model, MODEL_SAVE_PATH)
    print("Model saved. \n")

    stop = time.time()
    print(f"Time spent (min): {(stop - start) / 60}")


if __name__ == "__main__":
    main()
