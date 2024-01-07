import json
import logging
import os

import dvc.api
import joblib
import pandas as pd


log = logging.getLogger(__name__)


def read_data(
    data_path: str, repo, path2creds: str, encoding: str = "ISO-8859-1", sep="\t"
) -> pd.DataFrame:
    """Reads tabular data from DVC cloud storage

    Args:
        data_path (str): Git repository relative path to data
        repo (str): Git repository link
        path2creds(str): Path to json file with Google Drive service account credentials
        encoding (str, optional): File encoding. Defaults to "ISO-8859-1".

    Returns:
        pd.DataFrame: Downloaded table as pandas dataframe
    """

    # Add credentials to environment variable if json file exsists
    try:
        with open(path2creds) as f:
            os.environ["GDRIVE_CREDENTIALS_DATA"] = json.dumps(json.load(f))
    except FileNotFoundError as e:
        log.error("Error: %s", e)

    with dvc.api.open(data_path, repo=repo, encoding=encoding) as fd:
        data = pd.read_csv(fd, sep=sep)
    return data


def load_model(save_path: str) -> object:
    """Loads trained model's weights

    Args:
        save_path (str): Path where model's weights stored

    Returns:
        object: Trained model object
    """
    return joblib.load(save_path)


def save_model(model, save_path: str) -> None:
    """Saves trained model

    Args:
        model (object): Trained model object
        save_path (str): Path where model's weights stored
    """
    joblib.dump(model, save_path)
