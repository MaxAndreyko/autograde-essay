import dvc.api
import joblib
import pandas as pd


def read_data(
    data_path: str, repo, encoding: str = "ISO-8859-1", sep="\t"
) -> pd.DataFrame:
    """Reads tabular data from DVC cloud storage

    Args:
        data_path (str): Git repository relative path to data
        repo (str): Git repository link
        encoding (str, optional): File encoding. Defaults to "ISO-8859-1".

    Returns:
        pd.DataFrame: Downloaded table as pandas dataframe
    """
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
