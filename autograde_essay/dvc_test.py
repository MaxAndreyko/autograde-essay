import dvc.api


dvc.api.DVCFileSystem(
    url="https://drive.google.com/drive/folders/1DUrUAvbbFDHAgg1Yy_0u3CcRLlsjqH_y"
).get("training_set_rel3.tsv", "./data/")
