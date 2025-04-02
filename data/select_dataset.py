import re
import pandas as pd

def select_datasets(regex="[Rr]oot"):

    df = pd.read_csv("~/Descargas/scRNA-Seq_datasets.csv")
    regex = re.compile(regex)
    root_datasets = df.loc[df.tissue.apply(lambda x: bool(regex.match(str(x))))]["dataset ID"].to_list()
    return [f"transforms/HVG_{dataset}.csv" for dataset in root_datasets]


print(select_datasets())