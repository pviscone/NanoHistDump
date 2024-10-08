import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA


def df_to_dict(df, features, score):
    inout_dict = df[[*features, score]].to_dict(orient="list")
    return {key: np.array(entry) for key, entry in inout_dict.items()}


def load_from_df(df, features, ptkey="CryClu_pt", label="label", weight="weight", label2=2):
    df = df[df[ptkey] < 105]
    df.loc[df[label] == 2, label] = label2
    dmatrix = xgb.DMatrix(df[features], label=df[label], weight=df[weight] if isinstance(weight, str) else None)
    return df, dmatrix


def load_parquet(
    filename, features, ptkey="CryClu_pt", oldptkey="CryClu_pt", label="label", weight="weight", label2=2, scaler=None, pca=False
):
    df = pd.read_parquet(filename)
    df[ptkey] = df[oldptkey]

    if pca:
        X = df[features]
        if isinstance(pca, bool):
            pca = PCA(n_components=len(features))
            df[features] = pca.fit_transform(X)
        elif isinstance(pca, PCA):
            df[features] = pca.transform(X)

    if scaler is not None:
        if ptkey == oldptkey:
            raise ValueError(f"ptkey should not be {oldptkey} if you are scaling it")
        df = scaler.apply(df)

    if pca:
        return *load_from_df(df, features, label=label, weight=weight, label2=label2, ptkey=ptkey), pca
    return *load_from_df(df, features, label=label, weight=weight, label2=label2, ptkey=ptkey), None
