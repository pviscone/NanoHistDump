import numpy as np
import pandas as pd
import xgboost as xgb


def df_to_dict(df,features,score):
    inout_dict=df[[*features,score]].to_dict(orient="list")
    return {key:np.array(entry) for key,entry in inout_dict.items()}



def load_from_df(df,features,ptkey="CryClu_pt", label="label",weight="weight",label2=2):
    df=df[df[ptkey]<105]
    df.loc[df["label"]==2,"label"]=label2
    dmatrix = xgb.DMatrix(df[features], label=df[label], weight=df[weight] if isinstance(weight,str) else None)
    return df, dmatrix


def load_parquet(filename,features,ptkey="CryClu_pt",label="label",weight="weight",label2=2,scaler=None):
    df=pd.read_parquet(filename)
    if scaler is not None:
        if ptkey=="CryClu_pt":
            raise ValueError("ptkey should not be CryClu_pt if you are scaling it")
        df[ptkey]=df["CryClu_pt"]
        df=scaler.apply(df)
    return load_from_df(df,features,label=label,weight=weight,label2=label2, ptkey=ptkey)
