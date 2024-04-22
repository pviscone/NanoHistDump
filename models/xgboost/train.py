#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_parquet("131Xv3.parquet")
y=(df["label"].astype(int) & df["Tk_isReal"]==1).astype(int).to_numpy()

pt_weight=df["pt_weight"].to_numpy()
weight=pt_weight
weight[y==1]=weight[y==1]*np.sum(y==0)/np.sum(y==1)

df_train=df.drop(columns=["label","Tk_isReal","ev_id","pt_weight"])

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(df_train, y, weight, test_size=0.2, random_state=666)

