import numpy as np

def df_to_dict(df,features,score):
    inout_dict=df[[*features,score]].to_dict(orient="list")
    return {key:np.array(entry) for key,entry in inout_dict.items()}