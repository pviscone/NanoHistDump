import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


def load_data(filename,features,categoricals=None, test_size=0.2,seed=666,label2=2):
    test_size=0.2
    seed=666
    original_data=pd.read_parquet(filename)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if isinstance(label2,int):
        original_data.loc[original_data["label"]==2,"label"]=label2
    elif label2=="drop":
        original_data=original_data[original_data["label"]!=2]
    else:
        raise ValueError("label2 must be an integer or 'drop'")

    y=original_data["label"]

    weight = original_data["weight"]

    if categoricals is not None:
        for cat in categoricals:
            if cat in features:
                original_data[cat] = original_data[cat].astype("category")
            else:
                print(f"Warning: {cat} is in categoricals but not in features")


    data=original_data[features]


    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(data, y, weight, test_size=test_size, random_state=seed)

    dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train,enable_categorical=True)
    dtest = xgb.DMatrix(x_test, label=y_test, weight=w_test,enable_categorical=True)
    return original_data,dtrain, dtest

def predict(model, data,features):
    ddata=xgb.DMatrix(data[features], label=data["label"], weight=data["weight"],enable_categorical=True)
    return model.predict(ddata)
