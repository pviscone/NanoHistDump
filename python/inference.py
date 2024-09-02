import awkward as ak
import numpy as np
import xgboost as xgb


def xgb_wrapper(model, events, features=None, layout_template=None):
    if isinstance(model, str):
        model_json = model
        model = xgb.Booster()
        model.load_model(model_json)

    if features is None:
        print("Using default features from model. Things could go wrong.")
        features = model.feature_names

    for idx, feature in enumerate(features):
        if model.feature_names[idx] not in feature:
            print(model.feature_names)
            print(feature)
            raise ValueError(f"Feature name mismatch: {feature} instead of {model.feature_names[idx]}")

        if feature.startswith("abs_"):
            feature = feature.replace("abs_", "")
            array = np.abs(events[*(feature.split("_"))])
        else:
            array = events[*(feature.split("_"))]
        nested = True if array.ndim > 2 else False
        if nested:
            array = ak.flatten(array)
        array = ak.drop_none(array)
        array = ak.flatten(array).to_numpy(allow_missing=False)[:, None]
        if idx == 0:
            matrix = array
        else:
            matrix = np.concatenate((matrix, array), axis=1)

    dmatrix = xgb.DMatrix(matrix, feature_names=model.feature_names)
    scores = model.predict(dmatrix)
    if scores.ndim > 1:
        scores = 1 - scores[:, 0]

    if nested:

        def recur(layout_template, arr):
            if "_content" in layout_template.__dir__():
                layout_template._content = recur(layout_template._content, arr)
            else:
                layout_template = arr
            return layout_template

        recur(layout_template, ak.contents.NumpyArray(scores))
        awk_scores = ak.Array(layout_template)
    else:
        raise NotImplementedError("Not implemented for nested==False")

    return awk_scores
