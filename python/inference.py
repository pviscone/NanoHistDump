import awkward as ak
import numpy as np
import xgboost as xgb


def xgb_wrapper(model, events, features,nested=False, layout_template=None):
    if "str" in str(type(model)):
        model = xgb.Booster()
        model.load_model(model)

    for idx, feature in enumerate(features):
        feature_list = feature.split("/")

        if feature_list[-1] != model.feature_names[idx].split("_")[-1]:
            raise ValueError(f"Feature name mismatch: {feature} instead of {model.feature_names[idx]}")

        array=events[*feature_list]
        array=ak.drop_none(array)
        if nested:
            array=ak.flatten(events[*feature_list])
        array=ak.drop_none(array)
        array = ak.flatten(array).to_numpy(allow_missing=False)[:, None]
        if idx == 0:
            matrix = array
        else:
            matrix = np.concatenate((matrix, array), axis=1)

    dmatrix = xgb.DMatrix(matrix, feature_names=model.feature_names)
    scores = model.predict(dmatrix)


    if nested:
        def recur(layout_template, arr):
            if "_content" in layout_template.__dir__():
                layout_template._content=recur(layout_template._content,arr)
            else:
                layout_template=arr
            return layout_template

        recur(layout_template,ak.contents.NumpyArray(scores))
        awk_scores=ak.Array(layout_template)
    else:
        raise NotImplementedError("Not implemented for nested==False")


    return awk_scores
