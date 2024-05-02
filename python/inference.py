import awkward as ak
import numpy as np
import xgboost as xgb


def xgb_wrapper(model, events, features,nested=False):
    if "str" in str(type(model)):
        model = xgb.Booster()
        model.load_model(model)

    for idx, feature in enumerate(features):
        feature_list = feature.split("/")

        if feature_list[-1] != model.feature_names[idx].split("_")[-1]:
            raise ValueError(f"Feature name mismatch: {feature} instead of {model.feature_names[idx]}")

        array=events[*feature_list]
        if nested:
            array=ak.flatten(events[*feature_list])
        array = ak.flatten(array).to_numpy()[:, None]
        if idx == 0:
            matrix = array
        else:
            matrix = np.concatenate((matrix, array), axis=1)

    dmatrix = xgb.DMatrix(matrix, feature_names=model.feature_names)

    scores = model.predict(dmatrix)

    layout=ak.contents.NumpyArray(scores)
    if nested:
        layout=ak.contents.ListOffsetArray(events.layout.content.offsets, layout)
    awk_scores = ak.Array(ak.contents.ListOffsetArray(events.layout.offsets, layout))

    return awk_scores
