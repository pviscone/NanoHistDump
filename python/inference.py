import awkward as ak
import numpy as np
import xgboost as xgb
import copy
from scipy. special import softmax


def xgb_wrapper(model, events, features=None, layout_template=None, conifer_model= None):
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
        array = ak.drop_none(array)
        array = ak.ravel(array).to_numpy(allow_missing=False)[:, None]
        if idx == 0:
            matrix = array
        else:
            matrix = np.concatenate((matrix, array), axis=1)

    dmatrix = xgb.DMatrix(matrix, feature_names=model.feature_names)
    scores = model.predict(dmatrix)
    if scores.ndim > 1:
        scores = 1 - scores[:, 0]

    if conifer_model:
        if isinstance(conifer_model, str):
            import conifer
            hlsmodel=conifer.model.load_model(conifer_model)
            hlsmodel.compile()
        else:
            hlsmodel=conifer_model
        hlspreds=hlsmodel.decision_function(dmatrix.get_data().toarray())
        if hlspreds.ndim>1 and hlspreds.shape[1]>1:
            hlspreds = 1-softmax(hlspreds)[:,0]
        else:
            hlspreds=(hlspreds).ravel()
        conifer_layout=copy.deepcopy(layout_template)

    if nested:
        def recur(layout_template, arr):
            if "_content" in layout_template.__dir__():
                layout_template._content = recur(layout_template._content, arr)
            else:
                layout_template = arr
            return layout_template

        recur(layout_template, ak.contents.NumpyArray(scores))
        awk_scores = ak.Array(layout_template)
        if conifer_model:
            recur(conifer_layout, ak.contents.NumpyArray(hlspreds))
            conifer_scores= ak.Array(conifer_layout)

    else:
        raise NotImplementedError("Not implemented for array.ndim <= 2")

    if conifer_model:
        return awk_scores, conifer_scores
    else:
        return awk_scores, -1
