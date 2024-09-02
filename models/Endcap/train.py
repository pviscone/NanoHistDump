# %%
import importlib
import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import xgboost as xgb
from utils import plots, utils
from utils.bitscaler import BitScaler
from utils.plots import profile_int_dec

importlib.reload(plots)

import mplhep as hep
import numpy as np

hep.styles.cms.CMS["figure.autolayout"] = True
hep.style.use(hep.style.CMS)

train_file = "/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flatBDT/dataset/endcap_131Xv3_train.parquet"
test_file = "/afs/cern.ch/work/p/pviscone/NanoHistDump/models/flatBDT/dataset/endcap_131Xv3_test.parquet"

features = [
    #Kyung
    #"HGCalClu_showerlength",
    "HGCalClu_coreshowerlength",
    #"HGCalClu_eot",
    "HGCalClu_meanz",
    "HGCalClu_spptot",
    "HGCalClu_seetot",
    "HGCalClu_szz",
    #"HGCalClu_multiClassPuIdScore",
    "HGCalClu_multiClassPionIdScore",
    "HGCalClu_multiClassEmIdScore",
    #"HGCalClu_pt",
    #"HGCalClu_eta",


    #"HGCalClu_hbm",
    #"HGCalClu_hwQual",
    #"HGCalClu_maxlayer",
    #"HGCalClu_nTcs",
    #"HGCalClu_emaxe",
    #"HGCalClu_seemax",
    #"HGCalClu_ntc67",
    #"HGCalClu_ntc90",
    #"HGCalClu_phi",
    #"HGCalClu_pt",
    #"HGCalClu_ptEm",
    #"HGCalClu_sppmax",
    #"HGCalClu_srrmax",
    #"HGCalClu_srrmean",
    #"HGCalClu_srrtot",
    #"HGCalClu_varEtaEta",
    #"HGCalClu_varPhiPhi",
    #"HGCalClu_varZZ",
    #"HGCalClu_varRR",
    #"HGCalClu_pfPuIdPass",
    #"HGCalClu_pfEmIdPass",
    #"HGCalClu_pfPuIdScore",
    #"HGCalClu_pfEmIdScore",
    #"HGCalClu_egEmIdScore",
    #"HGCalClu_multiClassMaxScore",
    #"HGCalClu_multiClassPuIdScore",
    #"HGCalClu_multiClassPionIdScore",
    #"HGCalClu_multiClassEmIdScore"

    #Tracks
    #"Tk_chi2RPhi",
    "Tk_PtFrac",
    #Combined
    "PtRatio",
    #"nMatch",
    "abs_dEta",
    "abs_dPhi",
    #
    #"dEta",
    #"dPhi",
    # "CryClu_isSS",
    # "CryClu_isIso",
    # "CryClu_isLooseTkIso",
    # "CryClu_isLooseTkSS",
    # "CryClu_brems",
    # "Tk_hitPattern",
    #"Tk_nStubs",
    #"Tk_chi2Bend",
    #"Tk_chi2RZ",
    #
    # "Tk_pt",
    # "maxPtRatio_other",
    # "minPtRatio_other",
    # "meanPtRatio_other",
    # "stdPtRatio_other",
]

range_map = {
    "CryClu_pt": (0, 120),
    "CryClu_ss": (0, 1),
    "CryClu_relIso": (0, 170),
    "CryClu_standaloneWP": (0, 1),
    "CryClu_looseL1TkMatchWP": (0, 1),
    "Tk_chi2RPhi": (0, 200),
    "Tk_PtFrac": (0, 1),
    "PtRatio": (0, 51),
    "nMatch": (0, 14),
    "abs_dEta": (0, 0.03),
    "abs_dPhi": (0, 0.3),
}


scaler = BitScaler()
scaler.fit(range_map, target=(-1, 1))
scaler = None
pca = False
classes = 2


df_train, dtrain, pca = utils.load_parquet(
    train_file, features, scaler=scaler, ptkey="CC_pt",oldptkey="HGCalClu_pt", label2=2 if classes > 2 else 1, pca=pca
)
df_test, dtest, pca = utils.load_parquet(
    test_file, features, scaler=scaler, ptkey="CC_pt",oldptkey="HGCalClu_pt", label2=2 if classes > 2 else 1, pca=pca
)


""" w_train=dtrain.get_weight()
w_test=dtest.get_weight()

dtrain.set_weight(w_train*2)
dtest.set_weight(w_test*2) """


# %%
save_model = f"endcap{classes}_131Xv3.json"
save_model = False
load = False
def train(dtrain, dtest, save=False):
    params2 = {
        "tree_method": "hist",
        "max_depth": 10,
        "learning_rate": 0.35,
        "lambda": 5000,
        "alpha": 5000,
        #"colsample_bytree":0.8,
        "subsample": 0.825,
        "gamma":10,
        "min_split_loss":8,
        "min_child_weight": 80,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }
    params3 = {
        "tree_method": "hist",
        "max_depth": 12,
        "learning_rate": 0.5,
        "lambda": 1000,
        "alpha": 1000,
        # "colsample_bytree":0.9,
        "subsample": 0.8,
        # "gamma":5,
        "min_split_loss":5,
        "min_child_weight": 80,
        "objective": "multi:softprob",
        "num_class": classes,
        "eval_metric": "mlogloss",
    }

    params = params3 if classes > 2 else params2
    num_round = 12 if classes > 2 else 15

    evallist = [(dtrain, "train"), (dtest, "eval")]
    eval_result = {}
    model = xgb.train(params, dtrain, num_round, evallist, evals_result=eval_result)
    if save:
        model.save_model(save)
    return model, eval_result


if load:
    model = xgb.Booster()
    model.load_model(load)
else:
    model, eval_result = train(dtrain, dtest, save=save_model)
    plots.plot_loss(eval_result, loss="mlogloss" if classes > 2 else "logloss", save=f"fig/class{classes}/loss.pdf")

df_train["score"] = 1 - model.predict(dtrain)[:, 0] if classes > 2 else model.predict(dtrain)
df_test["score"] = 1 - model.predict(dtest)[:, 0] if classes > 2 else model.predict(dtest)

# %%
ax = profile_int_dec(
    df_train[[*features, "score"]], sign_prop=False, min_dec_bit=12, max_int_bit=15, nmax_differences=10, what="inputs"
)
plt.savefig(f"fig/class{classes}/inout_profile.pdf")

ax = profile_int_dec(model, sign_prop=False, min_dec_bit=12, max_int_bit=15, nmax_differences=10, what="thrs")
plt.savefig(f"fig/class{classes}/thr_profile.pdf")

ax = profile_int_dec(model, sign_prop=False, min_dec_bit=12, max_int_bit=15, nmax_differences=10, what="gains")
plt.savefig(f"fig/class{classes}/gain_profile.pdf")
# %%

rank = plots.plot_importance(model, save=f"fig/class{classes}")


# %%

plots.plot_scores(
    df_train["score"],
    df_train["label"],
    df_test["score"],
    df_test["label"],
    func=lambda x: np.arctanh(x),
    log=True,
    bins=np.linspace(0, 3.5, 30),
    save=f"fig/class{classes}/scores.pdf",
)
plots.plot_pt_roc(df_train, ptkey="CC_pt", save=f"fig/class{classes}/train_pt_roc.pdf")
plots.plot_pt_roc(df_test, ptkey="CC_pt", save=f"fig/class{classes}/test_pt_roc.pdf")

# %%
ax1, _ = plots.plot_best_pt_roc(
    df_train, ptkey="CC_pt", ids=["evId","HGCalClu_id"], thrs_to_select=[0.85, 0.7, 0.35, 0.3, 0.55, 0.85], pt_bins=(0, 5, 10, 20, 30, 50, 150)
)
ax1.text(0.325, 0.45, "Train")
plt.savefig(f"train_roc.pdf")

ax2, _ = plots.plot_best_pt_roc(
    df_test, ptkey="CC_pt",ids=["evId","HGCalClu_id"], thrs_to_select=[0.85, 0.7, 0.35, 0.3, 0.55, 0.85], pt_bins=(0, 5, 10, 20, 30, 50, 150)
)
ax2.text(0.325, 0.45, "Test")
plt.savefig(f"test_roc.pdf")

# %%
