# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#!-----------------Create a dataset-----------------!#
df = pd.read_parquet("131Xv3.parquet")
y = (df["label"].astype(int) & df["Tk_isReal"] == 1).astype(int).to_numpy()

pt_weight = df["pt_weight"].to_numpy()
weight = pt_weight
weight[y == 1] = weight[y == 1] * np.sum(pt_weight[y == 0]) / np.sum(pt_weight[y == 1])

df_train = df.drop(columns=["label", "Tk_isReal", "ev_id", "pt_weight"])

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    df_train, y, weight, test_size=0.2, random_state=666
)


# %%
#!-----------------Train a BDT-----------------!#
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test)
# create model instance
params = {
    "tree_method": "hist",
    "max_depth": 5,
    "learning_rate": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}
num_round = 35
evallist = [(dtrain, "train"), (dtest, "eval")]
# fit model
eval_result = {}
model = xgb.train(params, dtrain, num_round, evallist, evals_result=eval_result)
# make predictions
preds_test = model.predict(dtest)
preds_train = model.predict(dtrain)

# %%
#!-----------------Plot Loss-----------------!#
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

plt.plot(eval_result["train"]["logloss"], label="Train")
plt.plot(eval_result["eval"]["logloss"], label="Test")
# plt.yscale("log")
plt.xlabel("Boosting Round")
plt.ylabel("Log Loss")
plt.legend()
plt.grid()
plt.save("fig/BDT_Loss_131Xv3.pdf")
# %%
#!-----------------Plot BDT Score-----------------!#
func = lambda x: x
bins = np.linspace(0, 1, 10)

train_hist_signal = np.histogram(
    func(preds_train[y_train == 1]), density=True, bins=bins, weights=dtrain.get_weight()[y_train == 1]
)
train_hist_pu = np.histogram(
    func(preds_train[y_train == 0]), density=True, bins=bins, weights=dtrain.get_weight()[y_train == 0]
)
centers = (train_hist_signal[1][1:] + train_hist_signal[1][:-1]) / 2


fig, ax1 = plt.subplots()
#plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

ax1.hist(
    func(preds_test[y_test == 1]),
    hatch="/",
    bins=bins,
    label="Signal Test",
    density=True,
    histtype="step",
    linewidth=2,
    weights=dtest.get_weight()[y_test == 1],
)

ax1.hist(
    func(preds_test[y_test == 0]),
    hatch="\\",
    bins=bins,
    label="PU Test",
    density=True,
    histtype="step",
    linewidth=2,
    weights=dtest.get_weight()[y_test == 0],
)


ax1.plot(centers, train_hist_signal[0], "v", label="Signal Train", color="Blue")
ax1.plot(centers, train_hist_pu[0], "^", label="PU Train", color="red")

ax1.grid()
ax1.legend()
ax1.set_xlabel("BDT Score")
ax1.set_ylabel("Density")
hep.cms.text("Simulation Phase-2")
hep.cms.lumitext("PU200")
""" q=(train_hist_signal[0]**2)/(train_hist_signal[0]+train_hist_pu[0])
ax2.plot(centers,q)
ax2.grid()
ax2.set_yscale("log") """
fig.savefig("fig/BDT_Score_131Xv3.pdf")
# %%
#!-----------------Plot ROC Curve-----------------!#
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, preds_test, sample_weight=w_test)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid()
hep.cms.text("Simulation Phase-2")
hep.cms.lumitext("PU200")
plt.savefig("fig/BDT_ROC_131Xv3.pdf")
# %%
#!-----------------Plot Efficiency & FOM-----------------!#
plt.plot(thresholds, tpr, label="Signal efficiency")
plt.plot(thresholds, fpr, label="PU efficiency")

purity = np.nan_to_num(tpr / (tpr + fpr), 1)
plt.plot(thresholds, purity, "--", label="Purity")
plt.plot(thresholds, purity * tpr, ":", label="Purity*Signal efficiency")
hep.cms.text("Simulation Phase-2")
hep.cms.lumitext("PU200")

# approximation
fom = np.nan_to_num(tpr / np.sqrt(tpr + fpr), 0) * np.sqrt(len(y_test) / 2)
plt.plot(thresholds, fom / np.max(fom), "-.", label="$\\frac{S}{\\sqrt{S+B}}$ /max(FOM)")
plt.legend()
plt.grid()
plt.xlabel("BDT Threshold")
plt.ylabel("Efficiency")
hep.cms.text("Simulation Phase-2")
hep.cms.lumitext("PU200")
plt.savefig("fig/BDT_Efficiency_131Xv3.pdf")

#%%
xgb.save(model,"BDT_131Xv3.model")