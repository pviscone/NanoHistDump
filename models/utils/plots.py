from collections.abc import Iterable
from itertools import pairwise

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.lines import Line2D
from sklearn.metrics import auc, roc_curve

acab_palette = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

hep.styles.cms.CMS["patch.linewidth"] = 2
hep.styles.cms.CMS["lines.linewidth"] = 2
# hep.styles.cms.CMS["axes.prop_cycle"] = cycler("color", acab_palette)
# hep.styles.cms.CMS["figure.autolayout"]=True
hep.styles.cms.CMS["axes.grid"] = True

hep.style.use(hep.style.CMS)


def plot_loss(eval_result, loss="mlogloss", save=False):
    fig, ax = plt.subplots()
    ax.plot(eval_result["train"][loss], label="train")
    ax.plot(eval_result["eval"][loss], label="eval")
    # ax.set_yscale("log")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("LogLoss")
    ax.legend()
    plt.show()
    if save:
        fig.savefig(save)
    return ax


def plot_scores(
    preds_train, y_train, preds_test, y_test, bins=np.linspace(0, 1, 30), save=False, log=False, func=lambda x: x
):
    fig, ax = plt.subplots()

    classes = np.unique(y_train)
    colors = ["dodgerblue", "salmon", "green", "purple"]
    colors = colors[: len(classes)]
    for color, cls in zip(colors, classes):
        cls_pred_test = preds_test[y_test == cls]
        cls_pred_train = preds_train[y_train == cls]
        if cls_pred_train.ndim > 1:
            cls_pred_train = 1 - cls_pred_train[:, 0]
            cls_pred_test = 1 - cls_pred_test[:, 0]

        hatch = "/"
        marker = "v"
        if cls != 0:
            hatch = "\\"
            marker = "^"

        train = np.histogram(func(cls_pred_train), bins=bins, density=True)

        centers = (train[1][1:] + train[1][:-1]) / 2

        ax.plot(
            centers,
            train[0],
            label=f"y={int(cls)} Train",
            marker=marker,
            color=color,
            markersize=10,
            markeredgecolor="black",
            zorder=999,
        )

        ax.hist(
            func(cls_pred_test),
            bins=bins,
            label=f"y={int(cls)} Test",
            hatch=hatch,
            density=True,
            histtype="step",
            color=color,
        )

    ax.set_xlabel("1-Score(Bkg)")
    if log:
        ax.set_yscale("log")
    ax.legend()

    if save:
        fig.savefig(save)
    return fig, ax


def plot_pt_roc(
    data,
    score="score",
    labkey="label",
    ptkey="CryClu_pt",
    pt_bins=(0, 5, 10, 20, 30, 50, 150),
    save=False,
    lumitext="All Cluster-Track Couples",
    eff=False,
    thrs_to_select=False,
):
    thrs = [0.4, 0.6, 0.8, 0.9]
    colors = ["red", "dodgerblue", "green", "gold"]
    markers = ["o", "s", "^", "v", "X", "*"]

    fig, ax = plt.subplots()
    custom_lines = []
    lines = []
    for idx, (minpt, maxpt) in enumerate(pairwise(pt_bins)):
        pt_data = data[np.bitwise_and(data[ptkey] > minpt, data[ptkey] < maxpt)]

        # ddata=xgb.DMatrix(pt_data[features],label=pt_data["label"],enable_categorical=True)
        preds = pt_data[score].to_numpy()
        y = pt_data[labkey].to_numpy()
        y[y == 2] = 1
        # y=ddata.get_label()

        fpr, tpr, thresh = roc_curve(y, preds)
        # frac=len(pt_data[pt_data["label"]!=1])/len(data[data["label"]!=1])
        # fpr=fpr*frac
        # print(frac)

        if eff:
            if isinstance(eff, Iterable):
                e = eff[idx]
            else:
                e = eff

            score_eff_idx = np.argmin(np.abs(tpr - e))
            score_eff = thresh[score_eff_idx]
            print(f"pt=[{minpt},{maxpt}] GeV:tpr={e}  fpr={fpr[score_eff_idx]:.2f} thr={score_eff:.2f}")
        if thrs_to_select:
            if isinstance(thrs_to_select, Iterable):
                thr_to_select = thrs_to_select[idx]
            else:
                thr_to_select = thrs_to_select
            thr_idx = np.argmin(np.abs(thresh - thr_to_select))
            print(f"pt=[{minpt},{maxpt}] GeV:tpr={tpr[thr_idx]:.2f}  fpr={fpr[thr_idx]:.2f} thr={thr_to_select:.2f}")

        roc_auc = auc(fpr, tpr)
        lab = f"pt=[{minpt},{maxpt}] (AUC = {roc_auc:.2f}) "
        if eff:
            lab += rf"$\epsilon_S$={e:.2f}: {score_eff:.2f}"

        if thrs_to_select:
            lab += rf"$\epsilon_S$={tpr[thr_idx]:.2f} $\epsilon_B$={fpr[thr_idx]:.2f}: {thr_to_select:.2f}"

        for col, thr in zip(colors, thrs):
            label = ""
            if idx == 0:
                label = f"score={thr}"
            thridx = np.argmin(np.abs(thresh - thr))
            ax.plot(
                fpr[thridx],
                tpr[thridx],
                color=col,
                marker=markers[idx],
                label=label,
                markersize=12,
                markeredgecolor="black",
                zorder=999,
            )

        (line,) = ax.plot(fpr, tpr, label=lab, alpha=0.9)
        lines.append(line)
        custom_lines.append(
            Line2D(
                [0],
                [0],
                color=plt.gca().lines[-1].get_color(),
                lw=2,
                linestyle="-",
                label=f"{minpt}-{maxpt} GeV",
                marker=markers[idx],
                markersize=10,
                markeredgecolor="black",
            )
        )
    # ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([-0.05, 0.5])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    line_to_custom_line = {line: custom_line for line, custom_line in zip(lines, custom_lines)}
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [line_to_custom_line.get(handle, handle) for handle in handles]
    ax.legend(handles, labels, fontsize=19)
    hep.cms.text("Simulation", ax=ax)
    hep.cms.lumitext(lumitext, ax=ax)
    if save:
        fig.savefig(save)
    return ax


def plot_best_pt_roc(
    data,
    score="score",
    labkey="label",
    ptkey="CryClu_pt",
    pt_bins=(0, 5, 10, 20, 30, 50, 150),
    ids=["evId", "CryClu_id"],
    save=False,
    eff=False,
    thrs_to_select=False,
):
    new_data = data.astype(float)
    new_data = new_data.groupby(ids).max(score).reset_index()
    ax = plot_pt_roc(
        new_data,
        score=score,
        labkey=labkey,
        ptkey=ptkey,
        pt_bins=pt_bins,
        save=save,
        lumitext="Best Couple per Cluster",
        eff=eff,
        thrs_to_select=thrs_to_select,
    )
    return ax, new_data


# ex #profile(df_dict,sign_prop=False)
def profile(
    data,
    bins=np.logspace(-7, 12, 50, base=2),
    sign_prop=True,
    legend=True,
    ax=None,
    zero_dummy_value=None,
    sign_percent=False,
):
    if ax is None:
        fig, ax = plt.subplots()

    # Initialize base position for the y-axis
    base_position = 0

    # Store y-tick positions and labels
    yticks_positions = []
    yticks_labels = []

    for i, (entry, values) in enumerate(data.items()):
        # Separate positive and negative values
        pos_values = values[values > 0]
        neg_values = -values[values < 0]  # Make negative values positive for histogram

        if zero_dummy_value:
            zeros_values_pos = values[values == zero_dummy_value]
            pos_values = pos_values[pos_values != zero_dummy_value]
            zero_hist_pos, _ = np.histogram(zeros_values_pos, bins=bins)

            zeros_values_neg = -values[values == -zero_dummy_value]
            neg_values = neg_values[neg_values != zero_dummy_value]
            zero_hist_neg, _ = np.histogram(zeros_values_neg, bins=bins)

        # Calculate histogram data
        pos_hist, _ = np.histogram(pos_values, bins=bins)
        neg_hist, _ = np.histogram(neg_values, bins=bins)

        # Normalize histograms
        if sign_prop:
            norm_list = [pos_hist.max(), neg_hist.max()]
            if zero_dummy_value:
                norm_list.append(zero_hist_pos.max())
                norm_list.append(zero_hist_neg.max())
            norm = np.max(norm_list)
            pos_hist = pos_hist / norm
            neg_hist = neg_hist / norm
            if zero_dummy_value:
                zero_hist_pos = zero_hist_pos / norm
                zero_hist_neg = zero_hist_neg / norm
        else:
            pos_norm = [np.max(pos_hist)]
            neg_norm = [np.max(neg_hist)]
            if zero_dummy_value:
                pos_norm.append(zero_hist_pos.max())
                neg_norm.append(zero_hist_neg.max())
            pos_hist = pos_hist / np.max(pos_norm)
            neg_hist = neg_hist / np.max(neg_norm)
            if zero_dummy_value:
                zero_hist_pos = zero_hist_pos / np.max(pos_norm)
                zero_hist_neg = zero_hist_neg / np.max(neg_norm)

        ax.axhline(base_position, color="black", lw=1)
        ax.axhline(base_position + 1.05, color="black", alpha=0.2, linestyle="--", lw=1)
        ax.axhline(base_position - 1.05, color="black", alpha=0.2, linestyle="--", lw=1)

        if len(values) < 1:
            pos_frac = 1
            neg_frac = 1
        else:
            pos_frac = len(pos_values) / len(values)
            neg_frac = len(neg_values) / len(values)

        # Plot positive histogram horizontally with a specific color
        ax.fill_between(
            np.repeat(bins, 2)[1:-1],
            base_position,
            np.repeat(base_position + pos_hist, 2),
            alpha=1,
            color="red",
            label="Positive" if i == 0 else "",
        )

        # Plot negative histogram horizontally (mirrored) with another color
        ax.fill_between(
            np.repeat(bins, 2)[1:-1],
            base_position,
            np.repeat(base_position - neg_hist, 2),
            alpha=1,
            color="dodgerblue",
            label="Negative" if i == 0 else "",
        )

        if sign_percent:
            ax.text(bins[-1] / 8, base_position + 0.9, f"{pos_frac*100:.1f}%", color="red")
            ax.text(bins[-1] / 8, base_position + 0.3, f"{neg_frac*100:.1f}%", color="dodgerblue")

        if zero_dummy_value:
            zero_frac_pos = len(zeros_values_pos) / len(values)
            ax.fill_between(
                np.repeat(bins, 2)[1:-1],
                base_position,
                np.repeat(base_position + zero_hist_pos, 2),
                alpha=1,
                color="green",
                label="Zero+" if i == 0 else "",
            )
            ax.text(bins[-1] / 8, base_position + 0.7, f"{zero_frac_pos*100:.1f}%", color="green")

            zero_frac_neg = len(zeros_values_neg) / len(values)
            ax.fill_between(
                np.repeat(bins, 2)[1:-1],
                base_position,
                np.repeat(base_position - zero_hist_neg, 2),
                alpha=1,
                color="purple",
                label="Zero-" if i == 0 else "",
            )
            ax.text(bins[-1] / 8, base_position + 0.5, f"{zero_frac_neg*100:.1f}%", color="purple")

        ax.boxplot(
            pos_values,
            positions=[base_position + 0.25],
            widths=0.2,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="salmon", alpha=0.6),
            notch=False,
            whis=[2.5, 97.5],
            showfliers=False,
        )
        ax.boxplot(
            neg_values,
            positions=[base_position - 0.25],
            widths=0.2,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="deepskyblue", alpha=0.6),
            notch=False,
            whis=[2.5, 97.5],
            showfliers=False,
        )

        # Update y-ticks positions and labels
        yticks_positions.append(base_position)
        yticks_labels.append(entry)

        # Increment base position for the next entry
        base_position += 2.1  # Adjust spacing between histograms

    # Set the x-axis limit to match the bins range
    ax.set_xlim(bins[0], bins[-1])

    # Set y-ticks to the middle of each "violin" and label them with the entry names
    ax.set_yticks(yticks_positions)
    ax.set_yticklabels(yticks_labels)

    # Add legend
    if legend:
        ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True)

    return ax


# ex: ax=profile_int_dec(df_dict,sign_prop=False,title="Threshold",min_dec_bit=12,max_int_bit=15)
def profile_int_dec(
    data, zeros=True, max_int_bit=12, min_dec_bit=-14, sign_prop=True, nmax_differences=1000, what=None
):
    if what == "inputs":
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data should be a pandas DataFrame")
        data_dict = data.to_dict(orient="list")
        data_dict = {key: np.array(entry) for key, entry in data_dict.items()}
        data = data_dict
        title = "Input"
    elif what in ["thrs", "gains"]:
        if not isinstance(data, xgb.Booster):
            raise ValueError("data should be a xgboost.Booster")
        model = data
        features = model.feature_names
        value_dict = {feat: [] for feat in features}

        for feat in features:
            for tree in model:
                df_tree = tree.trees_to_dataframe()
                value_dict[feat].append(
                    df_tree[df_tree["Feature"] == feat]["Split" if what == "thrs" else "Gain"].to_numpy()
                )
            value_dict[feat] = np.concatenate(value_dict[feat])
        data = value_dict
        title = "Thrs." if what == "thrs" else "Gain"
    else:
        raise ValueError("what should be 'inputs','thrs' or 'gains'")

    min_dec_bit = np.abs(min_dec_bit)
    int_bins = np.logspace(-2, max_int_bit, max_int_bit + 3, base=2)
    dec_bins = np.logspace(-min_dec_bit, 1, min_dec_bit + 2, base=2)

    ints = {}
    decs = {}

    def calculate_differences(arr):
        arr = np.array(arr)
        diffs = np.abs(arr[:, np.newaxis] - arr)
        return diffs[np.triu_indices(len(arr), k=1)]

    delta = {}

    for key in data.keys():
        decimals, integers = np.modf(data[key])

        if zeros:
            integers[np.bitwise_and(integers == 0, decimals >= 0)] = 0.25
            integers[np.bitwise_and(integers == 0, decimals < 0)] = -0.25

        decs[key] = decimals
        ints[key] = integers

        arr = np.array([])
        #!
        for i in np.unique(integers):
            mask = integers == i
            dec_masked = decimals[mask]
            n_diff = np.min([len(dec_masked), nmax_differences])
            dec_masked = np.random.choice(dec_masked, size=n_diff, replace=False)
            arr = np.append(arr, calculate_differences(dec_masked)) * np.sign(i)

        delta[key] = arr
    fig, ax = plt.subplots(1, 3, figsize=(20, 5 * len(data)), sharey=True)
    plt.subplots_adjust(wspace=0)
    ax[0] = profile(ints, bins=int_bins, ax=ax[0], zero_dummy_value=0.25, sign_prop=sign_prop, sign_percent=True)
    ax[0].set_title(f"{title} Integer part")

    if zeros:
        xtickslabels = ax[0].get_xticklabels()
        xtickslabels[1].set_text("0")
        xtickslabels[-1].set_text("")
        ax[0].set_xticklabels(xtickslabels)
        ax[0].axvline(0.5, color="black", alpha=0.5)

    ax[1] = profile(decs, bins=dec_bins, ax=ax[1], legend=False, sign_prop=sign_prop)
    ax[1].set_title(f"{title} Decimal part")

    xtickslabels = ax[1].get_xticklabels()
    print(xtickslabels)
    xtickslabels[1] = ""
    ax[1].set_xticklabels(xtickslabels)

    ax[2] = profile(delta, bins=dec_bins, ax=ax[2], legend=False, sign_prop=sign_prop)
    ax[2].set_title(rf"{title} |$\Delta$ dec| (fixed int)")
    xtickslabels = ax[2].get_xticklabels()
    xtickslabels[1] = ""
    ax[2].set_xticklabels(xtickslabels)

    return ax


def plot_importance(model, save=None):
    xgb.plot_importance(model, importance_type="gain", show_values=False)

    if save:
        plt.savefig(f"{save}/importance_average_gain.pdf")
    xgb.plot_importance(model, importance_type="weight", show_values=False)

    if save:
        plt.savefig(f"{save}/importance_weight.pdf")

    rank = {}
    for key in model.get_score():
        rank[key] = model.get_score(importance_type="weight")[key] * model.get_score(importance_type="gain")[key]

    fig, ax = plt.subplots()
    sorted_rank = {k: v for k, v in sorted(rank.items(), key=lambda item: item[1])}

    ax.barh(list(sorted_rank.keys()), width=sorted_rank.values())
    ax.set_xlabel("Gain")

    if save:
        plt.savefig(f"{save}/importance_gain.pdf")
    return sorted_rank
