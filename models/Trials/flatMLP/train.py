#%%
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential

sys.path.append("..")
from utils.plots import plot_scores,plot_best_pt_roc

features=[
    "CryClu_pt",
    "CryClu_ss",
    "CryClu_relIso",
    "CryClu_standaloneWP",
    "CryClu_looseL1TkMatchWP",
    "Tk_chi2RPhi",
    "Tk_PtFrac",
    "PtRatio",
    "nMatch",
    "abs_dEta",
    "abs_dPhi",
    #"dEta",
    #"dPhi",
    #"CryClu_isSS",
    #Comment for light model
    #"CryClu_isIso",
    #"CryClu_isLooseTkIso",
    #"CryClu_isLooseTkSS",
    #"CryClu_brems",
    #"Tk_hitPattern",
    #"Tk_nStubs",
    #"Tk_chi2Bend",
    #"Tk_chi2RZ",
    #
    #"Tk_pt",
    #"maxPtRatio_other",
    #"minPtRatio_other",
    #"meanPtRatio_other",
    #"stdPtRatio_other",
]


# Sample DataFrame
df = pd.read_parquet("131Xv3.parquet")

# Split into features (X) and labels (y)
y = df["label"]
X = df[features]

#y to one-hot encoding
enc = OneHotEncoder()
y = enc.fit_transform(y.to_numpy().reshape(-1, 1)).toarray()

#sample weights
w = df["weight"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.33, random_state=666)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(3, activation="softmax"))  # Use 'softmax' for multi-class classification

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"]
              )

# Train the model
history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=64,
                    validation_data=(X_test,y_test,w_test),
                    sample_weight=w_train
                    )

# Evaluate the model
#loss, accuracy = model.evaluate(X_test, y_test)

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="test")
plt.legend()

# Make predictions
preds_test = model.predict(X_test)
preds_train = model.predict(X_train)

#%%

fig,ax=plot_scores(preds_train, np.where(y_train==1)[1], preds_test, np.where(y_test==1)[1], save=False, log=False)

#%%
df["score"]=1-model.predict(df[features])[:,0]
df_train,df_test=train_test_split(df,test_size=0.33, random_state=666)
ax1,_=plot_best_pt_roc(df_train,thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],pt_bins=(0,5,10,20,30,50,150))
ax1.text(0.3,0.5,"Train",fontsize=28)

ax2,_=plot_best_pt_roc(df_test,thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],pt_bins=(0,5,10,20,30,50,150))
ax2.text(0.3,0.5,"Test",fontsize=28)

ax3,_=plot_best_pt_roc(df,thrs_to_select=[0.85,0.7,0.35,0.3,0.55,0.85],pt_bins=(0,5,10,20,30,50,150))
ax3.text(0.3,0.5,"All",fontsize=28)

# %%
