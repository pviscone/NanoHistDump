import numpy as np


class BitScaler:
    def __init__(self) -> None:
        self.fitted=False

    def fit(self,range_dict):
        self.range_dict = range_dict
        if self.fitted:
            raise ValueError("Scaler already fitted")

        self.scale_dict={}
        for key in self.range_dict:
            low,high=self.range_dict[key]
            mu=(high+low)/2
            div=2**np.ceil(np.log2((high-low)/2))
            self.scale_dict[key]=(mu,div)

        self.fitted=True

    def apply(self,df):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        for key in self.scale_dict:
            df[key]=(df[key]-self.scale_dict[key][0])/self.scale_dict[key][1]
        return df


    def save(self,filename):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        np.save(filename, self.scale_dict)


    def load(self,filename):
        self.scale_dict=np.load(filename,allow_pickle=True).item()
        self.fitted=True

