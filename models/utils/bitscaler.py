import numpy as np


# TODO test new save and load methods
class BitScaler:
    def __init__(self) -> None:
        self.fitted = False

    def fit(self, range_dict, target=(-1, 1)):
        self.range_dict = range_dict
        if self.fitted:
            raise ValueError("Scaler already fitted")

        self.scale_funcs = {}
        new_low, new_high = target

        for key in self.range_dict:
            low, high = self.range_dict[key]

            quant_range = 2 ** np.ceil(np.log2(high - low))

            #! God damn, python, you suck!!!
            #! Dirty workaround to avoid the lambda function to capture the last value of the loop
            self.scale_funcs[key] = eval(f"lambda x: {new_low}+((x-{low})*({new_high}-{new_low})/{quant_range})")

        self.range_dict["target"] = target
        self.fitted = True

    def apply(self, df):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        for key in self.scale_funcs:
            df[key] = self.scale_funcs[key](df[key])
        return df

    def save(self, filename):
        if not self.fitted:
            raise ValueError("Scaler not fitted")
        np.save(filename, self.range_dict)

    def load(self, filename):
        self.range_dict = np.load(filename, allow_pickle=True).item()
        target = self.range_dict.pop("target")
        self.fit(self.range_dict, target=target)
