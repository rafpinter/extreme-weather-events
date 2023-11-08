import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List


class Preprocess:
    def __init__(self) -> None:
        self.data = None

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        return self.data

    def preprocess_data(self, data, drop_cols=["SNo"], is_test=False):
        df = data.copy()

        # Droping cols
        if not is_test:
            y = df["Label"].copy()
            df = df.drop(["Label"], axis=1)

        # dummy variables
        # labels = pd.get_dummies(df["Label"], prefix="label")
        # df = pd.concat([df, labels], axis=1)
        # df = df.drop("Label", axis=1)

        # cyclical longitude
        df = self.cyclical_features(data=df, feature="lon", total=360)

        # difs
        df = self.dif_cols(df)

        # multi
        df = self.multi_cols(df)

        # date spli
        df = self.date_split(df)

        # Squaring cols
        df = self.square_cols(df)

        ## wind norm
        df = self.calculate_norm_of_wind(df)

        # Water per second
        df = self.water_per_second(df)

        # to numpy
        if not is_test:
            df["Label"] = y

        df = df.drop(drop_cols, axis=1)
        # df_numpy = df.to_numpy()

        return df

    def train_valid_split(self, data, test_size=0.33, random_state=42):
        X_train, X_valid, y_train, y_valid = train_test_split(
            data[:, :-1], data[:, -1:], test_size=test_size, random_state=random_state
        )

        return (
            X_train,
            y_train[:, -1],
            X_valid,
            y_valid[:, -1],
        )

    def calculate_norm_of_wind(self, data):
        data["norm_wind_850"] = data[["U850", "V850"]].apply(
            lambda x: (x["U850"] ** 2 + x["V850"] ** 2) ** (1 / 2), axis=1
        )
        data["norm_wind_bot"] = data[["UBOT", "VBOT"]].apply(
            lambda x: (x["UBOT"] ** 2 + x["VBOT"] ** 2) ** (1 / 2), axis=1
        )
        data["wind_diff"] = data["norm_wind_bot"] - data["norm_wind_850"]
        return data

    def date_split(self, df):
        # df["year"] = df["time"].apply(lambda x: str(x)[:4])
        df["month"] = df["time"].apply(lambda x: int(str(x)[4:6]))
        return df

    def water_per_second(self, data):
        data["total_water_per_s"] = data["TMQ"] / data["norm_wind_bot"]
        return data

    def normalize_data(self, data):
        mu = data[:, :].mean(axis=0)
        sigma = data[:, :].std(axis=0)
        data[:, :] = (data[:, :] - mu) / sigma
        return data

    def min_max_scale(self, data):
        minim = np.min(data, axis=0)
        maxim = np.max(data, axis=0)
        data = (data - minim) / (maxim - minim)
        return data

    def cyclical_features(self, data, feature, total):
        data[f"{feature}_sin"] = np.sin(2 * np.pi * data[feature] / total)
        data[f"{feature}_cos"] = np.cos(2 * np.pi * data[feature] / total)
        return data

    def abs_features(self, data, features: List):
        for col in features:
            data[f"abs_{col}"] = data[col].abs()
        return data

    def multi_cols(self, df):
        mults = [
            ["PS", "PSL"],
            ["T200", "T500"],
            ["TREFHT", "T200"],
            ["TREFHT", "T500"],
            ["U850", "UBOT"],
            ["V850", "VBOT"],
            ["Z200", "Z1000"],
            ["ZBOT", "Z1000"],
            ["ZBOT", "Z200"],
            ["V850", "U850"],
            ["UBOT", "VBOT"],
        ]

        for cols in mults:
            df["*".join(cols)] = df[cols[0]] * df[cols[1]]

        return df

    def dif_cols(self, df):
        difs = [
            ["PS", "PSL"],
            ["T200", "T500"],
            ["TREFHT", "T200"],
            ["TREFHT", "T500"],
            ["Z200", "Z1000"],
        ]

        for cols in difs:
            df["-".join(cols)] = df[cols[0]] - df[cols[1]]

        return df

    def square_cols(self, data):
        sq_cols = [
            "TMQ",
            "U850",
            "V850",
            "UBOT",
            "VBOT",
            "PS",
            "PSL",
            "T200",
            "T500",
            "PRECT",
            "TS",
            "Z1000",
            "Z200",
            "ZBOT",
        ]

        for col in sq_cols:
            data[f"{col}_{col}"] = data[col] * data[col]
        return data

    def is_north(self, data):
        data["is_northern_hemisphere"] = np.where(data["lat"] > 0)
