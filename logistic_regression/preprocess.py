import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List


class Preprocess:
    """
    A class used to preprocess weather data for machine learning analysis.

    Attributes:
    -----------
    data : DataFrame or None
        A variable to store data, initialized as None.

    Methods:
    --------
    load_data(filepath):
        Loads data from a CSV file into the 'data' attribute.

    preprocess_data(data, drop_cols, is_test):
        Performs preprocessing operations such as feature engineering and data normalization.

    train_valid_split(data, test_size, random_state):
        Splits the data into training and validation sets.

    calculate_norm_of_wind(data):
        Calculates the norm of the wind vector from its U and V components.

    date_split(df):
        Extracts the month from a date column.

    water_per_second(data):
        Calculates the water content per second using 'TMQ' and 'norm_wind_bot' features.

    normalize_data(data):
        Normalizes the data using the Z-score method.

    min_max_scale(data):
        Scales the data to a range between 0 and 1 using min-max scaling.

    cyclical_features(data, feature, total):
        Converts a linear feature into its cyclical components using sine and cosine transformations.

    abs_features(data, features):
        Creates absolute value features for the specified columns.

    multi_cols(df):
        Creates new features by multiplying pairs of columns.

    dif_cols(df):
        Creates new features by finding the difference between pairs of columns.

    square_cols(data):
        Creates new features by squaring selected columns.
    """

    def __init__(self) -> None:
        self.data = None

    def load_data(self, filepath):
        """Load data from a CSV file into the DataFrame."""
        self.data = pd.read_csv(filepath)
        return self.data

    def preprocess_data(self, data, drop_cols=["SNo"], is_test=False):
        """Perform preprocessing on the data such as dropping columns, feature engineering, and encoding."""
        df = data.copy()

        # Droping cols
        if not is_test:
            y = df["Label"].copy()
            df = df.drop(["Label"], axis=1)

        # cyclical longitude
        df = self.cyclical_features(data=df, feature="lon", total=360)

        # Difference
        df = self.dif_cols(df)

        # Multiplication
        df = self.multi_cols(df)

        # Date split: get month
        df = self.date_split(df)

        # Squaring cols
        df = self.square_cols(df)

        ## wind norm
        df = self.calculate_norm_of_wind(df)

        # Water per second
        df = self.water_per_second(df)

        # Adding label again
        if not is_test:
            df["Label"] = y

        df = df.drop(drop_cols, axis=1)

        return df

    def train_valid_split(self, data, test_size=0.33, random_state=42):
        X_train, X_valid, y_train, y_valid = train_test_split(
            data[:, :-1], data[:, -1:], test_size=test_size, random_state=random_state
        )
        """Split the data into training and validation sets."""

        return (
            X_train,
            y_train[:, -1],
            X_valid,
            y_valid[:, -1],
        )

    def calculate_norm_of_wind(self, data):
        """Calculate the norm of wind vectors at two different atmospheric levels."""
        data["norm_wind_850"] = data[["U850", "V850"]].apply(
            lambda x: (x["U850"] ** 2 + x["V850"] ** 2) ** (1 / 2), axis=1
        )
        data["norm_wind_bot"] = data[["UBOT", "VBOT"]].apply(
            lambda x: (x["UBOT"] ** 2 + x["VBOT"] ** 2) ** (1 / 2), axis=1
        )
        data["wind_diff"] = data["norm_wind_bot"] - data["norm_wind_850"]
        return data

    def date_split(self, df):
        """Extract the month from the 'time' column."""
        df["month"] = df["time"].apply(lambda x: int(str(x)[4:6]))
        return df

    def water_per_second(self, data):
        """Calculate total water content per second using 'TMQ' and 'norm_wind_bot'."""
        data["total_water_per_s"] = data["TMQ"] / data["norm_wind_bot"]
        return data

    def normalize_data(self, data):
        """Normalize the data using the Z-score normalization method."""
        mu = data[:, :].mean(axis=0)
        mu = data[:, :].mean(axis=0)
        sigma = data[:, :].std(axis=0)
        data[:, :] = (data[:, :] - mu) / sigma
        return data

    def min_max_scale(self, data):
        """Scale the data between 0 and 1 using the min-max scaling method."""
        minim = np.min(data, axis=0)
        maxim = np.max(data, axis=0)
        data = (data - minim) / (maxim - minim)
        return data

    def cyclical_features(self, data, feature, total):
        """Convert a linear feature into its cyclical components using sine and cosine."""
        data[f"{feature}_sin"] = np.sin(2 * np.pi * data[feature] / total)
        data[f"{feature}_cos"] = np.cos(2 * np.pi * data[feature] / total)
        return data

    def abs_features(self, data, features: List):
        """Create absolute value features for specified columns."""
        for col in features:
            data[f"abs_{col}"] = data[col].abs()
        return data

    def multi_cols(self, df):
        """Create new features by multiplying pairs of existing features."""
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
        """Create new features by finding the difference between pairs of existing features."""
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
        """Create new features by squaring selected existing features."""
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
