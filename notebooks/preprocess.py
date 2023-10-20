import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(data, label_subset, feature_subset, n_train):
    """Randomly split data into a train and valid set
    with the subset of classes in label_subset and the subset
    of features in feature_subset.
    ------
    Effectue une partition aléatoire des données en sous-ensembles
    train set et valid set avec le sous-ensemble de classes label_subset
    et le sous-ensemble de feature feature_subset
    """
    # extract only data with class label in label_subset
    # on extrait seulement les classes de label_subset
    data = data[np.isin(data[:, -1], label_subset), :]

    # remap labels to [-1, 1]
    # on transforme les classes pour qu'elles soient [-1, 1]
    if len(label_subset) != 2:
        raise UserWarning("We are exclusively  dealing with binary classification.")
    data[data[:, -1] == label_subset[0], -1] = -1
    data[data[:, -1] == label_subset[1], -1] = 1

    # extract chosen features + labels
    # on extrait les features et leurs étiquettes
    data = data[:, feature_subset + [-1]]

    # insert a column of 1s for the bias
    # on ajoute une colonne pour le biais
    data = np.insert(data, -1, 1, axis=1)

    # separate into train and valid
    # on sépare en train et valid
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)
    train_inds = inds[:n_train]
    valid_inds = inds[n_train:]
    trainset = data[train_inds]
    validset = data[valid_inds]

    # normalize train set to mean 0 and standard deviation 1 feature-wise
    # apply the same transformation to the valid set
    # on normalise les données pour qu'elles soient de moyenne 0
    # et d'écart-type 1 par caractéristique et on applique
    # ces mêmes transformations au valid set
    mu = trainset[:, :2].mean(axis=0)
    sigma = trainset[:, :2].std(axis=0)
    trainset[:, :2] = (trainset[:, :2] - mu) / sigma
    validset[:, :2] = (validset[:, :2] - mu) / sigma

    return trainset, validset


class Preprocess:
    def __init__(self) -> None:
        self.data = None

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        return self.data

    def preprocess_data(self, data, drop_cols=["SNo"]):
        # Droping cols
        df = self.data.drop(drop_cols, axis=1)

        # dummy variables
        # labels = pd.get_dummies(df["Label"], prefix="label")
        # df = pd.concat([df, labels], axis=1)
        # df = df.drop("Label", axis=1)

        # to numpy
        df_numpy = df.to_numpy()

        return df, df_numpy

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

    def normalize_data(self, data):
        mu = data[:, :2].mean(axis=0)
        sigma = data[:, :2].std(axis=0)
        data[:, :2] = (data[:, :2] - mu) / sigma
        return data
