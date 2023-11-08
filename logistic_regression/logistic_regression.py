"""
Some inspirations for the code
https://www.kaggle.com/code/vitorgamalemos/multinomial-logistic-regression-from-scratch/notebook#Multinomial-Logistic-Regression-(Softmax-Regression)
https://stats.stackexchange.com/questions/166958/multinomial-logistic-loss-vs-cross-entropy-vs-square-error#:~:text=The%20cost%20function%20of%20Multinomial,(i)))%5D.
And some corrections from chatgpt
"""

import numpy as np
import pandas as pd
import numpy as np
from logistic_regression import LogisticRegression, LogisticRegressionOvR
from sklearn.model_selection import train_test_split
from preprocess import Preprocess
from datetime import datetime
import mlflow
import mlflow.sklearn
from scipy.optimize import minimize


class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularizer=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularizer = regularizer

    def fit(self, X, y, valid_x, valid_y, collist=[]):
        mlflow.set_experiment("MultinomialLogisticRegression")

        self.num_classes = len(np.unique(y))
        self.weights = np.zeros((X.shape[1], self.num_classes))
        self.bias = np.zeros(self.num_classes)

        with mlflow.start_run(nested=True) as run:
            # Log hyperparameters
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("num_iterations", self.num_iterations)
            mlflow.log_param("collist", ",".join(collist))

            self.train_errors = []
            self.valid_errors = []
            self.iters = []
            self.gradients = []

            for i in range(self.num_iterations):
                scores = np.dot(X, self.weights) + self.bias
                probs = self.softmax(scores)
                loss = self.cross_entropy(probs, y)
                gradient = self.gradient(X, probs, y)

                self.weights -= self.learning_rate * gradient
                self.bias -= self.learning_rate * np.sum(gradient, axis=0)

                if i % 10 == 0:
                    self.train_errors.append(loss)

                    scores_valid = np.dot(valid_x, self.weights) + self.bias
                    probs_valid = self.softmax(scores_valid)
                    self.valid_errors.append(self.cross_entropy(probs_valid, valid_y))

                    self.gradients.append(gradient)
                    self.iters.append(i)

                if i % 100 == 0:
                    print(f"Epoch {i}, cross entropy loss: {loss}")

            # Define a file path for your list
            file_path = "weights.txt"
            # Write the list to a file
            with open(file_path, "w") as file:
                for item in self.weights:
                    file.write(f"{item}\n")
            mlflow.log_artifact(file_path)

            # Define a file path for your list
            file_path = "bias.txt"
            # Write the list to a file
            with open(file_path, "w") as file:
                for item in self.bias:
                    file.write(f"{item}\n")
            mlflow.log_artifact(file_path)

            ## Predict
            y_pred = self.predict(X)

            # Log metrics
            mlflow.log_metric("accuracy", self.accuracy(y, y_pred))
            mlflow.log_metric("precision", self.precision(y, y_pred))
            mlflow.log_metric("recall", self.recall(y, y_pred))
            mlflow.log_metric("f1_score", self.f1_score(y, y_pred))

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)

    def softmax(self, scores):
        return np.exp(scores - np.max(scores, axis=1, keepdims=True)) / np.sum(
            np.exp(scores - np.max(scores, axis=1, keepdims=True)),
            axis=1,
            keepdims=True,
        )

    def cross_entropy(self, probs, y):
        return -np.sum(np.log(probs[range(len(y)), y.astype(int)])) / len(
            y
        ) + 0.5 * self.regularizer * np.sum(self.weights**2)

    def gradient(self, X, probs, y):
        return np.dot(
            X.T,
            (probs - (np.arange(self.num_classes) == y[:, None]).astype(int)) / len(y),
        ) + (self.regularizer) * np.sum(self.weights)

    def separete_feat_folds(self, dataset):
        sep = np.ceil(np.linspace(start=0, stop=dataset.shape[0], num=5)).astype(int)
        print(sep)
        fold1 = dataset[0 : sep[1], :]
        fold2 = dataset[sep[1] : sep[2], :]
        fold3 = dataset[sep[2] : sep[3], :]
        fold4 = dataset[sep[3] : sep[4], :]
        fold5 = dataset[sep[4] :, :]
        return (
            fold1,
            fold2,
            fold3,
            fold4,
            fold5,
        )

    def separete_label_folds(self, dataset):
        sep = np.ceil(np.linspace(start=0, stop=dataset.shape[0], num=5)).astype(int)
        print(sep)
        fold1 = dataset[0 : sep[1]]
        fold2 = dataset[sep[1] : sep[2]]
        fold3 = dataset[sep[2] : sep[3]]
        fold4 = dataset[sep[3] : sep[4]]
        fold5 = dataset[sep[4] :]
        return (
            fold1,
            fold2,
            fold3,
            fold4,
            fold5,
        )

    def cross_validation_train(self, train_val_x, train_val_y, test_val_x, test_val_y):
        self.fit(train_val_x, train_val_y, test_val_x, test_val_y)
        predictions = self.predict(test_val_x)
        self.cross_validation_accuracy.append(self.accuracy(test_val_y, predictions))
        self.cross_valid_train_errors.append(self.train_errors)
        self.cross_valid_valid_errors.append(self.valid_errors)
        self.cross_valid_iters.append(self.iters)

    def cross_validation(self, X, y):
        """Adapted from
        https://www.kaggle.com/code/joycpkxatze/k-fold-cross-validation-from-scratch-python
        """

        self.cross_validation_accuracy = []
        self.cross_valid_train_errors = []
        self.cross_valid_valid_errors = []
        self.cross_valid_iters = []

        x_fold1, x_fold2, x_fold3, x_fold4, x_fold5 = self.separete_feat_folds(X)
        y_fold1, y_fold2, y_fold3, y_fold4, y_fold5 = self.separete_label_folds(y)

        train_val1_x = np.concatenate((x_fold1, x_fold2, x_fold3, x_fold4))
        train_val1_y = np.concatenate((y_fold1, y_fold2, y_fold3, y_fold4))
        test_val1_x = x_fold5
        test_val1_y = y_fold5
        self.cross_validation_train(
            train_val1_x, train_val1_y, test_val1_x, test_val1_y
        )

        train_val2_x = np.concatenate((x_fold1, x_fold2, x_fold3, x_fold5))
        train_val2_y = np.concatenate((y_fold1, y_fold2, y_fold3, y_fold5))
        test_val2_x = x_fold4
        test_val2_y = y_fold4
        self.cross_validation_train(
            train_val2_x, train_val2_y, test_val2_x, test_val2_y
        )

        train_val3_x = np.concatenate((x_fold1, x_fold2, x_fold4, x_fold5))
        train_val3_y = np.concatenate((y_fold1, y_fold2, y_fold4, y_fold5))
        test_val3_x = x_fold3
        test_val3_y = y_fold3
        self.cross_validation_train(
            train_val3_x, train_val3_y, test_val3_x, test_val3_y
        )

        train_val4_x = np.concatenate((x_fold1, x_fold3, x_fold4, x_fold5))
        train_val4_y = np.concatenate((y_fold1, y_fold3, y_fold4, y_fold5))
        test_val4_x = x_fold2
        test_val4_y = y_fold2
        self.cross_validation_train(
            train_val4_x, train_val4_y, test_val4_x, test_val4_y
        )

        train_val5_x = np.concatenate((x_fold2, x_fold3, x_fold4, x_fold5))
        train_val5_y = np.concatenate((y_fold2, y_fold3, y_fold4, y_fold5))
        test_val5_x = x_fold1
        test_val5_y = y_fold1
        self.cross_validation_train(
            train_val5_x, train_val5_y, test_val5_x, test_val5_y
        )

        print("Accuracies:", self.cross_validation_accuracy)
        print("Avg Accuracy:", np.mean(self.cross_validation_accuracy))

    def confusion_matrix(self, y_true, y_pred):
        """
        Compute confusion matrix for given true and predicted labels.

        :param y_true: Actual class labels
        :param y_pred: Predicted class labels
        :return: Confusion matrix
        """

        # Get unique class labels
        classes = np.unique(np.concatenate((y_true, y_pred)))
        num_classes = len(classes)

        # Create a mapping from class label to index
        label_to_index = {label: idx for idx, label in enumerate(classes)}

        # Initialize confusion matrix with zeros
        matrix = np.zeros((num_classes, num_classes), dtype=int)

        # Populate the confusion matrix
        for true_label, pred_label in zip(y_true, y_pred):
            matrix[label_to_index[true_label], label_to_index[pred_label]] += 1

        return matrix

    def accuracy(self, y_true, y_pred):
        acc = np.mean(y_true == y_pred)

        return acc

    def precision(self, y_true, y_pred):
        classes = np.unique(y_true)
        precisions = []

        for cls in classes:
            true_positives = np.sum((y_pred == cls) & (y_true == cls))
            false_positives = np.sum((y_pred == cls) & (y_true != cls))

            if true_positives + false_positives == 0:
                precision_cls = 0
            else:
                precision_cls = true_positives / (true_positives + false_positives)

            precisions.append(precision_cls)

        precision = np.mean(precisions)

        return precision

    def recall(self, y_true, y_pred):
        classes = np.unique(y_true)
        recalls = []

        for cls in classes:
            true_positives = np.sum((y_pred == cls) & (y_true == cls))
            false_negatives = np.sum((y_pred != cls) & (y_true == cls))

            if true_positives + false_negatives == 0:
                recall_cls = 0
            else:
                recall_cls = true_positives / (true_positives + false_negatives)

            recalls.append(recall_cls)

        rec = np.mean(recalls)
        return rec

    def f1_score(self, y_true, y_pred):
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)

        # Avoid division by zero
        if prec + rec == 0:
            return 0

        f1 = 2 * (prec * rec) / (prec + rec)

        return f1

    def get_metrics(self, y_true, y_pred, return_values=False):
        matrix = self.confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(matrix, "\n")

        acc = self.accuracy(y_true, y_pred)
        print("Accuracy:")
        print(round(acc, 4), "\n")

        prec = self.precision(y_true, y_pred)
        print("Precision:")
        print(round(prec, 4), "\n")

        rec = self.recall(y_true, y_pred)
        print("Recall:")
        print(round(rec, 4), "\n")

        f1 = self.f1_score(y_true, y_pred)
        print("F1 Score:")
        print(round(f1, 4))

        if return_values:
            return matrix, acc, prec, rec, f1


# Example usage:
if __name__ == "__main__":
    # Generate some random data for demonstration purposes.

    raw_train_data = "data/train.csv"
    raw_test_data = "data/test.csv"

    preproc = Preprocess()
    raw_data = preproc.load_data(raw_train_data)

    cols = [
        "lat",
        "Z1000",
        "Z200",
        "TMQ",
        "PSL",
        "U850",
        "VBOT",
        "TS",
        "QREFHT",
        "UBOT",
        "time",
    ]

    raw_data = raw_data[cols + ["Label"]]

    train_df, train_data = preproc.preprocess_data(raw_data, drop_cols=["time"])
    np.random.shuffle(train_data)
    X_train, y_train, X_valid, y_valid = preproc.train_valid_split(
        train_data, test_size=0.33, random_state=42
    )

    print(train_df.columns)

    X_train = preproc.normalize_data(X_train)
    X_valid = preproc.normalize_data(X_valid)

    # Create and train the model.
    model = MultinomialLogisticRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train, y_train, collist=raw_data.columns)

    # Make predictions.
    predictions = model.predict(X_valid)
    print(train_df.columns)

    model.get_metrics(y_valid, predictions, return_values=False)

    # test
    preproc = Preprocess()
    raw_data = preproc.load_data(raw_test_data)
    raw_data = raw_data[cols + ["SNo"]]

    test_df, test_data = preproc.preprocess_data(
        raw_data, drop_cols=["SNo", "time"], is_test=True
    )

    print(test_df.columns)

    test_data = preproc.normalize_data(test_data)
    y_pred_test = model.predict(test_data)

    submition = raw_data["SNo"].reset_index().copy()
    submition["Label"] = pd.Series(y_pred_test)
    submition.drop("index", axis=1, inplace=True)
    submition.to_csv(f"predictions_{datetime.now()}.csv", index=False)
