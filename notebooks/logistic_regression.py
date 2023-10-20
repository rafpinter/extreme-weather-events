import numpy as np


class LogisticRegressionOvR:
    def __init__(self, learning_rate=0.1, num_iterations=1000, lambda_reg=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg
        self.models = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.unique_labels = np.unique(y)
        self.models = []

        for label in self.unique_labels:
            y_binary = np.where(y == label, 1, 0)
            lr = LogisticRegression(
                self.learning_rate, self.num_iterations, self.lambda_reg
            )
            lr.fit(X, y_binary)
            self.models.append(lr)

    def predict(self, X):
        # Get the raw probabilities for each model
        scores = [model.predict(X) for model in self.models]

        # Select the label with the highest probability for each sample
        predictions = self.unique_labels[np.argmax(scores, axis=0)]

        return predictions

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

        print("Confusion Matrix:")
        print(matrix, "\n")
        return matrix

    def accuracy(self, y_true, y_pred):
        print("Accuracy:")
        acc = np.mean(y_true == y_pred)
        print(round(acc, 4), "\n")
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

        print("Precision:")
        precision = np.mean(precisions)
        print(round(precision, 4), "\n")
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

        print("Recall:")
        rec = np.mean(recalls)
        print(round(rec, 4))
        return rec

    def get_metrics(self, y_true, y_pred, return_values=False):
        matrix = self.confusion_matrix(y_true, y_pred)
        acc = self.accuracy(y_true, y_pred)
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        if return_values:
            return matrix, acc, prec, rec


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_reg=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg  # Regularization parameter
        self.weights = None

    def sigmoid(self, z):
        # print(1 / (1 + np.exp(-z)))
        return 1 / (1 + np.exp(-z))

    def loss(self, X, y):
        return np.mean(
            np.log(1 + np.exp(-y * self.predict(X)))
        ) + 0.5 * self.lambda_reg * np.sum(self.weights**2)

    def gradient(self, X, y):
        return (
            np.mean((-y / (np.exp(y * self.predict(X)) + 1))[:, np.newaxis] * X, axis=0)
            + self.lambda_reg * self.weights
        )

    def predict(self, X, augment=False):
        """Return f(x) for a batch X
        Retourne f(x) pour un batch X
        """
        if augment:
            num_samples, _ = X.shape
            X = np.hstack([np.ones((num_samples, 1)), X])
        return self.sigmoid(np.dot(X, self.weights))

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Augment X with a column of ones for the bias term
        X_augmented = np.hstack([np.ones((num_samples, 1)), X])

        # Initialize weights (including bias)
        self.weights = np.zeros(num_features + 1)

        for _ in range(self.num_iterations):
            # Update weights
            self.weights -= self.learning_rate * self.gradient(X_augmented, y)

    # def predict(self, X):
    #     num_samples = X.shape[0]

    #     # Augment X with a column of ones for the bias term
    #     X_augmented = np.hstack([np.ones((num_samples, 1)), X])

    #     linear_model = np.dot(X_augmented, self.weights)
    #     return self.sigmoid(linear_model)

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

        print("Confusion Matrix:")
        print(matrix, "\n")
        return matrix

    def accuracy(self, y_true, y_pred):
        print("Accuracy:")
        acc = np.mean(y_true == y_pred)
        print(round(acc, 4), "\n")
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

        print("Precision:")
        precision = np.mean(precisions)
        print(round(precision, 4), "\n")
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

        print("Recall:")
        rec = np.mean(recalls)
        print(round(rec, 4))
        return rec

    def get_metrics(self, y_true, y_pred, return_values=False):
        matrix = self.confusion_matrix(y_true, y_pred)
        acc = self.accuracy(y_true, y_pred)
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        if return_values:
            return matrix, acc, prec, rec


# Sample usage
if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 5], [1, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 1, 2, 0, 1, 2, 2])

    classifier = LogisticRegressionOvR(learning_rate=0.01, num_iterations=1000)
    classifier.fit(X, y)
    predictions = classifier.predict(X)

    print(predictions)
