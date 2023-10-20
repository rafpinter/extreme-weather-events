import numpy as np


class LogisticModel:
    def __init__(self, reg, learning_rate, n_iterations) -> None:
        self.w = None
        self.reg = reg
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        linear_output = np.dot(X, self.w)
        return self.sigmoid(linear_output)

    def test(self, X, y):
        ypred = self.predict(X)
        total_errors = 0
        for i in range(len(ypred)):
            if ypred[i] != y[i]:
                total_errors += 1

        return total_errors / len(ypred)

    def loss(self, X, y):
        return np.mean(
            np.log(1 + np.exp(-y * self.predict(X)))
        ) + 0.5 * self.reg * np.sum(self.w**2)

    def gradient(self, X, y):
        a = 1 / (1 + np.exp(y * self.predict(X)))
        b = -y
        return np.mean(a * b, axis=0) + self.reg * self.w

    def train(self, X, y):
        self.w = np.zeros(X.shape[1])

        losses = []
        errors = []

        for epoch in range(self.n_iterations):
            self.w -= self.learning_rate * self.gradient(X, y)

            losses.append(self.loss(X, y))
            errors.append(self.test(X, y))

        print("Training completed: the train error is {:.2f}%".format(errors[-1] * 100))
        return np.array(losses), np.array(errors)

    def accuracy(self, X, y):
        y_predicted = self.predict(X)
        accuracy_value = np.mean(y_predicted == y)
        return accuracy_value

    def precision(self, X, y):
        y_pred = self.predict(X)
        print(y_pred)
        true_positives = 0
        for i in range(len(y_pred)):
            if (y[i, 0] == 1) & (y_pred == 1):
                true_positives += 1
        predicted_positives = np.sum(y_pred == 1)
        print(true_positives)
        print(predicted_positives)
        if predicted_positives == 0:  # To handle the case where the denominator is 0
            return 0
        return true_positives / predicted_positives

    def recall(self, X, y):
        y_pred = self.predict(X)
        true_positives = np.sum((y_pred == 1) & (y == 1))
        actual_positives = np.sum(y == 1)
        if actual_positives == 0:  # To handle the case where the denominator is 0
            return 0
        return true_positives / actual_positives



class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for _ in range(self.num_iterations):
            # Compute model prediction
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in predictions]
        return y_pred_class
    
    
    