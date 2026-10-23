
import numpy as np
from data_partition import get_data


# Calculate predictions using weights
def forward(x, weights):

    scores = x @ weights.T

    return scores


# Calculate L2 loss (mean squared error)
def calculate_loss(scores, targets):

    loss = np.mean(np.sum((scores - targets) ** 2, axis=1))

    return loss


# Train linear classifier using gradient descent
def train_linear_classifier(x_train, y_train, epochs=100, learning_rate=0.1):

    # Number of input pixels and output classes
    num_features = x_train.shape[1]
    num_classes = 10

    # Start with small random weights
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 0.01, (num_classes, num_features))

    # Convert digit labels to one-hot vectors
    targets = np.eye(num_classes)[y_train]

    for epoch in range(epochs):

        # STEP 1: Forward
        scores = forward(x_train, weights)

        # STEP 2: Calculate L2 loss
        loss = calculate_loss(scores, targets)

        # STEP 3: Backward - calculate gradient
        error = scores - targets
        gradient = (2 / len(x_train)) * (error.T @ x_train)

        # STEP 4: Update weights
        weights = weights - learning_rate * gradient

        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss)

    return weights


# Predict a digit using the trained weights
def predict_linear_classifier(test, weights):

    scores = forward(test, weights)

    prediction = np.argmax(scores, axis=1)

    return prediction


# Run algorithm
if __name__ == "__main__":

    x_train, y_train, x_test, y_test = get_data()

    weights = train_linear_classifier(x_train, y_train)

    predictions = predict_linear_classifier(x_test, weights)

    accuracy = np.mean(predictions == y_test) * 100

    print("Accuracy:", accuracy, "%")
    print("First prediction:", predictions[0])
    print("Actual label:", y_test[0])
