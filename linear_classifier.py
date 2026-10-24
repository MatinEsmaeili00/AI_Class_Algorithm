
import numpy as np
import json
from pathlib import Path
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
def train_linear_classifier(x_train, y_train, epochs=100, learning_rate=0.001):

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

    # Train
    weights = train_linear_classifier(x_train, y_train)

    # Test all images
    predictions = predict_linear_classifier(x_test, weights)

    accuracy = np.mean(predictions == y_test) * 100

    print("Accuracy:", accuracy, "%")

    # Choose one image for visualization
    test_index = 0
    test_image = x_test[test_index]

    # Calculate class scores for this image
    scores = forward(test_image.reshape(1, -1), weights)[0]

    prediction = int(np.argmax(scores))

    print("Predicted digit:", prediction)
    print("Actual digit:", y_test[test_index])

    # Save data for p5.js
    result = {
        "algorithm": "linear_classifier",
        "test_image": test_image.tolist(),
        "prediction": prediction,
        "actual": int(y_test[test_index]),
        "scores": scores.tolist(),
        "accuracy": float(accuracy)
    }

    folder = Path(__file__).resolve().parent / "visualization"
    folder.mkdir(exist_ok=True)

    with open(folder / "linear_classifier_result.json", "w") as file:
        json.dump(result, file)

    print("Linear Classifier visualization data exported!")
