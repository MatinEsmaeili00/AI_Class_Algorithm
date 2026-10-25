
import sys
import json
from pathlib import Path

import numpy as np
import torch

# Locate the project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import existing algorithms
from data_partition import get_data
from linear_classifier import forward
from mlp_classifier import MLP
from cnn_classifier import CNN


# Folder locations
MODELS = ROOT / "models"
OUTPUT = ROOT / "visualization"


# Save results into JSON file
def save_result(filename, result):

    OUTPUT.mkdir(exist_ok=True)

    with open(OUTPUT / filename, "w") as file:
        json.dump(result, file)

    print("Exported:", filename)


# -------------- Start ------------- Linear Classifier ----

def export_linear(x_test, y_test):

    # Load trained weights
    weights = np.load(MODELS / "linear_weights.npy")

    # Calculate predictions for all test images
    all_scores = forward(x_test, weights)

    predictions = np.argmax(all_scores, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test) * 100

    # Choose one image
    test_index = 0
    test_image = x_test[test_index]

    scores = all_scores[test_index]

    prediction = int(predictions[test_index])

    result = {
        "algorithm": "linear_classifier",

        "test_image": test_image.tolist(),
        "prediction": prediction,
        "actual": int(y_test[test_index]),

        "scores": scores.tolist(),
        "accuracy": float(accuracy),

        # Learned weights for all 10 digits
        "weights": weights.tolist()
    }

    save_result("linear_classifier_result.json", result)


# -------------- End ------------- Linear Classifier ----


# -------------- Start ------------- MLP ----

def export_mlp(x_test, y_test):

    # Create MLP model
    model = MLP()

    # Load trained network weights
    model.load_state_dict(
        torch.load(
            MODELS / "mlp_model.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    # Convert test images into PyTorch tensors
    x = torch.tensor(x_test, dtype=torch.float32)

    # Calculate predictions and accuracy
    predictions = []

    with torch.no_grad():

        # Test images in batches
        for start in range(0, len(x), 128):

            batch = x[start:start + 128]

            scores = model(batch)

            prediction = torch.argmax(scores, dim=1)

            predictions.append(prediction)

        predictions = torch.cat(predictions)

        correct = torch.tensor(y_test, dtype=torch.long)

        accuracy = (
            (predictions == correct)
            .float()
            .mean()
            .item() * 100
        )

        # Choose one test image
        test_index = 0
        test_image = x[test_index:test_index + 1]

        # STEP 1: First hidden layer
        hidden1 = torch.relu(
            model.network[0](test_image)
        )

        # STEP 2: Second hidden layer
        hidden2 = torch.relu(
            model.network[2](hidden1)
        )

        # STEP 3: Output layer
        scores = model.network[4](hidden2)[0]

        prediction = int(
            torch.argmax(scores).item()
        )

    result = {
        "algorithm": "mlp",

        "test_image": x_test[test_index].tolist(),
        "prediction": prediction,
        "actual": int(y_test[test_index]),

        "scores": scores.tolist(),
        "accuracy": float(accuracy),

        # Neuron activations
        "activations": {
            "hidden1": hidden1[0].tolist(),
            "hidden2": hidden2[0].tolist()
        },

        # Learned weights for all layers
        "weights": {
            "layer1": model.network[0].weight.detach().tolist(),
            "layer2": model.network[2].weight.detach().tolist(),
            "layer3": model.network[4].weight.detach().tolist()
        }
    }

    save_result("mlp_result.json", result)


# -------------- End ------------- MLP ----


# -------------- Start ------------- CNN ----

def export_cnn(x_test, y_test):

    # Create CNN model
    model = CNN()

    # Load trained weights
    model.load_state_dict(
        torch.load(
            MODELS / "cnn_model.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    # Convert test images into PyTorch tensors
    x = torch.tensor(x_test, dtype=torch.float32)

    # Reshape flattened images into 1 x 28 x 28
    x = x.reshape(-1, 1, 28, 28)

    # Calculate predictions
    predictions = []

    with torch.no_grad():

        # Test images in batches
        for start in range(0, len(x), 128):

            batch = x[start:start + 128]

            scores = model(batch)

            prediction = torch.argmax(scores, dim=1)

            predictions.append(prediction)

        predictions = torch.cat(predictions)

        # Calculate accuracy
        correct = torch.tensor(y_test, dtype=torch.long)

        accuracy = (
            (predictions == correct)
            .float()
            .mean()
            .item() * 100
        )

        # Choose one test image
        test_index = 0
        test_image = x[test_index:test_index + 1]

        # STEP 1: First convolution + ReLU
        conv1 = model.relu(
            model.conv1(test_image)
        )

        # STEP 2: First pooling
        pool1 = model.pool(conv1)

        # STEP 3: Second convolution + ReLU
        conv2 = model.relu(
            model.conv2(pool1)
        )

        # STEP 4: Second pooling
        pool2 = model.pool(conv2)

        # STEP 5: Flatten
        flattened = torch.flatten(
            pool2,
            start_dim=1
        )

        # STEP 6: Fully connected output
        scores = model.fc(flattened)[0]

        prediction = int(
            torch.argmax(scores).item()
        )

    result = {
        "algorithm": "cnn",

        "test_image": x_test[test_index].tolist(),
        "prediction": prediction,
        "actual": int(y_test[test_index]),

        "scores": scores.tolist(),
        "accuracy": float(accuracy),

        # Feature maps after convolution and pooling
        "feature_maps": {
            "conv1": conv1[0].tolist(),
            "pool1": pool1[0].tolist(),
            "conv2": conv2[0].tolist(),
            "pool2": pool2[0].tolist()
        },

        # Learned convolution filters
        "filters": {

            # 32 filters, each 3 x 3
            "conv1": model.conv1.weight[:, 0]
                .detach().tolist(),

            # 64 filters, first input channel only
            "conv2": model.conv2.weight[:, 0]
                .detach().tolist()
        }
    }

    save_result("cnn_result.json", result)


# -------------- End ------------- CNN ----


# -------------- Start ------------- Run Exporter ----

if __name__ == "__main__":

    # Load the same MNIST dataset and test partition
    x_train, y_train, x_test, y_test = get_data()

    # Export results for each algorithm
    export_linear(x_test, y_test)

    export_mlp(x_test, y_test)

    export_cnn(x_test, y_test)

    print("All visualization results exported!")


# -------------- End ------------- Run Exporter ----
