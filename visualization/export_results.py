
import sys
import json
from pathlib import Path

import numpy as np
import torch

# Locate the project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_partition import get_data
from linear_classifier import forward
from mlp_classifier import MLP


MODELS = ROOT / "models"
OUTPUT = ROOT / "visualization"


def save_result(filename, result):

    OUTPUT.mkdir(exist_ok=True)

    with open(OUTPUT / filename, "w") as file:
        json.dump(result, file)

    print("Exported:", filename)


# Export Linear Classifier visualization
def export_linear(x_test, y_test):

    weights = np.load(MODELS / "linear_weights.npy")

    # Predict all test images
    all_scores = forward(x_test, weights)
    predictions = np.argmax(all_scores, axis=1)

    accuracy = np.mean(predictions == y_test) * 100

    test_index = 0
    scores = all_scores[test_index]

    result = {
        "algorithm": "linear_classifier",
        "test_image": x_test[test_index].tolist(),
        "prediction": int(predictions[test_index]),
        "actual": int(y_test[test_index]),
        "scores": scores.tolist(),
        "accuracy": float(accuracy),

        # Learned weights for every output digit
        "weights": weights.tolist()
    }

    save_result("linear_classifier_result.json", result)


# Export MLP visualization
def export_mlp(x_test, y_test):

    model = MLP()

    model.load_state_dict(
        torch.load(
            MODELS / "mlp_model.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    x = torch.tensor(x_test, dtype=torch.float32)

    with torch.no_grad():

        all_scores = model(x)

        predictions = torch.argmax(all_scores, dim=1)

        accuracy = (
            (predictions == torch.tensor(y_test))
            .float()
            .mean()
            .item() * 100
        )

        # Choose test image
        test_index = 0
        test_image = x[test_index:test_index + 1]

        # Extract layer activations
        hidden1 = torch.relu(model.network[0](test_image))

        hidden2 = torch.relu(model.network[2](hidden1))

        scores = model.network[4](hidden2)[0]

    result = {
        "algorithm": "mlp",
        "test_image": x_test[test_index].tolist(),
        "prediction": int(predictions[test_index]),
        "actual": int(y_test[test_index]),
        "scores": scores.tolist(),
        "accuracy": float(accuracy),

        # Hidden neuron activations
        "activations": {
            "hidden1": hidden1[0].tolist(),
            "hidden2": hidden2[0].tolist()
        },

        # Learned weights for each layer
        "weights": {
            "layer1": model.network[0].weight.detach().tolist(),
            "layer2": model.network[2].weight.detach().tolist(),
            "layer3": model.network[4].weight.detach().tolist()
        }
    }

    save_result("mlp_result.json", result)


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = get_data()

    export_linear(x_test, y_test)
    export_mlp(x_test, y_test)
