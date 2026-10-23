import numpy as np
from data_partition import get_data
import json
from pathlib import Path


# Calculate probability using Laplace smoothing
def laplace_smoothing(pixel_count, total_images):
    probability = (pixel_count + 1) / (total_images + 2)
    return probability


# Train Naive Bayes using training images and labels
def train_naive_bayes(x_train, y_train):
    # Convert pixels to binary (0 or 1)
    x_train = (x_train > 0.5).astype(int)

    pixel_probabilities = {}
    class_probabilities = {}

    labels = np.unique(y_train)

    for label in labels:
        # Get all images for this label
        images = x_train[y_train == label]

        # Probability of this label
        class_probabilities[label] = len(images) / len(x_train)

        # Count how many times each pixel is ON
        pixel_count = np.zeros(x_train.shape[1])

        for image in images:
            pixel_count += image

        # Calculate probabilities using our function
        pixel_probabilities[label] = laplace_smoothing(
            pixel_count,
            len(images)
        )

    return pixel_probabilities, class_probabilities


# Predict the label of one image
def predict_naive_bayes(pixel_probabilities, class_probabilities, test):
    # Convert image to binary
    test = (np.asarray(test).reshape(-1) > 0.5).astype(int)

    scores = {}

    for label in class_probabilities:
        probability = np.log(class_probabilities[label])

        # Check all 784 pixels
        for i in range(len(test)):
            pixel_probability = pixel_probabilities[label][i]

            if test[i] == 1:
                probability += np.log(pixel_probability)
            else:
                probability += np.log(1 - pixel_probability)

        scores[label] = probability

    # Find the label with the highest score
    prediction = max(scores, key=scores.get)
    return prediction, scores


# Convert log scores to percentages for the p5.js chart
def get_probabilities(scores):
    # Subtract maximum first so exponentials do not underflow
    highest_score = max(scores.values())
    probabilities = {}
    total = 0

    for label in scores:
        probabilities[label] = np.exp(scores[label] - highest_score)
        total += probabilities[label]

    for label in probabilities:
        probabilities[label] = float(probabilities[label] / total * 100)

    return probabilities


# Run training and save results for the p5.js visualizer
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_data()

    pixel_probabilities, class_probabilities = train_naive_bayes(
        x_train, y_train
    )

    # Change this number to try a different test image
    test_index = 0
    test_image = x_test[test_index]

    prediction, scores = predict_naive_bayes(
        pixel_probabilities,
        class_probabilities,
        test_image
    )

    probabilities = get_probabilities(scores)

    print("Actual digit:", y_test[test_index])
    print("Predicted digit:", prediction)
    print("Probabilities:", probabilities)

    result = {
        "algorithm": "naive_bayes",
        "test_image": test_image.tolist(),
        "prediction": int(prediction),
        "actual": int(y_test[test_index]),
        "probabilities": {str(label): value for label, value in probabilities.items()}
    }

    folder = Path(__file__).resolve().parent / "visualization"
    folder.mkdir(exist_ok=True)

    with open(folder / "naive_bayes_result.json", "w") as file:
        json.dump(result, file)

    print("Naive Bayes visualization data exported!")
