
import numpy as np


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
