
# X = Input data (features), Y = Correct labels (answers)
#
# x_train: Training images used by the algorithm to learn or find patterns.
# y_train: Correct class labels corresponding to each image in x_train.
#
# x_test:  Testing images that the algorithm has not seen during training.
# y_test:  Correct class labels for x_test, used to check prediction accuracy.
#
# For KNN:
# - x_train is used to calculate distances to the test image.
# - y_train is used to identify the classes of the nearest neighbors.
# - x_test contains the images we want to classify.
# - y_test is used to compare the predictions with the actual classes.
#
# Example with MNIST:
# x_train.shape = (48000, 784)  # 48000 images, each with 784 pixels
# y_train.shape = (48000,)      # 48000 corresponding digit labels
# x_test.shape  = (12000, 784)  # 12000 testing images
# y_test.shape  = (12000,)      # 12000 correct testing labels
#
# The example shapes assume 60000 images split 80% training / 20% testing.
#
# IMPORTANT: Always keep each image and its label at the same index.
# Both KNN and Naive Bayes will use these same data partitions.


import numpy as np
from pathlib import Path
from PIL import Image


# -------------- Start ------------- Creating Data Partitions ----

def create_partitions(images, labels, train_ratio=0.8, seed=42):

    # Make sure each image has a corresponding label
    if len(images) != len(labels):
        raise ValueError("Images and labels must have the same length")

    if not 0 < train_ratio < 1:
        raise ValueError("Training ratio must be between 0 and 1")

    # Create an array of indices
    indices = np.arange(len(images))

    # Shuffle the indices randomly
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    # Calculate how many images go into training
    train_size = int(len(images) * train_ratio)

    if train_size == 0 or train_size == len(images):
        raise ValueError("Both partitions must contain at least one image")

    # Separate training and testing indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Use the same indices for images and labels
    x_train = images[train_indices]
    y_train = labels[train_indices]

    x_test = images[test_indices]
    y_test = labels[test_indices]

    return x_train, y_train, x_test, y_test


# -------------- End ------------- Creating Data Partitions ----

# no loading and percentage!!
# # -------------- Start ------------- Loading MNIST Images ----

# def load_mnist():

#     images = []
#     labels = []

#     # Find MNIST folder in the same directory as data_partition.py
#     mnist_folder = Path(__file__).resolve().parent / "MNIST"

#     # Read folders from 0 to 9
#     for digit in range(10):

#         # Convert digit to string to match folder name
#         class_folder = mnist_folder / str(digit)

#         if not class_folder.is_dir():
#             raise FileNotFoundError(f"Missing folder: {class_folder}")

#         # Read each image inside this digit folder
#         for image_path in sorted(class_folder.iterdir()):

#             # Only accept image files
#             if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
#                 continue

#             # Open image using Pillow
#             with Image.open(image_path) as image:

#                 # Convert RGB or other formats to grayscale
#                 image = image.convert("L")

#                 # Verify the image dimensions
#                 if image.size != (28, 28):
#                     raise ValueError(f"Expected 28x28: {image_path}")

#                 # Convert image pixels into NumPy array
#                 image_array = np.array(image, dtype=np.float32)

#             # Normalize pixel values from 0-255 into 0-1
#             image_array = image_array / 255.0

#             # Flatten 28x28 pixels into 784 features
#             image_array = image_array.flatten()

#             # Save image and corresponding digit label
#             images.append(image_array)
#             labels.append(digit)

#     if len(images) == 0:
#         raise ValueError("No MNIST images were found")

#     # Convert lists to NumPy arrays
#     images = np.array(images, dtype=np.float32)
#     labels = np.array(labels, dtype=np.int64)

#     return images, labels

# # -------------- End ------------- Loading MNIST Images ----




# -------------- Start ------------- Loading MNIST Images ----

def load_mnist(load_percentage=100):

    images = []
    labels = []
    image_files = []

    # Find MNIST folder
    mnist_folder = Path(__file__).resolve().parent / "MNIST"

    # First, collect all image paths and their labels
    for digit in range(10):

        class_folder = mnist_folder / str(digit)

        if not class_folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {class_folder}")

        for image_path in sorted(class_folder.iterdir()):

            if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                continue

            # Save the image path and corresponding label
            image_files.append((image_path, digit))

    # Total number of images
    total_images = len(image_files)

    if total_images == 0:
        raise ValueError("No MNIST images were found")

    print("Total images to load:", total_images)

    # Now actually open and process the images
    for i in range(total_images):

        image_path, digit = image_files[i]

        with Image.open(image_path) as image:

            image = image.convert("L")

            if image.size != (28, 28):
                raise ValueError(f"Expected 28x28: {image_path}")

            image_array = np.array(image, dtype=np.float32)

        # Normalize and flatten
        image_array = image_array / 255.0
        image_array = image_array.flatten()

        # Save the image and its label
        images.append(image_array)
        labels.append(digit)

        # -------- CALL LOADING PROGRESS HERE --------

        loaded_images = i + 1

        if loaded_images % 500 == 0 or loaded_images == total_images:
            loading_progress(loaded_images, total_images)

    print()  # Move to the next line after loading finishes

    # Convert lists to NumPy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    return images, labels

# -------------- End ------------- Loading MNIST Images ----


# testing with numbers !!
# def get_data():

#     images = np.array([
#         [1, 1],
#         [2, 2],
#         [3, 3],
#         [4, 4],
#         [5, 5],
#         [6, 6],
#         [7, 7],
#         [8, 8],
#         [9, 9],
#         [10, 10]
#     ], dtype=np.float32)

#     labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

#     x_train, y_train, x_test, y_test = create_partitions(
#         images, labels, train_ratio=0.8, seed=42
#     )

#     return x_train, y_train, x_test, y_test


# -------------- Start ------------- Getting Dataset ----

def get_data():

    # Load real MNIST images and labels
    images, labels = load_mnist()

    print("Total images:", len(images))
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    # Create random training and testing partitions
    x_train, y_train, x_test, y_test = create_partitions(
        images, labels, train_ratio=0.8, seed=42
    )

    print("Training images:", len(x_train))
    print("Testing images:", len(x_test))

    return x_train, y_train, x_test, y_test

# -------------- End ------------- Getting Dataset ----

# -------------- Start ------------- Loading Progress ----

def loading_progress(current, total):

    # Calculate percentage
    percentage = current / total * 100

    # Create progress bar
    bar_length = 30

    filled_length = int(bar_length * current / total)

    bar = "#" * filled_length
    empty = "-" * (bar_length - filled_length)

    # Display progress
    print(
        f"\rLoading MNIST: [{bar}{empty}] "
        f"{percentage:.1f}% ({current}/{total})",
        end="",
        flush=True
    )

# -------------- End ------------- Loading Progress ----

