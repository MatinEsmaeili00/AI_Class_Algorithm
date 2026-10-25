
import torch
import torch.nn as nn
from pathlib import Path
from data_partition import get_data


# Create the Convolutional Neural Network
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Activation function
        self.relu = nn.ReLU()

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer
        self.fc = nn.Linear(64 * 7 * 7, 10)


    # Calculate predictions using the network
    def forward(self, x):

        # STEP 1: First convolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # STEP 2: Second convolution
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # STEP 3: Flatten the image features
        x = torch.flatten(x, start_dim=1)

        # STEP 4: Fully connected layer
        scores = self.fc(x)

        return scores


# Train CNN using SGD
def train_cnn(x_train, y_train, epochs=5, learning_rate=0.05, batch_size=128):

    torch.manual_seed(42)

    # Convert NumPy arrays to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # Reshape 784 pixels into 1 x 28 x 28 images
    x_train = x_train.reshape(-1, 1, 28, 28)

    # Create CNN model
    model = CNN()

    # Calculate classification loss
    criterion = nn.CrossEntropyLoss()

    # Update weights using SGD
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        # Shuffle images for each epoch
        indices = torch.randperm(len(x_train))

        for start in range(0, len(x_train), batch_size):

            batch_indices = indices[start:start + batch_size]

            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            # STEP 1: Forward
            scores = model(x_batch)

            # STEP 2: Calculate loss
            loss = criterion(scores, y_batch)

            # STEP 3: Reset old gradients
            optimizer.zero_grad()

            # STEP 4: Backward
            loss.backward()

            # STEP 5: Update weights
            optimizer.step()

            total_loss += loss.item() * len(x_batch)

        average_loss = total_loss / len(x_train)

        print("Epoch:", epoch + 1, "Loss:", average_loss)

    return model


# Predict digits using the trained CNN
def predict_cnn(model, test, batch_size=128):

    model.eval()

    # Convert input into PyTorch tensor
    if not isinstance(test, torch.Tensor):
        test = torch.tensor(test, dtype=torch.float32)

    # Reshape into images
    test = test.reshape(-1, 1, 28, 28)

    predictions = []

    with torch.no_grad():

        for start in range(0, len(test), batch_size):

            x_batch = test[start:start + batch_size]

            # Forward
            scores = model(x_batch)

            # Find class with highest score
            prediction = torch.argmax(scores, dim=1)

            predictions.append(prediction)

    return torch.cat(predictions)


# Run algorithm
if __name__ == "__main__":

    # Load MNIST dataset
    x_train, y_train, x_test, y_test = get_data()

    # Train CNN
    model = train_cnn(x_train, y_train)

    # Test all images
    predictions = predict_cnn(model, x_test)

    # Convert correct labels into tensor
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Calculate accuracy
    accuracy = (
        (predictions == y_test_tensor)
        .float()
        .mean()
        .item() * 100
    )

    print("Accuracy:", accuracy, "%")

    # Save trained CNN weights
    folder = Path(__file__).resolve().parent / "models"
    folder.mkdir(exist_ok=True)

    torch.save(model.state_dict(), folder / "cnn_model.pth")

    print("CNN trained weights saved!")
