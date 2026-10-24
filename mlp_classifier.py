
import torch
import torch.nn as nn
import json
from pathlib import Path
from data_partition import get_data


# Create the neural network
class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    # Calculate predictions using the network
    def forward(self, x):
        scores = self.network(x)
        return scores


# Train MLP using SGD
def train_mlp(x_train, y_train, epochs=10, learning_rate=0.1, batch_size=128):

    torch.manual_seed(42)

    # Convert NumPy arrays to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # Create the model
    model = MLP()

    # Calculate classification loss
    criterion = nn.CrossEntropyLoss()

    # Update weights using SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        # Shuffle images each epoch
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


# Predict digits using the trained network
def predict_mlp(model, test):

    model.eval()

    with torch.no_grad():
        scores = model(test)
        predictions = torch.argmax(scores, dim=1)

    return predictions


# Run algorithm
if __name__ == "__main__":

    x_train, y_train, x_test, y_test = get_data()

    # Train
    model = train_mlp(x_train, y_train)

    # Convert test data into tensors
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Test all images
    predictions = predict_mlp(model, x_test_tensor)

    accuracy = (predictions == y_test_tensor).float().mean().item() * 100

    print("Accuracy:", accuracy, "%")

    # Choose one image for visualization
    test_index = 0
    test_image = x_test[test_index]

    with torch.no_grad():
        scores = model(x_test_tensor[test_index:test_index + 1])[0]

    prediction = int(torch.argmax(scores).item())

    print("Predicted digit:", prediction)
    print("Actual digit:", y_test[test_index])

    # Save data for p5.js
    result = {
        "algorithm": "mlp",
        "test_image": test_image.tolist(),
        "prediction": prediction,
        "actual": int(y_test[test_index]),
        "scores": scores.tolist(),
        "accuracy": float(accuracy)
    }

    folder = Path(__file__).resolve().parent / "visualization"
    folder.mkdir(exist_ok=True)

    with open(folder / "mlp_result.json", "w") as file:
        json.dump(result, file)

    # Save trained network weights
    torch.save(model.state_dict(), folder / "mlp_model.pth")

    print("MLP visualization data exported!")
