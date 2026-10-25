# MNIST Classification Algorithms

In this project, I implemented five machine learning algorithms to recognize handwritten digits (0–9) from the MNIST dataset. I used **NumPy** for the classical algorithms and **PyTorch** for the MLP and CNN. I also made a **p5.js visualization** to inspect predictions and see how the algorithms work.

**Repository:** https://github.com/MatinEsmaeili00/AI_Class_Algorithm

## Algorithms

| Algorithm | Main idea | File |
|---|---|---|
| K-Nearest Neighbors (KNN) | Finds the closest training images and votes | `KNN_Simple.py` |
| Naive Bayes | Predicts using pixel probabilities | `naive_bayes.py` |
| Linear Classifier | Learns weights with gradient descent | `linear_classifier.py` |
| Multilayer Perceptron (MLP) | Learns nonlinear patterns using hidden layers | `mlp_classifier.py` |
| Convolutional Neural Network (CNN) | Learns image features using convolution filters | `cnn_classifier.py` |

All five use the shared MNIST loading and training/testing split in `data_partition.py`.

## Requirements

- Python 3.10+ (a version compatible with PyTorch)
- Git (for cloning)
- A web browser (for the p5.js visualization)
- Internet access the first time you install dependencies and load the p5.js library

Python dependencies are listed in `requirements.txt`: `numpy`, `Pillow`, and `torch`. Training runs on the CPU; a dedicated GPU is not required.

## 1. Download the project

Open a terminal and run:

```bash
git clone https://github.com/MatinEsmaeili00/AI_Class_Algorithm.git
cd AI_Class_Algorithm
```

If you already cloned it, update your local copy with:

```bash
git pull
```

## 2. Create a virtual environment and install packages

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can use the virtual environment directly without activating it: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and then use that interpreter for the commands below.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Extract the MNIST dataset

The repository includes **`MNIST.zip`**. Extract it in the project root before running any Python algorithm or regenerating results.

### Windows (PowerShell)

```powershell
Expand-Archive -Path MNIST.zip -DestinationPath . -Force
```

### macOS / Linux

```bash
unzip MNIST.zip -d .
```

After extraction, the images must be inside digit folders like this:

```text
AI_Class_Algorithm/
├── MNIST.zip
├── MNIST/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── ...
│   └── 9/
├── data_partition.py
└── ...
```

If extraction creates another nested `MNIST` directory, move the inner digit folders so their paths are `MNIST/0/`, `MNIST/1/`, etc. The extracted `MNIST/` folder is ignored by Git, so only the ZIP is committed.

## 4. View the existing results (no training needed)

The repository already contains JSON result files in `visualization/` for the five algorithms. To view them, start a local web server **from the project root**:

```bash
python -m http.server 8000
```

Then open **http://localhost:8000/visualization/** and choose an algorithm from the dropdown.

The visualizer shows the test digit, predicted and actual labels, and algorithm-specific information such as nearest neighbors, probabilities, learned weights, MLP layers, or CNN feature maps.

> Do not just double-click `index.html`. Running a local server lets the page load its JSON files correctly. Press `Ctrl+C` in the terminal to stop the server.

## 5. Run the algorithms yourself

Run these commands separately from the **project root** after extracting MNIST and installing dependencies:

```bash
python KNN_Simple.py
python naive_bayes.py
python linear_classifier.py
python mlp_classifier.py
python cnn_classifier.py
```

The scripts load the MNIST dataset as needed. `data_partition.py` uses an **80% training / 20% testing** split with a fixed seed of `42` (with all 60,000 images: 48,000 training and 12,000 testing).

- **KNN:** Predicts a test digit using its nearest training images and writes `visualization/result.json`.
- **Naive Bayes:** Predicts a test digit using pixel probabilities and writes `visualization/naive_bayes_result.json`.
- **Linear Classifier:** Trains and saves `models/linear_weights.npy`.
- **MLP:** Trains and saves `models/mlp_model.pth`.
- **CNN:** Trains and saves `models/cnn_model.pth`.

Training the neural networks and loading thousands of images can take some time, depending on your computer.

## 6. Update the p5.js results after training

Once the Linear Classifier, MLP, and CNN model files exist, run:

```bash
python visualization/export_results.py
```

This reads the saved model weights and creates or updates:

```text
visualization/linear_classifier_result.json
visualization/mlp_result.json
visualization/cnn_result.json
```

The exporter **does not retrain** these models. KNN and Naive Bayes currently create their own result JSON when their Python scripts run.

Refresh **http://localhost:8000/visualization/** to see the updated results.

## Project structure

```text
AI_Class_Algorithm/
├── KNN_Simple.py
├── naive_bayes.py
├── linear_classifier.py
├── mlp_classifier.py
├── cnn_classifier.py
├── data_partition.py
├── requirements.txt
├── MNIST.zip
├── MNIST/                        # Extract locally; ignored by Git
├── models/
│   ├── linear_weights.npy
│   ├── mlp_model.pth
│   └── cnn_model.pth
└── visualization/
    ├── index.html
    ├── sketch.js
    ├── export_results.py
    ├── result.json
    ├── naive_bayes_result.json
    ├── linear_classifier_result.json
    ├── mlp_result.json
    └── cnn_result.json
```

## Results from my recorded runs

| Algorithm | Recorded result |
|---|---|
| KNN | Correctly predicted the example digit 8; full test accuracy not measured |
| Naive Bayes | Predicted 3 for an example with actual digit 8; full test accuracy not measured |
| Linear Classifier | 66.71% test accuracy |
| MLP | 97.32% test accuracy |
| CNN | 97.89% test accuracy |

These results are from my runs; retraining may produce different results. KNN and Naive Bayes values above are **single-image examples**, not dataset accuracy percentages.

## Troubleshooting

- **`FileNotFoundError: Missing folder .../MNIST/0`**: Extract `MNIST.zip` and check the `MNIST/0/` through `MNIST/9/` folder layout.
- **`ModuleNotFoundError`**: Activate `.venv` and run `python -m pip install -r requirements.txt`.
- **Visualization shows “Could not load ... .json”**: Run the matching algorithm or the exporter, then refresh the page.
- **Exporter cannot find a model in `models/`**: Train the Linear Classifier, MLP, and/or CNN first.
- **Port 8000 is busy**: Use `python -m http.server 8001` and open `http://localhost:8001/visualization/`.

## What I learned

I started with KNN and worked through probability-based classification, gradient descent, neural networks, and convolution. I used NumPy for the first three algorithms and PyTorch for the MLP and CNN. The p5.js visualization helped me see the predictions and some of the learned features.
