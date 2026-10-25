
 // MNIST Digit Recognition Visualizer
// This code only reads JSON files exported by Python.
// It does not train or modify any machine learning algorithm.

let result = null;

let algorithmSelect;
let digitSelect;
let neuronSelect;

let cnnLayerSelect;
let cnnFilterSelect;

let message = "Loading results...";
let currentRequest = 0;


// JSON files for each algorithm
const FILES = {
    knn: "result.json",
    naive_bayes: "naive_bayes_result.json",
    linear_classifier: "linear_classifier_result.json",
    mlp: "mlp_result.json",
    cnn: "cnn_result.json"
};


// ---------------- START SETUP ----------------

function setup() {

    createCanvas(1280, 850);
    pixelDensity(1);
    noLoop();
    textFont("Arial");

    // Select algorithm
    algorithmSelect = createSelect();
    algorithmSelect.position(30, 66);

    algorithmSelect.option("KNN", "knn");
    algorithmSelect.option("Naive Bayes", "naive_bayes");
    algorithmSelect.option("Linear Classifier", "linear_classifier");
    algorithmSelect.option("MLP (PyTorch)", "mlp");
    algorithmSelect.option("CNN (PyTorch)", "cnn");

    algorithmSelect.changed(loadAlgorithm);


    // Linear Classifier digit selection
    digitSelect = createSelect();
    digitSelect.position(920, 424);

    for (let i = 0; i < 10; i++) {
        digitSelect.option("Digit " + i, String(i));
    }

    digitSelect.changed(() => redraw());
    digitSelect.hide();


    // MLP neuron selection
    neuronSelect = createSelect();
    neuronSelect.position(920, 424);

    neuronSelect.changed(() => redraw());
    neuronSelect.hide();


    // CNN convolutional layer selection
    cnnLayerSelect = createSelect();
    cnnLayerSelect.position(700, 525);

    cnnLayerSelect.option("Convolution 1", "conv1");
    cnnLayerSelect.option("Convolution 2", "conv2");

    cnnLayerSelect.changed(updateCNNFilters);
    cnnLayerSelect.hide();


    // CNN filter selection
    cnnFilterSelect = createSelect();
    cnnFilterSelect.position(910, 525);

    cnnFilterSelect.changed(() => redraw());
    cnnFilterSelect.hide();


    loadAlgorithm();
}


// ---------------- END SETUP ----------------


// ---------------- START LOAD ALGORITHM ----------------

function loadAlgorithm() {

    const algorithm = algorithmSelect.value();
    const requestId = ++currentRequest;

    message = "Loading " + algorithm + "...";
    result = null;

    // Hide all algorithm-specific controls
    digitSelect.hide();
    neuronSelect.hide();
    cnnLayerSelect.hide();
    cnnFilterSelect.hide();

    redraw();

    // Load the exported JSON file
    loadJSON(
        FILES[algorithm] + "?r=" + Date.now(),

        function(data) {

            // Ignore previous requests
            if (requestId !== currentRequest) {
                return;
            }

            result = data;
            message = "";

            // Linear Classifier controls
            if (
                algorithm === "linear_classifier" &&
                Array.isArray(data.weights)
            ) {

                digitSelect.value(String(data.prediction));
                digitSelect.show();
            }


            // MLP controls
            if (
                algorithm === "mlp" &&
                data.weights &&
                data.activations
            ) {

                neuronSelect.elt.innerHTML = "";

                let totalNeurons = data.activations.hidden1.length;

                for (let i = 0; i < totalNeurons; i++) {

                    neuronSelect.option(
                        "Hidden 1 neuron " + i,
                        String(i)
                    );
                }

                neuronSelect.value("0");
                neuronSelect.show();
            }


            // CNN controls
            if (
                algorithm === "cnn" &&
                data.feature_maps &&
                data.filters
            ) {

                cnnLayerSelect.value("conv1");

                cnnLayerSelect.show();
                cnnFilterSelect.show();

                updateCNNFilters();
            }

            redraw();

        },

        function(error) {

            if (requestId !== currentRequest) {
                return;
            }

            message =
                "Could not load " + FILES[algorithm] +
                ". Run visualization/export_results.py first.";

            redraw();
        }
    );
}


// ---------------- END LOAD ALGORITHM ----------------


// ---------------- START MAIN DRAW ----------------

function draw() {

    background(19, 23, 31);

    noStroke();

    fill(245);
    textSize(27);
    text("MNIST Digit Recognition", 30, 40);

    fill(163, 178, 196);
    textSize(13);
    text("Choose algorithm", 30, 62);


    if (!result) {

        fill(240);
        textSize(18);
        text(message, 30, 160);

        return;
    }


    // Main background panels
    drawPanel(25, 115, 410, 330);
    drawPanel(455, 115, 800, 330);
    drawPanel(25, 460, 1230, 360);


    // Draw test image
    fill(238);
    textSize(19);
    text("Test image", 45, 150);

    if (
        result.test_image &&
        result.test_image.length >= 784
    ) {

        drawDigit(result.test_image, 50, 170, 6.8);
    }


    // Display prediction
    fill(166);
    textSize(16);
    text("Prediction", 274, 205);

    fill(64, 217, 149);
    textSize(60);
    text(result.prediction, 272, 276);


    // Display actual label
    fill(230);
    textSize(15);

    text("Actual: " + result.actual, 274, 310);


    // Check if prediction is correct
    if (result.prediction === result.actual) {

        fill(64, 217, 149);
        text("Correct prediction", 274, 339);

    } else {

        fill(249, 111, 111);
        text("Incorrect prediction", 274, 339);
    }


    // Show accuracy
    if (Number.isFinite(result.accuracy)) {

        fill(170);
        textSize(13);

        text(
            "Test accuracy: " +
            result.accuracy.toFixed(2) + "%",
            48,
            425
        );
    }


    // Display selected algorithm
    const chosen = algorithmSelect.value();

    if (chosen === "knn") {

        drawKNN();

    }
    else if (chosen === "naive_bayes") {

        drawProbabilityScores();

    }
    else if (chosen === "linear_classifier") {

        drawRawScores("Linear classifier output scores");
        drawLinearWeights();

    }
    else if (chosen === "mlp") {

        drawRawScores("MLP output scores (logits)");
        drawMLPNetwork();

    }
    else if (chosen === "cnn") {

        drawRawScores("CNN output scores (logits)");
        drawCNN();
    }
}


// ---------------- END MAIN DRAW ----------------


// ---------------- DRAW PANEL ----------------

function drawPanel(x, y, w, h) {

    noStroke();
    fill(29, 35, 46);

    rect(x, y, w, h, 13);
}


// ---------------- DRAW MNIST IMAGE ----------------

function drawDigit(pixels, x, y, scale) {

    noStroke();

    for (let row = 0; row < 28; row++) {

        for (let col = 0; col < 28; col++) {

            let index = row * 28 + col;

            let pixel = constrain(
                Number(pixels[index]) || 0,
                0,
                1
            );

            fill(pixel * 255);

            rect(
                x + col * scale,
                y + row * scale,
                scale + 0.1,
                scale + 0.1
            );
        }
    }
}


// ---------------- CLASS SCORES ----------------

function drawRawScores(title) {

    fill(240);
    textSize(19);
    text(title, 478, 150);

    let scores = result.scores;

    if (!Array.isArray(scores) || scores.length !== 10) {
        return;
    }

    let minValue = Math.min(...scores);
    let maxValue = Math.max(...scores);
    let range = maxValue - minValue || 1;

    for (let digit = 0; digit < 10; digit++) {

        let y = 170 + digit * 24;

        fill(200);
        textSize(13);
        text(String(digit), 480, y + 12);

        // Background bar
        fill(54, 62, 77);
        rect(510, y, 600, 17, 4);

        // Normalize score for visualization
        let barWidth = Math.max(
            2,
            (scores[digit] - minValue) / range * 600
        );

        // Highlight predicted digit
        if (digit === result.prediction) {
            fill(67, 216, 151);
        } else {
            fill(110, 160, 245);
        }

        rect(510, y, barWidth, 17, 4);

        // Display score
        fill(220);
        text(scores[digit].toFixed(3), 1122, y + 13);
    }

    fill(152);
    textSize(12);

    text(
        "Bar lengths are relative scores, not probabilities.",
        479,
        428
    );
}


// ---------------- NAIVE BAYES ----------------

function drawProbabilityScores() {

    fill(240);
    textSize(19);

    text("Naive Bayes class probabilities", 478, 150);

    for (let digit = 0; digit < 10; digit++) {

        let raw = result.probabilities
            ? Number(result.probabilities[String(digit)])
            : 0;

        let value = Number.isFinite(raw) ? raw : 0;

        let y = 170 + digit * 24;

        fill(200);
        textSize(13);
        text(String(digit), 480, y + 12);

        fill(54, 62, 77);
        rect(510, y, 600, 17, 4);

        if (digit === result.prediction) {
            fill(67, 216, 151);
        } else {
            fill(110, 160, 245);
        }

        rect(
            510,
            y,
            constrain(value / 100, 0, 1) * 600,
            17,
            4
        );

        fill(220);
        text(value.toFixed(2) + "%", 1122, y + 13);
    }

    fill(194);
    textSize(17);

    text(
        "Predicted class: " + result.prediction,
        48,
        500
    );
}


// ---------------- KNN ----------------

function drawKNN() {

    fill(240);
    textSize(19);
    text("Nearest neighbors", 478, 150);

    let neighbors = Array.isArray(result.neighbors)
        ? result.neighbors
        : [];

    fill(180);
    textSize(14);

    text(
        "Returned neighbors: " + neighbors.length,
        478,
        182
    );

    for (
        let i = 0;
        i < Math.min(neighbors.length, 7);
        i++
    ) {

        let neighbor = neighbors[i];
        let x = 48 + i * 170;

        if (
            neighbor.pixels &&
            neighbor.pixels.length >= 784
        ) {

            drawDigit(neighbor.pixels, x, 520, 4.1);
        }

        fill(224);
        textSize(12);

        text("Label: " + neighbor.label, x, 656);

        text(
            "Dist: " + Number(neighbor.distance).toFixed(3),
            x,
            674
        );
    }

    fill(164);
    textSize(12);

    text(
        "Displaying up to 7 neighbors from exported JSON.",
        48,
        787
    );
}


// ---------------- WEIGHT HEATMAP ----------------

// Green = positive weights
// Red = negative weights

function drawWeightHeatmap(weights, x, y, scale) {

    if (
        !Array.isArray(weights) ||
        weights.length !== 784
    ) {
        return;
    }

    let limit = Math.max(
        0.000001,
        ...weights.map(v => Math.abs(v))
    );

    noStroke();

    for (let i = 0; i < 784; i++) {

        let w = weights[i];

        let strength = Math.sqrt(
            Math.abs(w) / limit
        );

        if (w >= 0) {

            fill(
                24 + 35 * strength,
                51 + 168 * strength,
                57 + 70 * strength
            );

        } else {

            fill(
                55 + 193 * strength,
                35 + 36 * strength,
                50 + 35 * strength
            );
        }

        let col = i % 28;
        let row = Math.floor(i / 28);

        rect(
            x + col * scale,
            y + row * scale,
            scale + 0.15,
            scale + 0.15
        );
    }
}


// ---------------- LINEAR CLASSIFIER ----------------

function drawLinearWeights() {

    fill(239);
    textSize(20);

    text(
        "Learned weights: 784 inputs -> 10 outputs",
        48,
        495
    );

    if (
        !Array.isArray(result.weights) ||
        result.weights.length !== 10
    ) {

        fill(245, 189, 98);
        textSize(15);

        text(
            "No weights found. Run export_results.py",
            48,
            555
        );

        return;
    }

    fill(165);
    textSize(14);

    text("Select output class:", 768, 438);

    let digit = constrain(
        parseInt(digitSelect.value() || "0"),
        0,
        9
    );

    let weights = result.weights[digit];

    // Display learned weights
    drawWeightHeatmap(weights, 65, 525, 8.5);

    fill(229);
    textSize(17);

    text("Output digit " + digit, 365, 565);

    fill(172);
    textSize(14);

    text("Each square = one pixel weight", 365, 596);

    // Positive weight legend
    fill(67, 216, 151);
    rect(365, 627, 14, 14);

    fill(203);
    text("Positive weight", 388, 639);

    // Negative weight legend
    fill(244, 88, 92);
    rect(365, 654, 14, 14);

    fill(203);
    text("Negative weight", 388, 666);

    fill(168);
    textSize(13);

    text(
        "A linear classifier has no hidden layers.",
        740,
        562
    );

    text(
        "Each output connects to all 784 input pixels.",
        740,
        587
    );
}


// ---------------- MLP NETWORK ----------------

function drawMLPNetwork() {

    let A = result.activations;
    let W = result.weights;

    if (
        !A ||
        !W ||
        !W.layer1 ||
        !W.layer2 ||
        !W.layer3
    ) {

        fill(245, 189, 98);
        textSize(15);

        text(
            "Missing MLP weights. Run export_results.py",
            48,
            520
        );

        return;
    }

    fill(239);
    textSize(20);

    text(
        "MLP architecture and neuron weights",
        48,
        493
    );

    fill(171);
    textSize(13);

    text(
        "Sampled neurons with real learned connection weights.",
        48,
        516
    );


    // Choose a smaller number of visible neurons
    let inputIndices = sampleIndices(784, 11);
    let h1Indices = sampleIndices(A.hidden1.length, 12);
    let h2Indices = sampleIndices(A.hidden2.length, 10);

    let outIndices = Array.from(
        { length: 10 },
        (_, i) => i
    );

    let layers = [
        inputIndices,
        h1Indices,
        h2Indices,
        outIndices
    ];

    let xs = [105, 355, 605, 845];


    // Calculate neuron positions
    let yPositions = layers.map(layer =>
        layer.map(
            (_, i) =>
                568 +
                i * (185 / Math.max(1, layer.length - 1))
        )
    );

    let weights = [
        W.layer1,
        W.layer2,
        W.layer3
    ];


    // Draw connections
    for (let layer = 0; layer < 3; layer++) {

        for (let j = 0; j < layers[layer + 1].length; j++) {

            for (let i = 0; i < layers[layer].length; i++) {

                let value =
                    weights[layer]
                    [layers[layer + 1][j]]
                    [layers[layer][i]];

                let alpha = constrain(
                    Math.abs(value) * 260,
                    10,
                    115
                );

                if (value >= 0) {
                    stroke(74, 219, 153, alpha);
                } else {
                    stroke(255, 109, 109, alpha);
                }

                strokeWeight(0.85);

                line(
                    xs[layer],
                    yPositions[layer][i],
                    xs[layer + 1],
                    yPositions[layer + 1][j]
                );
            }
        }
    }


    // Draw neurons
    noStroke();

    let values = [
        result.test_image,
        A.hidden1,
        A.hidden2,
        result.scores
    ];

    for (let layer = 0; layer < 4; layer++) {

        let layerValues = values[layer];

        let maxAbs = Math.max(
            0.001,
            ...layerValues.map(v => Math.abs(v))
        );

        for (let j = 0; j < layers[layer].length; j++) {

            let v = layerValues[layers[layer][j]];

            let normalized = constrain(
                Math.abs(v) / maxAbs,
                0,
                1
            );

            if (v >= 0) {

                fill(
                    59,
                    95 + 140 * normalized,
                    131
                );

            } else {

                fill(226, 76, 98);
            }

            circle(
                xs[layer],
                yPositions[layer][j],
                10 + 8 * normalized
            );
        }
    }


    // Layer labels
    fill(219);
    textSize(13);

    let labels = [
        "Input 784",
        "Hidden 256 + ReLU",
        "Hidden 128 + ReLU",
        "Output 10"
    ];

    for (let i = 0; i < labels.length; i++) {

        text(
            labels[i],
            xs[i] - 45,
            551
        );
    }

    fill(171);
    textSize(12);
    text("Sampled connections", 49, 785);


    // Selected first hidden-layer neuron
    let selected = constrain(
        parseInt(neuronSelect.value() || "0"),
        0,
        W.layer1.length - 1
    );

    fill(230);
    textSize(15);

    text(
        "Hidden 1 neuron " + selected,
        946,
        555
    );

    fill(166);
    textSize(12);

    text("28x28 incoming weights", 946, 576);


    // Draw selected neuron's weights
    drawWeightHeatmap(
        W.layer1[selected],
        952,
        587,
        6.8
    );

    fill(198);
    textSize(12);

    text(
        "Activation: " +
        A.hidden1[selected].toFixed(5),
        951,
        797
    );
}


// ---------------- SAMPLE NEURON INDICES ----------------

function sampleIndices(total, count) {

    return Array.from(
        { length: count },
        (_, i) =>
            Math.round(
                i * (total - 1) /
                Math.max(1, count - 1)
            )
    );
}


// ---------------- CNN FILTER SELECTION ----------------

function updateCNNFilters() {

    if (!result || !result.feature_maps) {
        return;
    }

    let layer = cnnLayerSelect.value();

    let totalFilters = result.feature_maps[layer].length;

    // Clear previous filter options
    cnnFilterSelect.elt.innerHTML = "";

    for (let i = 0; i < totalFilters; i++) {

        cnnFilterSelect.option(
            "Filter " + i,
            String(i)
        );
    }

    cnnFilterSelect.value("0");

    redraw();
}


// ---------------- CNN FEATURE MAP ----------------

// Draw a 2D feature map in grayscale
function drawFeatureMap(map, x, y, size) {

    if (!Array.isArray(map) || map.length === 0) {
        return;
    }

    let height = map.length;
    let width = map[0].length;

    let values = map.flat();

    let minValue = Math.min(...values);
    let maxValue = Math.max(...values);

    let range = maxValue - minValue || 1;

    let pixelSize = size / width;

    noStroke();

    for (let row = 0; row < height; row++) {

        for (let col = 0; col < width; col++) {

            let value = map[row][col];

            let brightness =
                (value - minValue) / range * 255;

            fill(brightness);

            rect(
                x + col * pixelSize,
                y + row * pixelSize,
                pixelSize + 0.1,
                pixelSize + 0.1
            );
        }
    }
}


// ---------------- CNN KERNEL ----------------

// Visualize the learned 3x3 convolution weights
function drawCNNKernel(kernel, x, y, size) {

    if (!Array.isArray(kernel)) {
        return;
    }

    let values = kernel.flat();

    let maxAbs = Math.max(
        0.000001,
        ...values.map(v => Math.abs(v))
    );

    let pixelSize = size / 3;

    for (let row = 0; row < 3; row++) {

        for (let col = 0; col < 3; col++) {

            let weight = kernel[row][col];

            let strength = Math.abs(weight) / maxAbs;

            if (weight >= 0) {

                // Positive weight = green
                fill(
                    35,
                    70 + strength * 170,
                    100
                );

            } else {

                // Negative weight = red
                fill(
                    80 + strength * 170,
                    45,
                    65
                );
            }

            stroke(35);

            rect(
                x + col * pixelSize,
                y + row * pixelSize,
                pixelSize,
                pixelSize
            );
        }
    }

    noStroke();
}


// ---------------- CNN VISUALIZATION ----------------

function drawCNN() {

    if (!result.feature_maps || !result.filters) {

        fill(245, 189, 98);
        textSize(15);

        text(
            "Missing CNN data. Run export_results.py",
            48,
            520
        );

        return;
    }

    let layer = cnnLayerSelect.value();

    let filterIndex = parseInt(
        cnnFilterSelect.value() || "0"
    );

    let featureMaps = result.feature_maps;
    let filters = result.filters;


    // Selected convolution output
    let convMap = featureMaps[layer][filterIndex];

    // Corresponding pooling output
    let poolLayer = layer === "conv1"
        ? "pool1"
        : "pool2";

    let poolMap = featureMaps[poolLayer][filterIndex];

    // Learned 3x3 kernel
    let kernel = filters[layer][filterIndex];


    // Title
    fill(239);
    textSize(20);

    text(
        "CNN convolution filters and feature maps",
        48,
        494
    );

    fill(168);
    textSize(13);

    text(
        "Choose a layer and filter to inspect what the CNN detects.",
        48,
        519
    );


    // Input image
    fill(235);
    textSize(15);
    text("Input image", 55, 565);

    drawDigit(
        result.test_image,
        55,
        585,
        5
    );


    // Kernel
    fill(235);
    textSize(15);
    text("Learned 3x3 kernel", 250, 565);

    drawCNNKernel(
        kernel,
        275,
        600,
        105
    );


    // Convolution feature map
    fill(235);
    textSize(15);
    text("After Conv + ReLU", 465, 565);

    drawFeatureMap(
        convMap,
        480,
        590,
        150
    );


    // Pooling feature map
    fill(235);
    textSize(15);
    text("After MaxPool", 690, 565);

    drawFeatureMap(
        poolMap,
        710,
        590,
        150
    );


    // Information about selected layer
    fill(225);
    textSize(16);

    let totalFilters = layer === "conv1" ? 32 : 64;

    text(
        "Selected: " + layer +
        " / Filter " + filterIndex,
        910,
        610
    );

    fill(170);
    textSize(13);

    text(
        "Total output channels: " + totalFilters,
        910,
        641
    );

    let convSize = layer === "conv1"
        ? "28 x 28"
        : "14 x 14";

    let poolSize = layer === "conv1"
        ? "14 x 14"
        : "7 x 7";

    text(
        "Conv feature map: " + convSize,
        910,
        672
    );

    text(
        "Pooled feature map: " + poolSize,
        910,
        700
    );


    // Color legend
    fill(69, 215, 153);
    rect(55, 776, 13, 13);

    fill(180);
    textSize(12);
    text("Positive kernel weight", 75, 788);

    fill(245, 92, 97);
    rect(270, 776, 13, 13);

    fill(180);
    text("Negative kernel weight", 290, 788);

    fill(150);

    text(
        "Feature maps are normalized for display.",
        690,
        788
    );

    // Note about Conv2 kernels
    if (layer === "conv2") {

        fill(160);
        textSize(11);

        text(
            "Conv2 kernel shows input channel 0 only.",
            910,
            727
        );
    }
}
