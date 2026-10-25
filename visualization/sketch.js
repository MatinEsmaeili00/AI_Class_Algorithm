
 // MNIST visualizer
// This file only reads exported JSON. It never trains models.

let result = null;
let algorithmSelect, digitSelect, neuronSelect;
let message = "Loading results...";
let currentRequest = 0;

const FILES = {
    knn: "result.json",
    naive_bayes: "naive_bayes_result.json",
    linear_classifier: "linear_classifier_result.json",
    mlp: "mlp_result.json"
};


// ---------------- SETUP ----------------

function setup() {

    createCanvas(1280, 850);
    pixelDensity(1);
    noLoop();
    textFont("Arial");

    // Algorithm selection
    algorithmSelect = createSelect();
    algorithmSelect.position(30, 66);

    algorithmSelect.option("KNN", "knn");
    algorithmSelect.option("Naive Bayes", "naive_bayes");
    algorithmSelect.option("Linear Classifier", "linear_classifier");
    algorithmSelect.option("MLP (PyTorch)", "mlp");

    algorithmSelect.changed(loadAlgorithm);

    // Linear Classifier digit selection
    digitSelect = createSelect();
    digitSelect.position(920, 424);

    for (let i = 0; i < 10; i++) {
        digitSelect.option("Digit " + i, String(i));
    }

    digitSelect.changed(() => redraw());

    // MLP neuron selection
    neuronSelect = createSelect();
    neuronSelect.position(920, 424);
    neuronSelect.changed(() => redraw());

    loadAlgorithm();
}


// ---------------- LOAD JSON ----------------

function loadAlgorithm() {

    const algorithm = algorithmSelect.value();
    const requestId = ++currentRequest;

    message = "Loading " + algorithm + "...";
    result = null;

    digitSelect.hide();
    neuronSelect.hide();

    redraw();

    // Prevent browser from using outdated JSON
    loadJSON(
        FILES[algorithm] + "?r=" + Date.now(),

        function(data) {

            if (requestId !== currentRequest) return;

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

                const n = data.activations.hidden1.length;

                for (let i = 0; i < n; i++) {
                    neuronSelect.option(
                        "Hidden 1 neuron " + i,
                        String(i)
                    );
                }

                neuronSelect.value("0");
                neuronSelect.show();
            }

            redraw();
        },

        function(error) {

            if (requestId !== currentRequest) return;

            message =
                "Could not load " + FILES[algorithm] +
                ". Run the exporter first.";

            redraw();
        }
    );
}


// ---------------- MAIN DRAW ----------------

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

    // Main panels
    drawPanel(25, 115, 410, 330);
    drawPanel(455, 115, 800, 330);
    drawPanel(25, 460, 1230, 360);

    // Test image
    fill(238);
    textSize(19);
    text("Test image", 45, 150);

    if (
        result.test_image &&
        result.test_image.length >= 784
    ) {
        drawDigit(result.test_image, 50, 170, 6.8);
    }

    // Prediction
    fill(166);
    textSize(16);
    text("Prediction", 274, 205);

    fill(64, 217, 149);
    textSize(60);
    text(result.prediction, 272, 276);

    // Actual label
    fill(230);
    textSize(15);
    text("Actual: " + result.actual, 274, 310);

    if (result.prediction === result.actual) {
        fill(64, 217, 149);
        text("Correct prediction", 274, 339);
    } else {
        fill(249, 111, 111);
        text("Incorrect prediction", 274, 339);
    }

    // Accuracy
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

    // Draw selected algorithm
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
}


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

            const index = row * 28 + col;

            const pixel = constrain(
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

    const scores = result.scores;

    if (!Array.isArray(scores) || scores.length !== 10) {
        return;
    }

    const minV = Math.min(...scores);
    const maxV = Math.max(...scores);
    const range = maxV - minV || 1;

    for (let digit = 0; digit < 10; digit++) {

        const y = 170 + digit * 24;

        fill(200);
        textSize(13);
        text(String(digit), 480, y + 12);

        // Bar background
        fill(54, 62, 77);
        rect(510, y, 600, 17, 4);

        // Normalize score for display
        const barWidth = Math.max(
            2,
            (scores[digit] - minV) / range * 600
        );

        if (digit === result.prediction) {
            fill(67, 216, 151);
        } else {
            fill(110, 160, 245);
        }

        rect(510, y, barWidth, 17, 4);

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

        const raw = result.probabilities
            ? Number(result.probabilities[String(digit)])
            : 0;

        const value = Number.isFinite(raw) ? raw : 0;

        const y = 170 + digit * 24;

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

    const neighbors = Array.isArray(result.neighbors)
        ? result.neighbors
        : [];

    textSize(14);
    fill(180);

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

        const neighbor = neighbors[i];
        const x = 48 + i * 170;

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
        "Displaying up to 7 neighbors from the exported JSON.",
        48,
        787
    );
}


// ---------------- WEIGHT HEATMAP ----------------

// Convert 784 learned weights into a 28x28 heatmap.
// Green = positive weights
// Red = negative weights

function drawWeightHeatmap(weights, x, y, scale) {

    if (
        !Array.isArray(weights) ||
        weights.length !== 784
    ) {
        return;
    }

    const limit = Math.max(
        0.000001,
        ...weights.map(v => Math.abs(v))
    );

    noStroke();

    for (let i = 0; i < 784; i++) {

        const w = weights[i];

        const strength = Math.sqrt(
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

        const col = i % 28;
        const row = Math.floor(i / 28);

        rect(
            x + col * scale,
            y + row * scale,
            scale + 0.15,
            scale + 0.15
        );
    }
}


// ---------------- LINEAR CLASSIFIER WEIGHTS ----------------

function drawLinearWeights() {

    fill(239);
    textSize(20);

    text(
        "Learned weights: 784 input pixels -> 10 output classes",
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
            "No weights found. Run visualization/export_results.py",
            48,
            555
        );

        return;
    }

    fill(165);
    textSize(14);
    text("Select output class:", 768, 438);

    const digit = constrain(
        parseInt(digitSelect.value() || "0"),
        0,
        9
    );

    const weights = result.weights[digit];

    // Show actual learned weights
    drawWeightHeatmap(weights, 65, 525, 8.5);

    fill(229);
    textSize(17);

    text("Output digit " + digit, 365, 565);

    fill(172);
    textSize(14);

    text("Each square = one pixel weight", 365, 596);

    // Color legend
    fill(67, 216, 151);
    rect(365, 627, 14, 14);

    fill(203);
    text("Positive weight", 388, 639);

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
        "Each of the 10 outputs connects to all 784 inputs.",
        740,
        587
    );

    text(
        "Heatmap uses weights exported from NumPy.",
        740,
        612
    );
}


// ---------------- MLP NETWORK ----------------

// Shows a sampled version of the real network.
// All displayed weights and activations come from Python.

function drawMLPNetwork() {

    const A = result.activations;
    const W = result.weights;

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
            "Missing activations/weights. Run visualization/export_results.py",
            48,
            520
        );

        return;
    }

    fill(239);
    textSize(20);

    text(
        "MLP architecture and selected neuron weights",
        48,
        493
    );

    fill(171);
    textSize(13);

    text(
        "Visible neurons are sampled. Lines show actual learned weights.",
        48,
        516
    );

    // Choose a smaller number of visible neurons
    const inputIndices = sampleIndices(784, 11);
    const h1Indices = sampleIndices(A.hidden1.length, 12);
    const h2Indices = sampleIndices(A.hidden2.length, 10);

    const outIndices = Array.from(
        { length: 10 },
        (_, i) => i
    );

    const layers = [
        inputIndices,
        h1Indices,
        h2Indices,
        outIndices
    ];

    const xs = [105, 355, 605, 845];

    // Calculate each neuron position
    const yPositions = layers.map(layer =>
        layer.map(
            (_, i) =>
                568 +
                i * (185 / Math.max(1, layer.length - 1))
        )
    );

    const weights = [
        W.layer1,
        W.layer2,
        W.layer3
    ];

    // -------- DRAW CONNECTIONS --------

    for (let layer = 0; layer < 3; layer++) {

        for (
            let j = 0;
            j < layers[layer + 1].length;
            j++
        ) {

            for (
                let i = 0;
                i < layers[layer].length;
                i++
            ) {

                const value =
                    weights[layer]
                    [layers[layer + 1][j]]
                    [layers[layer][i]];

                const alpha = constrain(
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

    // -------- DRAW NEURONS --------

    noStroke();

    const values = [
        result.test_image,
        A.hidden1,
        A.hidden2,
        result.scores
    ];

    for (let layer = 0; layer < 4; layer++) {

        const layerValues = values[layer];

        const maxAbs = Math.max(
            0.001,
            ...layerValues.map(v => Math.abs(v))
        );

        for (
            let j = 0;
            j < layers[layer].length;
            j++
        ) {

            const v = layerValues[layers[layer][j]];

            const normalized = constrain(
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

    // -------- LAYER LABELS --------

    fill(219);
    textSize(13);

    const labels = [
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

    // -------- SELECTED HIDDEN NEURON --------

    const selected = constrain(
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

    text(
        "28x28 incoming weights",
        946,
        576
    );

    // Display the selected neuron's 784 input weights
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

// Select evenly spaced neurons so the network
// remains readable without displaying all 784 inputs.

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
