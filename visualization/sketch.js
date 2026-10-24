
let result;
let algorithmSelect;
let loadMessage = "Loading results...";


function setup() {

    createCanvas(1250, 650);
    noLoop();

    // Create algorithm selection dropdown
    algorithmSelect = createSelect();

    algorithmSelect.option("Naive Bayes", "naive_bayes");
    algorithmSelect.option("KNN", "knn");
    algorithmSelect.option("Linear Classifier", "linear_classifier");
    algorithmSelect.option("MLP (PyTorch)", "mlp");

    algorithmSelect.position(30, 65);
    algorithmSelect.changed(loadAlgorithm);

    loadAlgorithm();
}


// Load results from Python JSON files
function loadAlgorithm() {

    result = null;
    loadMessage = "Loading results...";
    redraw();

    let algorithm = algorithmSelect.value();
    let file = "";

    if (algorithm === "knn") {
        file = "result.json";
    }
    else if (algorithm === "naive_bayes") {
        file = "naive_bayes_result.json";
    }
    else if (algorithm === "linear_classifier") {
        file = "linear_classifier_result.json";
    }
    else if (algorithm === "mlp") {
        file = "mlp_result.json";
    }

    loadJSON(file, function(data) {

        // Ignore old responses if user switched algorithms
        if (algorithmSelect.value() !== algorithm) {
            return;
        }

        result = data;
        loadMessage = "";
        redraw();

    }, function(error) {

        if (algorithmSelect.value() !== algorithm) {
            return;
        }

        loadMessage = "Could not load " + file +
                      ". Run its Python file first.";
        redraw();
    });
}


// Main visualization
function draw() {

    background(22);

    fill(255);
    textSize(26);
    text("MNIST Digit Recognition", 30, 40);

    if (!result) {

        textSize(18);
        text(loadMessage, 40, 145);
        return;
    }

    // Draw test image
    fill(255);
    textSize(17);
    text("Test Image", 40, 140);

    drawDigit(result.test_image, 40, 155, 7);

    // Show prediction
    textSize(18);
    fill(255);
    text("Predicted Digit:", 300, 180);

    textSize(60);
    fill(70, 220, 130);
    text(result.prediction, 300, 250);

    // Show correct label
    textSize(18);
    fill(255);
    text("Actual Digit: " + result.actual, 300, 295);

    fill(
        result.prediction === result.actual
            ? "limegreen"
            : "red"
    );

    text(
        result.prediction === result.actual
            ? "Correct!"
            : "Incorrect",
        300, 330
    );

    // Display selected algorithm results
    if (algorithmSelect.value() === "knn") {
        drawNeighbors();
    }
    else if (algorithmSelect.value() === "naive_bayes") {
        drawProbabilities();
    }
    else if (algorithmSelect.value() === "linear_classifier") {
        drawLinearScores();
    }
    else if (algorithmSelect.value() === "mlp") {
        drawMLPScores();
    }
}


// KNN nearest neighbors visualization
function drawNeighbors() {

    fill(255);
    textSize(20);

    text(
        result.neighbors.length + " Nearest Neighbors",
        40,
        390
    );

    for (let i = 0; i < result.neighbors.length; i++) {

        let neighbor = result.neighbors[i];
        let x = 40 + i * 230;

        drawDigit(neighbor.pixels, x, 405, 4);

        fill(255);
        textSize(14);

        text("Class: " + neighbor.label, x, 545);

        text(
            "Distance: " + neighbor.distance.toFixed(2),
            x,
            570
        );
    }
}


// Naive Bayes class probabilities visualization
function drawProbabilities() {

    fill(255);
    textSize(20);
    text("Naive Bayes Class Probabilities", 590, 130);

    for (let digit = 0; digit < 10; digit++) {

        let value = result.probabilities[String(digit)] || 0;
        let y = 155 + digit * 41;

        fill(255);
        textSize(16);
        text(digit, 590, y + 17);

        // Background bar
        fill(55);
        rect(620, y, 430, 23);

        // Highlight prediction
        fill(
            digit === result.prediction
                ? color(70, 220, 130)
                : color(100, 155, 240)
        );

        rect(620, y, 430 * value / 100, 23);

        fill(255);
        text(value.toFixed(2) + "%", 1065, y + 17);
    }
}


// Linear Classifier class scores visualization
function drawLinearScores() {

    fill(255);
    textSize(20);
    text("Linear Classifier Scores", 590, 130);

    let scores = result.scores;

    let minScore = Math.min(...scores);
    let maxScore = Math.max(...scores);
    let range = maxScore - minScore;

    for (let digit = 0; digit < 10; digit++) {

        let score = scores[digit];
        let y = 155 + digit * 41;

        // Normalize score for visualization
        let barLength = range === 0
            ? 0
            : (score - minScore) / range * 430;

        fill(255);
        textSize(16);
        text(digit, 590, y + 17);

        // Background bar
        fill(55);
        rect(620, y, 430, 23);

        // Highlight predicted digit
        if (digit === result.prediction) {
            fill(70, 220, 130);
        } else {
            fill(100, 155, 240);
        }

        rect(620, y, barLength, 23);

        // Display actual score
        fill(255);
        text(score.toFixed(2), 1065, y + 17);
    }

    fill(190);
    textSize(15);

    text(
        "Test Accuracy: " + result.accuracy.toFixed(2) + "%",
        590, 595
    );
}


// MLP neural network class scores visualization
function drawMLPScores() {

    fill(255);
    textSize(20);
    text("MLP Neural Network Scores", 590, 130);

    let scores = result.scores;

    let minScore = Math.min(...scores);
    let maxScore = Math.max(...scores);
    let range = maxScore - minScore;

    for (let digit = 0; digit < 10; digit++) {

        let score = scores[digit];
        let y = 155 + digit * 41;

        // Normalize score for bar visualization
        let barLength = range === 0
            ? 0
            : (score - minScore) / range * 430;

        fill(255);
        textSize(16);
        text(digit, 590, y + 17);

        // Background bar
        fill(55);
        rect(620, y, 430, 23);

        // Highlight predicted digit
        if (digit === result.prediction) {
            fill(70, 220, 130);
        } else {
            fill(100, 155, 240);
        }

        rect(620, y, barLength, 23);

        // Display actual network score (logit)
        fill(255);
        text(score.toFixed(2), 1065, y + 17);
    }

    // Display MLP accuracy
    fill(190);
    textSize(15);

    text(
        "Test Accuracy: " + result.accuracy.toFixed(2) + "%",
        590, 595
    );

    // Show network architecture
    fill(190);
    textSize(14);

    text(
        "Architecture: 784 -> 256 -> 128 -> 10",
        590, 625
    );
}


// Draw flattened 28x28 MNIST image
function drawDigit(pixels, x, y, pixelSize) {

    noStroke();

    for (let row = 0; row < 28; row++) {

        for (let col = 0; col < 28; col++) {

            let index = row * 28 + col;
            let brightness = pixels[index] * 255;

            fill(brightness);

            rect(
                x + col * pixelSize,
                y + row * pixelSize,
                pixelSize,
                pixelSize
            );
        }
    }
}
