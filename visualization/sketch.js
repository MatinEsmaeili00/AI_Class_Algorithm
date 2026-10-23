let result;
let algorithmSelect;
let loadMessage = "Loading results...";

function setup() {
    createCanvas(1250, 620);
    noLoop();

    algorithmSelect = createSelect();
    algorithmSelect.option("Naive Bayes", "naive_bayes");
    algorithmSelect.option("KNN", "knn");
    algorithmSelect.position(30, 65);
    algorithmSelect.changed(loadAlgorithm);

    loadAlgorithm();
}

// Read the JSON written by the selected Python algorithm
function loadAlgorithm() {
    result = null;
    loadMessage = "Loading results...";
    redraw();

    let file = algorithmSelect.value() === "knn"
        ? "result.json"
        : "naive_bayes_result.json";

    loadJSON(file, function(data) {
        result = data;
        loadMessage = "";
        redraw();
    }, function(error) {
        loadMessage = "Could not load " + file + ". Run its Python file first.";
        redraw();
    });
}

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

    fill(255);
    textSize(17);
    text("Test Image", 40, 140);
    drawDigit(result.test_image, 40, 155, 7);

    textSize(18);
    fill(255);
    text("Predicted Digit:", 300, 180);
    textSize(60);
    fill(70, 220, 130);
    text(result.prediction, 300, 250);

    textSize(18);
    fill(255);
    text("Actual Digit: " + result.actual, 300, 295);
    fill(result.prediction === result.actual ? "limegreen" : "red");
    text(result.prediction === result.actual ? "Correct!" : "Incorrect", 300, 330);

    if (algorithmSelect.value() === "knn") {
        drawNeighbors();
    } else {
        drawProbabilities();
    }
}

// Keep the original KNN nearest-neighbors visualization
function drawNeighbors() {
    fill(255);
    textSize(20);
    text(result.neighbors.length + " Nearest Neighbors", 40, 390);

    for (let i = 0; i < result.neighbors.length; i++) {
        let neighbor = result.neighbors[i];
        let x = 40 + i * 230;

        drawDigit(neighbor.pixels, x, 405, 4);
        fill(255);
        textSize(14);
        text("Class: " + neighbor.label, x, 545);
        text("Distance: " + neighbor.distance.toFixed(2), x, 570);
    }
}

// Display class probabilities learned by Naive Bayes
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

        fill(55);
        rect(620, y, 430, 23);
        fill(digit === result.prediction ? color(70, 220, 130) : color(100, 155, 240));
        rect(620, y, 430 * value / 100, 23);

        fill(255);
        text(value.toFixed(2) + "%", 1065, y + 17);
    }
}

// Draw a flattened 28x28 MNIST image
function drawDigit(pixels, x, y, pixelSize) {
    noStroke();
    for (let row = 0; row < 28; row++) {
        for (let col = 0; col < 28; col++) {
            let index = row * 28 + col;
            let brightness = pixels[index] * 255;
            fill(brightness);
            rect(x + col * pixelSize, y + row * pixelSize,
                pixelSize, pixelSize);
        }
    }
}
