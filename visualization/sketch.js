
let result;

// Load the Python prediction data
function preload() {
    result = loadJSON("result.json");
}

function setup() {
    createCanvas(800, 550);
    noLoop();
}

function draw() {
    background(22);

    fill(255);
    textSize(26);
    text("KNN Digit Recognition", 30, 45);

    // Draw the test image
    textSize(17);
    text("Test Image", 40, 95);

    drawDigit(result.test_image, 40, 110, 7);

    // Prediction information
    textSize(18);
    fill(255);
    text("Predicted Digit:", 300, 140);

    textSize(60);
    fill(70, 220, 130);
    text(result.prediction, 300, 210);

    textSize(18);
    fill(255);
    text("Actual Digit: " + result.actual, 300, 255);

    let correct = result.prediction === result.actual;

    fill(correct ? "limegreen" : "red");
    text(correct ? "Correct!" : "Incorrect", 300, 290);

    // Draw nearest neighbors
    fill(255);
    textSize(20);
    text("3 Nearest Neighbors", 40, 350);

    for (let i = 0; i < result.neighbors.length; i++) {

        let neighbor = result.neighbors[i];
        let x = 40 + i * 230;

        drawDigit(neighbor.pixels, x, 365, 4);

        fill(255);
        textSize(14);
        text("Class: " + neighbor.label, x, 505);
        text("Distance: " + neighbor.distance.toFixed(2), x, 530);
    }
}

// Draw a flattened 28x28 MNIST image
function drawDigit(pixels, x, y, pixelSize) {

    noStroke();

    for (let row = 0; row < 28; row++) {
        for (let col = 0; col < 28; col++) {

            let index = row * 28 + col;

            // Convert normalized [0,1] to grayscale [0,255]
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
