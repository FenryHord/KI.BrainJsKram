// Schritte bei Deep Learning

// 1. Rohdaten beschaffen
// 2. Rohdaten in Trainingsdaten umformatieren
// 3. Neuronales Netz bauen
// 4. Training
// 5. Validierung

// 1. Rohdaten beschaffen --> Pixel des Bilds
const original = document.getElementById('original');
const WIDTH = original.width;
const HEIGHT = original.height;
const canvas = document.getElementById('canvas');
canvas.width = WIDTH;
canvas.height = HEIGHT;
const ctx = canvas.getContext('2d');
ctx.drawImage(original, 0, 0);
const originalImageData = Array.from(
  ctx.getImageData(0, 0, WIDTH, HEIGHT).data
);
ctx.clearRect(0, 0, WIDTH, HEIGHT);
// originalImageData[0] --> 0ter Pixel, Rot-Wert
// originalImageData[1] --> 0ter Pixel, Grün-Wert
// originalImageData[2] --> 0ter Pixel, Blau-Wert
// originalImageData[3] --> 0ter Pixel, Alpha-Wert
// originalImageData[4] --> 1ter Pixel, Rot-Wert
// originalImageData[5] --> 1ter Pixel, Grün-Wert
// originalImageData[6] --> 1ter Pixel, Blau-Wert
// originalImageData[7] --> 1ter Pixel, Alpha-Wert
// originalImageData[8] --> 2ter Pixel, Rot-Wert
// usw...

// 2. Rohdaten in Trainingsdaten umformatieren
// hier: [
//           {input: [34/WIDTH, 145/HEIGHT], output: [255/255]},
//           {input: [300/WIDTH, 400/HEIGHT], output: [12/255]}
//       ]
let trainingData = [];
for (let x = 0; x < WIDTH; x++) {
  for (let y = 0; y < HEIGHT; y++) {
    let index = x * 4 + y * 4 * WIDTH;

    let red = originalImageData[index];
    let green = originalImageData[index + 1];
    let blue = originalImageData[index + 2];

    let data = {
      input: [x / WIDTH, y / HEIGHT],
      output: [red / 255, green / 255, blue / 255]
    };
    trainingData.push(data);
  }
}

/* paintImageFromTrainingData();

function paintImageFromTrainingData() {
  trainingData.forEach(data => {
    let red = data.output[0] * 255;
    let green = data.output[1] * 255;
    let blue = data.output[2] * 255;

    ctx.beginPath();
    ctx.lineWidth = '1';
    ctx.strokeStyle = 'rgb(' + red + ', ' + green + ', ' + blue + ')';
    ctx.rect(data.input[0] * WIDTH, data.input[1] * HEIGHT, 1, 1);
    ctx.stroke();
  });
} */

// 3. Neuronales Netz bauen
const net = new brain.NeuralNetwork({ hiddenLayers: [5] });
net.setActivation('sigmoid');

console.log('train');
train();

function randomTrainingData() {
  let currTrainingData = [];
  trainingData.forEach(data => {
    if (Math.random() < 0.99) {
      currTrainingData.push(data);
    }
  });
  //console.log(currTrainingData.length + ' samples');
  return currTrainingData;
}

async function train() {

  const ITERATIONS = 20;
  let currIter = 0;
  for (let i = 0; i < 1000; i++) {
    // 4. Training
    let currData = randomTrainingData();
    console.log("laenge" +  currData.length);
    await net.trainAsync(currData, {
      iterations: ITERATIONS,
      //callback: paintCurrentNetImage,
      //callbackPeriod: 20,
      log: true,
      logPeriod: 20
    });

    console.log(currIter += 20);
    if (i % 10 == 0) {
      paintCurrentNetImage(i);
    }
  }
}
//paintCurrentNetImage();

// 5. Validierung
function paintCurrentNetImage(i) {
  for (let x = 0; x < WIDTH; x++) {
    for (let y = 0; y < HEIGHT; y++) {
      let output = net.run([x / WIDTH, y / HEIGHT]);
      let red = output[0] * 255;
      let green = output[1] * 255;
      let blue = output[2] * 255;
      ctx.beginPath();
      ctx.lineWidth = '1';
      ctx.strokeStyle = 'rgb(' + red + ', ' + green + ', ' + blue + ')';
      ctx.rect(x, y, 1, 1);
      ctx.stroke();
    }
  }
  console.log('drawn' + i);
}
