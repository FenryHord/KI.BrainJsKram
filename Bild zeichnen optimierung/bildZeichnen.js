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

const tmpCanv = document.getElementById("tmpCanv");
tmpCanv.width = WIDTH;
tmpCanv.height = HEIGHT;
const tmpCtx = tmpCanv.getContext("2d");

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

    let grauwert = originalImageData[index];

    let data = {
      input: [x / WIDTH, y / HEIGHT],
      output: [grauwert / 255]
    };
    trainingData.push(data);
  }
}

//paintImageFromTrainingData(trainingData);

function paintImageFromTrainingData(data, canv) {
  canv.clearRect(0, 0, WIDTH, HEIGHT);
  data.forEach(data => {
    let color = data.output * 255;
    canv.beginPath();
    canv.lineWidth = '1';
    canv.strokeStyle = 'rgb(' + color + ', ' + color + ', ' + color + ')';
    canv.rect(data.input[0] * WIDTH, data.input[1] * HEIGHT, 1, 1);
    canv.stroke();
  });
}

// 3. Neuronales Netz bauen
const net = new brain.NeuralNetwork({ hiddenLayers: [10, 10] });
net.setActivation('sigmoid');

const MAXITERATIONS = 40;
const MAXSAMPLES = 3000;
const conf = {
  iterations: MAXITERATIONS,
  log: false,
  learningRate: 0.3,
  errorThresh: 0.001
};

myTrain();

async function myTrain() {
 // trainStep();
  while(true) {
    await trainStep().then(data => paintCurrentNetImage(data));

  }
}

async function trainStep() {
  // 4. Training
  let currentTrainData = [];
  for (let i = 0; i < MAXSAMPLES; i++) {
    let rnd = Math.round(Math.random() * trainingData.length);
    currentTrainData.push(trainingData[rnd]);
  }

  paintImageFromTrainingData(currentTrainData, tmpCtx);
  return net.trainAsync(currentTrainData, conf);
}
let iterations = 0;
// 5. Validierung
async function paintCurrentNetImage(data) {
  iterations += data.iterations;
  document.getElementById('fortschritt').innerText =
   'error: ' + data.error + ' \n Iterationen: ' + iterations;
  for (let x = 0; x < WIDTH; x++) {
    for (let y = 0; y < HEIGHT; y++) {
      let grauwert = net.run([x / WIDTH, y / HEIGHT]);
      let color = grauwert * 255;
      ctx.beginPath();
      ctx.lineWidth = '1';
      ctx.strokeStyle = 'rgb(' + color + ', ' + color + ', ' + color + ')';
      ctx.rect(x, y, 1, 1);
      ctx.stroke();
    }
  }
}
