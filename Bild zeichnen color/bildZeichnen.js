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

    let r = originalImageData[index];
    let g = originalImageData[index + 1];
    let b = originalImageData[index + 2];

    let data = {
      input: [x / WIDTH, y / HEIGHT],
      output: [r / 255, g / 255, b / 255]
    };
    trainingData.push(data);
  }
}

//paintImageFromTrainingData(trainingData);

function paintImageFromTrainingData(data, canv) {
  canv.clearRect(0, 0, WIDTH, HEIGHT);
  data.forEach(data => {
    let r = data.output[0] * 255;
    let g = data.output[1] * 255;
    let b = data.output[2] * 255;

    canv.beginPath();
    canv.lineWidth = '1';
    canv.strokeStyle = 'rgb(' + r + ', ' + g + ', ' + b + ')';
    canv.rect(data.input[0] * WIDTH, data.input[1] * HEIGHT, 1, 1);
    canv.stroke();
  });
}

// 3. Neuronales Netz bauen
const net = new brain.NeuralNetwork({ hiddenLayers: [20, 20] });
net.setActivation('sigmoid');

const MAXITERATIONS = 25;
const MAXSAMPLES = 5000;
const conf = {
  iterations: MAXITERATIONS,
  log: false,
  learningRate: 0.3,
  errorThresh: 0.001
};

let iterations = 0;
let trainStepCNT = 0;
let drawStepCNT = 0;
myTrain();

function myTrain() {
  if (trainStepCNT - drawStepCNT < 1) {

    trainStepCNT++;
    trainStep().then((data) => { paintCurrentNetImage(data) });

  }
  setTimeout(myTrain, 5);
}

async function trainStep() {
  // 4. Training
  let currentTrainData = [];
  let rnd = 0;
  for (let i = 0; i < MAXSAMPLES; i++) {
    rnd = Math.round(Math.random() * trainingData.length);
    currentTrainData.push(trainingData[rnd]);
  }
  console.log("cnt ", currentTrainData.length);


  paintImageFromTrainingData(currentTrainData, tmpCtx);
  return net.trainAsync(currentTrainData, conf);
}
// 5. Validierung
async function paintCurrentNetImage(data) {
  drawStepCNT++;

  console.count("draw");
  iterations += data.iterations;
  document.getElementById('fortschritt').innerText =
    'error: ' + data.error + ' \n Iterationen: ' + iterations;
  for (let x = 0; x < WIDTH; x++) {
    for (let y = 0; y < HEIGHT; y++) {
      let rgb = net.run([x / WIDTH, y / HEIGHT]);
      let r = rgb[0] * 255;
      let g = rgb[1] * 255;
      let b = rgb[2] * 255;
      ctx.beginPath();
      ctx.lineWidth = '1';
      ctx.strokeStyle = 'rgb(' + r + ', ' + g + ', ' + b + ')';
      ctx.rect(x, y, 1, 1);
      ctx.stroke();
    }
  }
}
