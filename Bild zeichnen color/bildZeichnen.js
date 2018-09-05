// Schritte bei Deep Learning

// 1. Rohdaten beschaffen
// 2. Rohdaten in Trainingsdaten umformatieren
// 3. Neuronales Netz bauen
// 4. Training
// 5. Validierung

// 1. Rohdaten beschaffen --> Pixel des Bilds
const MAXITERATIONS = 20;
const MAXSAMPLES = 3000;
const conf = {
  iterations: MAXITERATIONS,
  log: false,
  //learningRate: 0.3,
  errorThresh: 0.000005
};

let iterations = 0;
let trainStepCNT = 0;
let drawStepCNT = 0;


const original = document.getElementById('original');
const WIDTH = original.width;
const HEIGHT = original.height;
const canvas = document.getElementById('canvas');
canvas.width = WIDTH;
canvas.height = HEIGHT;
const ctx = canvas.getContext('2d');
ctx.drawImage(original, 0, 0);

const originalImageData = ctx.getImageData(0, 0, WIDTH, HEIGHT);
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
    let r = originalImageData.data[index];
    let b = originalImageData.data[index + 2];
    let g = originalImageData.data[index + 1];

    let data = {
      input: [x / WIDTH, y / HEIGHT],
      output: [r / 255, g / 255, b / 255]
    };
    trainingData.push(data);
  }
}

//paintImageFromTrainingData(trainingData);

function paintImageFromTrainingData(data, canv) {

  console.time("draw1");
  originalImageData.data.map((pixel, index) => {
    originalImageData.data[index] = 0;
  });
  let index1 = 0;
  data.forEach(element => {
    index1 = (Math.round(element.input[0] * WIDTH) + Math.round(element.input[1] * HEIGHT) * WIDTH) * 4;
    originalImageData.data[index1] = element.output[0] * 225;
    originalImageData.data[index1 + 1] = element.output[1] * 255;
    originalImageData.data[index1 + 2] = element.output[2] * 255;
    originalImageData.data[index1 + 3] = 255;
  });

  canv.putImageData(originalImageData, 0, 0);
  console.timeEnd("draw1");

  /*

    console.time("draw2");
    canv.clearRect(0, 0, WIDTH, HEIGHT);
    data.forEach((pixel) => {
      let r = pixel.output[0] * 255;
      let g = pixel.output[1] * 255;
      let b = pixel.output[2] * 255;
      let a = 1;
      testClamped.push(r, g, b, a);
      canv.beginPath();
      canv.lineWidth = '1';
      canv.strokeStyle = 'rgb(' + r + ', ' + g + ', ' + b + ')';
      canv.rect(pixel.input[0] * WIDTH, pixel.input[1] * HEIGHT, 1, 1);
      canv.stroke();
    });
    console.timeEnd("draw2");*/
}

// 3. Neuronales Netz bauen
const net = new brain.NeuralNetwork({ hiddenLayers: [20, 20] });
net.setActivation('sigmoid');


myTrain();

function myTrain() {
  if (trainStepCNT - drawStepCNT < 1) {

    trainStepCNT++;
    trainStep().then(
      (data) => { paintCurrentNetImage(data) },
      (error) => console.error(error));
  }
  setTimeout(myTrain, 5);
}

async function trainStep() {
  let currentTrainData = [];
  let rnd = 0;
  let dat = undefined;
  //currentTrainData = trainingData;
  for (let i = 0; i < MAXSAMPLES; i++) {
    rnd = Math.round(Math.random() * trainingData.length);

    dat = trainingData[rnd];
    if (dat != undefined) {
      currentTrainData.push(dat);
    }
  }/*
  let proz = MAXSAMPLES / trainingData.length;
  trainingData.forEach((element) => {
    rnd = Math.random();
    if(rnd < proz){
      currentTrainData.push(element);
    }
  });
  console.log("länge", currentTrainData.length, "proz", proz);*/

  paintImageFromTrainingData(currentTrainData, tmpCtx);

  // 4. Training
  return net.trainAsync(currentTrainData, conf);

}
// 5. Validierung
async function paintCurrentNetImage(data) {
  drawStepCNT++;

  console.count("draw");
  iterations += data.iterations;
  document.getElementById('fortschritt').innerText =
    'error: ' + data.error + ' \n Iterationen: ' + iterations;
  console.time("runDraw");
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
  console.timeEnd("runDraw");


}
